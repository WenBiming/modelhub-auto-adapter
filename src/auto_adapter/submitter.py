"""提交与调度层（spec §4.5）。M4 实现。

限流参数只能调小不能绕过（CLAUDE.md 铁律）。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from .models import AddTaskRequest, TaskStatus, ensure_utc
from .platform_client import (
    PlatformClient,
    PlatformClientError,
    escalate_if_credential_error,
)
from .settings import Settings
from .storage import Storage

logger = logging.getLogger(__name__)
EST_ADAPT_HOURS = 2
_MAX_DT = datetime.max.replace(tzinfo=timezone.utc)


def drain(storage: Storage, client: PlatformClient, settings: Settings, now: datetime | None = None) -> int:
    """提交 QUEUED 任务，返回本 tick 实际提交数。

    - kill_switch 打开时直接返回 0；
    - 排序键 (priority, bounty_deadline, model_id)；
    - 令牌桶：max_submits_per_minute；在途 (PENDING+RUNNING) ≥ max_inflight 时停止；
    - 悬赏剩余时间 < 预估适配时长×2 仍未提交 → 标记 ABANDONED 并告警；
    - 成功：写回 task_id/submit_time/status=PENDING；
    - 平台明确拒绝（业务码非 50000/50001）：退回 QUEUED，下个 tick 重试；
    - 结果不可知（传输层异常/HTTPError/50000/50001）：留在 PENDING + task_id=None，
      由 monitor 对账（宁漏勿重，绝不重复提交）。

    Safety: Mark PENDING before network call, revert ONLY on definite rejection,
    set kill_switch on update_task failure.
    """
    now = now or datetime.now(timezone.utc)
    if storage.kill_switch():
        logger.warning("kill switch on; submission paused")
        return 0

    # 排序键里的 deadline 也要归一化：naive 与 aware 混排会在比较时抛 TypeError。
    queued = sorted(storage.tasks_by_status(TaskStatus.QUEUED),
                    key=lambda r: (r.priority, ensure_utc(r.bounty_deadline) or _MAX_DT, r.model_id))

    # First pass: mark expiring bounties as ABANDONED (independent of budget)
    for record in queued:
        deadline = ensure_utc(record.bounty_deadline)
        if deadline is not None and \
                deadline - now < timedelta(hours=2 * EST_ADAPT_HOURS):
            record.status = TaskStatus.ABANDONED
            storage.update_task(record)
            logger.warning("bounty %s abandoned: deadline too close", record.model_id)

    # Second pass: submit with rate limit
    inflight = len(storage.tasks_by_status(TaskStatus.PENDING, TaskStatus.RUNNING))
    budget = min(settings.max_submits_per_minute, settings.max_inflight - inflight)
    if budget <= 0:
        return 0

    submitted = 0
    for record in queued:
        # Skip if already abandoned in first pass
        if record.status != TaskStatus.QUEUED:
            continue
        if submitted >= budget:
            break

        req = AddTaskRequest(
            model_address=record.model_url, task_type=record.task_type,
            target_gpu=record.target_gpu, framework=record.framework,
            config_params=record.config_params, strategy_id=settings.strategy_id)

        if settings.dry_run:
            # 演练模式：完整走到"请求已组装好"这一步，然后停手。记录保持 QUEUED、
            # task_id 保持 None——绝不能伪造 PENDING/task_id，否则 monitor 对账时
            # 会在平台侧找不到它而误判"任务被清理"并拉闸。
            # 每个 (model_id, target_gpu) 只详细打印一次，避免每分钟刷屏。
            key = f"dryrun_logged:{record.model_id}@{record.target_gpu}"
            if not storage.get_counter(key):
                storage.set_counter(key, 1)
                logger.info("DRY RUN would submit: %s", json.dumps({
                    "modelAddress": req.model_address, "taskType": req.task_type,
                    "targetGpu": req.target_gpu, "framework": req.framework,
                    "strategyId": req.strategy_id, "configParams": req.config_params,
                }, ensure_ascii=False))
            submitted += 1  # 占用限流预算，让演练的节奏与真实运行一致
            continue

        # Mark PENDING BEFORE network call (宁漏勿重: prevent duplicate submission)
        record.status = TaskStatus.PENDING
        record.submit_time = now
        try:
            storage.update_task(record)
        except Exception:
            logger.exception("failed to mark %s PENDING before submit", record.model_id)
            # Revert to QUEUED so retry happens next tick
            record.status = TaskStatus.QUEUED
            record.submit_time = None
            try:
                storage.update_task(record)
            except Exception:
                logger.error("failed to revert %s to QUEUED after update_task failure", record.model_id)
            continue

        try:
            record.task_id = client.add_task(req)
        except PlatformClientError as e:
            if e.is_transient:
                # 50000/50001：平台内部异常，任务是否已建单不可知。退回 QUEUED 会在
                # 下个 tick 对同一 (model_id, target_gpu) 再提交一次——平台一旦判定
                # 重复提交会清空账号全部任务。留在 PENDING + task_id=None（Task 7 的
                # 安全态，monitor.poll 会用 modelId/gpuType 去平台列表里认领回来）。
                logger.error(
                    "submit outcome UNKNOWN for %s (transient platform error %s); "
                    "record left PENDING with task_id=None for reconciliation",
                    record.model_id, e)
                continue

            # 明确拒绝（40100/40101/40400 等业务码）：平台没有建单，退回 QUEUED 安全。
            record.status = TaskStatus.QUEUED
            record.submit_time = None
            try:
                storage.update_task(record)
            except Exception:
                logger.error("failed to revert %s to QUEUED after PlatformClientError", record.model_id)

            if escalate_if_credential_error(storage, e):
                return submitted
            logger.warning("submit failed for %s: %s", record.model_id, e)
            continue
        except Exception:
            # 传输层异常（ReadTimeout/ConnectionError）与 HTTPError：请求可能已经
            # 到达平台并建单，只是响应没回来。结果不可知 → 同样留在 PENDING。
            logger.error(
                "submit outcome UNKNOWN for %s (transport/HTTP error); record left PENDING "
                "with task_id=None for reconciliation", record.model_id, exc_info=True)
            continue

        # Persist task_id after successful submission
        try:
            storage.update_task(record)
        except Exception:
            logger.error("failed to persist task_id %d for %s (MANUAL RECONCILIATION REQUIRED)",
                        record.task_id, record.model_id)
            storage.set_kill_switch(True, f"update_task failed for {record.model_id} with task_id {record.task_id}")
            # Storage is unreliable; stop all submissions in this tick
            submitted += 1
            return submitted

        submitted += 1
    return submitted
