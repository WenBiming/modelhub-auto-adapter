"""提交与调度层（spec §4.5）。M4 实现。

限流参数只能调小不能绕过（CLAUDE.md 铁律）。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from .models import AddTaskRequest, TaskStatus
from .platform_client import PlatformClient, PlatformClientError
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
    - 成功：写回 task_id/submit_time/status=PENDING；失败：保持 QUEUED 记录原因。

    Safety: Mark PENDING before network call, revert on failure, set kill_switch on update_task failure.
    """
    now = now or datetime.now(timezone.utc)
    if storage.kill_switch():
        logger.warning("kill switch on; submission paused")
        return 0

    queued = sorted(storage.tasks_by_status(TaskStatus.QUEUED),
                    key=lambda r: (r.priority, r.bounty_deadline or _MAX_DT, r.model_id))

    # First pass: mark expiring bounties as ABANDONED (independent of budget)
    for record in queued:
        if record.bounty_deadline is not None and \
                record.bounty_deadline - now < timedelta(hours=2 * EST_ADAPT_HOURS):
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
            # Revert to QUEUED on platform error
            record.status = TaskStatus.QUEUED
            record.submit_time = None
            try:
                storage.update_task(record)
            except Exception:
                logger.error("failed to revert %s to QUEUED after PlatformClientError", record.model_id)

            if e.is_credential_error:
                storage.set_kill_switch(True, str(e))
                return submitted
            logger.warning("submit failed for %s: %s", record.model_id, e)
            continue
        except Exception:
            # Revert to QUEUED on any other error
            record.status = TaskStatus.QUEUED
            record.submit_time = None
            try:
                storage.update_task(record)
            except Exception:
                logger.error("failed to revert %s to QUEUED after exception", record.model_id)
            logger.exception("submit failed for %s", record.model_id)
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
