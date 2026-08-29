"""提交与调度层（spec §4.5）。M4 实现。

限流参数只能调小不能绕过（CLAUDE.md 铁律）。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from .models import ACTIVE_STATUSES, AddTaskRequest, TaskStatus
from .platform_client import PlatformClient, PlatformClientError
from .settings import Settings
from .storage import Storage

logger = logging.getLogger(__name__)
EST_ADAPT_HOURS = 2
_MAX_DT = datetime.max.replace(tzinfo=timezone.utc)


def drain(storage: Storage, client: PlatformClient, settings: Settings, now: datetime | None = None) -> int:
    """提交 QUEUED 任务，返回本 tick 实际提交数。

    - kill_switch 打开时直接返回 0；
    - 排序键 (priority, bounty_deadline, discovered_at)；
    - 令牌桶：max_submits_per_minute；在途 (PENDING+RUNNING) ≥ max_inflight 时停止；
    - 悬赏剩余时间 < 预估适配时长×2 仍未提交 → 标记 ABANDONED 并告警；
    - 成功：写回 task_id/submit_time/status=PENDING；失败：保持 QUEUED 记录原因。
    """
    now = now or datetime.now(timezone.utc)
    if storage.kill_switch():
        logger.warning("kill switch on; submission paused")
        return 0
    inflight = len(storage.tasks_by_status(TaskStatus.PENDING, TaskStatus.RUNNING))
    budget = min(settings.max_submits_per_minute, settings.max_inflight - inflight)
    if budget <= 0:
        return 0
    queued = sorted(storage.tasks_by_status(TaskStatus.QUEUED),
                    key=lambda r: (r.priority, r.bounty_deadline or _MAX_DT, r.model_id))
    submitted = 0
    for record in queued:
        if submitted >= budget:
            break
        if record.bounty_deadline is not None and \
                record.bounty_deadline - now < timedelta(hours=2 * EST_ADAPT_HOURS):
            record.status = TaskStatus.ABANDONED
            storage.update_task(record)
            logger.warning("bounty %s abandoned: deadline too close", record.model_id)
            continue
        req = AddTaskRequest(
            model_address=record.model_url, task_type=record.task_type,
            target_gpu=record.target_gpu, framework=record.framework,
            config_params=record.config_params, strategy_id=settings.strategy_id)
        try:
            record.task_id = client.add_task(req)
        except PlatformClientError as e:
            if e.is_credential_error:
                storage.set_kill_switch(True, str(e))
                return submitted
            logger.warning("submit failed for %s: %s", record.model_id, e)
            continue
        except Exception:
            logger.exception("submit failed for %s", record.model_id)
            continue
        record.submit_time = now
        record.status = TaskStatus.PENDING
        storage.update_task(record)
        submitted += 1
    return submitted
