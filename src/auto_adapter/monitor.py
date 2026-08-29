"""监控与日志分析层（spec §4.6）。M5 实现。"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from . import rules
from .models import TaskStatus
from .platform_client import PlatformClient
from .settings import Settings
from .storage import Storage

logger = logging.getLogger(__name__)


def poll(storage: Storage, client: PlatformClient, settings: Settings,
         now: datetime | None = None) -> None:
    """对账并更新本地任务表：

    - list_my_tasks（onlyMine=true）分页拉取，以平台 status/verifyResult 为准更新 TaskRecord；
    - 本地记录 task_id 为 None（PENDING 意图已落盘但 add_task 未完成）→ NEEDS_HUMAN，
      不算平台侧消失，不触发 kill switch（该告警只保留给平台已接单又丢失的任务）；
    - 本地在途但平台侧消失的任务 → ABANDONED + set_kill_switch(True)
      （可能被违规清理，最高级别告警，暂停提交待人工确认）；
    - 失败任务拉日志存入 record.last_log，留给 failure 层分类；
    - SUCCESS 时清零 consecutive_engine_failures 计数（成功即重置熔断）；
    - PENDING/RUNNING 超过 task_timeout_hours → 标记 TIMEOUT。
    """
    now = now or datetime.now(timezone.utc)
    records = storage.tasks_by_status(TaskStatus.PENDING, TaskStatus.RUNNING)
    if not records:
        return

    # 分离出 task_id 缺失的记录：Task 7 会在调用 add_task 前先落盘 PENDING 意图，
    # 若中途写入平台失败，task_id 会是 None——这不是"平台侧消失"，须先于对账处理。
    orphans = [r for r in records if r.task_id is None]
    records = [r for r in records if r.task_id is not None]
    for rec in orphans:
        rec.status = TaskStatus.NEEDS_HUMAN
        storage.update_task(rec)
        logger.warning(
            "task record for model %s has no task_id (submit likely failed mid-flight); "
            "marked NEEDS_HUMAN", rec.model_id)

    if not records:
        return

    try:
        page = client.list_my_tasks(page_size=100)
    except Exception:
        logger.exception("list_my_tasks failed; will retry next tick")
        return

    platform_rows = {row["taskId"]: row for row in page.get("records", [])}
    timeout = timedelta(hours=settings.task_timeout_hours)

    for rec in records:
        row = platform_rows.get(rec.task_id)
        if row is None:
            rec.status = TaskStatus.ABANDONED
            storage.update_task(rec)
            storage.set_kill_switch(
                True, f"task {rec.task_id} vanished from platform (possible violation cleanup)")
            logger.error("task %s vanished from platform; kill switch ON", rec.task_id)
            continue

        mapped = rules.map_platform_status(row.get("status"))

        if mapped == "failed":
            try:
                rec.last_log = client.get_task_log(rec.task_id)
            except Exception:
                logger.exception("get_task_log failed for %s", rec.task_id)
                rec.last_log = ""
            from .failure import classify  # 局部导入避免模块环
            from .models import FailureKind
            kind = classify(rec.last_log)
            rec.status = (TaskStatus.QUALITY_FAILED if kind == FailureKind.QUALITY
                          else TaskStatus.ENGINE_FAILED)
            storage.update_task(rec)
            continue

        if mapped == TaskStatus.SUCCESS:
            storage.set_counter("consecutive_engine_failures", 0)  # 成功即重置熔断计数
            rec.status = TaskStatus.SUCCESS
            storage.update_task(rec)
            continue

        # 仍在活跃态（或状态未知）：先同步状态，再做超时判定（两者不互斥）
        if mapped is None:
            logger.warning("unknown platform status %r for task %s", row.get("status"), rec.task_id)
        elif mapped != rec.status:
            rec.status = mapped
            storage.update_task(rec)

        if rec.submit_time is not None and now - rec.submit_time > timeout:
            rec.status = TaskStatus.TIMEOUT
            storage.update_task(rec)
