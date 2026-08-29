"""监控与日志分析层（spec §4.6）。M5 实现。"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from . import rules
from .failure import classify
from .models import FailureKind, TaskStatus
from .platform_client import PlatformClient
from .settings import Settings
from .storage import Storage

logger = logging.getLogger(__name__)

# 单个 tick 内 list_my_tasks 分页枚举的硬上限：list_my_tasks 是账号全部任务历史的分页
# 列表（onlyMine=true），服务长期运行后总任务数必然超过一页；上限约束单次 tick 的
# 工作量，保证每个 tick 步骤在 30s 内完成（优雅停机要求）。
MAX_PAGES = 20


def _fetch_platform_rows(client: PlatformClient) -> tuple[dict[int, dict], str]:
    """拉取 list_my_tasks 全部分页，累积为 {taskId: row}。

    返回 (platform_rows, outcome)，outcome ∈ {"complete", "truncated", "failed"}：

    - "complete"：已读完平台报告的全部页（响应的 pages 字段），或提前遇到空页——
      这是唯一允许据此判定"平台侧消失"的情形；
    - "truncated"：达到 MAX_PAGES 上限仍未读完，只返回已读到的行；调用方不得据此
      判定任务消失（可能只是还没翻到那一页）；
    - "failed"：第 2 页起请求异常，只返回已读到的行，同样不得判定消失。

    第一页请求异常直接向上抛出，由调用方决定是否完全跳过本轮、不触碰任何本地状态
    （list_my_tasks 完全不可用时，连"部分枚举"都做不到）。
    """
    page = client.list_my_tasks(current=1, page_size=100)  # 第一页异常向上抛出
    rows = page.get("records", [])
    platform_rows: dict[int, dict] = {row["taskId"]: row for row in rows}
    total_pages = page.get("pages", 1)
    pages_read = 1

    while rows and pages_read < total_pages:
        if pages_read >= MAX_PAGES:
            logger.warning(
                "list_my_tasks enumeration hit MAX_PAGES=%d cap (platform reports %d pages); "
                "vanish check skipped this tick", MAX_PAGES, total_pages)
            return platform_rows, "truncated"
        try:
            page = client.list_my_tasks(current=pages_read + 1, page_size=100)
        except Exception:
            logger.exception(
                "list_my_tasks page %d failed; enumeration incomplete, vanish check skipped",
                pages_read + 1)
            return platform_rows, "failed"
        rows = page.get("records", [])
        for row in rows:
            platform_rows[row["taskId"]] = row
        pages_read += 1

    return platform_rows, "complete"


def poll(storage: Storage, client: PlatformClient, settings: Settings,
         now: datetime | None = None) -> None:
    """对账并更新本地任务表：

    - list_my_tasks（onlyMine=true）分页拉取账号全部任务历史（非仅在途任务），以平台
      status/verifyResult 为准更新 TaskRecord；分页读取见 `_fetch_platform_rows`——
      "消失"判定只在完整读完所有页时才成立，避免把"还没翻到的页"误判为任务丢失；
    - 本地记录 task_id 为 None（PENDING 意图已落盘但 add_task 未完成）→ NEEDS_HUMAN，
      不算平台侧消失，不触发 kill switch（该告警只保留给平台已接单又丢失的任务）；
    - 本地在途但平台侧（完整枚举后）消失的任务 → ABANDONED + set_kill_switch(True)
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
        platform_rows, outcome = _fetch_platform_rows(client)
    except Exception:
        logger.exception("list_my_tasks failed; will retry next tick")
        return

    enumeration_complete = outcome == "complete"
    timeout = timedelta(hours=settings.task_timeout_hours)

    for rec in records:
        row = platform_rows.get(rec.task_id)
        if row is None:
            if not enumeration_complete:
                logger.warning(
                    "task %s not found in partial platform listing (%s); vanish check "
                    "skipped this tick, record left unchanged", rec.task_id, outcome)
                continue
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
