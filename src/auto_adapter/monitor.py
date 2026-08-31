"""监控与日志分析层（spec §4.6）。M5 实现。"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone

from . import rules
from .failure import classify
from .models import FailureKind, Priority, TaskRecord, TaskStatus, ensure_utc
from .platform_client import PlatformClient, escalate_if_credential_error
from .settings import Settings
from .storage import Storage
from .storage.base import DuplicateTaskError

logger = logging.getLogger(__name__)

# 单个 tick 内 list_my_tasks 分页枚举的硬上限：list_my_tasks 是账号全部任务历史的分页
# 列表（onlyMine=true），服务长期运行后总任务数必然超过一页。
#
# 注意 MAX_PAGES 本身**不足以**保证 30s 优雅停机预算：20 页 × 10s HTTP 超时最坏 200s。
# 真正的停机保证来自 poll(stop_event=...)——SIGTERM 一到就中断翻页并返回 "truncated"
# （从而自动关闭"消失"判定）。MAX_PAGES 只限制单 tick 的正常工作量上限。
MAX_PAGES = 20


def _as_int(raw, default: int = 0) -> int:
    """平台把 int64 字段序列化成 **字符串**（实测：taskId "33260"、creatorId "64"、
    machineId "8"），int32 字段（verifyResult）才是真数字——典型的 Java 防 JS 精度
    丢失做法，附录 A.3 里标 int64 的字段都要按这个处理。

    不统一转换会踩两个坑：分页比较 `int < str` 直接抛 TypeError（线上实测），
    以及 platform_rows 用字符串键、本地 task_id 是 int，`.get()` 永远匹配不上——
    每个在途任务都会被判成"平台侧消失"，触发最高级别的熔断误报。
    """
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _fetch_platform_rows(
    client: PlatformClient,
    storage: Storage | None = None,
    stop_event: threading.Event | None = None,
) -> tuple[dict[int, dict], str]:
    """拉取 list_my_tasks 全部分页，累积为 {taskId: row}。

    返回 (platform_rows, outcome)，outcome ∈ {"complete", "truncated", "failed"}：

    - "complete"：已读完平台报告的全部页（响应的 pages 字段），或提前遇到空页——
      这是唯一允许据此判定"平台侧消失"的情形；
    - "truncated"：达到 MAX_PAGES 上限、或收到停机信号提前中断，只返回已读到的行；
      调用方不得据此判定任务消失（可能只是还没翻到那一页）；
    - "failed"：第 2 页起请求异常，只返回已读到的行，同样不得判定消失。

    第一页请求异常直接向上抛出，由调用方决定是否完全跳过本轮、不触碰任何本地状态
    （list_my_tasks 完全不可用时，连"部分枚举"都做不到）。
    """
    page = client.list_my_tasks(current=1, page_size=100)  # 第一页异常向上抛出
    rows = page.get("records", [])
    platform_rows: dict[int, dict] = {_as_int(row.get("taskId")): row for row in rows}
    total_pages = _as_int(page.get("pages"), 1)
    pages_read = 1

    while rows and pages_read < total_pages:
        if stop_event is not None and stop_event.is_set():
            # SIGTERM：立刻停止翻页（每页最坏 10s，翻完可能远超 30s 宽限期）。
            # 返回 truncated 而非 complete——"消失"判定本 tick 自动失效。
            logger.warning(
                "shutdown requested during list_my_tasks pagination after %d/%d pages; "
                "vanish check skipped this tick", pages_read, total_pages)
            return platform_rows, "truncated"
        if pages_read >= MAX_PAGES:
            logger.warning(
                "list_my_tasks enumeration hit MAX_PAGES=%d cap (platform reports %d pages); "
                "vanish check skipped this tick", MAX_PAGES, total_pages)
            return platform_rows, "truncated"
        try:
            page = client.list_my_tasks(current=pages_read + 1, page_size=100)
        except Exception as e:
            if storage is not None:
                escalate_if_credential_error(storage, e)
            logger.exception(
                "list_my_tasks page %d failed; enumeration incomplete, vanish check skipped",
                pages_read + 1)
            return platform_rows, "failed"
        rows = page.get("records", [])
        for row in rows:
            platform_rows[_as_int(row.get("taskId"))] = row
        pages_read += 1

    return platform_rows, "complete"


def _parse_platform_time(raw) -> datetime | None:
    """解析平台 date-time 字段（附录 A.3 未给出精确格式，尽力而为）。"""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        return ensure_utc(datetime.fromisoformat(text))
    except ValueError:
        logger.debug("unparsable platform time %r", raw)
        return None


def _progress_time(rec: TaskRecord, row: dict) -> datetime | None:
    """超时判定基准：优先平台行的 updateTime（附录 A.3 指定的进度信号），
    缺失或无法解析时退回本地 submit_time。

    只看 submit_time 会把稳步推进中的长任务在 6h 处误杀；updateTime 才反映平台侧
    是否还在动。
    """
    return _parse_platform_time(row.get("updateTime")) or ensure_utc(rec.submit_time)


def _check_timeout(storage: Storage, rec: TaskRecord, row: dict,
                   now: datetime, timeout: timedelta) -> None:
    progressed_at = _progress_time(rec, row)
    if progressed_at is not None and now - progressed_at > timeout:
        rec.status = TaskStatus.TIMEOUT
        storage.update_task(rec)


def _match_orphan(rec: TaskRecord, platform_rows: dict[int, dict]) -> dict | None:
    """按 (modelId, gpuType) 在平台行里找孤儿记录对应的任务（附录 A.3）。

    同一对可能有多行（历史重试）；取 taskId 最大的一行，即最近创建的那个。
    """
    matches = [row for row in platform_rows.values()
               if row.get("modelId") == rec.model_id and row.get("gpuType") == rec.target_gpu]
    if not matches:
        return None
    # 按 taskId 数值取最新一条：字符串排序会把 "9" 排在 "10" 后面。
    return max(matches, key=lambda r: _as_int(r.get("taskId")))


def poll(storage: Storage, client: PlatformClient, settings: Settings,
         now: datetime | None = None,
         stop_event: threading.Event | None = None) -> None:
    """对账并更新本地任务表：

    - list_my_tasks（onlyMine=true）分页拉取账号全部任务历史（非仅在途任务），以平台
      status/verifyResult 为准更新 TaskRecord；分页读取见 `_fetch_platform_rows`——
      "消失"判定只在完整读完所有页时才成立，避免把"还没翻到的页"误判为任务丢失；
    - 本地记录 task_id 为 None（PENDING 意图已落盘但 add_task 结果不可知）→ **仅在枚举
      完整时**按 (modelId, gpuType) 认领回 task_id 恢复正常对账，无匹配行才标
      NEEDS_HUMAN；枚举不完整（truncated/failed）时认领和判死都不做，原样留到下个
      tick——部分列表里的"最新一行"可能只是历史旧行，认错了会引出重复提交；
    - 本地在途但平台侧（完整枚举后）消失的任务 → ABANDONED + set_kill_switch(True)
      （可能被违规清理，最高级别告警，暂停提交待人工确认）；
    - 失败任务拉日志存入 record.last_log，留给 failure 层分类；日志拉取失败时**不**
      改状态（classify("") 会误判成 ENGINE，触发 spec §4.7 明令禁止的质量失败重试）；
    - SUCCESS 时清零 consecutive_engine_failures 计数（成功即重置熔断）；
    - PENDING/RUNNING 距 updateTime（缺失时 submit_time）超过 task_timeout_hours
      → 标记 TIMEOUT；
    - stop_event 置位时中断分页（优雅停机），本 tick 不做消失判定。
    """
    now = now or datetime.now(timezone.utc)
    records = storage.tasks_by_status(TaskStatus.PENDING, TaskStatus.RUNNING)
    if not records:
        return

    # task_id 缺失的记录：submitter 在调用 add_task 前先落盘 PENDING 意图，若提交结果
    # 不可知（传输层超时/50000），task_id 会是 None。平台很可能已经建了单——所以先按
    # (modelId, gpuType) 去平台列表认领，而不是直接判死丢弃。
    orphans = [r for r in records if r.task_id is None]
    records = [r for r in records if r.task_id is not None]

    try:
        platform_rows, outcome = _fetch_platform_rows(client, storage, stop_event)
    except Exception as e:
        escalate_if_credential_error(storage, e)
        logger.exception("list_my_tasks failed; will retry next tick")
        return

    enumeration_complete = outcome == "complete"

    for rec in orphans:
        if not enumeration_complete:
            # 认领和判死一样，必须建立在完整枚举之上。list_my_tasks 是账号**全部历史**，
            # 同一 (modelId, gpuType) 走过重试梯子/悬赏重排队后会留下多条旧行；部分枚举
            # 里"taskId 最大的一行"完全可能只是一条陈旧的 FAILED 行，而真正新建的那条还
            # 在没翻到的页上。认领到旧行 → 状态被同步成 engine_failed → failure.handle
            # 重新入队 → 下一轮 drain 为同一对提交第二个任务，而第一个可能仍在平台上跑。
            logger.warning(
                "task record for model %s has no task_id; platform enumeration was %s, "
                "leaving it untouched for the next tick", rec.model_id, outcome)
            continue
        row = _match_orphan(rec, platform_rows)
        if row is not None:
            rec.task_id = _as_int(row.get("taskId"))
            storage.update_task(rec)
            logger.warning(
                "reattached orphan record %s@%s to platform task %s (submit outcome was "
                "unknown; the platform did create it)",
                rec.model_id, rec.target_gpu, rec.task_id)
            records.append(rec)
        else:
            rec.status = TaskStatus.NEEDS_HUMAN
            storage.update_task(rec)
            logger.warning(
                "task record for model %s has no task_id and no matching platform row "
                "(enumeration complete); marked NEEDS_HUMAN", rec.model_id)

    if not records:
        return

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

        # status 与 verifyResult 必须成对判读：status=success 只说明作业跑完了，
        # 适配是否通过看 verifyResult（详见 rules.map_platform_result）。
        mapped = rules.map_platform_result(row.get("status"), row.get("verifyResult"))

        if mapped == rules.NEEDS_CLASSIFICATION:
            try:
                rec.last_log = client.get_task_log(rec.task_id)
            except Exception as e:
                # 拿不到日志就分不了类。这里曾经写 last_log=""，而 classify("") 返回
                # ENGINE——一次瞬时的日志拉取失败会让 QUALITY 失败被自动重试最多 3 次，
                # 正是 spec §4.7 明令禁止的。保持当前状态，下个 tick 再试。
                escalate_if_credential_error(storage, e)
                logger.exception(
                    "get_task_log failed for task %s; leaving %s@%s unclassified, retry next tick",
                    rec.task_id, rec.model_id, rec.target_gpu)
                _check_timeout(storage, rec, row, now, timeout)  # 超时仍然兜底
                continue
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
            logger.warning("unknown platform status %r (verifyResult=%r) for task %s",
                           row.get("status"), row.get("verifyResult"), rec.task_id)
        elif mapped != rec.status:
            rec.status = mapped
            storage.update_task(rec)

        _check_timeout(storage, rec, row, now, timeout)


def adopt_orphaned_platform_tasks(storage: Storage, client: PlatformClient) -> int:
    """启动时把平台上仍在途的任务认领回本地任务表，返回认领条数。

    为什么必须有这一步：平台不保证挂载持久卷，容器存储是临时的。Pod 一重启，
    本地任务表清零，而平台上那些 waiting/running 的任务还在跑。下一个 tick 会
    重新发现同一模型，去问 search_model——**正在跑的任务不会出现在已完成的
    verifyResult 里**，于是去重放行，同一 (model_id, target_gpu) 被提交第二次。
    平台判定重复提交的处理是清空账号下全部任务。

    只认领活跃态（waiting/running）：已完成的任务由 eligibility 查 search_model
    覆盖得到，不需要也不应该在本地重建。

    枚举不完整（truncated/failed）时**不做任何认领**并返回 0——宁可这一轮不认领
    （下个 tick 再来），也不能基于半份名单做判断。调用方据此决定是否放行提交。
    """
    platform_rows, outcome = _fetch_platform_rows(client, storage)
    if outcome != "complete":
        logger.warning(
            "startup adoption skipped: platform listing %s; submissions stay paused "
            "until a complete listing is read", outcome)
        return -1  # 负数表示"未能确认"，与"确认了但一条都没有"区分开

    adopted = 0
    for task_id, row in platform_rows.items():
        mapped = rules.map_platform_result(row.get("status"), _as_int(row.get("verifyResult")))
        if mapped not in (TaskStatus.PENDING, TaskStatus.RUNNING):
            continue  # 只认领在途任务；已完成的靠 search_model 去重
        model_id, gpu = row.get("modelId"), row.get("gpuType")
        if not model_id or not gpu:
            continue
        if storage.get_task(model_id, gpu) is not None:
            continue  # 本地已有记录，无需认领
        try:
            storage.insert_task(TaskRecord(
                model_id=model_id, target_gpu=gpu,
                framework="", status=mapped, priority=Priority.NEW_ADAPTATION,
                task_id=_as_int(task_id),
                submit_time=_parse_platform_time(row.get("updateTime")),
                model_url="", task_type=""))
            adopted += 1
        except DuplicateTaskError:
            pass
    if adopted:
        logger.warning(
            "adopted %d in-flight platform task(s) after a storage reset; "
            "they will not be resubmitted", adopted)
    return adopted
