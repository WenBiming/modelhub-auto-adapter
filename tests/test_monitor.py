import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from auto_adapter import monitor
from auto_adapter.models import Priority, TaskRecord, TaskStatus
from auto_adapter.platform_client import PlatformClientError
from auto_adapter.settings import Settings
from auto_adapter.storage.sqlite import SqliteStorage

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
SETTINGS = Settings(xc_token="t", strategy_id="s", base_url="https://x", task_timeout_hours=6)


@pytest.fixture
def store(tmp_path):
    return SqliteStorage(str(tmp_path / "t.db"))


def _pending(model_id, task_id, submit_time=NOW):
    return TaskRecord(model_id=model_id, target_gpu="MetaX_c-500", framework="vllm",
                      status=TaskStatus.PENDING, priority=Priority.NEW_MODEL,
                      task_id=task_id, submit_time=submit_time)


def _page(*rows):
    return {"records": list(rows), "total": len(rows), "current": 1, "pages": 1, "size": 100}


def test_status_sync_success_and_failure(store):
    store.insert_task(_pending("org/a", 1))
    store.insert_task(_pending("org/b", 2))
    client = Mock()
    client.list_my_tasks.return_value = _page(
        {"taskId": 1, "status": "success", "verifyResult": 1},
        {"taskId": 2, "status": "failed", "verifyResult": 1},
    )
    client.get_task_log.return_value = "CUDA out of memory"
    monitor.poll(store, client, SETTINGS, now=NOW)
    assert store.get_task("org/a", "MetaX_c-500").status == TaskStatus.SUCCESS
    failed = store.get_task("org/b", "MetaX_c-500")
    assert failed.status == TaskStatus.ENGINE_FAILED
    assert failed.last_log == "CUDA out of memory"


def test_vanished_task_triggers_kill_switch(store):
    store.insert_task(_pending("org/a", 1))
    client = Mock()
    client.list_my_tasks.return_value = _page()
    monitor.poll(store, client, SETTINGS, now=NOW)
    assert store.get_task("org/a", "MetaX_c-500").status == TaskStatus.ABANDONED
    assert store.kill_switch() is True


def test_stuck_task_marked_timeout(store):
    store.insert_task(_pending("org/a", 1, submit_time=NOW - timedelta(hours=7)))
    client = Mock()
    client.list_my_tasks.return_value = _page({"taskId": 1, "status": "running", "verifyResult": 1})
    monitor.poll(store, client, SETTINGS, now=NOW)
    assert store.get_task("org/a", "MetaX_c-500").status == TaskStatus.TIMEOUT


def test_unknown_status_left_unchanged(store):
    store.insert_task(_pending("org/a", 1))
    client = Mock()
    client.list_my_tasks.return_value = _page({"taskId": 1, "status": "WEIRD_STATE", "verifyResult": 1})
    monitor.poll(store, client, SETTINGS, now=NOW)
    assert store.get_task("org/a", "MetaX_c-500").status == TaskStatus.PENDING


def test_multi_page_listing_syncs_status_without_false_vanish(store):
    """FIX 1: list_my_tasks is a paged listing of the whole account task history, not just
    in-flight tasks. A record whose taskId sits on page 2 must not be mistaken for a
    vanished task just because page 1 didn't contain it."""
    store.insert_task(_pending("org/a", 99))
    client = Mock()

    def list_my_tasks(current=1, page_size=50, **filters):
        if current == 1:
            return {"records": [{"taskId": 1000, "status": "success", "verifyResult": 1}],
                    "total": 2, "current": 1, "pages": 2, "size": 100}
        if current == 2:
            return {"records": [{"taskId": 99, "status": "running", "verifyResult": 1}],
                    "total": 2, "current": 2, "pages": 2, "size": 100}
        return {"records": [], "total": 2, "current": current, "pages": 2, "size": 100}

    client.list_my_tasks.side_effect = list_my_tasks
    monitor.poll(store, client, SETTINGS, now=NOW)
    rec = store.get_task("org/a", "MetaX_c-500")
    assert rec.status == TaskStatus.RUNNING
    assert rec.status != TaskStatus.ABANDONED
    assert store.kill_switch() is False


def test_page_cap_truncation_never_vanishes_a_task(store):
    """FIX 1: if the platform reports more pages than MAX_PAGES, enumeration is truncated
    and must NOT be treated as proof the task vanished — the record is left untouched and
    the kill switch stays off."""
    store.insert_task(_pending("org/a", 99))
    client = Mock()

    def list_my_tasks(current=1, page_size=50, **filters):
        # every page has a decoy record (never taskId 99) and reports far more pages
        # than MAX_PAGES, forcing the cap to trigger before task 99 could ever be found.
        return {"records": [{"taskId": 9000 + current, "status": "success", "verifyResult": 1}],
                "total": 999, "current": current, "pages": 999, "size": 100}

    client.list_my_tasks.side_effect = list_my_tasks
    monitor.poll(store, client, SETTINGS, now=NOW)
    rec = store.get_task("org/a", "MetaX_c-500")
    assert rec.status == TaskStatus.PENDING
    assert rec.status != TaskStatus.ABANDONED
    assert store.kill_switch() is False


def test_single_page_complete_enumeration_still_vanishes(store):
    """Sanity check for FIX 1: a complete single-page enumeration (pages == 1) where the
    task is genuinely absent is still the one case where ABANDONED + kill switch is
    legitimate."""
    store.insert_task(_pending("org/a", 1))
    client = Mock()
    client.list_my_tasks.return_value = _page()
    monitor.poll(store, client, SETTINGS, now=NOW)
    assert store.get_task("org/a", "MetaX_c-500").status == TaskStatus.ABANDONED
    assert store.kill_switch() is True


def test_success_resets_consecutive_engine_failures_counter(store):
    store.insert_task(_pending("org/a", 1))
    store.set_counter("consecutive_engine_failures", 4)
    client = Mock()
    client.list_my_tasks.return_value = _page({"taskId": 1, "status": "success", "verifyResult": 1})
    monitor.poll(store, client, SETTINGS, now=NOW)
    assert store.get_counter("consecutive_engine_failures") == 0


def test_list_my_tasks_failure_on_first_call_leaves_state_untouched(store):
    store.insert_task(_pending("org/a", 1))
    store.insert_task(_pending("org/b", 2, submit_time=NOW - timedelta(hours=7)))
    client = Mock()
    client.list_my_tasks.side_effect = RuntimeError("network down")
    monitor.poll(store, client, SETTINGS, now=NOW)
    assert store.get_task("org/a", "MetaX_c-500").status == TaskStatus.PENDING
    assert store.get_task("org/b", "MetaX_c-500").status == TaskStatus.PENDING
    assert store.kill_switch() is False


def test_pending_with_no_task_id_marked_needs_human(store, caplog):
    """Task 7 ruling: a PENDING record with task_id=None means the intent was persisted
    but add_task never completed (storage write succeeded, submit failed mid-flight).
    monitor must treat this as NEEDS_HUMAN, not as a vanished-from-platform task, and
    must NOT trip the kill switch for it."""
    store.insert_task(_pending("org/a", None))
    client = Mock()
    client.list_my_tasks.return_value = _page()
    with caplog.at_level("WARNING"):
        monitor.poll(store, client, SETTINGS, now=NOW)
    rec = store.get_task("org/a", "MetaX_c-500")
    assert rec.status == TaskStatus.NEEDS_HUMAN
    assert store.kill_switch() is False
    assert "org/a" in caplog.text


def _row(task_id, status="running", model_id="org/a", gpu="MetaX_c-500", **extra):
    return {"taskId": task_id, "status": status,
            "modelId": model_id, "gpuType": gpu, **extra}


def test_log_fetch_failure_does_not_classify_as_engine(store):
    """I4: classify("") returns ENGINE, so writing last_log="" on a get_task_log failure
    turned a QUALITY failure into an engine failure eligible for up to 3 automatic
    retries — the one thing spec §4.7 says must never be auto-retried. On a log-fetch
    failure the record must keep its current status and be retried next tick."""
    store.insert_task(_pending("org/a", 1))
    client = Mock()
    client.list_my_tasks.return_value = _page(_row(1, "FAILED"))
    client.get_task_log.side_effect = ConnectionError("log service down")

    monitor.poll(store, client, SETTINGS, now=NOW)

    rec = store.get_task("org/a", "MetaX_c-500")
    assert rec.status == TaskStatus.PENDING
    assert rec.status != TaskStatus.ENGINE_FAILED


def test_log_fetch_failure_still_honours_timeout(store):
    """The unclassified record is still bounded: a stale task times out normally."""
    store.insert_task(_pending("org/a", 1, submit_time=NOW - timedelta(hours=7)))
    client = Mock()
    client.list_my_tasks.return_value = _page(_row(1, "FAILED"))
    client.get_task_log.side_effect = ConnectionError("log service down")

    monitor.poll(store, client, SETTINGS, now=NOW)

    assert store.get_task("org/a", "MetaX_c-500").status == TaskStatus.TIMEOUT


def test_orphan_is_reattached_by_model_and_gpu(store):
    """I5: a PENDING record with task_id=None means the submit outcome was unknown — the
    platform may well have created the task. The listing carries modelId/gpuType
    (appendix A.3), so the task_id is recoverable and normal reconciliation resumes.
    Discarding it as NEEDS_HUMAN threw away a live task."""
    store.insert_task(_pending("org/a", None))
    client = Mock()
    client.list_my_tasks.return_value = _page(_row(777, "RUNNING", model_id="org/a"))

    monitor.poll(store, client, SETTINGS, now=NOW)

    rec = store.get_task("org/a", "MetaX_c-500")
    assert rec.task_id == 777
    assert rec.status == TaskStatus.RUNNING  # reconciled in the same tick
    assert store.kill_switch() is False


def test_orphan_with_no_matching_row_needs_human(store):
    """Only a COMPLETE enumeration with no matching (modelId, gpuType) row proves the
    platform never created the task."""
    store.insert_task(_pending("org/a", None))
    client = Mock()
    client.list_my_tasks.return_value = _page(_row(5, "RUNNING", model_id="org/other"))

    monitor.poll(store, client, SETTINGS, now=NOW)

    assert store.get_task("org/a", "MetaX_c-500").status == TaskStatus.NEEDS_HUMAN
    assert store.kill_switch() is False


def test_orphan_left_untouched_when_enumeration_incomplete(store):
    """If the listing is truncated/failed we cannot tell whether the task exists, so the
    orphan must survive to the next tick rather than be written off."""
    store.insert_task(_pending("org/a", None))
    client = Mock()

    def list_my_tasks(current=1, page_size=50, **filters):
        if current == 1:
            return {"records": [_row(1, model_id="org/other")],
                    "total": 2, "current": 1, "pages": 2, "size": 100}
        raise ConnectionError("page 2 down")

    client.list_my_tasks.side_effect = list_my_tasks

    monitor.poll(store, client, SETTINGS, now=NOW)

    rec = store.get_task("org/a", "MetaX_c-500")
    assert rec.status == TaskStatus.PENDING and rec.task_id is None


def test_stop_event_during_pagination_never_escalates_a_vanish(store):
    """I6: MAX_PAGES(20) x 10s HTTP timeout can far exceed the 30s shutdown budget, so
    poll takes a stop_event. When it trips mid-pagination the enumeration is truncated,
    which must disable the vanish check — otherwise a SIGTERM would abandon live tasks
    and trip the kill switch."""
    store.insert_task(_pending("org/a", 99))
    stop_event = threading.Event()
    client = Mock()

    def list_my_tasks(current=1, page_size=50, **filters):
        stop_event.set()  # SIGTERM arrives while page 1 is being handled
        return {"records": [_row(9000 + current, model_id="org/decoy")],
                "total": 500, "current": current, "pages": 500, "size": 100}

    client.list_my_tasks.side_effect = list_my_tasks

    monitor.poll(store, client, SETTINGS, now=NOW, stop_event=stop_event)

    rec = store.get_task("org/a", "MetaX_c-500")
    assert rec.status == TaskStatus.PENDING
    assert rec.status != TaskStatus.ABANDONED
    assert store.kill_switch() is False
    assert client.list_my_tasks.call_count == 1  # stopped paginating immediately


def test_timeout_uses_platform_update_time(store):
    """M6: appendix A.3 designates updateTime as the progress signal. A task that has
    been submitted for 7h but reported progress a minute ago is alive, not stuck —
    timing it out would kill a task making steady progress (and cost a submission)."""
    store.insert_task(_pending("org/a", 1, submit_time=NOW - timedelta(hours=7)))
    client = Mock()
    client.list_my_tasks.return_value = _page(
        _row(1, "RUNNING", updateTime=(NOW - timedelta(minutes=1)).isoformat()))

    monitor.poll(store, client, SETTINGS, now=NOW)

    assert store.get_task("org/a", "MetaX_c-500").status == TaskStatus.RUNNING


def test_timeout_triggers_when_update_time_is_stale(store):
    """The other half of M6: no progress for longer than the budget is still a timeout,
    even though the record was only just submitted."""
    store.insert_task(_pending("org/a", 1, submit_time=NOW - timedelta(minutes=5)))
    client = Mock()
    client.list_my_tasks.return_value = _page(
        _row(1, "RUNNING", updateTime=(NOW - timedelta(hours=7)).isoformat()))

    monitor.poll(store, client, SETTINGS, now=NOW)

    assert store.get_task("org/a", "MetaX_c-500").status == TaskStatus.TIMEOUT


def test_credential_error_during_poll_trips_kill_switch(store):
    """I3: monitor used to swallow a 40100 into its generic handler, so an expired
    Xc-Token silently turned the whole agent into a no-op with no alarm."""
    store.insert_task(_pending("org/a", 1))
    client = Mock()
    client.list_my_tasks.side_effect = PlatformClientError(40101, "no auth")

    monitor.poll(store, client, SETTINGS, now=NOW)

    assert store.kill_switch() is True


def test_orphan_not_reattached_to_stale_row_on_partial_enumeration(store):
    """I5 re-review: reattaching must be gated on a COMPLETE enumeration, exactly like the
    NEEDS_HUMAN escalation.

    list_my_tasks is the account's full task history, so a pair that has been through the
    retry ladder or a bounty requeue has stale rows. Reproduced chain when the gate is
    missing: an ambiguous submit leaves PENDING/task_id=None; next tick page 1 returns an
    old FAILED row for the same (modelId, gpuType) while page 2 — which carries the newly
    created task — errors; the orphan reattaches to the stale taskId, monitor marks it
    engine_failed, failure.handle requeues it, and the next drain submits a SECOND task for
    that pair while the first may still be live on the platform."""
    store.insert_task(_pending("org/a", None))
    client = Mock()

    def list_my_tasks(current=1, page_size=50, **filters):
        if current == 1:
            # stale historical row for the very same (modelId, gpuType)
            return {"records": [_row(11, "FAILED", model_id="org/a")],
                    "total": 2, "current": 1, "pages": 2, "size": 100}
        raise ConnectionError("page 2 down")  # the page holding the real new task

    client.list_my_tasks.side_effect = list_my_tasks

    monitor.poll(store, client, SETTINGS, now=NOW)

    rec = store.get_task("org/a", "MetaX_c-500")
    assert rec.task_id is None, "must not adopt a taskId from an incomplete listing"
    assert rec.status == TaskStatus.PENDING  # not engine_failed, so nothing requeues it
    client.get_task_log.assert_not_called()


def test_orphan_not_reattached_to_stale_row_on_truncated_enumeration(store):
    """Same gate for the MAX_PAGES/stop_event truncation outcome."""
    store.insert_task(_pending("org/a", None))
    client = Mock()
    client.list_my_tasks.side_effect = lambda current=1, page_size=50, **f: {
        "records": [_row(9000 + current, "FAILED", model_id="org/a")],
        "total": 999, "current": current, "pages": 999, "size": 100}

    monitor.poll(store, client, SETTINGS, now=NOW)

    rec = store.get_task("org/a", "MetaX_c-500")
    assert rec.task_id is None and rec.status == TaskStatus.PENDING


def test_success_with_failed_verify_result_is_not_a_success(store):
    """平台的 status 与 verifyResult 是正交字段（实测自平台 UI 筛选，2026-08-29）：
    status=success 只表示"作业跑完了"，适配是否通过看 verifyResult。

    这是本系统最危险的一条误判：只看 status 会把每一个 verifyResult=-1（验证失败）
    的任务记成适配成功——既不会进入重试/拉黑流程，成功率指标也全是假的。
    用户账号里绝大多数历史记录正是 status=success + verifyResult=-1。
    """
    store.insert_task(_pending("org/a", 1))
    client = Mock()
    client.list_my_tasks.return_value = _page(
        {"taskId": 1, "status": "success", "verifyResult": -1,
         "statusText": "成功", "verifyResultText": "验证失败"})
    client.get_task_log.return_value = "LLM judge score below threshold"

    monitor.poll(store, client, SETTINGS, now=NOW)

    rec = store.get_task("org/a", "MetaX_c-500")
    assert rec.status != TaskStatus.SUCCESS
    assert rec.status == TaskStatus.QUALITY_FAILED  # 日志把它归为质量失败


def test_success_without_verify_result_is_not_assumed_passed(store):
    """verifyResult 缺失/未知时同样不得宣布成功——只有明确的 1 才算通过。"""
    store.insert_task(_pending("org/a", 1))
    client = Mock()
    client.list_my_tasks.return_value = _page({"taskId": 1, "status": "success"})
    client.get_task_log.return_value = "CUDA out of memory"

    monitor.poll(store, client, SETTINGS, now=NOW)

    assert store.get_task("org/a", "MetaX_c-500").status == TaskStatus.ENGINE_FAILED


def test_waiting_status_syncs_to_pending(store):
    """平台排队态的真实值是小写 waiting；早先的映射表只有 QUEUED/PENDING，
    会把它当成未知状态而永不同步，任务一直卡到 6 小时超时。"""
    store.insert_task(_pending("org/a", 1))
    client = Mock()
    client.list_my_tasks.return_value = _page(
        {"taskId": 1, "status": "waiting", "verifyResult": 1})

    monitor.poll(store, client, SETTINGS, now=NOW)

    assert store.get_task("org/a", "MetaX_c-500").status == TaskStatus.PENDING


def _plat_row(task_id, model_id, gpu, status="running", verify=1):
    return {"taskId": task_id, "modelId": model_id, "gpuType": gpu,
            "status": status, "verifyResult": verify,
            "updateTime": "2026-08-29 11:00:00"}


def test_adoption_reclaims_in_flight_tasks_after_storage_reset(store):
    """容器存储是临时的：Pod 重启后本地任务表清零，而平台上 waiting/running 的任务
    还在跑。不认领回来，下个 tick 会重新发现同一模型——而**在跑的任务不会出现在
    search_model 的已完成结果里**——去重放行，同一 (model_id, gpu) 被提交第二次。"""
    client = Mock()
    client.list_my_tasks.return_value = {
        "records": [_plat_row(501, "org/running", "MetaX_c-500", "running"),
                    _plat_row(502, "org/queued", "Ascend_910-b4", "waiting")],
        "total": 2, "current": 1, "pages": 1, "size": 100}

    assert monitor.adopt_orphaned_platform_tasks(store, client) == 2

    assert store.get_task("org/running", "MetaX_c-500").task_id == 501
    assert store.get_task("org/queued", "Ascend_910-b4").status == TaskStatus.PENDING


def test_adoption_ignores_finished_tasks(store):
    """已完成的任务不重建：它们由 eligibility 查 search_model 覆盖到，
    在本地凭空造记录反而会挡住合法的重新适配。"""
    client = Mock()
    client.list_my_tasks.return_value = {
        "records": [_plat_row(601, "org/done", "MetaX_c-500", "success", 1),
                    _plat_row(602, "org/failed", "MetaX_c-500", "success", -1)],
        "total": 2, "current": 1, "pages": 1, "size": 100}

    assert monitor.adopt_orphaned_platform_tasks(store, client) == 0
    assert store.get_task("org/done", "MetaX_c-500") is None


def test_adoption_reports_unconfirmed_on_partial_enumeration(store):
    """只读到半份名单时返回 -1（未能确认），调用方据此拉闸暂停提交——
    宁可这一轮不提交，也不能在"可能有在途任务"的状态下放行。"""
    client = Mock()
    client.list_my_tasks.side_effect = [
        {"records": [_plat_row(701, "org/a", "MetaX_c-500")],
         "total": 200, "current": 1, "pages": 3, "size": 100},
        ConnectionError("page 2 down"),
    ]

    assert monitor.adopt_orphaned_platform_tasks(store, client) == -1
    assert store.get_task("org/a", "MetaX_c-500") is None  # 半份名单不做任何认领


def test_adoption_does_not_clobber_existing_local_records(store):
    store.insert_task(_pending("org/a", 999))
    client = Mock()
    client.list_my_tasks.return_value = {
        "records": [_plat_row(888, "org/a", "MetaX_c-500")],
        "total": 1, "current": 1, "pages": 1, "size": 100}

    assert monitor.adopt_orphaned_platform_tasks(store, client) == 0
    assert store.get_task("org/a", "MetaX_c-500").task_id == 999  # 本地记录优先
