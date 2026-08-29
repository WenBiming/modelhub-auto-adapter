from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from auto_adapter import monitor
from auto_adapter.models import Priority, TaskRecord, TaskStatus
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
        {"taskId": 1, "status": "SUCCESS"},
        {"taskId": 2, "status": "FAILED"},
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
    client.list_my_tasks.return_value = _page({"taskId": 1, "status": "RUNNING"})
    monitor.poll(store, client, SETTINGS, now=NOW)
    assert store.get_task("org/a", "MetaX_c-500").status == TaskStatus.TIMEOUT


def test_unknown_status_left_unchanged(store):
    store.insert_task(_pending("org/a", 1))
    client = Mock()
    client.list_my_tasks.return_value = _page({"taskId": 1, "status": "WEIRD_STATE"})
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
            return {"records": [{"taskId": 1000, "status": "SUCCESS"}],
                    "total": 2, "current": 1, "pages": 2, "size": 100}
        if current == 2:
            return {"records": [{"taskId": 99, "status": "RUNNING"}],
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
        return {"records": [{"taskId": 9000 + current, "status": "SUCCESS"}],
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
    client.list_my_tasks.return_value = _page({"taskId": 1, "status": "SUCCESS"})
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
