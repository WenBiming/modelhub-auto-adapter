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
