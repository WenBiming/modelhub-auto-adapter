from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from auto_adapter import submitter
from auto_adapter.models import Priority, TaskRecord, TaskStatus
from auto_adapter.platform_client import PlatformClientError
from auto_adapter.settings import Settings
from auto_adapter.storage.sqlite import SqliteStorage

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
SETTINGS = Settings(xc_token="t", strategy_id="s", base_url="https://x",
                    max_submits_per_minute=2, max_inflight=3)


@pytest.fixture
def store(tmp_path):
    return SqliteStorage(str(tmp_path / "t.db"))


def _queued(model_id, priority=Priority.NEW_MODEL, deadline=None):
    return TaskRecord(model_id=model_id, target_gpu="MetaX_c-500", framework="vllm",
                      status=TaskStatus.QUEUED, priority=priority,
                      model_url=f"https://huggingface.co/{model_id}",
                      task_type="text-generation", config_params="framework: vllm\n",
                      bounty_deadline=deadline)


def test_drain_respects_priority_and_rate_limit(store):
    store.insert_task(_queued("org/new"))
    store.insert_task(_queued("org/bounty", Priority.BOUNTY, NOW + timedelta(days=2)))
    store.insert_task(_queued("org/adapt", Priority.NEW_ADAPTATION))
    client = Mock()
    client.add_task.side_effect = [1, 2]
    assert submitter.drain(store, client, SETTINGS, now=NOW) == 2  # 限流=2
    submitted_types = [c.args[0].model_address for c in client.add_task.call_args_list]
    assert submitted_types == ["https://huggingface.co/org/bounty",
                               "https://huggingface.co/org/new"]
    pending = store.tasks_by_status(TaskStatus.PENDING)
    assert {r.task_id for r in pending} == {1, 2}
    assert all(r.submit_time == NOW for r in pending)


def test_drain_respects_inflight_cap(store):
    for i in range(3):
        rec = _queued(f"org/m{i}")
        rec.status = TaskStatus.RUNNING
        store.insert_task(rec)
    store.insert_task(_queued("org/new"))
    client = Mock()
    assert submitter.drain(store, client, SETTINGS, now=NOW) == 0
    client.add_task.assert_not_called()


def test_kill_switch_blocks(store):
    store.insert_task(_queued("org/new"))
    store.set_kill_switch(True, "test")
    assert submitter.drain(store, Mock(), SETTINGS, now=NOW) == 0


def test_credential_error_sets_kill_switch(store):
    store.insert_task(_queued("org/new"))
    client = Mock()
    client.add_task.side_effect = PlatformClientError(40100, "not login")
    assert submitter.drain(store, client, SETTINGS, now=NOW) == 0
    assert store.kill_switch() is True


def test_expiring_bounty_abandoned(store):
    store.insert_task(_queued("org/late", Priority.BOUNTY, NOW + timedelta(hours=1)))
    assert submitter.drain(store, Mock(), SETTINGS, now=NOW) == 0
    assert store.tasks_by_status(TaskStatus.ABANDONED)[0].model_id == "org/late"
