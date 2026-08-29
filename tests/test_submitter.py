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


def test_post_submit_update_failure_sets_kill_switch(store):
    """Safety property FIX 1: if update_task fails after successful submission,
    kill_switch is set to prevent resubmission. Record is NOT left in QUEUED.
    (task_id not persisted due to storage failure, but kill_switch blocks resubmission and
    task_id is logged for manual reconciliation)"""
    store.insert_task(_queued("org/risky"))
    client = Mock()
    client.add_task.return_value = 999

    original_update = store.update_task
    call_count = [0]
    def failing_update(rec):
        call_count[0] += 1
        if call_count[0] == 2:  # Second call (after successful add_task) fails
            raise RuntimeError("storage error")
        return original_update(rec)

    store.update_task = failing_update

    # drain should count it as submitted (add_task succeeded)
    assert submitter.drain(store, client, SETTINGS, now=NOW) == 1
    # kill_switch must be on to prevent resubmission (most critical safety property)
    assert store.kill_switch() is True
    # record must NOT be in QUEUED (it's in PENDING, preventing resubmission)
    queued = store.tasks_by_status(TaskStatus.QUEUED)
    assert len(queued) == 0
    pending = store.tasks_by_status(TaskStatus.PENDING)
    assert pending[0].model_id == "org/risky"
    # task_id not persisted (storage failed), but kill_switch+logs allow manual reconciliation


def test_client_error_reverts_to_queued(store):
    """Safety property FIX 1: if add_task fails (non-credential), record reverts to QUEUED
    so it can be resubmitted next tick."""
    store.insert_task(_queued("org/retry"))
    client = Mock()
    client.add_task.side_effect = PlatformClientError(500, "server error")

    assert submitter.drain(store, client, SETTINGS, now=NOW) == 0
    # Record must be reverted to QUEUED, not left in PENDING
    queued = store.tasks_by_status(TaskStatus.QUEUED)
    assert queued[0].model_id == "org/retry"
    assert queued[0].submit_time is None
    assert queued[0].task_id is None
    # kill_switch should NOT be set (non-credential error)
    assert store.kill_switch() is False


def test_expiring_bounty_abandoned_regardless_of_budget(store):
    """Safety property FIX 2: expiring bounties are marked ABANDONED even when
    in-flight is at max_inflight (independent of budget constraint)."""
    # Fill in-flight to max_inflight=3
    for i in range(3):
        rec = _queued(f"org/running{i}")
        rec.status = TaskStatus.RUNNING
        store.insert_task(rec)

    # Add an expiring bounty
    store.insert_task(_queued("org/expiring", Priority.BOUNTY, NOW + timedelta(hours=1)))

    client = Mock()
    result = submitter.drain(store, client, SETTINGS, now=NOW)

    # No submissions (budget=0) but bounty is abandoned
    assert result == 0
    abandoned = store.tasks_by_status(TaskStatus.ABANDONED)
    assert len(abandoned) == 1
    assert abandoned[0].model_id == "org/expiring"
