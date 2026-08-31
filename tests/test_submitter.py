from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
import requests

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


def test_task_id_persist_failure_stops_submissions(store):
    """Safety property ROUND 2: When update_task fails after successful submission
    (persisting task_id), drain must immediately stop and return, not continue to
    submit more records in the same tick. This prevents cascading failures when
    storage is unreliable."""
    store.insert_task(_queued("org/first"))
    store.insert_task(_queued("org/second"))

    client = Mock()
    client.add_task.side_effect = [111, 222]  # Would succeed for both if we got that far

    original_update = store.update_task
    call_count = [0]
    def failing_update(rec):
        call_count[0] += 1
        if call_count[0] == 2:  # Second call: persisting task_id for first record
            raise RuntimeError("storage unreliable")
        return original_update(rec)

    store.update_task = failing_update

    # Should submit first record and return, never submit second
    result = submitter.drain(store, client, SETTINGS, now=NOW)
    assert result == 1

    # Kill switch must be on (storage failed)
    assert store.kill_switch() is True

    # client.add_task called only once (second record never submitted)
    assert client.add_task.call_count == 1

    # First record is PENDING (submitted successfully)
    pending = store.tasks_by_status(TaskStatus.PENDING)
    assert len(pending) == 1
    assert pending[0].model_id == "org/first"

    # Second record still QUEUED (never submitted)
    queued = store.tasks_by_status(TaskStatus.QUEUED)
    assert len(queued) == 1
    assert queued[0].model_id == "org/second"


def test_transport_error_leaves_record_pending_and_never_resubmits(store):
    """C1 safety property: a ReadTimeout means the outcome is UNKNOWN — the platform may
    already have created the task. Reverting to QUEUED would resubmit the same
    (model_id, target_gpu) next tick, which is exactly what triggers the platform's
    purge-all-tasks duplicate detection. The record must stay out of QUEUED, and a
    second drain over the same storage must not call add_task again."""
    store.insert_task(_queued("org/timeout"))
    client = Mock()
    client.add_task.side_effect = requests.exceptions.ReadTimeout("read timed out")

    assert submitter.drain(store, client, SETTINGS, now=NOW) == 0

    assert store.tasks_by_status(TaskStatus.QUEUED) == []
    rec = store.get_task("org/timeout", "MetaX_c-500")
    assert rec.status == TaskStatus.PENDING and rec.task_id is None

    client.add_task.reset_mock()
    submitter.drain(store, client, SETTINGS, now=NOW)
    client.add_task.assert_not_called()


def test_http_error_leaves_record_pending(store):
    """An HTTPError from raise_for_status is equally ambiguous: the request reached the
    platform, and the response may have been lost after the task was created."""
    store.insert_task(_queued("org/http"))
    client = Mock()
    client.add_task.side_effect = requests.exceptions.HTTPError("502 Bad Gateway")

    assert submitter.drain(store, client, SETTINGS, now=NOW) == 0

    assert store.tasks_by_status(TaskStatus.QUEUED) == []
    assert store.get_task("org/http", "MetaX_c-500").status == TaskStatus.PENDING


@pytest.mark.parametrize("code", [50000, 50001])
def test_transient_platform_code_leaves_record_pending(store, code):
    """50000/50001 are platform-internal errors: whether the task got created is unknown,
    so the record must not go back to QUEUED."""
    store.insert_task(_queued("org/transient"))
    client = Mock()
    client.add_task.side_effect = PlatformClientError(code, "system error")

    assert submitter.drain(store, client, SETTINGS, now=NOW) == 0

    assert store.tasks_by_status(TaskStatus.QUEUED) == []
    assert store.get_task("org/transient", "MetaX_c-500").status == TaskStatus.PENDING


@pytest.mark.parametrize("code", [40100, 40101, 40400, 40001])
def test_definite_rejection_reverts_to_queued(store, code):
    """The other half of C1: a business-code rejection means the platform did NOT create
    a task, so returning the record to QUEUED for a later retry is safe (and required —
    otherwise every rejected candidate would need manual reconciliation)."""
    store.insert_task(_queued("org/rejected"))
    client = Mock()
    client.add_task.side_effect = PlatformClientError(code, "rejected")

    assert submitter.drain(store, client, SETTINGS, now=NOW) == 0

    rec = store.get_task("org/rejected", "MetaX_c-500")
    assert rec.status == TaskStatus.QUEUED
    assert rec.task_id is None and rec.submit_time is None


def test_naive_bounty_deadline_does_not_crash_drain(store):
    """I1: a hand-written bounty JSON deadline without an offset yields a naive datetime.
    Comparing it against an aware `now` (and sorting it alongside aware deadlines) used to
    raise TypeError, killing the submit stage on every tick."""
    naive = _queued("org/naive", Priority.BOUNTY)
    naive.bounty_deadline = datetime(2026, 9, 30)  # naive
    store.insert_task(naive)
    store.insert_task(_queued("org/aware", Priority.BOUNTY, NOW + timedelta(days=3)))
    client = Mock()
    client.add_task.side_effect = [1, 2]

    assert submitter.drain(store, client, SETTINGS, now=NOW) == 2


DRY_SETTINGS = Settings(xc_token="t", strategy_id="s", base_url="https://x",
                        max_submits_per_minute=2, max_inflight=3, dry_run=True)


def test_dry_run_never_calls_add_task(store):
    """演练模式必须走完组装、按限流节奏挑出记录，但一个平台请求都不发。"""
    store.insert_task(_queued("org/a"))
    store.insert_task(_queued("org/b"))
    client = Mock()

    assert submitter.drain(store, client, DRY_SETTINGS, now=NOW) == 2
    client.add_task.assert_not_called()


def test_dry_run_leaves_records_queued_with_no_task_id(store):
    """绝不能伪造 PENDING/task_id：monitor 对账时会在平台侧找不到它，
    误判为"任务被平台清理"从而拉下熔断闸——演练反而制造事故。"""
    store.insert_task(_queued("org/a"))

    submitter.drain(store, Mock(), DRY_SETTINGS, now=NOW)

    rec = store.get_task("org/a", "MetaX_c-500")
    assert rec.status == TaskStatus.QUEUED
    assert rec.task_id is None and rec.submit_time is None


def test_dry_run_respects_rate_limit(store):
    """演练的节奏要与真实运行一致，否则看到的行为不能代表上线后的行为。"""
    for i in range(5):
        store.insert_task(_queued(f"org/m{i}"))
    assert submitter.drain(store, Mock(), DRY_SETTINGS, now=NOW) == 2  # max_submits_per_minute


def test_dry_run_logs_the_intended_request_once(store, caplog):
    store.insert_task(_queued("org/a"))
    with caplog.at_level("INFO"):
        submitter.drain(store, Mock(), DRY_SETTINGS, now=NOW)
        first = caplog.text.count("DRY RUN would submit")
        submitter.drain(store, Mock(), DRY_SETTINGS, now=NOW)
        second = caplog.text.count("DRY RUN would submit")
    assert first == 1
    assert second == 1  # 同一 (model_id, target_gpu) 不再刷屏
    assert "org/a" in caplog.text


def test_dry_run_advances_through_the_queue_across_ticks(store):
    """演练时记录保持 QUEUED，若已报告过的仍占预算，每个 tick 都会排到同样的前两条，
    队列后面的永远看不到——演练也就只能看到冰山一角。"""
    for i in range(5):
        store.insert_task(_queued(f"org/m{i}"))
    client = Mock()

    seen = []
    for _ in range(3):
        submitter.drain(store, client, DRY_SETTINGS, now=NOW)
        seen.append(sum(1 for i in range(5)
                        if store.get_counter(f"dryrun_logged:org/m{i}@MetaX_c-500")))

    assert seen == [2, 4, 5]  # 每轮推进 2 条（限流），第三轮收尾
    client.add_task.assert_not_called()
