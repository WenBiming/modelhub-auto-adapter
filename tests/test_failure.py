"""M5：日志关键词分类（OOM/CUDA → ENGINE；judge → QUALITY）、
调参序列递进、重试上限拉黑、连续引擎失败熔断。"""
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
import yaml

from auto_adapter import config_gen, failure
from auto_adapter.models import FailureKind, Priority, TaskRecord, TaskStatus
from auto_adapter.platform_client import PlatformClientError
from auto_adapter.settings import Settings
from auto_adapter.storage.sqlite import SqliteStorage

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
SETTINGS = Settings(xc_token="t", strategy_id="s", base_url="https://x", max_retries=3)


@pytest.fixture
def store(tmp_path):
    return SqliteStorage(str(tmp_path / "t.db"))


def _failed(status=TaskStatus.ENGINE_FAILED, retry_count=0, deadline=None):
    return TaskRecord(model_id="org/m", target_gpu="MetaX_c-500", framework="vllm",
                      status=status, priority=Priority.NEW_MODEL, task_id=7,
                      retry_count=retry_count, bounty_deadline=deadline,
                      config_params=config_gen.render_config_params("vllm", 1))


def test_classify():
    assert failure.classify("CUDA out of memory on device") == FailureKind.ENGINE
    assert failure.classify("LLM judge score below threshold") == FailureKind.QUALITY
    assert failure.classify("some unknown garbage") == FailureKind.ENGINE


def test_next_config_ladder_composes_cumulatively():
    # Mirrors the real flow (handle() writes next_config's output back onto
    # record.config_params before the next retry), so each rung must still carry
    # every earlier rung's adjustment forward, not just its own.
    rec = _failed(retry_count=0)

    out0 = failure.next_config(rec)
    cfg0 = yaml.safe_load(out0)
    cmd0 = cfg0["sut_config"]["values"]["command"]
    assert cmd0[cmd0.index("--gpu-memory-utilization") + 1] == "0.95"

    rec.config_params = out0
    rec.retry_count = 1
    out1 = failure.next_config(rec)
    cfg1 = yaml.safe_load(out1)
    cmd1 = cfg1["sut_config"]["values"]["command"]
    assert cmd1[cmd1.index("--gpu-memory-utilization") + 1] == "0.95"  # rung 0 preserved
    assert cfg1["max_model_len"] == 2048

    rec.config_params = out1
    rec.retry_count = 2
    out2 = failure.next_config(rec)
    cfg2 = yaml.safe_load(out2)
    sut2 = cfg2["sut_config"]["values"]["command"]
    ref2 = cfg2["ref_config"]["values"]["command"]
    assert sut2[sut2.index("--gpu-memory-utilization") + 1] == "0.95"  # rung 0 preserved
    assert cfg2["max_model_len"] == 2048  # rung 1 preserved
    assert sut2[sut2.index("-tp") + 1] == "2" and ref2[ref2.index("-tp") + 1] == "2"
    assert cfg2["sut_config"]["gpu_num"] == 2 and cfg2["ref_config"]["gpu_num"] == 2

    rec.config_params = out2
    rec.retry_count = 3
    assert failure.next_config(rec) is None


def test_next_config_tp_cap_applies_without_overshoot():
    # Starting at tp=2, rung 2 should double to 4 (the cap), not 8.
    rec = _failed(retry_count=2)
    rec.config_params = config_gen.render_config_params("vllm", 2)
    cfg = yaml.safe_load(failure.next_config(rec))
    sut = cfg["sut_config"]["values"]["command"]
    ref = cfg["ref_config"]["values"]["command"]
    assert sut[sut.index("-tp") + 1] == "4" and ref[ref.index("-tp") + 1] == "4"
    assert cfg["sut_config"]["gpu_num"] == 4 and cfg["ref_config"]["gpu_num"] == 4


def test_next_config_tp_already_at_cap_exhausts_ladder():
    # A model already rendered at tp=4 (e.g. >70B) has no headroom left: doubling
    # would silently resubmit the exact config that just failed. Treat that as
    # ladder-exhausted rather than an ineffective resubmission.
    rec = _failed(retry_count=2)
    rec.config_params = config_gen.render_config_params("vllm", 4)
    assert failure.next_config(rec) is None


def test_handle_requeues_engine_failure_then_blacklists(store):
    store.insert_task(_failed(retry_count=0))
    failure.handle(store, Mock(), SETTINGS, now=NOW)
    rec = store.get_task("org/m", "MetaX_c-500")
    assert rec.status == TaskStatus.QUEUED and rec.retry_count == 1 and rec.task_id is None
    rec.status = TaskStatus.ENGINE_FAILED
    rec.retry_count = 3
    store.update_task(rec)
    failure.handle(store, Mock(), SETTINGS, now=NOW)
    assert store.is_blacklisted("org/m", "MetaX_c-500")


def test_handle_quality_failure_needs_human(store):
    store.insert_task(_failed(TaskStatus.QUALITY_FAILED))
    failure.handle(store, Mock(), SETTINGS, now=NOW)
    assert store.get_task("org/m", "MetaX_c-500").status == TaskStatus.NEEDS_HUMAN


def test_handle_timeout_stops_and_requeues_bounty(store):
    """Success path: a confirmed stop (stop_tasks → True) released the platform task,
    so requeueing the bounty cannot collide with a still-running task."""
    store.insert_task(_failed(TaskStatus.TIMEOUT, deadline=NOW + timedelta(days=1)))
    client = Mock()
    client.stop_tasks.return_value = True
    failure.handle(store, client, SETTINGS, now=NOW)
    client.stop_tasks.assert_called_once_with([7])
    assert store.get_task("org/m", "MetaX_c-500").status == TaskStatus.QUEUED


def test_handle_timeout_abandons_non_bounty(store):
    store.insert_task(_failed(TaskStatus.TIMEOUT))
    client = Mock()
    client.stop_tasks.return_value = True
    failure.handle(store, client, SETTINGS, now=NOW)
    assert store.get_task("org/m", "MetaX_c-500").status == TaskStatus.ABANDONED


def test_handle_timeout_does_not_requeue_when_stop_tasks_raises(store):
    """Safety property (C2): a bounty must NOT be requeued while the old platform task
    may still be running. stop_tasks raising means the stop was never confirmed, so
    resubmitting the same (model_id, target_gpu) risks the platform's duplicate-submission
    purge. The record stays in TIMEOUT and is retried next tick."""
    store.insert_task(_failed(TaskStatus.TIMEOUT, deadline=NOW + timedelta(days=1)))
    client = Mock()
    client.stop_tasks.side_effect = ConnectionError("boom")

    failure.handle(store, client, SETTINGS, now=NOW)

    client.stop_tasks.assert_called_once_with([7])
    assert store.get_task("org/m", "MetaX_c-500").status == TaskStatus.TIMEOUT


def test_handle_timeout_does_not_abandon_when_stop_tasks_returns_false(store):
    """stop_tasks returns bool (appendix A.5). A False return means the stop did not
    land, and is treated exactly like an exception: no transition, retry next tick."""
    store.insert_task(_failed(TaskStatus.TIMEOUT))
    client = Mock()
    client.stop_tasks.return_value = False

    failure.handle(store, client, SETTINGS, now=NOW)

    assert store.get_task("org/m", "MetaX_c-500").status == TaskStatus.TIMEOUT


def test_repeated_stop_failures_escalate_to_needs_human(store):
    """A record must not sit in TIMEOUT forever: after _MAX_STOP_ATTEMPTS ticks in which
    the stop never landed, a human has to see it. The attempt counter is separate from
    retry_count (the tuning-ladder index), which must not be consumed here."""
    store.insert_task(_failed(TaskStatus.TIMEOUT, deadline=NOW + timedelta(days=1)))
    client = Mock()
    client.stop_tasks.return_value = False

    for _ in range(failure._MAX_STOP_ATTEMPTS - 1):
        failure.handle(store, client, SETTINGS, now=NOW)
        rec = store.get_task("org/m", "MetaX_c-500")
        assert rec.status == TaskStatus.TIMEOUT
        assert rec.retry_count == 0  # ladder index untouched

    failure.handle(store, client, SETTINGS, now=NOW)
    rec = store.get_task("org/m", "MetaX_c-500")
    assert rec.status == TaskStatus.NEEDS_HUMAN
    assert rec.retry_count == 0


def test_stop_tasks_credential_error_trips_kill_switch(store):
    """I3: an expired Xc-Token surfacing in the failure stage must trip the kill switch
    rather than be swallowed by the generic handler (which turns the agent into a
    silent no-op)."""
    store.insert_task(_failed(TaskStatus.TIMEOUT))
    client = Mock()
    client.stop_tasks.side_effect = PlatformClientError(40100, "not login")

    failure.handle(store, client, SETTINGS, now=NOW)

    assert store.kill_switch() is True


def test_poison_config_params_does_not_starve_other_records(store):
    """I7: yaml.safe_load("") returns None and next_config then raises. That exception
    used to escape handle() on every tick, so no engine failure anywhere in the batch was
    ever processed again. The bad record goes NEEDS_HUMAN; the rest still get handled."""
    poison = _failed(retry_count=0)
    poison.model_id = "org/poison"
    poison.config_params = ""
    store.insert_task(poison)
    healthy = _failed(retry_count=0)
    healthy.model_id = "org/healthy"
    store.insert_task(healthy)

    failure.handle(store, Mock(), SETTINGS, now=NOW)

    assert store.get_task("org/poison", "MetaX_c-500").status == TaskStatus.NEEDS_HUMAN
    good = store.get_task("org/healthy", "MetaX_c-500")
    assert good.status == TaskStatus.QUEUED and good.retry_count == 1


def test_naive_bounty_deadline_does_not_crash_handle(store):
    """I1: a hand-written bounty deadline without an offset round-trips through storage as
    a naive datetime; comparing it with an aware `now` used to raise TypeError and kill
    the whole failure stage every tick."""
    rec = _failed(TaskStatus.TIMEOUT)
    rec.bounty_deadline = datetime(2026, 9, 30)  # naive, as hand-written JSON produces
    store.insert_task(rec)
    client = Mock()
    client.stop_tasks.return_value = True

    failure.handle(store, client, SETTINGS, now=NOW)

    assert store.get_task("org/m", "MetaX_c-500").status == TaskStatus.QUEUED


def test_engine_failure_streak_triggers_kill_switch(store):
    for i in range(5):
        rec = _failed(retry_count=3)
        rec.model_id = f"org/m{i}"
        store.insert_task(rec)
    failure.handle(store, Mock(), SETTINGS, now=NOW)
    assert store.kill_switch() is True
    assert store.get_counter("consecutive_engine_failures") == 5
