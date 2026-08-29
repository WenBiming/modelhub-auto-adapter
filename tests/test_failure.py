"""M5：日志关键词分类（OOM/CUDA → ENGINE；judge → QUALITY）、
调参序列递进、重试上限拉黑、连续引擎失败熔断。"""
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
import yaml

from auto_adapter import config_gen, failure
from auto_adapter.models import FailureKind, Priority, TaskRecord, TaskStatus
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
    store.insert_task(_failed(TaskStatus.TIMEOUT, deadline=NOW + timedelta(days=1)))
    client = Mock()
    failure.handle(store, client, SETTINGS, now=NOW)
    client.stop_tasks.assert_called_once_with([7])
    assert store.get_task("org/m", "MetaX_c-500").status == TaskStatus.QUEUED


def test_handle_timeout_abandons_non_bounty(store):
    store.insert_task(_failed(TaskStatus.TIMEOUT))
    client = Mock()
    failure.handle(store, client, SETTINGS, now=NOW)
    assert store.get_task("org/m", "MetaX_c-500").status == TaskStatus.ABANDONED


def test_handle_timeout_requeues_bounty_even_if_stop_tasks_fails(store):
    # stop_tasks is best-effort resource cleanup; its failure must not strand the
    # record in TIMEOUT.
    store.insert_task(_failed(TaskStatus.TIMEOUT, deadline=NOW + timedelta(days=1)))
    client = Mock()
    client.stop_tasks.side_effect = ConnectionError("boom")
    failure.handle(store, client, SETTINGS, now=NOW)
    client.stop_tasks.assert_called_once_with([7])
    assert store.get_task("org/m", "MetaX_c-500").status == TaskStatus.QUEUED


def test_handle_timeout_abandons_non_bounty_even_if_stop_tasks_fails(store):
    store.insert_task(_failed(TaskStatus.TIMEOUT))
    client = Mock()
    client.stop_tasks.side_effect = ConnectionError("boom")
    failure.handle(store, client, SETTINGS, now=NOW)
    assert store.get_task("org/m", "MetaX_c-500").status == TaskStatus.ABANDONED


def test_engine_failure_streak_triggers_kill_switch(store):
    for i in range(5):
        rec = _failed(retry_count=3)
        rec.model_id = f"org/m{i}"
        store.insert_task(rec)
    failure.handle(store, Mock(), SETTINGS, now=NOW)
    assert store.kill_switch() is True
    assert store.get_counter("consecutive_engine_failures") == 5
