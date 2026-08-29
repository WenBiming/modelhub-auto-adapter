from dataclasses import replace
from unittest.mock import Mock

import pytest

from auto_adapter.eligibility import Verdict, evaluate
from auto_adapter.models import ModelSearchResult, Priority
from auto_adapter.platform_client import PlatformClientError
from auto_adapter.storage.sqlite import SqliteStorage

GPU = "MetaX_c-500"


@pytest.fixture
def store(tmp_path):
    return SqliteStorage(str(tmp_path / "t.db"))


def _client(verify_result=None, error=False):
    client = Mock()
    if error:
        client.search_model.side_effect = ConnectionError("down")
    else:
        client.search_model.return_value = ModelSearchResult(
            is_in_db=bool(verify_result), model_info={}, verify_result=verify_result or {})
    return client


def test_new_model_enqueued(store, candidate):
    d = evaluate(candidate, GPU, store, _client({}))
    assert d.verdict == Verdict.ENQUEUE and d.priority == Priority.NEW_MODEL


def test_new_adaptation_when_other_gpu_verified(store, candidate):
    d = evaluate(candidate, GPU, store, _client({"other-gpu": {"passed": True}}))
    assert d.verdict == Verdict.ENQUEUE and d.priority == Priority.NEW_ADAPTATION


def test_same_gpu_verified_skipped(store, candidate):
    d = evaluate(candidate, GPU, store, _client({GPU: {"passed": True}}))
    assert d.verdict == Verdict.SKIP_DUPLICATE


def test_local_record_skipped_without_platform_query(store, candidate):
    from tests.test_storage import _record
    store.insert_task(_record())
    c = replace(candidate, model_id="org/model-a")
    client = _client({})
    d = evaluate(c, GPU, store, client)
    assert d.verdict == Verdict.SKIP_DUPLICATE
    client.search_model.assert_not_called()


def test_bounty_gets_top_priority(store, candidate):
    c = replace(candidate, is_bounty=True)
    d = evaluate(c, GPU, store, _client({"other-gpu": {}}))
    assert d.verdict == Verdict.ENQUEUE and d.priority == Priority.BOUNTY


def test_bounty_does_not_override_duplicate(store, candidate):
    c = replace(candidate, is_bounty=True)
    d = evaluate(c, GPU, store, _client({GPU: {"passed": True}}))
    assert d.verdict == Verdict.SKIP_DUPLICATE


def test_platform_error_skips_conservatively(store, candidate):
    d = evaluate(candidate, GPU, store, _client(error=True))
    assert d.verdict == Verdict.SKIP_UNCERTAIN


def _error_client(exc):
    client = Mock()
    client.search_model.side_effect = exc
    return client


def test_not_found_is_a_new_model_not_uncertainty(store, candidate):
    """I2: 40400 NOT_FOUND is the platform's normal answer for a model it has never seen
    — i.e. exactly the NEW_MODEL case this system exists to find. Folding it into
    SKIP_UNCERTAIN made those candidates re-queried every tick forever (SKIP_UNCERTAIN
    candidates are never marked processed) and left the NEW_MODEL path unreachable."""
    d = evaluate(candidate, GPU, store, _error_client(PlatformClientError(40400, "not found")))
    assert d.verdict == Verdict.ENQUEUE and d.priority == Priority.NEW_MODEL


def test_not_found_bounty_still_gets_bounty_priority(store, candidate):
    c = replace(candidate, is_bounty=True)
    d = evaluate(c, GPU, store, _error_client(PlatformClientError(40400, "not found")))
    assert d.verdict == Verdict.ENQUEUE and d.priority == Priority.BOUNTY


@pytest.mark.parametrize("exc", [
    PlatformClientError(50000, "system error"),
    PlatformClientError(50001, "operation error"),
    ConnectionError("down"),
])
def test_other_platform_errors_remain_uncertain(store, candidate, exc):
    """Everything that is not a definite "no such model" still means we cannot tell
    whether an adaptation already exists — 宁漏勿重."""
    assert evaluate(candidate, GPU, store, _error_client(exc)).verdict == Verdict.SKIP_UNCERTAIN


@pytest.mark.parametrize("code", [40100, 40101])
def test_credential_error_trips_kill_switch(store, candidate, code):
    """I3: an expired Xc-Token seen during eligibility must raise the alarm, not be
    swallowed as one more uncertain candidate."""
    d = evaluate(candidate, GPU, store, _error_client(PlatformClientError(code, "auth")))
    assert d.verdict == Verdict.SKIP_UNCERTAIN
    assert store.kill_switch() is True


def test_verify_result_populates_gpu_coverage(store, candidate):
    """I9: set_gpu_coverage had no caller, so gpu_coverage() was always empty and
    select_target_gpu always returned KNOWN_GPUS[0] — expanding KNOWN_GPUS (a documented
    pre-launch step) would have had no effect at all. eligibility already holds the
    per-GPU verification data, so it records it."""
    from auto_adapter import config_gen, rules

    evaluate(candidate, GPU, store, _client({"gpu-a": {"passed": True}}))
    other = replace(candidate, model_id="org/second")
    evaluate(other, GPU, store, _client({"gpu-a": {"passed": True}}))

    assert store.gpu_coverage()["gpu-a"] == 2

    # and the coverage actually steers card selection once more GPUs are known
    original = list(rules.KNOWN_GPUS)
    rules.KNOWN_GPUS[:] = ["gpu-a", "gpu-b"]
    try:
        assert config_gen.select_target_gpu(store) == "gpu-b"  # the least covered one
    finally:
        rules.KNOWN_GPUS[:] = original
