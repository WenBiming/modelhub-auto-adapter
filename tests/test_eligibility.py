from dataclasses import replace
from unittest.mock import Mock

import pytest

from auto_adapter.eligibility import Verdict, evaluate
from auto_adapter.models import ModelSearchResult, Priority
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
