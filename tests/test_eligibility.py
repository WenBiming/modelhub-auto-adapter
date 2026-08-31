"""去重与准入——全系统最关键模块。

选卡是逐候选做的：线上实测 Ascend_910-b4 覆盖了几乎所有热门模型，用一张全局的卡
评估所有候选会让每个候选都被判重复，智能体一个任务也提交不出去。
"""
from dataclasses import replace
from unittest.mock import Mock

import pytest

from auto_adapter import rules
from auto_adapter.eligibility import Verdict, evaluate
from auto_adapter.models import (ModelSearchResult, Priority, TaskRecord,
                                 TaskStatus)
from auto_adapter.platform_client import PlatformClientError
from auto_adapter.storage.sqlite import SqliteStorage

ALL_GPUS = set(rules.KNOWN_GPUS)


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


def _covered(*gpus):
    """平台侧 verify_result：按 GPU 型号分键（附录 A.4，线上实测确认）。"""
    return {g: {"passed": True} for g in gpus}


def test_new_model_enqueued(store, candidate):
    d = evaluate(candidate, store, _client({}))
    assert d.verdict == Verdict.ENQUEUE and d.priority == Priority.NEW_MODEL
    assert d.target_gpu in ALL_GPUS


def test_partially_covered_model_is_a_new_adaptation_on_an_uncovered_gpu(store, candidate):
    """线上实测的关键场景：热门模型在多数卡上适配过，但仍有空位。

    早先的实现拿一张全局卡去比，只要那张卡被覆盖就整个跳过——17 个候选全被判重复，
    而它们其实提供了大量"老模型上新卡"的机会。
    """
    uncovered = "hygon_k100-ai"
    d = evaluate(candidate, store, _client(_covered(*(ALL_GPUS - {uncovered}))))

    assert d.verdict == Verdict.ENQUEUE
    assert d.priority == Priority.NEW_ADAPTATION
    assert d.target_gpu == uncovered


def test_fully_covered_model_is_skipped(store, candidate):
    d = evaluate(candidate, store, _client(_covered(*ALL_GPUS)))
    assert d.verdict == Verdict.SKIP_DUPLICATE
    assert d.target_gpu is None


def test_coverage_outside_known_gpus_does_not_block(store, candidate):
    """平台 verify_result 里出现的型号比可提交列表多（实测有 Ascend_910-b3、
    Biren_166m、曦望 S2 等不在筛选下拉框里的卡）。它们不该影响我们能不能提交。"""
    d = evaluate(candidate, store, _client(_covered("Ascend_910-b3", "Biren_166m", "曦望 S2")))
    assert d.verdict == Verdict.ENQUEUE
    assert d.target_gpu in ALL_GPUS


def test_local_records_exclude_their_gpu(store, candidate):
    """本地已提交/已拉黑的组合同样要排除——那是我们自己占用的卡。"""
    taken = rules.KNOWN_GPUS[0]
    store.insert_task(TaskRecord(
        model_id=candidate.model_id, target_gpu=taken, framework="vllm",
        status=TaskStatus.QUEUED, priority=Priority.NEW_MODEL))

    d = evaluate(candidate, store, _client({}))

    assert d.verdict == Verdict.ENQUEUE and d.target_gpu != taken


def test_all_gpus_taken_locally_is_skipped(store, candidate):
    for gpu in rules.KNOWN_GPUS:
        store.insert_task(TaskRecord(
            model_id=candidate.model_id, target_gpu=gpu, framework="vllm",
            status=TaskStatus.QUEUED, priority=Priority.NEW_MODEL))

    assert evaluate(candidate, store, _client({})).verdict == Verdict.SKIP_DUPLICATE


def test_bounty_gets_top_priority(store, candidate):
    c = replace(candidate, is_bounty=True)
    d = evaluate(c, store, _client(_covered("Ascend_910-b3")))
    assert d.verdict == Verdict.ENQUEUE and d.priority == Priority.BOUNTY


def test_bounty_does_not_override_full_coverage(store, candidate):
    """悬赏优先级再高也不能提交一个所有卡都已适配的组合——那就是重复提交。"""
    c = replace(candidate, is_bounty=True)
    assert evaluate(c, store, _client(_covered(*ALL_GPUS))).verdict == Verdict.SKIP_DUPLICATE


def test_platform_error_skips_conservatively(store, candidate):
    d = evaluate(candidate, store, _client(error=True))
    assert d.verdict == Verdict.SKIP_UNCERTAIN


def test_not_found_is_a_new_model_not_uncertainty(store, candidate):
    """40400 是"平台没有这个模型"的正常业务应答，正是 NEW_MODEL 分支该收到的响应。"""
    client = Mock()
    client.search_model.side_effect = PlatformClientError(40400, "not found")
    d = evaluate(candidate, store, client)
    assert d.verdict == Verdict.ENQUEUE and d.priority == Priority.NEW_MODEL


def test_not_found_bounty_still_gets_bounty_priority(store, candidate):
    client = Mock()
    client.search_model.side_effect = PlatformClientError(40400, "not found")
    d = evaluate(replace(candidate, is_bounty=True), store, client)
    assert d.verdict == Verdict.ENQUEUE and d.priority == Priority.BOUNTY


@pytest.mark.parametrize("exc", [
    PlatformClientError(50000, "system error"),
    PlatformClientError(50001, "operation error"),
    ConnectionError("network down"),
])
def test_other_platform_errors_remain_uncertain(store, candidate, exc):
    client = Mock()
    client.search_model.side_effect = exc
    assert evaluate(candidate, store, client).verdict == Verdict.SKIP_UNCERTAIN


@pytest.mark.parametrize("code", [40100, 40101])
def test_credential_error_trips_kill_switch(store, candidate, code):
    client = Mock()
    client.search_model.side_effect = PlatformClientError(code, "auth")
    assert evaluate(candidate, store, client).verdict == Verdict.SKIP_UNCERTAIN
    assert store.kill_switch() is True


def test_verify_result_populates_gpu_coverage(store, candidate):
    """覆盖率缓存的唯一写入点：没有它 select_target_gpu 永远只会返回列表首位。"""
    evaluate(candidate, store, _client(_covered("Ascend_910-b4", "MetaX_c-500")))
    coverage = store.gpu_coverage()
    assert coverage["Ascend_910-b4"] == 1 and coverage["MetaX_c-500"] == 1


def test_uncovered_gpu_choice_prefers_the_least_covered(store, candidate):
    """多张可选时往稀疏处生长：让适配矩阵尽量铺开。"""
    store.set_gpu_coverage({g: 10 for g in rules.KNOWN_GPUS})
    store.set_gpu_coverage({**store.gpu_coverage(), "Vastai_va16": 0})

    d = evaluate(candidate, store, _client({}))

    assert d.target_gpu == "Vastai_va16"
