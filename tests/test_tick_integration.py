import json
import threading
from unittest.mock import Mock

from auto_adapter import rules
from auto_adapter.main import Deps, tick
from auto_adapter.models import ModelSearchResult, Priority, TaskRecord, TaskStatus
from auto_adapter.settings import Settings
from auto_adapter.storage.sqlite import SqliteStorage


# tick 通过 config_gen.select_target_gpu 选卡；空覆盖率时是 KNOWN_GPUS 首位。
# 不要硬编码型号——GPU 列表随平台扩容变化，流程测试不该因此而红。
SELECTED_GPU = rules.KNOWN_GPUS[0]


def test_tick_discovers_enqueues_and_submits(tmp_path, candidate):
    storage = SqliteStorage(str(tmp_path / "t.db"))
    client = Mock()
    client.search_model.return_value = ModelSearchResult(False, {}, {})
    client.add_task.return_value = 99
    client.list_my_tasks.return_value = {"records": [{"taskId": 99, "status": "RUNNING"}]}

    class Src:
        name = "fake"
        def fetch(self):
            return [candidate]

    deps = Deps(
        settings=Settings(xc_token="t", strategy_id="uuid-1", base_url="https://x"),
        storage=storage, client=client, sources=[Src()],
    )
    tick(deps, threading.Event())
    rec = storage.get_task(candidate.model_id, SELECTED_GPU)
    assert rec is not None and rec.task_id == 99
    assert rec.status == TaskStatus.RUNNING  # submit 后同 tick 内 monitor 已对账
    assert storage.pending_candidates() == []  # 候选已消费


def test_tick_skips_duplicate_candidate(tmp_path, candidate):
    storage = SqliteStorage(str(tmp_path / "t.db"))
    client = Mock()
    client.search_model.return_value = ModelSearchResult(
        True, {}, {SELECTED_GPU: {"passed": True}})
    client.list_my_tasks.return_value = {"records": []}

    class Src:
        name = "fake"
        def fetch(self):
            return [candidate]

    deps = Deps(settings=Settings(xc_token="t", strategy_id="s", base_url="https://x"),
                storage=storage, client=client, sources=[Src()])
    tick(deps, threading.Event())
    client.add_task.assert_not_called()
    assert storage.get_task(candidate.model_id, SELECTED_GPU) is None


def test_tick_stops_immediately_when_stop_event_already_set(tmp_path, candidate):
    """优雅停机（spec §4.9）：stop_event 在 tick 开始前已置位时，候选循环第一个
    检查点就必须退出——不得触碰平台 API（search_model/add_task/list_my_tasks/
    stop_tasks 均不应被调用）。"""
    storage = SqliteStorage(str(tmp_path / "t.db"))
    client = Mock()

    class Src:
        name = "fake"
        def fetch(self):
            return [candidate]

    deps = Deps(settings=Settings(xc_token="t", strategy_id="s", base_url="https://x"),
                storage=storage, client=client, sources=[Src()])
    stop_event = threading.Event()
    stop_event.set()

    tick(deps, stop_event)

    client.search_model.assert_not_called()
    client.add_task.assert_not_called()
    client.list_my_tasks.assert_not_called()
    client.stop_tasks.assert_not_called()


def test_tick_stops_between_monitor_and_failure(tmp_path, monkeypatch):
    """优雅停机的最后一个检查点：monitor.poll 完成后、failure.handle 之前，
    若 stop_event 已置位（例如 SIGTERM 恰好在此时到达），failure.handle
    （及其 client.stop_tasks 网络调用）不得执行。用一个已有 TIMEOUT 任务作为
    探针：若检查点缺失，failure.handle 会为它调用 client.stop_tasks。"""
    storage = SqliteStorage(str(tmp_path / "t.db"))
    client = Mock()
    client.list_my_tasks.return_value = {"records": []}
    storage.insert_task(TaskRecord(
        model_id="m", target_gpu=SELECTED_GPU, framework="vllm",
        status=TaskStatus.TIMEOUT, priority=Priority.NEW_MODEL, task_id=42))

    deps = Deps(settings=Settings(xc_token="t", strategy_id="s", base_url="https://x"),
                storage=storage, client=client, sources=[])
    stop_event = threading.Event()

    from auto_adapter import main as main_module

    def fake_poll(*args, **kwargs):
        stop_event.set()

    monkeypatch.setattr(main_module.monitor, "poll", fake_poll)

    tick(deps, stop_event)

    client.stop_tasks.assert_not_called()


def test_tick_resets_metrics_between_ticks(tmp_path, candidate, capsys):
    """metrics 是逐 tick 计数（spec §6：每 tick 打一行 JSON），不是进程累计值——
    否则日志流里每行数字只会单调增长，读不出"这一 tick 发生了什么"。"""
    storage = SqliteStorage(str(tmp_path / "t.db"))
    client = Mock()
    client.search_model.return_value = ModelSearchResult(False, {}, {})
    client.add_task.return_value = 99
    client.list_my_tasks.return_value = {"records": [{"taskId": 99, "status": "RUNNING"}]}

    class Src:
        name = "fake"
        def fetch(self):
            return [candidate]

    deps = Deps(settings=Settings(xc_token="t", strategy_id="s", base_url="https://x"),
                storage=storage, client=client, sources=[Src()])

    tick(deps, threading.Event())
    capsys.readouterr()  # 丢弃第一个 tick 的输出

    tick(deps, threading.Event())  # 候选已 processed，第二个 tick 不再入队
    out = capsys.readouterr().out
    second_tick_metrics = json.loads(out.strip().splitlines()[-1])["metrics"]

    assert "enqueued" not in second_tick_metrics


def _src(candidates):
    class Src:
        name = "fake"
        def fetch(self):
            return list(candidates)
    return Src()


def _deps(storage, client, sources=()):
    return Deps(settings=Settings(xc_token="t", strategy_id="s", base_url="https://x"),
                storage=storage, client=client, sources=list(sources))


def test_unresolvable_candidate_leaves_a_needs_human_record(tmp_path, candidate):
    """I8 (spec §4.4): a candidate whose task type cannot be resolved used to vanish with
    nothing but a log line — the candidate was marked processed and never looked at again,
    so a bounty could disappear silently. It must leave a record a human can find, with
    enough context (model_url/model_id/target_gpu) to act on."""
    from dataclasses import replace

    storage = SqliteStorage(str(tmp_path / "t.db"))
    client = Mock()
    client.search_model.return_value = ModelSearchResult(False, {}, {})
    client.list_my_tasks.return_value = {"records": []}
    mystery = replace(candidate, pipeline_tag=None, model_id="org/mystery")

    tick(_deps(storage, client, [_src([mystery])]), threading.Event())

    rec = storage.get_task("org/mystery", SELECTED_GPU)
    assert rec is not None and rec.status == TaskStatus.NEEDS_HUMAN
    assert rec.model_url == mystery.model_url
    client.add_task.assert_not_called()


def test_non_vllm_candidate_is_not_submitted(tmp_path, candidate):
    """Ruling: a non-vllm candidate must never reach add_task in v0.1; it lands as a
    NEEDS_HUMAN record instead of burning submissions on a placeholder template."""
    from dataclasses import replace

    storage = SqliteStorage(str(tmp_path / "t.db"))
    client = Mock()
    client.search_model.return_value = ModelSearchResult(False, {}, {})
    client.list_my_tasks.return_value = {"records": []}
    embedder = replace(candidate, model_id="org/bge-small", pipeline_tag="feature-extraction")

    tick(_deps(storage, client, [_src([embedder])]), threading.Event())

    client.add_task.assert_not_called()
    rec = storage.get_task("org/bge-small", SELECTED_GPU)
    assert rec is not None and rec.status == TaskStatus.NEEDS_HUMAN


def test_candidates_per_tick_is_capped_and_remainder_stays_pending(tmp_path, candidate):
    """I2: each evaluation can cost a 10s platform call, so an unbounded candidate loop
    blows the 30s shutdown budget. The overflow must stay pending, not be dropped."""
    from dataclasses import replace

    from auto_adapter import main as main_module

    storage = SqliteStorage(str(tmp_path / "t.db"))
    client = Mock()
    client.search_model.return_value = ModelSearchResult(True, {}, {SELECTED_GPU: {}})
    client.list_my_tasks.return_value = {"records": []}
    many = [replace(candidate, model_id=f"org/m{i}")
            for i in range(main_module.MAX_CANDIDATES_PER_TICK + 5)]

    tick(_deps(storage, client, [_src(many)]), threading.Event())

    assert client.search_model.call_count == main_module.MAX_CANDIDATES_PER_TICK
    assert len(storage.pending_candidates()) == 5


def test_metrics_are_flushed_even_when_the_tick_stops_early(tmp_path, candidate, capsys):
    """M3: a tick that returns early on stop_event (or raises) still has to emit its
    metrics line — otherwise the most interesting ticks are exactly the silent ones."""
    storage = SqliteStorage(str(tmp_path / "t.db"))
    stop_event = threading.Event()
    stop_event.set()

    tick(_deps(storage, Mock(), [_src([candidate])]), stop_event)

    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "metrics" in line


def test_metrics_line_reports_kill_switch_state_and_reason(tmp_path, capsys):
    """M4: the kill switch is the only safety brake and /health returns 200 regardless,
    so its state has to be visible somewhere an operator actually looks."""
    storage = SqliteStorage(str(tmp_path / "t.db"))
    storage.set_kill_switch(True, "task 42 vanished from platform")
    client = Mock()
    client.list_my_tasks.return_value = {"records": []}

    tick(_deps(storage, client), threading.Event())

    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["kill_switch"]["on"] is True
    assert "vanished" in line["kill_switch"]["reason"]


def test_persistently_uncertain_candidate_stops_blocking_the_queue(tmp_path, candidate):
    """Re-review ITEM 2: SKIP_UNCERTAIN candidates are never marked processed, and the
    per-tick cap slices a stable `WHERE processed = 0` scan. A full slice of candidates the
    platform never answers for would occupy that slice forever, starving everything behind
    them permanently while the agent still looks healthy.

    After MAX_UNCERTAIN_TICKS consecutive uncertain ticks the candidate is escalated to
    NEEDS_HUMAN and marked processed, so the queue advances — and because a task record now
    exists for the pair, a later evaluation short-circuits to SKIP_DUPLICATE rather than
    turning the release into a blind submission."""
    from dataclasses import replace

    from auto_adapter import main as main_module
    from auto_adapter.platform_client import PlatformClientError

    storage = SqliteStorage(str(tmp_path / "t.db"))
    client = Mock()
    client.add_task.return_value = 501
    # the listing carries the task that "org/behind" will create, so reconciliation in the
    # same tick is coherent (an empty listing would read as a vanished task)
    client.list_my_tasks.return_value = {"records": [
        {"taskId": 501, "status": "RUNNING", "modelId": "org/behind", "gpuType": SELECTED_GPU}]}

    blockers = [replace(candidate, model_id=f"org/blocker{i}")
                for i in range(main_module.MAX_CANDIDATES_PER_TICK)]
    behind = replace(candidate, model_id="org/behind")

    def search_model(model_id):
        if model_id.startswith("org/blocker"):
            raise PlatformClientError(50000, "system error")  # persistently unanswerable
        return ModelSearchResult(False, {}, {})

    client.search_model.side_effect = search_model
    deps = _deps(storage, client, [_src(blockers + [behind])])

    # Ticks 1..MAX-1: the blockers fill the slice and "org/behind" is never reached.
    for _ in range(main_module.MAX_UNCERTAIN_TICKS - 1):
        tick(deps, threading.Event())
    assert storage.get_task("org/behind", SELECTED_GPU) is None
    assert any(c.model_id == "org/behind" for c in storage.pending_candidates())

    # The escalating tick releases the whole slice...
    tick(deps, threading.Event())
    for blocker in blockers:
        rec = storage.get_task(blocker.model_id, SELECTED_GPU)
        assert rec is not None and rec.status == TaskStatus.NEEDS_HUMAN
        assert rec.model_url == blocker.model_url

    # ...and the candidate behind them is evaluated and enqueued on the next tick.
    tick(deps, threading.Event())
    rec = storage.get_task("org/behind", SELECTED_GPU)
    assert rec is not None and rec.status not in (TaskStatus.NEEDS_HUMAN, TaskStatus.ABANDONED)
    assert client.add_task.call_args.args[0].model_address == behind.model_url


def test_uncertain_streak_resets_when_the_platform_answers(tmp_path, candidate):
    """The escalation counts CONSECUTIVE failures: a candidate that is merely flaky must
    never be escalated, or a transient outage would dump the whole queue on a human."""
    from auto_adapter import main as main_module
    from auto_adapter.platform_client import PlatformClientError

    storage = SqliteStorage(str(tmp_path / "t.db"))
    client = Mock()
    client.list_my_tasks.return_value = {"records": []}
    client.add_task.return_value = 502
    deps = _deps(storage, client, [_src([candidate])])

    client.search_model.side_effect = PlatformClientError(50000, "system error")
    for _ in range(main_module.MAX_UNCERTAIN_TICKS - 1):
        tick(deps, threading.Event())

    client.search_model.side_effect = None
    client.search_model.return_value = ModelSearchResult(False, {}, {})
    tick(deps, threading.Event())  # answered: streak resets, candidate enqueued normally

    assert storage.get_counter(main_module._uncertain_key(candidate.model_id)) == 0
    rec = storage.get_task(candidate.model_id, SELECTED_GPU)
    assert rec is not None and rec.status != TaskStatus.NEEDS_HUMAN
