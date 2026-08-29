import json
import threading
from unittest.mock import Mock

from auto_adapter.main import Deps, tick
from auto_adapter.models import ModelSearchResult, Priority, TaskRecord, TaskStatus
from auto_adapter.settings import Settings
from auto_adapter.storage.sqlite import SqliteStorage


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
    rec = storage.get_task(candidate.model_id, "MetaX_c-500")
    assert rec is not None and rec.task_id == 99
    assert rec.status == TaskStatus.RUNNING  # submit 后同 tick 内 monitor 已对账
    assert storage.pending_candidates() == []  # 候选已消费


def test_tick_skips_duplicate_candidate(tmp_path, candidate):
    storage = SqliteStorage(str(tmp_path / "t.db"))
    client = Mock()
    client.search_model.return_value = ModelSearchResult(
        True, {}, {"MetaX_c-500": {"passed": True}})
    client.list_my_tasks.return_value = {"records": []}

    class Src:
        name = "fake"
        def fetch(self):
            return [candidate]

    deps = Deps(settings=Settings(xc_token="t", strategy_id="s", base_url="https://x"),
                storage=storage, client=client, sources=[Src()])
    tick(deps, threading.Event())
    client.add_task.assert_not_called()
    assert storage.get_task(candidate.model_id, "MetaX_c-500") is None


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
        model_id="m", target_gpu="MetaX_c-500", framework="vllm",
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
