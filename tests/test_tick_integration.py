import threading
from unittest.mock import Mock

import pytest

from auto_adapter.main import Deps, tick
from auto_adapter.models import ModelSearchResult, TaskStatus
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
