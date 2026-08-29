import json
from datetime import datetime, timezone

import responses

from auto_adapter.discovery.base import run
from auto_adapter.discovery.bounty import ManualBountySource
from auto_adapter.discovery.huggingface import HuggingFaceSource
from auto_adapter.storage.sqlite import SqliteStorage


@responses.activate
def test_huggingface_fetch_and_throttle():
    responses.get("https://huggingface.co/api/models", json=[
        {"modelId": "Qwen/Qwen2.5-7B-Instruct", "pipeline_tag": "text-generation"},
        {"id": "org/other-13B", "pipeline_tag": "text-generation"},
    ])
    src = HuggingFaceSource(limit=2, min_interval_seconds=3600)
    got = src.fetch()
    assert [c.model_id for c in got] == ["Qwen/Qwen2.5-7B-Instruct", "org/other-13B"]
    assert got[0].params_size == "7B"
    assert got[0].model_url == "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"
    assert src.fetch() == []  # 节流：1h 内第二次返回空
    assert len(responses.calls) == 1


def test_manual_bounty_source(tmp_path):
    path = tmp_path / "bounty.json"
    path.write_text(json.dumps([{
        "model_id": "org/bounty-model",
        "model_url": "https://huggingface.co/org/bounty-model",
        "deadline": "2026-09-30T00:00:00+00:00",
    }]))
    got = ManualBountySource(str(path)).fetch()
    assert got[0].is_bounty and got[0].bounty_deadline == datetime(2026, 9, 30, tzinfo=timezone.utc)
    assert ManualBountySource(str(tmp_path / "missing.json")).fetch() == []


def test_run_dedups_and_persists(tmp_path, candidate):
    store = SqliteStorage(str(tmp_path / "t.db"))

    class Fake:
        name = "fake"
        def fetch(self):
            return [candidate, candidate]

    class Broken:
        name = "broken"
        def fetch(self):
            raise ConnectionError("down")

    assert run([Fake(), Broken()], store) == 1  # 去重 + 单源故障不影响整体
    assert len(store.pending_candidates()) == 1
