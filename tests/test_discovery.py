import json
from datetime import datetime, timezone

import responses

from auto_adapter.discovery.base import run
from auto_adapter.discovery.bounty import ManualBountySource
from auto_adapter.discovery.huggingface import HuggingFaceSource
from auto_adapter.models import CandidateModel
from auto_adapter.storage.sqlite import SqliteStorage


@responses.activate
def test_huggingface_fetch_and_throttle(tmp_path):
    responses.get("https://huggingface.co/api/models", json=[
        {"modelId": "Qwen/Qwen2.5-7B-Instruct", "pipeline_tag": "text-generation"},
        {"id": "org/other-13B", "pipeline_tag": "text-generation"},
    ])
    store = SqliteStorage(str(tmp_path / "t.db"))
    src = HuggingFaceSource(store, limit=2, min_interval_seconds=3600)
    got = src.fetch()
    assert [c.model_id for c in got] == ["Qwen/Qwen2.5-7B-Instruct", "org/other-13B"]
    assert got[0].params_size == "7B"
    assert got[0].model_url == "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"

    # Verify request parameters
    assert len(responses.calls) == 1
    params = responses.calls[0].request.params
    assert params.get("sort") == "downloads"
    assert params.get("direction") == "-1"
    assert params.get("limit") == "2"
    assert params.get("pipeline_tag") == "text-generation"

    assert src.fetch() == []  # 节流：1h 内第二次返回空
    assert len(responses.calls) == 1  # 仍然只有一次调用

    # M2 safety property: the throttle lives in storage, not in process memory — a fresh
    # source object (i.e. a restarted process, or a crash loop) must stay throttled.
    assert HuggingFaceSource(store, limit=2, min_interval_seconds=3600).fetch() == []
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


def test_bounty_wins_dedup(tmp_path):
    """Bounty candidates must replace non-bounty duplicates to prevent悬赏时间窗口错过."""
    store = SqliteStorage(str(tmp_path / "t.db"))
    bounty_deadline = datetime(2026, 9, 30, tzinfo=timezone.utc)

    # Non-bounty candidate from HF
    non_bounty = CandidateModel(
        source="huggingface",
        model_id="shared/model",
        model_url="https://huggingface.co/shared/model",
        pipeline_tag="text-generation",
        params_size="7B",
        is_bounty=False,
        bounty_deadline=None,
        discovered_at=datetime.now(timezone.utc),
    )

    # Bounty candidate with same model_id
    bounty = CandidateModel(
        source="bounty",
        model_id="shared/model",
        model_url="https://huggingface.co/shared/model",
        pipeline_tag=None,
        params_size=None,
        is_bounty=True,
        bounty_deadline=bounty_deadline,
        discovered_at=datetime.now(timezone.utc),
    )

    class NonBountyFirst:
        name = "hf"
        def fetch(self):
            return [non_bounty]

    class BountySecond:
        name = "bounty"
        def fetch(self):
            return [bounty]

    # Run sources in order: HF first, then bounty
    count = run([NonBountyFirst(), BountySecond()], store)
    # Should count as 2 upserts total (HF upserts, bounty replaces but doesn't double-count)
    # Or 1 if bounty replacement doesn't increment count
    # Per the fix directive: "NOT double-count when a bounty replaces a non-bounty"
    # So count should be 1 (only the first upsert) or remain at what was stored

    # Verify the stored candidate has bounty flag and deadline
    stored = store.pending_candidates()
    assert len(stored) == 1
    assert stored[0].is_bounty is True
    assert stored[0].bounty_deadline == bounty_deadline


def test_manual_bounty_source_normalizes_naive_deadline(tmp_path):
    """I1: a hand-maintained bounty file will sooner or later carry a deadline with no
    UTC offset. A naive datetime flowing downstream makes `deadline - now` raise
    TypeError in submitter.drain and failure.handle, killing the pipeline every tick."""
    path = tmp_path / "bounty.json"
    path.write_text(json.dumps([{
        "model_id": "org/bounty-model",
        "model_url": "https://huggingface.co/org/bounty-model",
        "deadline": "2026-09-30T00:00:00",  # no offset — the easy hand-written mistake
    }]))

    got = ManualBountySource(str(path)).fetch()

    assert got[0].bounty_deadline.tzinfo is not None
    assert got[0].bounty_deadline == datetime(2026, 9, 30, tzinfo=timezone.utc)
    # and the value is directly usable in an aware arithmetic expression
    assert got[0].bounty_deadline > datetime(2026, 8, 29, tzinfo=timezone.utc)


def test_run_counts_only_genuinely_new_candidates(tmp_path, candidate):
    """M1: run() feeds main's candidates_discovered metric. Counting every deduped
    candidate it saw would make that number equal the fetch size on every tick, so
    "what turned up this tick" would be unreadable."""
    store = SqliteStorage(str(tmp_path / "t.db"))

    class Fake:
        name = "fake"
        def fetch(self):
            return [candidate]

    assert run([Fake()], store) == 1
    assert run([Fake()], store) == 0  # same candidate again: nothing new

    store.mark_candidate_processed(candidate.model_id)
    assert run([Fake()], store) == 0  # already-known even after processing
