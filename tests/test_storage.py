"""M1：storage 幂等与状态流转（spec §3 不变式、§4.8）。"""
from datetime import datetime, timezone

import pytest

from auto_adapter.models import Priority, TaskRecord, TaskStatus
from auto_adapter.storage.base import DuplicateTaskError
from auto_adapter.storage.sqlite import SqliteStorage


@pytest.fixture
def store(tmp_path):
    return SqliteStorage(str(tmp_path / "test.db"))


def _record(status=TaskStatus.QUEUED):
    return TaskRecord(
        model_id="org/model-a", target_gpu="MetaX_c-500", framework="vllm",
        status=status, priority=Priority.NEW_MODEL,
        model_url="https://huggingface.co/org/model-a", task_type="text-generation",
    )


def test_duplicate_task_rejected(store):
    store.insert_task(_record())
    with pytest.raises(DuplicateTaskError):
        store.insert_task(_record())


def test_task_roundtrip_and_status_query(store):
    rec = _record()
    store.insert_task(rec)
    rec.status = TaskStatus.PENDING
    rec.task_id = 42
    rec.submit_time = datetime(2026, 8, 29, tzinfo=timezone.utc)
    store.update_task(rec)
    got = store.tasks_by_status(TaskStatus.PENDING)
    assert len(got) == 1 and got[0].task_id == 42
    assert got[0].submit_time == rec.submit_time
    assert store.tasks_by_status(TaskStatus.QUEUED) == []


def test_blacklist(store):
    rec = _record(TaskStatus.BLACKLISTED)
    store.insert_task(rec)
    assert store.is_blacklisted("org/model-a", "MetaX_c-500")
    assert not store.is_blacklisted("org/model-a", "other-gpu")


def test_kill_switch_roundtrip(store):
    assert store.kill_switch() is False
    store.set_kill_switch(True, "credential error")
    assert store.kill_switch() is True


def test_counter_roundtrip(store):
    assert store.get_counter("consecutive_engine_failures") == 0
    store.set_counter("consecutive_engine_failures", 3)
    assert store.get_counter("consecutive_engine_failures") == 3


def test_gpu_coverage_roundtrip(store):
    assert store.gpu_coverage() == {}
    store.set_gpu_coverage({"MetaX_c-500": 3})
    assert store.gpu_coverage() == {"MetaX_c-500": 3}


def test_candidate_flow(store, candidate):
    store.upsert_candidate(candidate)
    store.upsert_candidate(candidate)  # 幂等
    assert [c.model_id for c in store.pending_candidates()] == [candidate.model_id]
    store.mark_candidate_processed(candidate.model_id)
    assert store.pending_candidates() == []
    store.upsert_candidate(candidate)  # 已处理的候选再 upsert 不复活
    assert store.pending_candidates() == []


def test_creates_missing_parent_directory(tmp_path):
    """平台不挂载卷，默认路径的父目录可能根本不存在——线上实测就是这里崩的
    （sqlite3.OperationalError: unable to open database file）。"""
    nested = tmp_path / "no" / "such" / "dir" / "agent.db"
    store = SqliteStorage(str(nested))
    store.set_counter("k", 1)
    assert store.get_counter("k") == 1
    assert nested.exists()


def test_unwritable_path_raises_storage_unavailable(tmp_path):
    """不可写时必须抛 StorageUnavailableError（由 main 转成"存活但不工作"），
    而不是让裸 sqlite3 异常炸穿启动流程。"""
    from auto_adapter.storage.base import StorageUnavailableError
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)  # r-x：不能在其中创建文件
    try:
        with pytest.raises(StorageUnavailableError):
            SqliteStorage(str(blocked / "sub" / "agent.db"))
    finally:
        blocked.chmod(0o700)
