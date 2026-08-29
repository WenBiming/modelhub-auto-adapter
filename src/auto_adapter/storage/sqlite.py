"""SQLite 默认实现。M1 里程碑实现；文件路径须指向挂载卷（STORAGE_PATH）。"""
from __future__ import annotations

import sqlite3

from ..models import CandidateModel, TaskRecord, TaskStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    model_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,          -- CandidateModel JSON
    processed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS tasks (
    model_id TEXT NOT NULL,
    target_gpu TEXT NOT NULL,
    payload TEXT NOT NULL,          -- TaskRecord JSON
    status TEXT NOT NULL,
    PRIMARY KEY (model_id, target_gpu)
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL             -- gpu_coverage / kill_switch 等
);
"""


class SqliteStorage:
    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA)

    # 接口实现：M1 里程碑，见 storage/base.py 的 Protocol 契约
    def upsert_candidate(self, candidate: CandidateModel) -> None:
        raise NotImplementedError

    def pending_candidates(self) -> list[CandidateModel]:
        raise NotImplementedError

    def mark_candidate_processed(self, model_id: str) -> None:
        raise NotImplementedError

    def insert_task(self, record: TaskRecord) -> None:
        raise NotImplementedError

    def update_task(self, record: TaskRecord) -> None:
        raise NotImplementedError

    def tasks_by_status(self, *statuses: TaskStatus) -> list[TaskRecord]:
        raise NotImplementedError

    def get_task(self, model_id: str, target_gpu: str) -> TaskRecord | None:
        raise NotImplementedError

    def is_blacklisted(self, model_id: str, target_gpu: str) -> bool:
        raise NotImplementedError

    def gpu_coverage(self) -> dict[str, int]:
        raise NotImplementedError

    def set_gpu_coverage(self, coverage: dict[str, int]) -> None:
        raise NotImplementedError

    def kill_switch(self) -> bool:
        raise NotImplementedError

    def set_kill_switch(self, on: bool, reason: str) -> None:
        raise NotImplementedError
