"""SQLite 默认实现。M1 里程碑实现；文件路径须指向挂载卷（STORAGE_PATH）。"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime

from ..models import CandidateModel, Priority, TaskRecord, TaskStatus
from .base import DuplicateTaskError

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

_DT_FIELDS_TASK = ("submit_time", "bounty_deadline")
_DT_FIELDS_CAND = ("bounty_deadline", "discovered_at")


def _dump(obj, dt_fields) -> str:
    d = asdict(obj)
    for f in dt_fields:
        if d.get(f) is not None:
            d[f] = d[f].isoformat()
    return json.dumps(d)


def _load_task(payload: str) -> TaskRecord:
    d = json.loads(payload)
    for f in _DT_FIELDS_TASK:
        if d.get(f) is not None:
            d[f] = datetime.fromisoformat(d[f])
    d["status"] = TaskStatus(d["status"])
    d["priority"] = Priority(d["priority"])
    return TaskRecord(**d)


def _load_candidate(payload: str) -> CandidateModel:
    d = json.loads(payload)
    for f in _DT_FIELDS_CAND:
        if d.get(f) is not None:
            d[f] = datetime.fromisoformat(d[f])
    return CandidateModel(**d)


class SqliteStorage:
    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA)

    def upsert_candidate(self, candidate: CandidateModel) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO candidates (model_id, payload) VALUES (?, ?) "
                "ON CONFLICT(model_id) DO UPDATE SET payload = excluded.payload",
                (candidate.model_id, _dump(candidate, _DT_FIELDS_CAND)),
            )

    def pending_candidates(self) -> list[CandidateModel]:
        rows = self._conn.execute(
            "SELECT payload FROM candidates WHERE processed = 0").fetchall()
        return [_load_candidate(r[0]) for r in rows]

    def has_candidate(self, model_id: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM candidates WHERE model_id = ?", (model_id,)).fetchone() is not None

    def mark_candidate_processed(self, model_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE candidates SET processed = 1 WHERE model_id = ?", (model_id,))

    def insert_task(self, record: TaskRecord) -> None:
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO tasks (model_id, target_gpu, payload, status) "
                    "VALUES (?, ?, ?, ?)",
                    (record.model_id, record.target_gpu,
                     _dump(record, _DT_FIELDS_TASK), record.status.value),
                )
        except sqlite3.IntegrityError as e:
            raise DuplicateTaskError(
                f"{record.model_id} @ {record.target_gpu}") from e

    def update_task(self, record: TaskRecord) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE tasks SET payload = ?, status = ? "
                "WHERE model_id = ? AND target_gpu = ?",
                (_dump(record, _DT_FIELDS_TASK), record.status.value,
                 record.model_id, record.target_gpu),
            )

    def tasks_by_status(self, *statuses: TaskStatus) -> list[TaskRecord]:
        marks = ",".join("?" * len(statuses))
        rows = self._conn.execute(
            f"SELECT payload FROM tasks WHERE status IN ({marks})",
            [s.value for s in statuses]).fetchall()
        return [_load_task(r[0]) for r in rows]

    def get_task(self, model_id: str, target_gpu: str) -> TaskRecord | None:
        row = self._conn.execute(
            "SELECT payload FROM tasks WHERE model_id = ? AND target_gpu = ?",
            (model_id, target_gpu)).fetchone()
        return _load_task(row[0]) if row else None

    def is_blacklisted(self, model_id: str, target_gpu: str) -> bool:
        rec = self.get_task(model_id, target_gpu)
        return rec is not None and rec.status == TaskStatus.BLACKLISTED

    def _kv_get(self, key: str, default):
        row = self._conn.execute(
            "SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def _kv_set(self, key: str, value) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )

    def get_counter(self, key: str) -> int:
        return int(self._kv_get(f"counter:{key}", 0))

    def set_counter(self, key: str, value: int) -> None:
        self._kv_set(f"counter:{key}", value)

    def gpu_coverage(self) -> dict[str, int]:
        return self._kv_get("gpu_coverage", {})

    def set_gpu_coverage(self, coverage: dict[str, int]) -> None:
        self._kv_set("gpu_coverage", coverage)

    def kill_switch(self) -> bool:
        return self.kill_switch_state()["on"]

    def kill_switch_state(self) -> dict:
        state = self._kv_get("kill_switch", {"on": False, "reason": ""})
        return {"on": bool(state.get("on")), "reason": state.get("reason", "")}

    def set_kill_switch(self, on: bool, reason: str) -> None:
        self._kv_set("kill_switch", {"on": on, "reason": reason})
