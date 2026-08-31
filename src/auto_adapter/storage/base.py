"""持久化接口（spec §4.8）。所有跨 tick 状态必须经此接口落盘，禁止内存态。"""
from __future__ import annotations

from typing import Protocol

from ..models import CandidateModel, TaskRecord, TaskStatus


class Storage(Protocol):
    # --- 候选表 ---
    def upsert_candidate(self, candidate: CandidateModel) -> None: ...
    def pending_candidates(self) -> list[CandidateModel]: ...
    def mark_candidate_processed(self, model_id: str) -> None: ...

    def has_candidate(self, model_id: str) -> bool:
        """候选表里是否已有该 model_id（含已 processed 的）——discovery.run 据此
        只把真正新出现的候选计入 candidates_discovered。"""
        ...

    # --- 任务表（唯一键 (model_id, target_gpu) 保证幂等，spec §3 不变式）---
    def insert_task(self, record: TaskRecord) -> None:
        """同 (model_id, target_gpu) 已存在活跃/成功/拉黑记录时抛 DuplicateTaskError。"""
        ...

    def update_task(self, record: TaskRecord) -> None: ...
    def tasks_by_status(self, *statuses: TaskStatus) -> list[TaskRecord]: ...
    def get_task(self, model_id: str, target_gpu: str) -> TaskRecord | None: ...

    # --- 黑名单 ---
    def is_blacklisted(self, model_id: str, target_gpu: str) -> bool: ...

    # --- GPU 覆盖率缓存（config_gen 选卡依据）---
    def gpu_coverage(self) -> dict[str, int]: ...
    def set_gpu_coverage(self, coverage: dict[str, int]) -> None: ...

    # --- 熔断开关（monitor 发现对账异常时暂停提交）---
    def kill_switch(self) -> bool: ...
    def set_kill_switch(self, on: bool, reason: str, source: str = "") -> None: ...

    def kill_switch_state(self) -> dict:
        """{"on": bool, "reason": str}——每 tick 打进 metrics 日志行，让运维能看见
        刹车状态和原因（清除方式见 README）。"""
        ...

    # --- 计数器（连续失败熔断用，Task 8/9 用）---
    def get_counter(self, key: str) -> int: ...
    def set_counter(self, key: str, value: int) -> None: ...


class DuplicateTaskError(Exception):
    """违反 (model_id, target_gpu) 唯一性——防重复提交的最终防线。"""


class StorageUnavailableError(Exception):
    """存储无法打开（目录不存在/不可写）。

    平台不保证挂载任何卷，容器里的默认路径可能根本不存在。由 main 捕获后转成
    "存活但不工作"，而不是让进程崩掉、把诊断信息埋进重启循环里。
    """
