"""核心数据模型。全部为不可变或简单可变的 dataclass，模块间只传递这些类型。

契约见 spec §3。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime


class Priority(enum.IntEnum):
    """数值越小优先级越高，可直接用于排序。"""

    BOUNTY = 0
    NEW_MODEL = 1
    NEW_ADAPTATION = 2


class TaskStatus(enum.StrEnum):
    QUEUED = "queued"
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ENGINE_FAILED = "engine_failed"
    QUALITY_FAILED = "quality_failed"
    TIMEOUT = "timeout"
    BLACKLISTED = "blacklisted"
    NEEDS_HUMAN = "needs_human"
    ABANDONED = "abandoned"


# 活跃状态：同一 (model_id, target_gpu) 在这些状态中最多一条（spec §3 不变式）
ACTIVE_STATUSES = {TaskStatus.QUEUED, TaskStatus.PENDING, TaskStatus.RUNNING}


class FailureKind(enum.StrEnum):
    ENGINE = "engine"
    QUALITY = "quality"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class CandidateModel:
    source: str  # "huggingface" | "modelscope" | "bounty"
    model_id: str
    model_url: str
    pipeline_tag: str | None
    params_size: str | None
    is_bounty: bool
    bounty_deadline: datetime | None
    discovered_at: datetime


@dataclass
class TaskRecord:
    model_id: str
    target_gpu: str
    framework: str
    status: TaskStatus
    priority: Priority
    config_params: str = ""  # YAML 字符串（spec 附录 A.1.1），重试时解析调整后重渲染
    task_id: int | None = None  # 平台 AsyncTaskVO.id，int64
    retry_count: int = 0
    submit_time: datetime | None = None
    bounty_deadline: datetime | None = None
    last_log: str | None = None


@dataclass(frozen=True)
class AddTaskRequest:
    """POST /api/adapt/task/add 请求体（spec 附录 A.1）。

    序列化为 JSON 时字段名转 camelCase：modelAddress/taskType/targetGpu/
    framework/configParams/strategyId。
    """

    model_address: str
    task_type: str
    target_gpu: str
    framework: str
    config_params: str  # YAML 字符串，含 sut_config/ref_config（附录 A.1.1）
    strategy_id: str


@dataclass(frozen=True)
class ModelSearchResult:
    """GET search-by-model-id 响应 data（spec 附录 A.4）。"""

    is_in_db: bool
    model_info: dict
    # 按 GPU 型号分键的验证结果——eligibility 分类的直接依据
    verify_result: dict[str, dict]
