"""失败分类与重试层（spec §4.7）。M5 实现。"""
from __future__ import annotations

from .models import FailureKind, TaskRecord
from .platform_client import PlatformClient
from .settings import Settings
from .storage import Storage

# 日志关键词 → 失败类型（实现时依据真实日志样本补全）
_ENGINE_PATTERNS = ("CUDA out of memory", "CUDA error", "container failed", "OOM")
_QUALITY_PATTERNS = ("judge", "quality check failed")


def classify(log_text: str) -> FailureKind:
    """基于日志关键词分类。无法判定时按 ENGINE 处理（重试成本低于误拉黑）。"""
    raise NotImplementedError


def next_config(record: TaskRecord) -> dict | None:
    """引擎失败的调参序列：降精度 → 降并行 → 换框架，按 retry_count 递进；
    序列耗尽返回 None（调用方拉黑）。"""
    raise NotImplementedError


def handle(storage: Storage, client: PlatformClient, settings: Settings) -> None:
    """处理 monitor 标记的失败任务：

    - ENGINE：next_config 调参后重新入队，retry_count+1；≥max_retries → BLACKLISTED；
    - QUALITY：NEEDS_HUMAN + 黑名单，不自动重试；
    - TIMEOUT：stop_task 释放资源；悬赏未过期可重排队一次，否则 ABANDONED。
    """
    raise NotImplementedError
