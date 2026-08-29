"""失败分类与重试层（spec §4.7）。M5 实现。"""
from __future__ import annotations

from .models import FailureKind, TaskRecord
from .platform_client import PlatformClient
from .settings import Settings
from .storage import Storage

# 日志关键词 → 失败类型（实现时依据真实日志样本补全）
_ENGINE_PATTERNS = ("CUDA out of memory", "CUDA error", "container failed", "OOM")
_QUALITY_PATTERNS = ("judge", "quality check failed", "score below")


def classify(log_text: str) -> FailureKind:
    """基于日志关键词分类。无法判定时按 ENGINE 处理（重试成本低于误拉黑）。"""
    text = (log_text or "").lower()
    if any(kw in text for kw in _QUALITY_PATTERNS):
        return FailureKind.QUALITY
    return FailureKind.ENGINE


def next_config(record: TaskRecord) -> str | None:
    """引擎失败的调参序列：解析 record.config_params（YAML，spec 附录 A.1.1），
    按 retry_count 递进调整——降 gpu-memory-utilization / 降 max_model_len /
    调 tp_size（sut 与 ref 保持一致）→ 换框架——重渲染为 YAML 字符串返回；
    序列耗尽返回 None（调用方拉黑）。"""
    raise NotImplementedError


def handle(storage: Storage, client: PlatformClient, settings: Settings) -> None:
    """处理 monitor 标记的失败任务：

    - ENGINE：next_config 调参后重新入队，retry_count+1；≥max_retries → BLACKLISTED；
    - QUALITY：NEEDS_HUMAN + 黑名单，不自动重试；
    - TIMEOUT：stop_tasks 批量释放资源；悬赏未过期可重排队一次，否则 ABANDONED。
    """
    raise NotImplementedError
