"""兜底规则表：taskType 推断、框架支持列表。允许人工白名单修正（spec 风险表）。

内容为占位示例，实现 M4 时依据平台实际枚举补全（spec §9 开放问题）。
"""
from __future__ import annotations

from .models import TaskStatus

# pipeline_tag 缺失时按模型名关键词兜底
FALLBACK_TASK_TYPE_RULES: list[tuple[str, str]] = [
    ("instruct", "text-generation"),
    ("chat", "text-generation"),
    ("embedding", "feature-extraction"),
    ("bge-", "feature-extraction"),
    ("whisper", "automatic-speech-recognition"),
]

# vllm 支持的模型架构（config.json 的 architectures 字段）。
# 预留用于后续迭代的架构级 framework 判断；v0.1 未使用（按 task_type 代替）。
VLLM_SUPPORTED_ARCHITECTURES: set[str] = {
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "MistralForCausalLM",
}

FALLBACK_FRAMEWORK = "transformers"  # 平台实际备选框架名待确认

# 已确认的 GPU 型号列表（唯一已确认型号，后续人工扩充）
KNOWN_GPUS = ["MetaX_c-500"]

# 人工白名单：model_id → 强制指定的 (task_type, framework)
MANUAL_OVERRIDES: dict[str, tuple[str, str]] = {}

# 平台任务状态字符串 → 本地 TaskStatus 映射（状态枚举尚未确认，spec §9 开放问题）。
# 值 "failed" 是一个中间态标记：表示该状态需要拉取日志后经 failure.classify 分类为
# ENGINE_FAILED 或 QUALITY_FAILED，而不是可以直接赋给 TaskRecord.status 的最终态。
PLATFORM_STATUS_MAP: dict[str, object] = {
    "PENDING": TaskStatus.PENDING, "QUEUED": TaskStatus.PENDING,
    "RUNNING": TaskStatus.RUNNING,
    "SUCCESS": TaskStatus.SUCCESS, "SUCCEED": TaskStatus.SUCCESS,
    "FAILED": "failed", "FAIL": "failed", "ERROR": "failed",
}


def map_platform_status(status: str):
    """平台状态字符串 → TaskStatus | "failed" | None（未知，保持原状并 warning）。"""
    return PLATFORM_STATUS_MAP.get((status or "").upper())
