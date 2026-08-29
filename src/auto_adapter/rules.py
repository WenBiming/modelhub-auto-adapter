"""兜底规则表：taskType 推断、框架支持列表。允许人工白名单修正（spec 风险表）。

内容为占位示例，实现 M4 时依据平台实际枚举补全（spec §9 开放问题）。
"""

# pipeline_tag 缺失时按模型名关键词兜底
FALLBACK_TASK_TYPE_RULES: list[tuple[str, str]] = [
    ("instruct", "text-generation"),
    ("chat", "text-generation"),
    ("embedding", "feature-extraction"),
    ("bge-", "feature-extraction"),
    ("whisper", "automatic-speech-recognition"),
]

# vllm 支持的模型架构（config.json 的 architectures 字段）
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
