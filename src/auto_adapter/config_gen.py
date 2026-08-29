"""任务配置生成层（spec §4.4）。M4 实现。"""
from __future__ import annotations

import re
from pathlib import Path

from . import rules

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def resolve_task_type(candidate) -> str | None:
    """pipeline_tag 直映射；缺失时查 rules.FALLBACK_TASK_TYPE_RULES；
    仍无法判定返回 None（调用方标记 NEEDS_HUMAN，不盲提）。"""
    if candidate.model_id in rules.MANUAL_OVERRIDES:
        return rules.MANUAL_OVERRIDES[candidate.model_id][0]
    if candidate.pipeline_tag:
        return candidate.pipeline_tag
    lowered = candidate.model_id.lower()
    for keyword, task_type in rules.FALLBACK_TASK_TYPE_RULES:
        if keyword in lowered:
            return task_type
    return None


def resolve_tp_size(params_size: str | None) -> int:
    """参数量 → tensor 并行度：<14B→1，14–70B→2，>70B→4；未知按 1。"""
    if not params_size:
        return 1
    m = re.match(r"([\d.]+)\s*B", params_size, re.IGNORECASE)
    if not m:
        return 1
    billions = float(m.group(1))
    if billions < 14:
        return 1
    if billions <= 70:
        return 2
    return 4


def resolve_framework(candidate) -> str:
    """架构在 vllm 支持列表 → vllm；否则退化到备选框架（rules.py 维护）。"""
    if candidate.model_id in rules.MANUAL_OVERRIDES:
        return rules.MANUAL_OVERRIDES[candidate.model_id][1]
    # v0.1：text-generation 走 vllm，其余退化（架构级判断见 spec §6 迭代方向）
    if resolve_task_type(candidate) == "text-generation":
        return "vllm"
    return rules.FALLBACK_FRAMEWORK


def select_target_gpu(storage) -> str:
    """选平台覆盖率最低的 GPU 型号（覆盖率缓存于 storage）。"""
    coverage = storage.gpu_coverage()
    return min(rules.KNOWN_GPUS, key=lambda g: coverage.get(g, 0))


def render_config_params(framework: str, tp_size: int, max_model_len: int = 4096,
                         gpu_mem_util: float = 0.9) -> str:
    """渲染 templates/{framework}.yaml 为 configParams YAML 字符串（spec 附录 A.1.1）。"""
    template = (_TEMPLATE_DIR / f"{framework}.yaml").read_text()
    return template.format(tp_size=tp_size, max_model_len=max_model_len,
                           gpu_mem_util=gpu_mem_util)


def build_request(candidate, target_gpu, strategy_id):
    """组装完整请求体：上述推导结果 + render_config_params。"""
    task_type = resolve_task_type(candidate)
    if task_type is None:
        raise ValueError(f"cannot resolve task type for {candidate.model_id}")
    framework = resolve_framework(candidate)
    config = render_config_params(framework, resolve_tp_size(candidate.params_size))
    from .models import AddTaskRequest
    return AddTaskRequest(
        model_address=candidate.model_url, task_type=task_type,
        target_gpu=target_gpu, framework=framework,
        config_params=config, strategy_id=strategy_id,
    )
