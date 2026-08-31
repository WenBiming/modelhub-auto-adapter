"""任务配置生成层（spec §4.4）。M4 实现。"""
from __future__ import annotations

import re
from pathlib import Path

from . import rules

_TEMPLATE_DIR = Path(__file__).parent / "templates"

# v0.1 只允许提交 vllm 候选（理由见 build_request）。
SUPPORTED_FRAMEWORKS = ("vllm",)


class UnresolvableCandidateError(ValueError):
    """候选无法自动组装成一次可信的提交：调用方必须落 NEEDS_HUMAN 记录，绝不盲提。

    `reason` 同时用作 metrics 计数名：
    - "unresolvable_task_type"：taskType 推不出来（spec §4.4）；
    - "unsupported_framework"：解析出的框架不在 v0.1 支持列表内。

    继承 ValueError 只为兼容既有调用方；调用方应捕获本类型而非裸 ValueError——
    模板格式化本身抛出的 ValueError 是另一回事，不能被算成"无法解析的候选"。
    """

    def __init__(self, reason: str, message: str, framework: str = "") -> None:
        super().__init__(message)
        self.reason = reason
        self.framework = framework


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
    # v0.1 范围：按 task_type 选框架（计划 Global Constraints 的 YAGNI 决定）。
    # spec §4.4 要求的"按模型架构判断 vllm 支持"是后续迭代项——CandidateModel
    # 目前不携带 architecture 字段，需先扩展发现层才能实现。
    if resolve_task_type(candidate) == "text-generation":
        return "vllm"
    return rules.FALLBACK_FRAMEWORK


def select_target_gpu(storage) -> str:
    """选平台覆盖率最低的 GPU 型号（覆盖率缓存于 storage，由 eligibility 写入）。

    v0.1 的实际效果：rules.KNOWN_GPUS 只有一个已确认型号，所以这里恒返回它。
    扩充 KNOWN_GPUS（spec §9 上线前人工步骤）后本函数立即开始按覆盖率分流。
    注意候选的 processed 标记按 model_id 记（不含 GPU）：一个模型在一段时间内只会
    被适配到一张卡上，多卡覆盖靠不同模型分流实现，而不是同一模型逐卡重复提交——
    这是防重复提交的保守取舍，要改成"每模型每卡各一次"必须同时改候选表主键。
    """
    coverage = storage.gpu_coverage()
    return min(rules.KNOWN_GPUS, key=lambda g: coverage.get(g, 0))


def render_config_params(framework: str, tp_size: int, max_model_len: int = 4096,
                         gpu_mem_util: float = 0.9) -> str:
    """渲染 templates/{framework}.yaml 为 configParams YAML 字符串（spec 附录 A.1.1）。"""
    template = (_TEMPLATE_DIR / f"{framework}.yaml").read_text()
    return template.format(tp_size=tp_size, max_model_len=max_model_len,
                           gpu_mem_util=gpu_mem_util)


def build_request(candidate, target_gpu, strategy_id):
    """组装完整请求体：上述推导结果 + render_config_params。

    无法可信组装时抛 UnresolvableCandidateError，调用方落 NEEDS_HUMAN 记录。
    """
    task_type = resolve_task_type(candidate)
    if task_type is None:
        raise UnresolvableCandidateError(
            "unresolvable_task_type", f"cannot resolve task type for {candidate.model_id}")
    framework = resolve_framework(candidate)
    if framework not in SUPPORTED_FRAMEWORKS:
        # v0.1 不提交非 vllm 候选：templates/transformers.yaml 的 command 字段是显式
        # 占位符（"待平台确认"），而失败重试梯子只会调 tp/显存/上下文长度，改不了一条
        # 错误的启动命令——这类候选只会白烧 3 次提交然后被永久拉黑，还平白增加平台
        # 侧的重复提交风险。等平台确认真实框架列表与启动命令后解除（spec §9）。
        raise UnresolvableCandidateError(
            "unsupported_framework",
            f"framework {framework!r} is not submittable in v0.1 "
            f"(only {SUPPORTED_FRAMEWORKS}); {candidate.model_id} needs human review",
            framework=framework)
    config = render_config_params(framework, resolve_tp_size(candidate.params_size))
    from .models import AddTaskRequest
    return AddTaskRequest(
        model_address=candidate.model_url, task_type=task_type,
        target_gpu=target_gpu, framework=framework,
        config_params=config, strategy_id=strategy_id,
    )
