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


# 首次提交的并行度。恒为 1 是刻意的保守取值：gpu_num 同时意味着"向平台申请几张卡"，
# 而各家算力机器的实际卡数我们并不知道。猜大了会因"机器没这么多卡"直接失败，
# 而失败重试梯子的调整方向是**加大** tp——它修不好这类失败，只会越修越糟。
# 反过来，猜小导致的 OOM 恰恰是梯子能修的（rung 2 会翻倍 tp）。
# 代价：超大模型（>70B）首次几乎必然 OOM，要靠梯子爬上去。
INITIAL_TP_SIZE = 1


def resolve_tp_size(params_size: str | None) -> int:
    """首次提交的 tensor 并行度：一律 1（见 INITIAL_TP_SIZE）。

    参数量只作为日志线索保留，不再直接决定初始并行度。
    """
    return INITIAL_TP_SIZE


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


def select_target_gpu(storage, exclude: set[str] | None = None) -> str | None:
    """在 rules.KNOWN_GPUS 里挑一张目标卡，排除 exclude 里的型号；无可选时返回 None。

    exclude 由调用方传入"这个模型已经适配过的卡"——**必须逐候选计算**。
    早先的实现每个 tick 只算一张全局的卡，线上实测的后果是：Ascend_910-b4 覆盖了
    几乎所有热门模型，于是每个候选都被判重复跳过，智能体永远提交不出任何任务。
    而同一批候选里其实有大量"这个模型还没上过那张卡"的新适配机会
    （例：某 32B 模型只覆盖 8 张卡，我们能提交的 10 张里有 4 张是空的）。

    多张可选时按平台覆盖率从低到高挑，让适配矩阵往稀疏处生长。
    """
    coverage = storage.gpu_coverage()
    available = [g for g in rules.KNOWN_GPUS if g not in (exclude or set())]
    if not available:
        return None
    return min(available, key=lambda g: coverage.get(g, 0))


def render_config_params(framework: str, tp_size: int, max_model_len: int = 4096,
                         gpu_mem_util: float = 0.9) -> str:
    """渲染 templates/{framework}.yaml 为 configParams YAML 字符串（spec 附录 A.1.1）。

    模板顶部的开发注释会被剥掉：它们随 configParams 一起发给平台没有意义，而且
    注释里写的占位符也会被 str.format 一并替换（线上演练里看到 "# 占位符 2/4096/0.9"
    这种被替换过的残句）。只去掉整行注释，行内的 YAML 值不受影响。
    """
    template = (_TEMPLATE_DIR / f"{framework}.yaml").read_text()
    body = "\n".join(line for line in template.splitlines()
                     if not line.lstrip().startswith("#"))
    return body.format(tp_size=tp_size, max_model_len=max_model_len,
                       gpu_mem_util=gpu_mem_util) + "\n"


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
