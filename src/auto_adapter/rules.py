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

# 平台支持的全部 GPU 型号，取自平台任务列表页「GPU类型」筛选下拉框（2026-08-29）。
KNOWN_GPUS = [
    "Ascend_910-b4",
    "MetaX_c-500",
    "Cambricon_mlu-370-x4",
    "Kunlunxin_p-800",
    "Kunlunxin_r-200-8f",
    "Iluvatar_bi-150",
    "Iluvatar_mrv-100",
    "hygon_k100-ai",
    "Vastai_va16",
    "Sunrise_pt-200-x1",
]

# 人工白名单：model_id → 强制指定的 (task_type, framework)
MANUAL_OVERRIDES: dict[str, tuple[str, str]] = {}

# 平台的 status 与 verifyResult 是两个**正交**字段，必须成对判读。
# 实测自平台自身 UI 的筛选请求与 /api/adapt/task/page 响应（2026-08-29）：
#
#   UI 标签      status     verifyResult   含义
#   验证排队中    waiting        1         作业排队中
#   验证进行中    running        1         作业运行中
#   验证异常      failed         1         作业本身出错（引擎/容器层）
#   验证通过      success        1         作业跑完 且 验证通过 ← 唯一的真正成功
#   验证失败      success       -1         作业跑完 但 验证未通过
#
# 关键陷阱：`status == "success"` 只表示"任务作业执行完毕"，不表示适配成功。
# 用户账号里大量记录正是 status=success + verifyResult=-1（statusText"成功" /
# verifyResultText"验证失败"）。只看 status 会把每一个验证失败都记成适配成功——
# 既不会重试，成功率指标也全是假的。
PLATFORM_STATUS_WAITING = "waiting"
PLATFORM_STATUS_RUNNING = "running"
PLATFORM_STATUS_FAILED = "failed"
PLATFORM_STATUS_SUCCESS = "success"

VERIFY_PASSED = 1
VERIFY_FAILED = -1

# 需要拉日志后经 failure.classify 分成 ENGINE_FAILED / QUALITY_FAILED 的中间态标记，
# 不是可以直接赋给 TaskRecord.status 的最终态。
NEEDS_CLASSIFICATION = "failed"


def map_platform_result(status: str, verify_result) -> object:
    """(status, verifyResult) → TaskStatus | NEEDS_CLASSIFICATION | None（未知）。

    保守规则：只有 status=success 且 verifyResult 明确为 1 才算 SUCCESS。
    success 配上任何其他 verifyResult（-1、缺失、未知值）都交给日志分类——
    宁可多跑一次分类，也绝不无凭据地宣布适配成功。
    """
    s = (status or "").strip().lower()
    if s == PLATFORM_STATUS_WAITING:
        return TaskStatus.PENDING
    if s == PLATFORM_STATUS_RUNNING:
        return TaskStatus.RUNNING
    if s == PLATFORM_STATUS_FAILED:
        return NEEDS_CLASSIFICATION
    if s == PLATFORM_STATUS_SUCCESS:
        if verify_result == VERIFY_PASSED:
            return TaskStatus.SUCCESS
        return NEEDS_CLASSIFICATION
    return None
