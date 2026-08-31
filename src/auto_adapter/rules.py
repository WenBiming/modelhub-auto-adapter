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

# 平台支持的推理框架（由平台使用者确认）。configParams 的启动命令按框架不同，
# 目前只有 vllm 的模板得到过官方样例佐证（spec 附录 A.1.1）。
PLATFORM_FRAMEWORKS = ("vllm", "llama.cpp")

FALLBACK_FRAMEWORK = "transformers"  # 平台实际备选框架名待确认

# GGUF 是 llama.cpp 的模型格式，vllm 跑不了它。
# 佐证：账号历史里的 GGUF 任务（Mistral-7B-Instruct-v0.3-GGUF、BitCPM4-0.5B-GGUF、
# Mistral-7B-v0.3-Chinese-Chat-GGUF）在 Ascend_910-b4 上全部验证失败。
# 这类失败重试梯子修不了——它只调并行度和显存，改不了格式与框架不匹配。
GGUF_MARKERS = ("gguf",)


def is_gguf(model_id: str) -> bool:
    lowered = (model_id or "").lower()
    return any(marker in lowered for marker in GGUF_MARKERS)


# llama.cpp 在平台上的框架名是 "llamacpp"（无点号）——取自一次真实的手工提交配置。
LLAMACPP_FRAMEWORK = "llamacpp"

# llama-server 二进制路径**按厂商编译**：Ascend 用 build_ascend，参照系统用 build。
# 目前只有 Ascend 这一条实证，其余 9 张卡的路径未知——猜错了容器根本起不来，
# 而重试梯子只调并行度和显存，修不了一个不存在的可执行文件路径。
# 因此 GGUF 候选只投这里有记录的卡（见 submittable_gpus_for）。
LLAMACPP_SUT_BINARIES = {
    "Ascend_910-b4": "/workspace/llama.cpp/build_ascend/bin/llama-server",
}

# 量化档偏好：Q4_K_M 是公认的体积/质量平衡点，往下依次退让。
# 避开 f32/f16（体积巨大）与 IQ1/IQ2（质量损失过大，LLM Judge 大概率过不了）。
GGUF_QUANT_PREFERENCE = (
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q4_0",
    "Q6_K", "Q8_0", "Q3_K_M", "Q3_K_L",
)


def pick_gguf_file(file_names) -> str | None:
    """按量化偏好从仓库文件列表里挑一个 .gguf；都不匹配时返回 None。

    GGUF 仓库通常含十几个量化档（实测 Mistral-7B-Instruct-v0.3-GGUF 有 23 个），
    而 llama-server 的 --model 要求具体文件名，必须选一个。
    """
    ggufs = [n for n in (file_names or []) if str(n).lower().endswith(".gguf")]
    for quant in GGUF_QUANT_PREFERENCE:
        for name in ggufs:
            if quant.lower() in name.lower():
                return name
    return None


def submittable_gpus_for(model_id: str) -> list[str]:
    """该模型可以投哪些卡。

    GGUF 走 llama.cpp，而 llama-server 的路径按厂商编译，只有已知路径的卡能投。
    """
    if is_gguf(model_id):
        return [g for g in KNOWN_GPUS if g in LLAMACPP_SUT_BINARIES]
    return list(KNOWN_GPUS)

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


# 发现层的下载量门槛：新发布的模型绝大多数是个人试验品（实测 ModelScope 最新列表
# 里下载量普遍是 1~3），不设门槛会把验证算力浪费在没人用的模型上。
# text-generation 是主力赛道、竞争充分，门槛高些；其他类型基数小，门槛低些。
MIN_DOWNLOADS_TEXT_GENERATION = 50
MIN_DOWNLOADS_OTHER = 5


def min_downloads_for(task_type: str | None) -> int:
    """该任务类型进入候选队列所需的最小下载量。"""
    if task_type == "text-generation":
        return MIN_DOWNLOADS_TEXT_GENERATION
    return MIN_DOWNLOADS_OTHER


def passes_download_threshold(task_type: str | None, downloads) -> bool:
    """下载量未知（None/非数字）时视为不达标——宁可漏掉也不浪费验证算力。"""
    try:
        return int(downloads) > min_downloads_for(task_type)
    except (TypeError, ValueError):
        return False
