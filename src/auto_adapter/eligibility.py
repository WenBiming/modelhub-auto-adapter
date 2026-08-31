"""去重与准入判断层——全系统最关键模块（spec §4.3）。M3 实现。

铁律：双重校验（本地 storage + 平台 search）都通过才入队；
平台查询失败时保守跳过（宁漏勿重），绝不在不确定时提交。
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

from .models import CandidateModel, ModelSearchResult, Priority
from .platform_client import (
    CODE_NOT_FOUND,
    PlatformClient,
    PlatformClientError,
    escalate_if_credential_error,
)
from .storage import Storage

logger = logging.getLogger(__name__)


class Verdict(enum.StrEnum):
    ENQUEUE = "enqueue"
    SKIP_DUPLICATE = "skip_duplicate"      # 同模型同 GPU 已适配/在途/拉黑
    SKIP_UNCERTAIN = "skip_uncertain"      # 平台查询失败，保守跳过本 tick


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    priority: Priority | None = None       # 仅 ENQUEUE 时有值
    reason: str = ""


def evaluate(
    candidate: CandidateModel,
    target_gpu: str,
    storage: Storage,
    client: PlatformClient,
) -> Decision:
    """分类矩阵（spec §4.3）：

    - 任一已存在记录，不限状态             → SKIP_DUPLICATE（不查平台）
    平台侧依据 client.search_model().verify_result（按 GPU 型号分键，spec 附录 A.4）：
    - verify_result 无任何键             → ENQUEUE, NEW_MODEL
    - 有键但无当前 target_gpu 的键       → ENQUEUE, NEW_ADAPTATION
    - 有 target_gpu 的键且已通过         → SKIP_DUPLICATE
    - 候选命中悬赏                      → ENQUEUE, BOUNTY（不覆盖 SKIP_DUPLICATE）
    - 40400 NOT_FOUND                   → 平台从未见过该模型 = 空 verify_result（新模型）
    - 其余异常/网络失败                  → SKIP_UNCERTAIN（宁漏勿重）
    """
    if storage.get_task(candidate.model_id, target_gpu) is not None:
        return Decision(Verdict.SKIP_DUPLICATE, reason="local record exists")
    try:
        result = client.search_model(candidate.model_id)
    except PlatformClientError as e:
        if e.code == CODE_NOT_FOUND:
            # 40400 是"平台没有这个模型"的正常业务应答（附录 A.6），也是本系统存在
            # 意义所在的 NEW_MODEL 分支最可能收到的响应——不是不确定。以前它被吞进
            # SKIP_UNCERTAIN，而 SKIP_UNCERTAIN 的候选永不标 processed，于是每个 tick
            # 重查一次、永远进不了队。
            result = ModelSearchResult(is_in_db=False, model_info={}, verify_result={})
        else:
            escalate_if_credential_error(storage, e)
            return Decision(Verdict.SKIP_UNCERTAIN, reason=f"platform query failed: {e}")
    except Exception as e:  # 平台不可知时宁漏勿重
        return Decision(Verdict.SKIP_UNCERTAIN, reason=f"platform query failed: {e}")

    _record_gpu_coverage(storage, result.verify_result)

    logger.info("platform coverage for %s: %s",
                candidate.model_id, sorted(result.verify_result.keys()) or "(none)")
    if target_gpu in result.verify_result:
        # GpuVerifyResult 内部字段未确认前，键存在即视为已覆盖（保守）
        return Decision(Verdict.SKIP_DUPLICATE, reason=f"already verified on {target_gpu}")
    if candidate.is_bounty:
        return Decision(Verdict.ENQUEUE, Priority.BOUNTY, reason="bounty")
    if not result.verify_result:
        return Decision(Verdict.ENQUEUE, Priority.NEW_MODEL, reason="no adaptation record")
    return Decision(Verdict.ENQUEUE, Priority.NEW_ADAPTATION, reason="new gpu for model")


def _record_gpu_coverage(storage: Storage, verify_result: dict) -> None:
    """把平台返回的 verify_result 键（该模型已验证过的 GPU 型号）累加进覆盖率缓存。

    这是 storage.set_gpu_coverage 的唯一写入点，也是让 config_gen.select_target_gpu
    真正"选覆盖率最低的卡"的数据来源——在此之前覆盖率永远是空 dict，选卡恒等于
    KNOWN_GPUS[0]，扩充 KNOWN_GPUS（spec §9 的上线前人工步骤）不会有任何效果。

    计数口径：每评估一个候选，为它已验证的每个 GPU 各 +1。候选评估后会被标记
    processed，所以同一模型不会被反复计入。覆盖率只用于卡之间的相对比较，绝对值
    不需要精确。
    """
    if not verify_result:
        return
    try:
        coverage = dict(storage.gpu_coverage())
        for gpu in verify_result:
            coverage[gpu] = coverage.get(gpu, 0) + 1
        storage.set_gpu_coverage(coverage)
    except Exception:
        # 覆盖率只影响选卡偏好，写失败不该改变准入判定（准入才是防重复提交的关键路径）。
        logger.exception("failed to update gpu coverage cache")
