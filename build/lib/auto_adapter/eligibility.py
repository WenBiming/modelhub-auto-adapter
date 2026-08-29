"""去重与准入判断层——全系统最关键模块（spec §4.3）。M3 实现。

铁律：双重校验（本地 storage + 平台 search）都通过才入队；
平台查询失败时保守跳过（宁漏勿重），绝不在不确定时提交。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

from .models import CandidateModel, Priority
from .platform_client import PlatformClient
from .storage import Storage


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
    - search_model 抛异常/网络失败       → SKIP_UNCERTAIN（宁漏勿重）
    """
    if storage.get_task(candidate.model_id, target_gpu) is not None:
        return Decision(Verdict.SKIP_DUPLICATE, reason="local record exists")
    try:
        result = client.search_model(candidate.model_id)
    except Exception as e:  # 平台不可知时宁漏勿重
        return Decision(Verdict.SKIP_UNCERTAIN, reason=f"platform query failed: {e}")
    if target_gpu in result.verify_result:
        # GpuVerifyResult 内部字段未确认前，键存在即视为已覆盖（保守）
        return Decision(Verdict.SKIP_DUPLICATE, reason=f"already verified on {target_gpu}")
    if candidate.is_bounty:
        return Decision(Verdict.ENQUEUE, Priority.BOUNTY, reason="bounty")
    if not result.verify_result:
        return Decision(Verdict.ENQUEUE, Priority.NEW_MODEL, reason="no adaptation record")
    return Decision(Verdict.ENQUEUE, Priority.NEW_ADAPTATION, reason="new gpu for model")
