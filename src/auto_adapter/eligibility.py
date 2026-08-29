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

    - 本地已有活跃/成功/拉黑记录        → SKIP_DUPLICATE
    - 平台无记录                        → ENQUEUE, NEW_MODEL
    - 平台有记录但目标 GPU 不同          → ENQUEUE, NEW_ADAPTATION
    - 平台同模型同 GPU 已适配            → SKIP_DUPLICATE
    - 候选命中悬赏                      → ENQUEUE, BOUNTY（覆盖上面的优先级）
    - search_adaptations 抛异常          → SKIP_UNCERTAIN
    """
    raise NotImplementedError
