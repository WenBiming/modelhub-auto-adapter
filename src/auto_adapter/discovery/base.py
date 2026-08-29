"""模型发现层接口（spec §4.2）。"""
from __future__ import annotations

import logging
from typing import Protocol

from ..models import CandidateModel

logger = logging.getLogger(__name__)


class DiscoverySource(Protocol):
    name: str

    def fetch(self) -> list[CandidateModel]:
        """拉取候选模型。实现自行负责节流（HF/MS 1h 一次，悬赏每 tick）。"""
        ...


def run(sources: list[DiscoverySource], storage) -> int:
    """执行所有来源，统一去重后写入候选表，返回新候选数。M3 实现。

    悬赏候选优先：若同一 model_id 既来自非悬赏源又来自悬赏源，保留悬赏版本以防止
    错失悬赏时间窗口。替换操作不重复计数。
    """
    candidates_by_id = {}  # model_id -> CandidateModel

    for src in sources:
        try:
            candidates = src.fetch()
        except Exception:
            logger.exception("discovery source %s failed", src.name)
            continue
        for c in candidates:
            if c.model_id not in candidates_by_id:
                # 新候选
                candidates_by_id[c.model_id] = c
            elif c.is_bounty and not candidates_by_id[c.model_id].is_bounty:
                # 悬赏候选替换非悬赏重复（不重复计数）
                candidates_by_id[c.model_id] = c
            # 其他情况保留已有候选

    # 批量写入最终候选
    for c in candidates_by_id.values():
        storage.upsert_candidate(c)

    return len(candidates_by_id)
