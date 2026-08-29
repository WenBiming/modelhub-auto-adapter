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


def run(sources, storage) -> int:
    """执行所有来源，统一去重后写入候选表，返回新候选数。M3 实现。"""
    count, seen = 0, set()
    for src in sources:
        try:
            candidates = src.fetch()
        except Exception:
            logger.exception("discovery source %s failed", src.name)
            continue
        for c in candidates:
            if c.model_id in seen:
                continue
            seen.add(c.model_id)
            storage.upsert_candidate(c)
            count += 1
    return count
