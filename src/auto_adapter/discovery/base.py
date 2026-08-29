"""模型发现层接口（spec §4.2）。"""
from __future__ import annotations

from typing import Protocol

from ..models import CandidateModel


class DiscoverySource(Protocol):
    name: str

    def fetch(self) -> list[CandidateModel]:
        """拉取候选模型。实现自行负责节流（HF/MS 1h 一次，悬赏每 tick）。"""
        ...


def run(sources: list[DiscoverySource], storage) -> int:
    """执行所有来源，统一去重后写入候选表，返回新候选数。M3 实现。"""
    raise NotImplementedError
