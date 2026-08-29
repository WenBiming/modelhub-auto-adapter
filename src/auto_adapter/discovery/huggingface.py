"""HuggingFace Hub 发现源：按 downloads/trending/pipeline_tag 筛选。M3 实现。"""
from __future__ import annotations

from ..models import CandidateModel


class HuggingFaceSource:
    name = "huggingface"

    def __init__(self, min_interval_seconds: int = 3600) -> None:
        self._min_interval = min_interval_seconds

    def fetch(self) -> list[CandidateModel]:
        raise NotImplementedError
