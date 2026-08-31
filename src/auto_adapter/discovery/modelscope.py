"""ModelScope 发现源（v0.1 范围外）。"""
from __future__ import annotations

from ..models import CandidateModel


class ModelScopeSource:
    name = "modelscope"

    def __init__(self, min_interval_seconds: int = 3600) -> None:
        self._min_interval = min_interval_seconds

    def fetch(self) -> list[CandidateModel]:
        raise NotImplementedError
