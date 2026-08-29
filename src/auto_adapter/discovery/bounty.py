"""平台悬赏列表发现源——优先级最高，每 tick 执行。M3 实现。

待实现（真实 API 路径待确认，spec §9）。
"""
from __future__ import annotations

from ..models import CandidateModel
from ..platform_client import PlatformClient


class BountySource:
    name = "bounty"

    def __init__(self, client: PlatformClient) -> None:
        self._client = client

    def fetch(self) -> list[CandidateModel]:
        raise NotImplementedError
