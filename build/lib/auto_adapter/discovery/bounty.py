"""悬赏来源：平台无悬赏 API（spec §9），v0.1 用人工维护的 JSON 配置文件。

文件格式：[{"model_id": str, "model_url": str, "deadline": ISO8601}]

deadline 由人手写，很容易漏掉时区偏移量（"2026-09-30T00:00:00"）。这里统一按 UTC
归一化：naive datetime 一旦流进 submitter/failure 的时间比较就会抛 TypeError，
让整条流水线每个 tick 都崩在同一处。
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..models import CandidateModel, ensure_utc

logger = logging.getLogger(__name__)


class ManualBountySource:
    name = "bounty"

    def __init__(self, path: str) -> None:
        self._path = Path(path) if path else None

    def fetch(self) -> list[CandidateModel]:
        if self._path is None or not self._path.exists():
            return []
        items = json.loads(self._path.read_text())
        return [CandidateModel(
            source="bounty", model_id=i["model_id"], model_url=i["model_url"],
            pipeline_tag=i.get("pipeline_tag"), params_size=i.get("params_size"),
            is_bounty=True,
            bounty_deadline=ensure_utc(
                datetime.fromisoformat(i["deadline"])) if i.get("deadline") else None,
            discovered_at=datetime.now(timezone.utc),
        ) for i in items]
