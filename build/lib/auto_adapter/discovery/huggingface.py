"""HuggingFace Hub 发现源：按 downloads/trending/pipeline_tag 筛选。M3 实现。"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone

import requests

from ..models import CandidateModel

_API = "https://huggingface.co/api/models"
_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)[bB]\b")


class HuggingFaceSource:
    name = "huggingface"

    def __init__(self, limit: int = 50, min_interval_seconds: int = 3600) -> None:
        self._limit = limit
        self._min_interval = min_interval_seconds
        self._last_fetch = 0.0

    def fetch(self) -> list[CandidateModel]:
        if time.monotonic() - self._last_fetch < self._min_interval and self._last_fetch:
            return []
        resp = requests.get(_API, params={
            "sort": "downloads", "direction": -1,
            "limit": self._limit, "pipeline_tag": "text-generation",
        }, timeout=10)
        resp.raise_for_status()
        self._last_fetch = time.monotonic()
        out = []
        for item in resp.json():
            model_id = item.get("modelId") or item.get("id")
            if not model_id:
                continue
            m = _PARAMS_RE.search(model_id)
            out.append(CandidateModel(
                source="huggingface", model_id=model_id,
                model_url=f"https://huggingface.co/{model_id}",
                pipeline_tag=item.get("pipeline_tag"),
                params_size=f"{m.group(1)}B" if m else None,
                is_bounty=False, bounty_deadline=None,
                discovered_at=datetime.now(timezone.utc),
            ))
        return out
