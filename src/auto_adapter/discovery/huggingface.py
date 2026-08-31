"""HuggingFace Hub 发现源：按 downloads/trending/pipeline_tag 筛选。M3 实现。"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone

import requests

from .. import rules
from ..models import CandidateModel

# 平台内网连不上 huggingface.co（线上实测 connect timeout）。HF_ENDPOINT 是 HF 生态
# 的既有约定（hf-mirror.com 等镜像都认这个变量），留出替换入口而不写死镜像地址。
_DEFAULT_ENDPOINT = "https://huggingface.co"


def _api_url() -> str:
    return os.environ.get("HF_ENDPOINT", _DEFAULT_ENDPOINT).rstrip("/") + "/api/models"
_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)[bB]\b")

# 上次成功拉取的墙钟时间戳（epoch 秒），存 storage 的 kv 计数器（跨进程重启有效）。
LAST_FETCH_KEY = "hf_last_fetch_epoch"


class HuggingFaceSource:
    name = "huggingface"

    def __init__(self, storage, limit: int = 50, min_interval_seconds: int = 3600) -> None:
        self._storage = storage
        self._limit = limit
        self._min_interval = min_interval_seconds

    def fetch(self) -> list[CandidateModel]:
        # 节流状态走 storage：进程内变量在崩溃重启循环里每次都从零开始，等于每次
        # 重启都打一遍 HF（CLAUDE.md：禁止业务模块自建内存态）。
        now = int(time.time())
        last_fetch = self._storage.get_counter(LAST_FETCH_KEY)
        if last_fetch and 0 <= now - last_fetch < self._min_interval:
            return []
        resp = requests.get(_api_url(), params={
            "sort": "downloads", "direction": -1,
            "limit": self._limit, "pipeline_tag": "text-generation",
        }, timeout=10)
        resp.raise_for_status()
        self._storage.set_counter(LAST_FETCH_KEY, now)
        out = []
        for item in resp.json():
            model_id = item.get("modelId") or item.get("id")
            if not model_id:
                continue
            if rules.is_gguf(model_id):
                continue  # GGUF 是 llama.cpp 格式，v0.1 提交不了（见 config_gen）
            if not rules.passes_download_threshold(
                    item.get("pipeline_tag"), item.get("downloads")):
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
