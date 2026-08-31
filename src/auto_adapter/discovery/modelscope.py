"""ModelScope 发现源。

为什么它不再是"v0.1 范围外"：平台运行环境**连不上 huggingface.co**（线上实测
connect timeout），而 ModelScope 在国内可达，平台自己 API 文档的样例用的也是
modelscope.cn 的模型地址。在这个部署环境里它是唯一能真正产出候选的来源。

接口（2026-08-31 实测）：
    PUT https://www.modelscope.cn/api/v1/dolphin/models
    body: {PageNumber, PageSize, SortBy: "DownloadsCount", Target: "", SingleCriterion: []}
    resp: Data.Model.Models[]，每项含 Path（组织）/Name（模型名）/Tasks[].Name/
          Downloads/Architectures

按任务筛选走**客户端过滤**而非请求参数：SingleCriterion 的 schema 未公开，实测
传入猜测的形状会被静默忽略（返回结果与不传完全一致）——那种"看起来生效其实没有"
的筛选比不筛选更危险。
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone

import requests

from .. import rules
from ..models import CandidateModel

logger = logging.getLogger(__name__)

_API = "https://www.modelscope.cn/api/v1/dolphin/models"
_MODEL_URL = "https://www.modelscope.cn/models/{model_id}"
_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)[bB]\b")
# 按发布时间倒序：新模型才是"平台从没见过"的那批（5 积分/个）；按下载量排序拿到的
# 全是热门老模型，实测 17 个里 5 个所有可提交的卡都已适配、其余也只剩零星空位。
_SORT_NEWEST = "GmtCreated"

# 每次向 ModelScope 要多少条再客户端过滤。实测 100 条里约 17 条是 text-generation，
# 取 200 是为了在 limit=50 时基本能填满，同时不至于一次拉太多。
_FETCH_PAGE_SIZE = 200

_THROTTLE_KEY = "modelscope_last_fetch"


class ModelScopeSource:
    name = "modelscope"

    def __init__(self, storage, limit: int = 50, min_interval_seconds: int = 3600,
                 task_types: tuple[str, ...] = ("text-generation",)) -> None:
        # 节流时间戳落盘：容器崩溃重启循环时不能每次重启都再打一遍上游
        # （CLAUDE.md：禁止业务模块自建内存态）。
        self._storage = storage
        self._limit = limit
        self._min_interval = min_interval_seconds
        self._task_types = tuple(task_types)

    def fetch(self) -> list[CandidateModel]:
        now = int(time.time())
        last = self._storage.get_counter(_THROTTLE_KEY)
        if last and now - last < self._min_interval:
            return []

        resp = requests.put(_API, json={
            "PageNumber": 1, "PageSize": _FETCH_PAGE_SIZE,
            "SortBy": _SORT_NEWEST, "Target": "", "SingleCriterion": [],
        }, timeout=10)
        resp.raise_for_status()
        models = (resp.json().get("Data") or {}).get("Model", {}).get("Models") or []
        self._storage.set_counter(_THROTTLE_KEY, now)

        out: list[CandidateModel] = []
        below_threshold = 0
        for item in models:
            if len(out) >= self._limit:
                break
            tasks = [t.get("Name") for t in (item.get("Tasks") or [])]
            task_type = next((t for t in tasks if t in self._task_types), None)
            if task_type is None:
                continue
            if not rules.passes_download_threshold(task_type, item.get("Downloads")):
                below_threshold += 1
                continue
            org, name = item.get("Path"), item.get("Name")
            if not org or not name:
                continue
            model_id = f"{org}/{name}"
            m = _PARAMS_RE.search(name)
            out.append(CandidateModel(
                source="modelscope", model_id=model_id,
                model_url=_MODEL_URL.format(model_id=model_id),
                pipeline_tag=task_type,
                params_size=f"{m.group(1)}B" if m else None,
                is_bounty=False, bounty_deadline=None,
                discovered_at=datetime.now(timezone.utc),
            ))
        logger.info("modelscope: %d candidates from %d newest models "
                    "(%d below the download threshold)", len(out), len(models), below_threshold)
        return out
