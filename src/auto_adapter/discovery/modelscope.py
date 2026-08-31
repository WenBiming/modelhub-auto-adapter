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

# 单页条数：实测传 200 也只返回 100，接口自身封顶。
_FETCH_PAGE_SIZE = 100

# 最多翻几页。"最新"和"下载量门槛"天然互相拉扯——今天刚发布的模型不可能有 50 次
# 下载，能同时满足的是发布了几天到几周、刚积累起热度的那一段。实测第一页 100 条只
# 筛出 2 个候选（11 条 text-generation，9 条没过门槛），只看一页产出率太低，智能体
# 大部分时间会空转。往深翻几页才能覆盖到那一段。
_MAX_PAGES = 5

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

    def _admits(self, item: dict) -> bool:
        """任务类型在配置内、且下载量过门槛。"""
        tasks = [t.get("Name") for t in (item.get("Tasks") or [])]
        task_type = next((t for t in tasks if t in self._task_types), None)
        return task_type is not None and rules.passes_download_threshold(
            task_type, item.get("Downloads"))

    def fetch(self) -> list[CandidateModel]:
        now = int(time.time())
        last = self._storage.get_counter(_THROTTLE_KEY)
        if last and now - last < self._min_interval:
            return []

        models: list[dict] = []
        for page in range(1, _MAX_PAGES + 1):
            resp = requests.put(_API, json={
                "PageNumber": page, "PageSize": _FETCH_PAGE_SIZE,
                "SortBy": _SORT_NEWEST, "Target": "", "SingleCriterion": [],
            }, timeout=10)
            resp.raise_for_status()
            batch = (resp.json().get("Data") or {}).get("Model", {}).get("Models") or []
            models.extend(batch)
            if len(batch) < _FETCH_PAGE_SIZE:
                break  # 翻到底了
            if sum(1 for m in models if self._admits(m)) >= self._limit:
                break  # 够了就别再打上游
        self._storage.set_counter(_THROTTLE_KEY, now)

        out: list[CandidateModel] = []
        below_threshold = 0
        gguf_skipped = 0
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
            if rules.is_gguf(model_id):
                gguf_skipped += 1
                continue  # llama.cpp 格式，v0.1 提交不了（见 config_gen）
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
                    "(%d below the download threshold, %d GGUF)",
                    len(out), len(models), below_threshold, gguf_skipped)
        return out
