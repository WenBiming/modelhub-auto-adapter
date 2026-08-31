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
_FILES_API = "https://www.modelscope.cn/api/v1/models/{model_id}/repo/files"
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
                 task_types: tuple[str, ...] = ("text-generation",),
                 stop_event=None) -> None:
        # 节流时间戳落盘：容器崩溃重启循环时不能每次重启都再打一遍上游
        # （CLAUDE.md：禁止业务模块自建内存态）。
        self._storage = storage
        self._limit = limit
        self._min_interval = min_interval_seconds
        self._task_types = tuple(task_types)
        self._stop_event = stop_event

    def _admits(self, item: dict) -> bool:
        """任务类型在配置内、且下载量过门槛。"""
        tasks = [t.get("Name") for t in (item.get("Tasks") or [])]
        task_type = next((t for t in tasks if t in self._task_types), None)
        return task_type is not None and rules.passes_download_threshold(
            task_type, item.get("Downloads"))

    def _resolve_gguf_file(self, model_id: str) -> str | None:
        """列出仓库文件并按量化偏好挑一个 .gguf（rules.pick_gguf_file）。

        网络失败时返回 None（跳过该候选）——组不出可信的启动命令就别提交。
        """
        try:
            resp = requests.get(_FILES_API.format(model_id=model_id),
                                params={"Revision": "master", "Root": ""}, timeout=10)
            resp.raise_for_status()
            files = (resp.json().get("Data") or {}).get("Files") or []
        except Exception as e:
            logger.warning("cannot list files for %s: %s", model_id, e)
            return None
        return rules.pick_gguf_file([f.get("Name") for f in files])

    def fetch(self) -> list[CandidateModel]:
        now = int(time.time())
        last = self._storage.get_counter(_THROTTLE_KEY)
        if last and now - last < self._min_interval:
            return []

        models: list[dict] = []
        for page in range(1, _MAX_PAGES + 1):
            if self._stop_event is not None and self._stop_event.is_set():
                # 每页最坏 10s，翻满 5 页会超出平台 30s 的停机宽限期（之后是 SIGKILL）。
                # 已拿到的页照常使用，剩下的留给下个 tick。
                logger.warning("shutdown requested during modelscope pagination "
                               "after %d page(s)", page - 1)
                break
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
            gguf_file = None
            if rules.is_gguf(model_id):
                gguf_file = self._resolve_gguf_file(model_id)
                if gguf_file is None:
                    gguf_skipped += 1
                    continue  # 没有可用的量化档，投出去也是白费
            m = _PARAMS_RE.search(name)
            out.append(CandidateModel(
                source="modelscope", model_id=model_id,
                model_url=_MODEL_URL.format(model_id=model_id),
                pipeline_tag=task_type,
                params_size=f"{m.group(1)}B" if m else None,
                is_bounty=False, bounty_deadline=None,
                discovered_at=datetime.now(timezone.utc),
                model_file=gguf_file,
            ))
        logger.info("modelscope: %d candidates from %d newest models "
                    "(%d below the download threshold, %d GGUF unusable)",
                    len(out), len(models), below_threshold, gguf_skipped)
        return out
