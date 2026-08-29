"""ModelHub 开放平台客户端——全仓库唯一允许调用平台 API 的模块。契约见 spec §4.1。

M2 里程碑实现。所有方法 HTTP 超时 ≤ 10s（优雅停机要求）。
"""
from __future__ import annotations

import requests

from .models import AddTaskRequest

REQUEST_TIMEOUT = 10


class PlatformClientError(Exception):
    """4xx 类错误：请求本身有问题，不应重试。"""


class PlatformClient:
    def __init__(self, base_url: str, xc_token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers["Xc-Token"] = xc_token

    def add_task(self, req: AddTaskRequest) -> str:
        """POST /api/adapt/task/add，返回平台 task_id。"""
        raise NotImplementedError

    def list_tasks(self, page: int = 1, size: int = 50) -> list[dict]:
        """GET /api/adapt/task/page，返回本策略提交的任务分页。"""
        raise NotImplementedError

    def get_task_log(self, task_id: str) -> str:
        """GET /api/adapt/task/log?taskId=..."""
        raise NotImplementedError

    def search_adaptations(self, model_id: str) -> list[dict]:
        """GET /api/computility/models/search-by-model-id?modelId=...

        去重准入的平台侧依据；调用方在此接口失败时必须保守跳过（宁漏勿重）。
        """
        raise NotImplementedError

    def stop_task(self, task_id: str) -> None:
        """PUT /api/async/task/stop-create-contest-task"""
        raise NotImplementedError

    def list_bounties(self) -> list[dict]:
        """悬赏列表。真实路径待确认（spec §9 开放问题）。"""
        raise NotImplementedError
