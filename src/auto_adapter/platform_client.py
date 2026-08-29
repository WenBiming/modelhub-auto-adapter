"""ModelHub 开放平台客户端——全仓库唯一允许调用平台 API 的模块。

契约见 spec §4.1 与附录 A（抓取自平台在线文档，2026-08-29）。
统一响应信封 {code, message, data}，code == 0 为成功。
M2 里程碑实现。所有方法 HTTP 超时 ≤ 10s（优雅停机要求）。
"""
from __future__ import annotations

import requests

from .models import AddTaskRequest, ModelSearchResult

REQUEST_TIMEOUT = 10

# 附录 A.6 错误码
CODE_OK = 0
CODE_NOT_LOGIN = 40100
CODE_NO_AUTH = 40101
CODE_NOT_FOUND = 40400
CODE_SYSTEM_ERROR = 50000
CODE_OPERATION_ERROR = 50001


class PlatformClientError(Exception):
    """业务码非 0。凭据类（40100/40101）由调用方触发 kill_switch。"""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code

    @property
    def is_credential_error(self) -> bool:
        return self.code in (CODE_NOT_LOGIN, CODE_NO_AUTH)


class PlatformClient:
    def __init__(self, base_url: str, xc_token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers["Xc-Token"] = xc_token

    def add_task(self, req: AddTaskRequest) -> int:
        """POST /api/adapt/task/add → AsyncTaskVO.id（int64）。

        请求体字段转 camelCase；configParams 为 YAML 字符串（附录 A.1.1）。
        """
        raise NotImplementedError

    def list_my_tasks(self, current: int = 1, page_size: int = 50, **filters) -> dict:
        """GET /api/adapt/task/page，固定 onlyMine=true。

        返回 Page<AsyncModelVerifyTaskVO>（附录 A.3）；filters 支持
        modelId/gpuType/status/stage/verifyResult/taskId 等。
        """
        raise NotImplementedError

    def get_task_log(self, task_id: int) -> str:
        """GET /api/adapt/task/log?taskId=... → 日志全文（data: string）。"""
        raise NotImplementedError

    def search_model(self, model_id: str) -> ModelSearchResult:
        """GET /api/computility/models/search-by-model-id?modelId=...

        去重准入的平台侧依据：verify_result 按 GPU 型号分键（附录 A.4）。
        调用方在此接口失败时必须保守跳过（宁漏勿重）。
        """
        raise NotImplementedError

    def stop_tasks(self, task_ids: list[int]) -> bool:
        """PUT /api/async/task/stop-create-contest-task，批量终止（附录 A.5）。"""
        raise NotImplementedError

    def list_bounties(self) -> list[dict]:
        """悬赏列表。开放平台 API 文档无此接口（spec §9）：
        需改为爬取「模型适配挑战」页面或读取人工配置列表。"""
        raise NotImplementedError
