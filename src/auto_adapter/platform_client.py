"""ModelHub 开放平台客户端——全仓库唯一允许调用平台 API 的模块。

契约见 spec §4.1 与附录 A（抓取自平台在线文档，2026-08-29）。
统一响应信封 {code, message, data}，code == 0 为成功。
M2 里程碑实现。所有方法 HTTP 超时 ≤ 10s（优雅停机要求）。
"""
from __future__ import annotations

import logging
import os

import requests

from .models import AddTaskRequest, ModelSearchResult

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 10

# 平台开放平台 API 的鉴权头。文档写明是 Xc-Token，但平台注入给智能体的
# EXTERNAL_SERVICE_TOKEN 未必就是同一种令牌（线上实测被 401 拒绝）——留一个不改代码
# 就能换头名的口子，便于排查。值本身永远不进日志。
AUTH_HEADER = os.environ.get("AUTH_HEADER", "Xc-Token")

# 附录 A.6 错误码
CODE_OK = 0
CODE_NOT_LOGIN = 40100
CODE_NO_AUTH = 40101
CODE_NOT_FOUND = 40400
CODE_SYSTEM_ERROR = 50000
CODE_OPERATION_ERROR = 50001


# 平台侧的临时系统错误：请求可能已经被处理，也可能没有——结果不可知。
TRANSIENT_CODES = (CODE_SYSTEM_ERROR, CODE_OPERATION_ERROR)


class PlatformClientError(Exception):
    """业务码非 0。凭据类（40100/40101）由调用方触发 kill_switch。"""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code

    @property
    def is_credential_error(self) -> bool:
        return self.code in (CODE_NOT_LOGIN, CODE_NO_AUTH)

    @property
    def is_transient(self) -> bool:
        """50000/50001：平台内部异常，请求是否已生效不可知。

        提交路径上必须按"可能已建单"处理——绝不能回滚成 QUEUED 重提交。
        """
        return self.code in TRANSIENT_CODES

    @property
    def is_definite_rejection(self) -> bool:
        """业务码明确拒绝了这次请求（40100/40101/40400 等）：平台没有建单。

        只有这种情况下把记录退回 QUEUED 才是安全的。
        """
        return not self.is_transient


def _is_credential_failure(exc: BaseException) -> bool:
    """凭据失效的两种形态：业务码 40100/40101，以及 HTTP 401/403。

    线上实测：令牌无效时平台在 HTTP 层就返回 401，`raise_for_status()` 抛的是
    requests.HTTPError，根本走不到业务码解析——只认业务码会让最该告警的情形
    （令牌过期）静默退化成一个笼统的"接口失败"。
    """
    if isinstance(exc, PlatformClientError) and exc.is_credential_error:
        return True
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None) in (401, 403)


def escalate_if_credential_error(storage, exc: BaseException) -> bool:
    """凭据错误（40100/40101）出现在任何阶段都必须拉闸，返回是否拉了闸。

    过期的 Xc-Token 会让 eligibility/monitor/failure 的每一次平台调用都失败；
    若各处只是吞进通用 handler，智能体会静默退化成空转（既不提交也不告警）。
    调用方在每个通用 except 之前调用本函数。异常信息里只有平台返回的
    code/message，不含凭据本身（CLAUDE.md：凭据不写日志）。
    """
    if not _is_credential_failure(exc):
        return False
    try:
        storage.set_kill_switch(True, f"credential error from platform: {exc}")
    except Exception:
        logger.exception("failed to persist kill switch after credential error")
    logger.error(
        "credential error from platform (%s); kill switch ON. "
        "The agent sent header %r; if the platform injected EXTERNAL_SERVICE_TOKEN is not "
        "an open-API xcToken, set AUTH_HEADER or supply a valid token — retrying will not "
        "fix an invalid credential.", exc, AUTH_HEADER)
    return True


class PlatformClient:
    def __init__(self, base_url: str, xc_token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers[AUTH_HEADER] = xc_token
        # 官方 API 文档的调用示例带 Accept，照做以免在内容协商上出现差异。
        self._session.headers["Accept"] = "application/json"

    def _request(self, method: str, path: str, *, params=None, json_body=None):
        resp = self._session.request(
            method, self._base_url + path,
            params=params, json=json_body, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != CODE_OK:
            raise PlatformClientError(body.get("code", -1), body.get("message", ""))
        return body.get("data")

    def add_task(self, req: AddTaskRequest) -> int:
        """POST /api/adapt/task/add → AsyncTaskVO.id（int64）。

        请求体字段转 camelCase；configParams 为 YAML 字符串（附录 A.1.1）。
        """
        data = self._request("POST", "/api/adapt/task/add", json_body={
            "modelAddress": req.model_address, "taskType": req.task_type,
            "targetGpu": req.target_gpu, "framework": req.framework,
            "configParams": req.config_params, "strategyId": req.strategy_id,
        })
        return int(data["id"])

    def list_my_tasks(self, current: int = 1, page_size: int = 50, **filters) -> dict:
        """GET /api/adapt/task/page，固定 onlyMine=true。

        返回 Page<AsyncModelVerifyTaskVO>（附录 A.3）；filters 支持
        modelId/gpuType/status/stage/verifyResult/taskId 等。
        """
        params = {"current": current, "pageSize": page_size, "onlyMine": "true", **filters}
        return self._request("GET", "/api/adapt/task/page", params=params)

    def get_task_log(self, task_id: int) -> str:
        """GET /api/adapt/task/log?taskId=... → 日志全文（data: string）。"""
        return self._request("GET", "/api/adapt/task/log", params={"taskId": task_id})

    def search_model(self, model_id: str) -> ModelSearchResult:
        """GET /api/computility/models/search-by-model-id?modelId=...

        去重准入的平台侧依据：verify_result 按 GPU 型号分键（附录 A.4）。
        调用方在此接口失败时必须保守跳过（宁漏勿重）。
        """
        data = self._request("GET", "/api/computility/models/search-by-model-id",
                             params={"modelId": model_id})
        return ModelSearchResult(
            is_in_db=bool(data.get("isInDB")),
            model_info=data.get("modelInfo") or {},
            verify_result=data.get("verifyResult") or {},
        )

    def stop_tasks(self, task_ids: list[int]) -> bool:
        """PUT /api/async/task/stop-create-contest-task，批量终止（附录 A.5）。"""
        return bool(self._request("PUT", "/api/async/task/stop-create-contest-task",
                                  json_body={"taskIds": task_ids}))
