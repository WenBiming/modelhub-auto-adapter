"""监控与日志分析层（spec §4.6）。M5 实现。"""
from __future__ import annotations

from .platform_client import PlatformClient
from .settings import Settings
from .storage import Storage


def poll(storage: Storage, client: PlatformClient, settings: Settings) -> None:
    """对账并更新本地任务表：

    - list_my_tasks（onlyMine=true）分页拉取，以平台 status/verifyResult 为准更新 TaskRecord；
    - 本地在途但平台侧消失的任务 → ABANDONED + set_kill_switch(True)
      （可能被违规清理，最高级别告警，暂停提交待人工确认）；
    - 失败任务拉日志存入 record.last_log，留给 failure 层分类；
    - PENDING/RUNNING 超过 task_timeout_hours → 标记 TIMEOUT。
    """
    raise NotImplementedError
