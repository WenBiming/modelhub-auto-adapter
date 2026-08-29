"""提交与调度层（spec §4.5）。M4 实现。

限流参数只能调小不能绕过（CLAUDE.md 铁律）。
"""
from __future__ import annotations

from .platform_client import PlatformClient
from .settings import Settings
from .storage import Storage


def drain(storage: Storage, client: PlatformClient, settings: Settings) -> int:
    """提交 QUEUED 任务，返回本 tick 实际提交数。

    - kill_switch 打开时直接返回 0；
    - 排序键 (priority, bounty_deadline, discovered_at)；
    - 令牌桶：max_submits_per_minute；在途 (PENDING+RUNNING) ≥ max_inflight 时停止；
    - 悬赏剩余时间 < 预估适配时长×2 仍未提交 → 标记 ABANDONED 并告警；
    - 成功：写回 task_id/submit_time/status=PENDING；失败：保持 QUEUED 记录原因。
    """
    raise NotImplementedError
