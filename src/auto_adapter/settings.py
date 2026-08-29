"""集中读取环境变量配置。凭据只在此处进入进程，禁止写日志。契约见 spec §5。"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    xc_token: str
    strategy_id: str
    base_url: str
    storage_path: str = "/data/agent.db"
    tick_seconds: int = 60
    max_submits_per_minute: int = 2
    max_inflight: int = 5
    max_retries: int = 3
    task_timeout_hours: int = 6

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            xc_token=os.environ["XC_TOKEN"],
            strategy_id=os.environ["STRATEGY_ID"],
            base_url=os.environ["MODELHUB_BASE_URL"],
            storage_path=os.environ.get("STORAGE_PATH", cls.storage_path),
            tick_seconds=int(os.environ.get("TICK_SECONDS", cls.tick_seconds)),
            max_submits_per_minute=int(
                os.environ.get("MAX_SUBMITS_PER_MINUTE", cls.max_submits_per_minute)
            ),
            max_inflight=int(os.environ.get("MAX_INFLIGHT", cls.max_inflight)),
            max_retries=int(os.environ.get("MAX_RETRIES", cls.max_retries)),
            task_timeout_hours=int(
                os.environ.get("TASK_TIMEOUT_HOURS", cls.task_timeout_hours)
            ),
        )
