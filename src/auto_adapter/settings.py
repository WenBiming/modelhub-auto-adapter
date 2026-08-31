"""集中读取环境变量配置。凭据只在此处进入进程，禁止写日志。契约见 spec §5。

平台运行时契约（实测自官方 demo 仓库 xc_agent_platform_demo 与提交说明文档）：
平台注入的凭据环境变量名是 **EXTERNAL_SERVICE_TOKEN**，不是 XC_TOKEN；平台也不
注入 MODELHUB_BASE_URL。本地开发仍可用 XC_TOKEN，两者都接受。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://modelhub.org.cn"

# 平台不保证挂载任何卷，/data 在容器里根本不存在（线上实测 sqlite3 报
# "unable to open database file"）。默认落在镜像自带、必定可写的目录；
# 若运行环境挂了持久卷，用 STORAGE_PATH 指过去即可获得跨重启的持久化。
DEFAULT_STORAGE_PATH = "/app/data/agent.db"

# 平台注入的凭据变量名优先，其次是本地开发惯用名
TOKEN_ENV_VARS = ("EXTERNAL_SERVICE_TOKEN", "XC_TOKEN")


class ConfigError(Exception):
    """配置缺失/非法。由 main 捕获后转为"存活但不工作"，而不是崩溃循环。"""


@dataclass(frozen=True)
class Settings:
    xc_token: str
    strategy_id: str
    base_url: str = DEFAULT_BASE_URL
    dry_run: bool = False
    storage_path: str = DEFAULT_STORAGE_PATH
    tick_seconds: int = 60
    max_submits_per_minute: int = 2
    max_inflight: int = 5
    max_retries: int = 3
    task_timeout_hours: int = 6
    bounty_config_path: str = ""
    hf_fetch_limit: int = 50
    hf_discovery_enabled: bool = True
    modelscope_discovery_enabled: bool = True
    # v0.1 只提交 vllm（即 text-generation）。放开其他类型会让候选队列被无法提交的
    # 模型占满——每个候选都要花一次平台 search_model（10s 超时），而单 tick 只评估
    # 20 个。等平台确认了其他框架的启动命令再放开（spec §9）。
    discovery_task_types: tuple[str, ...] = ("text-generation",)

    @classmethod
    def from_env(cls) -> "Settings":
        """读取环境变量。缺失/非法一律抛 ConfigError，由 main 转成可诊断的存活态。"""
        try:
            tick_seconds = int(os.environ.get("TICK_SECONDS", cls.tick_seconds))
        except ValueError as e:
            raise ConfigError(f"TICK_SECONDS must be an integer: {e}") from e
        if tick_seconds < 60:
            raise ConfigError(
                "TICK_SECONDS must be >= 60 (rate limit assumes one drain per minute)")

        token = next((os.environ[k] for k in TOKEN_ENV_VARS if os.environ.get(k)), "")
        if not token:
            raise ConfigError(
                "missing platform credential: set EXTERNAL_SERVICE_TOKEN "
                "(injected by the platform) or XC_TOKEN for local runs")
        strategy_id = os.environ.get("STRATEGY_ID", "")
        if not strategy_id:
            raise ConfigError("missing STRATEGY_ID (the platform injects it at runtime)")

        return cls(
            xc_token=token,
            strategy_id=strategy_id,
            base_url=os.environ.get("MODELHUB_BASE_URL") or DEFAULT_BASE_URL,
            dry_run=os.environ.get("DRY_RUN", "").strip().lower() in ("1", "true", "yes"),
            storage_path=os.environ.get("STORAGE_PATH", cls.storage_path),
            tick_seconds=tick_seconds,
            max_submits_per_minute=int(
                os.environ.get("MAX_SUBMITS_PER_MINUTE", cls.max_submits_per_minute)
            ),
            max_inflight=int(os.environ.get("MAX_INFLIGHT", cls.max_inflight)),
            max_retries=int(os.environ.get("MAX_RETRIES", cls.max_retries)),
            task_timeout_hours=int(
                os.environ.get("TASK_TIMEOUT_HOURS", cls.task_timeout_hours)
            ),
            bounty_config_path=os.environ.get("BOUNTY_CONFIG_PATH", cls.bounty_config_path),
            hf_fetch_limit=int(os.environ.get("HF_FETCH_LIMIT", cls.hf_fetch_limit)),
            hf_discovery_enabled=os.environ.get(
                "HF_DISCOVERY_ENABLED", str(cls.hf_discovery_enabled)
            ).strip().lower() not in ("false", "0"),
            discovery_task_types=tuple(
                s.strip() for s in os.environ.get(
                    "DISCOVERY_TASK_TYPES", "text-generation").split(",") if s.strip()),
            modelscope_discovery_enabled=os.environ.get(
                "MODELSCOPE_DISCOVERY_ENABLED", str(cls.modelscope_discovery_enabled)
            ).strip().lower() not in ("false", "0"),
        )
