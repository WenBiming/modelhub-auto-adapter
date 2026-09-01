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

# 凭据来源，按优先级从高到低。
#
# XC_TOKEN 在前是刻意的：平台注入的 EXTERNAL_SERVICE_TOKEN 被开放平台 API 以 401
# 拒绝（线上实测），运维只能自行配一个有效的 xcToken——**显式配置必须能压过平台
# 的默认注入**，否则那个无效令牌会一直盖掉你配的这个，而症状只是一句 401。
TOKEN_ENV_VARS = (
    "XC_TOKEN",                # 本地/运维显式配置，优先级最高
    "XcToken",                 # 平台 API 文档里的请求头名，注入时可能同名
    "xcToken",
    "XCTOKEN",
    "EXTERNAL_SERVICE_TOKEN",  # 官方 demo 读的名字（线上实测被 401 拒绝）
)

# 兜底扫描时排除：含 token 字样但不是平台凭据。
_TOKEN_SCAN_EXCLUDE = frozenset({"AUTH_HEADER"})


def token_env_candidates() -> list[str]:
    """环境里所有名字含 token 的变量名（不区分大小写），排序返回。**只有名字**。

    平台到底注入了什么，文档没说全，官方 demo 读的那个又被 401 拒绝。穷举候选之外
    再扫一遍，让智能体自己找到能用的那个，同时把找到的名字暴露出来供诊断。
    """
    return sorted(
        name for name, value in os.environ.items()
        if "token" in name.lower() and value and name not in _TOKEN_SCAN_EXCLUDE)


def resolve_token_env_name() -> str | None:
    """选出实际使用的令牌变量名，**只在 TOKEN_ENV_VARS 白名单内取**。

    刻意不拿兜底扫描的结果直接用：环境里任何含 token 字样的变量都可能是别的服务的
    凭据（CI 令牌之类），把它当作 Xc-Token 发给 modelhub.org.cn 就是把无关凭据
    泄露给第三方。扫描结果只用于诊断（token_env_candidates），要用哪个由人决定，
    加进白名单或直接设 XC_TOKEN。
    """
    for name in TOKEN_ENV_VARS:
        if os.environ.get(name):
            return name
    return None


class ConfigError(Exception):
    """配置缺失/非法。由 main 捕获后转为"存活但不工作"，而不是崩溃循环。"""


@dataclass(frozen=True)
class Settings:
    xc_token: str
    strategy_id: str
    base_url: str = DEFAULT_BASE_URL
    dry_run: bool = False
    token_env_name: str = ""  # 令牌取自哪个环境变量（只记名字，便于诊断）
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

    @staticmethod
    def token_env_candidates_snapshot() -> list[str]:
        """诊断用：环境里所有含 token 字样的变量名。只有名字，没有值。"""
        return token_env_candidates()

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

        token_env = resolve_token_env_name()
        if token_env is None:
            seen = token_env_candidates()
            raise ConfigError(
                "missing platform credential: none of "
                f"{', '.join(TOKEN_ENV_VARS)} is set. "
                + (f"Environment does contain token-looking variables: {', '.join(seen)} "
                   "— if one of those is the platform credential, add its name to "
                   "TOKEN_ENV_VARS (it is not used automatically, because an unrelated "
                   "token must never be sent to the platform)."
                   if seen else "No token-looking variable is present at all."))
        token = os.environ[token_env]
        strategy_id = os.environ.get("STRATEGY_ID", "")
        if not strategy_id:
            raise ConfigError("missing STRATEGY_ID (the platform injects it at runtime)")

        return cls(
            xc_token=token,
            token_env_name=token_env,
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
