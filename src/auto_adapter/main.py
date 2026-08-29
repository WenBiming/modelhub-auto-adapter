"""入口：健康检查线程 + 主调度循环 + SIGTERM 优雅停机（spec §2、§4.9）。M1 实现。

优雅停机约束：每个 tick 步骤必须 < 30s（HTTP 超时 ≤ 10s 已保证），
收到 SIGTERM 后完成当前步骤即退出，状态已全部持久化，重启自动恢复。
"""
from __future__ import annotations

import logging
import signal
import threading
import time

from . import health
from .settings import Settings

logger = logging.getLogger(__name__)


def run_loop(tick_fn, stop_event: threading.Event, interval_seconds: float) -> None:
    while not stop_event.is_set():
        try:
            tick_fn()
        except Exception:
            logger.exception("tick failed; will retry next tick")
        stop_event.wait(interval_seconds)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    health.start_in_background(port=8080)
    deps = build_deps(settings)
    run_loop(lambda: tick(deps, stop_event), stop_event, settings.tick_seconds)


def build_deps(settings: Settings):
    """构造 storage/client/sources。Task 10 完成实现，这里先占位。"""
    raise NotImplementedError


def tick(deps, stop_event: threading.Event) -> None:
    """Task 10 接线。"""
    raise NotImplementedError


if __name__ == "__main__":
    main()
