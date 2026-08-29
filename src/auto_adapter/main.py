"""入口：健康检查线程 + 主调度循环 + SIGTERM 优雅停机（spec §2、§4.9）。M1 实现。

优雅停机约束：每个 tick 步骤必须 < 30s（HTTP 超时 ≤ 10s 已保证），
收到 SIGTERM 后完成当前步骤即退出，状态已全部持久化，重启自动恢复。
"""
from __future__ import annotations

import signal
import threading
import time

from . import health
from .settings import Settings


def main() -> None:
    settings = Settings.from_env()
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    health.start_in_background(port=8080)

    # M1 后续接线：storage/client/sources 的构造与各层调用
    while not stop_event.is_set():
        tick(settings, stop_event)
        stop_event.wait(settings.tick_seconds)


def tick(settings: Settings, stop_event: threading.Event) -> None:
    """单个调度周期，顺序执行（spec §2）：
    discovery.run → eligibility → submitter.drain → monitor.poll → failure.handle
    每步之间检查 stop_event。M1~M5 逐步接线。
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
