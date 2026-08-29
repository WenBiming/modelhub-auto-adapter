"""可观测性（spec §6）：进程内计数器，每 tick 以结构化 JSON 打 stdout。M6 实现。"""
from __future__ import annotations

import json
import sys
from collections import Counter

_counters: Counter = Counter()


def incr(name: str, n: int = 1) -> None:
    _counters[name] += n


def flush_tick_summary() -> None:
    print(json.dumps({"metrics": dict(_counters)}), file=sys.stdout, flush=True)


def reset() -> None:
    _counters.clear()
