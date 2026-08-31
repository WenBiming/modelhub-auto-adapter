"""可观测性（spec §6）：进程内计数器，每 tick 以结构化 JSON 打 stdout。M6 实现。"""
from __future__ import annotations

import json
import sys
from collections import Counter

_counters: Counter = Counter()


def incr(name: str, n: int = 1) -> None:
    _counters[name] += n


def flush_tick_summary(kill_switch: dict | None = None) -> None:
    """打一行 tick 汇总。

    kill_switch 状态随每个 tick 一起打出来：熔断开关是本系统唯一的安全刹车，但它
    只存在 storage 的 kv 表里，/health 又恒返回 200——不打进日志流的话，运维除非
    去翻 sqlite，否则看不出智能体已经停摆、更看不出为什么停摆。
    清除方法见 README「熔断开关（kill switch）」。
    """
    line = {"metrics": dict(_counters)}
    if kill_switch is not None:
        line["kill_switch"] = kill_switch
    print(json.dumps(line, ensure_ascii=False), file=sys.stdout, flush=True)


def reset() -> None:
    _counters.clear()
