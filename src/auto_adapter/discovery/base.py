"""模型发现层接口（spec §4.2）。"""
from __future__ import annotations

import logging

import requests
from typing import Protocol

from ..models import CandidateModel

logger = logging.getLogger(__name__)


class DiscoverySource(Protocol):
    name: str

    def fetch(self) -> list[CandidateModel]:
        """拉取候选模型。实现自行负责节流（HF/MS 1h 一次，悬赏每 tick）。

        多页拉取的实现必须能被停机信号中断（stop_event 在构造时注入）：单页 10s
        超时 × 多页会超出平台 30s 的宽限期，超时后是 SIGKILL，不是优雅退出。
        """
        ...


def run(sources: list[DiscoverySource], storage) -> int:
    """执行所有来源，统一去重后写入候选表，返回**新**候选数。M3 实现。

    悬赏候选优先：若同一 model_id 既来自非悬赏源又来自悬赏源，保留悬赏版本以防止
    错失悬赏时间窗口。替换操作不重复计数。

    返回值只数候选表里此前不存在的 model_id。主循环把它作为 candidates_discovered
    上报：若把每轮看到的全部去重候选都算进去，稳态下这个指标会恒等于每次拉取的
    条数，"这个 tick 发现了什么新东西"就读不出来了。
    """
    candidates_by_id = {}  # model_id -> CandidateModel

    for src in sources:
        try:
            candidates = src.fetch()
        except requests.RequestException as e:
            # 上游不可达是可预期的常态（平台内网连不上 huggingface.co），
            # 每个 tick 打一整页 traceback 只会淹没真正的信号。
            logger.warning("discovery source %s unreachable: %s", src.name, e)
            continue
        except Exception:
            logger.exception("discovery source %s failed", src.name)
            continue
        for c in candidates:
            if c.model_id not in candidates_by_id:
                # 新候选
                candidates_by_id[c.model_id] = c
            elif c.is_bounty and not candidates_by_id[c.model_id].is_bounty:
                # 悬赏候选替换非悬赏重复（不重复计数）
                candidates_by_id[c.model_id] = c
            # 其他情况保留已有候选

    # 批量写入最终候选，同时统计其中真正新出现的（upsert 之前问，问完再写）
    new_count = 0
    for c in candidates_by_id.values():
        if not storage.has_candidate(c.model_id):
            new_count += 1
        storage.upsert_candidate(c)

    return new_count
