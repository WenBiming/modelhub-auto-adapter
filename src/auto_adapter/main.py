"""入口：健康检查线程 + 主调度循环 + SIGTERM 优雅停机（spec §2、§4.9）。M1 实现。

优雅停机约束：每个 tick 步骤必须 < 30s（HTTP 超时 ≤ 10s 已保证），
收到 SIGTERM 后完成当前步骤即退出，状态已全部持久化，重启自动恢复。

Task 10（M6）：接线全链路——discovery.run → eligibility.evaluate（逐候选）→
config_gen.build_request + storage.insert_task → submitter.drain → monitor.poll →
failure.handle，步骤间检查 stop_event，metrics 逐项计数并在 tick 末尾 flush。
"""
from __future__ import annotations

import logging
import signal
import threading
import time
from dataclasses import dataclass, field

from . import config_gen, eligibility, failure, health, metrics, monitor, submitter
from .discovery import base as discovery
from .discovery.bounty import ManualBountySource
from .discovery.huggingface import HuggingFaceSource
from .eligibility import Verdict
from .models import TaskRecord, TaskStatus
from .platform_client import PlatformClient
from .settings import Settings
from .storage import SqliteStorage
from .storage.base import DuplicateTaskError

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


@dataclass
class Deps:
    settings: Settings
    storage: object
    client: object
    sources: list = field(default_factory=list)


def build_deps(settings: Settings) -> Deps:
    """构造 storage/client/sources（spec §2）。

    悬赏源固定在前：discovery.run 对同一 model_id 的重复候选保留悬赏版本
    （见 discovery/base.py），来源顺序体现这一优先级意图。HuggingFace 源
    仅在 settings.hf_discovery_enabled 为真时纳入。
    """
    sources = [ManualBountySource(settings.bounty_config_path)]
    if settings.hf_discovery_enabled:
        sources.append(HuggingFaceSource(limit=settings.hf_fetch_limit))
    return Deps(
        settings=settings,
        storage=SqliteStorage(settings.storage_path),
        client=PlatformClient(settings.base_url, settings.xc_token),
        sources=sources,
    )


def tick(deps: Deps, stop_event: threading.Event) -> None:
    """按 spec §2 顺序执行一次完整调度：discovery → eligibility（逐候选）→
    config_gen + storage.insert_task → submitter.drain → monitor.poll → failure.handle。

    SKIP_UNCERTAIN 的候选不标记 processed，下个 tick 重试；其余判定结果
    （ENQUEUE 成功/失败、SKIP_DUPLICATE）均标记 processed。
    """
    s = deps
    metrics.incr("candidates_discovered", discovery.run(s.sources, s.storage))
    for cand in s.storage.pending_candidates():
        if stop_event.is_set():
            return
        target_gpu = config_gen.select_target_gpu(s.storage)
        decision = eligibility.evaluate(cand, target_gpu, s.storage, s.client)
        if decision.verdict == Verdict.SKIP_UNCERTAIN:
            metrics.incr("skipped_uncertain")
            continue  # 不标记 processed，下个 tick 重试
        if decision.verdict == Verdict.ENQUEUE:
            try:
                req = config_gen.build_request(cand, target_gpu, s.settings.strategy_id)
                s.storage.insert_task(TaskRecord(
                    model_id=cand.model_id, target_gpu=target_gpu,
                    framework=req.framework, status=TaskStatus.QUEUED,
                    priority=decision.priority, model_url=cand.model_url,
                    task_type=req.task_type, config_params=req.config_params,
                    bounty_deadline=cand.bounty_deadline))
                metrics.incr("enqueued")
            except ValueError:
                metrics.incr("unresolvable_task_type")
                logger.warning("cannot resolve task type for %s; needs human", cand.model_id)
            except DuplicateTaskError:
                metrics.incr("skipped_duplicate")
        else:
            metrics.incr("skipped_duplicate")
        s.storage.mark_candidate_processed(cand.model_id)
    if stop_event.is_set():
        return
    metrics.incr("submitted", submitter.drain(s.storage, s.client, s.settings))
    if stop_event.is_set():
        return
    monitor.poll(s.storage, s.client, s.settings)
    failure.handle(s.storage, s.client, s.settings)
    metrics.flush_tick_summary()


if __name__ == "__main__":
    main()
