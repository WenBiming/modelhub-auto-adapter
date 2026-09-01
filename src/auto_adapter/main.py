"""入口：健康检查线程 + 主调度循环 + SIGTERM 优雅停机（spec §2、§4.9）。M1 实现。

优雅停机约束：每个 tick 步骤必须 < 30s（HTTP 超时 ≤ 10s 已保证），
收到 SIGTERM 后完成当前步骤即退出，状态已全部持久化，重启自动恢复。

Task 10（M6）：接线全链路——discovery.run → eligibility.evaluate（逐候选）→
config_gen.build_request + storage.insert_task → submitter.drain → monitor.poll →
failure.handle，五个步骤边界均检查 stop_event（候选循环内、drain 前、poll 前、
handle 前），metrics 逐项计数，tick 末尾 flush 后清零（逐 tick 而非累计值）。
"""
from __future__ import annotations

import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field

from . import (config_gen, eligibility, failure, health, metrics, monitor,
               platform_client, rules, submitter)
from .discovery import base as discovery
from .discovery.bounty import ManualBountySource
from .discovery.huggingface import HuggingFaceSource
from .discovery.modelscope import ModelScopeSource
from .eligibility import Verdict
from .models import Priority, TaskRecord, TaskStatus
from .platform_client import PlatformClient, escalate_if_credential_error
from .settings import ConfigError, Settings
from .storage import SqliteStorage, StorageUnavailableError
from .storage.base import DuplicateTaskError

logger = logging.getLogger(__name__)

# 单 tick 内最多评估多少个候选。每个候选的 eligibility.evaluate 都要打一次平台
# search_model（HTTP 超时 10s），30s 优雅停机预算下不能让候选循环无限长：待处理
# 候选表可能一次涌入几百条（一次 HF 拉取 50 条 × 多个 tick 累积）。超出的部分不标
# processed，留到下个 tick，顺序不变。
MAX_CANDIDATES_PER_TICK = 20

# 同一候选连续多少个 tick 拿不到平台结果后升级为 NEEDS_HUMAN 并放行队列。
# 与上面的切片配套：SKIP_UNCERTAIN 不标 processed，若干个"永远查不出结果"的候选
# 会占满切片、把后面的候选无限期饿死（见 _note_uncertain）。
MAX_UNCERTAIN_TICKS = 5


def run_loop(tick_fn, stop_event: threading.Event, interval_seconds: float) -> None:
    while not stop_event.is_set():
        try:
            tick_fn()
        except Exception:
            logger.exception("tick failed; will retry next tick")
        stop_event.wait(interval_seconds)


def _idle_until_stopped(stop_event: threading.Event, reason: str) -> None:
    """配置不可用时保持存活并周期性复述原因，直到 SIGTERM。

    平台看到的是一个健康但明显在报错的 Pod，日志里每分钟一条原因——比崩溃三次后
    只剩一个"失败"状态可诊断得多。绝不在此状态下做任何平台调用。
    """
    while not stop_event.is_set():
        logger.error("agent idle: %s (fix the environment and restart)", reason)
        stop_event.wait(60)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    # 健康检查必须先于配置校验启动：平台 livenessProbe 探不到 /health 就重启 Pod，
    # 三次后直接标记失败——配置写错时那等于什么诊断信息都拿不到。先起来，再校验，
    # 校验失败就"存活但不工作"，把原因写进 / 端点和日志（见下）。
    health.start_in_background(port=8080)

    # 平台到底注入了哪些环境变量，文档没说全（只提了 STRATEGY_ID，demo 用的是
    # EXTERNAL_SERVICE_TOKEN 而它被 401 拒绝）。把**变量名**打出来是唯一可靠的
    # 判定方式——名字不是机密，值才是，值永远不进日志。
    logger.info("environment variables visible to the agent: %s",
                ", ".join(sorted(os.environ)) or "(none)")

    try:
        settings = Settings.from_env()
    except ConfigError as e:
        health.set_state(status="misconfigured", config_error=str(e),
                         token_env_candidates=Settings.token_env_candidates_snapshot())
        logger.error("configuration error: %s", e)
        _idle_until_stopped(stop_event, str(e))
        return

    logger.info("using credential from environment variable %r (value never logged); "
                "auth header %r", settings.token_env_name, platform_client.AUTH_HEADER)
    if settings.dry_run:
        logger.warning(
            "DRY RUN enabled: the agent will discover, dedup and build requests, "
            "but will NOT submit anything to the platform")

    # 存储打不开（平台未挂卷、路径不可写）同样不能崩：崩了只剩重启循环，
    # 平台三次后标记失败，运维看不到 "unable to open database file" 之外的线索。
    try:
        deps = build_deps(settings, stop_event)
    except StorageUnavailableError as e:
        health.set_state(status="storage-unavailable", config_error=str(e))
        logger.error("storage unavailable: %s", e)
        _idle_until_stopped(stop_event, str(e))
        return

    _adopt_in_flight_tasks(deps)  # 失败不阻塞启动：每个 tick 会重试并在成功后自动解闸
    health.set_state(status="running", config_error=None, dry_run=settings.dry_run,
                     token_env=settings.token_env_name)
    run_loop(lambda: tick(deps, stop_event), stop_event, settings.tick_seconds)


ADOPTION_SOURCE = "startup_adoption"


def _adopt_in_flight_tasks(deps: "Deps") -> bool:
    """认领平台上仍在途的任务，返回是否确认成功（容器存储是临时的，见 monitor）。

    确认不了（平台不可达或只读到部分名单）就拉下熔断闸：本地是空表而平台可能有在途
    任务，此时放行提交等于赌"没有重复"，而平台对重复提交的处理是清空账号全部任务。

    确认成功且当前闸门正是本函数拉的 → 自动解除。这一步很重要：认领原先只在启动时
    跑一次，网络抖一下就把智能体永久锁死，只能靠重启恢复。"暂时不可知"和"出事了"
    必须区别对待——前者自愈，后者等人。
    """
    credential_failure = False
    try:
        adopted = monitor.adopt_orphaned_platform_tasks(deps.storage, deps.client)
    except Exception as e:
        credential_failure = escalate_if_credential_error(deps.storage, e)
        logger.error("adoption failed; submissions stay paused until reconciled: %s", e)
        adopted = -1

    if adopted < 0:
        if credential_failure:
            # 凭据错误已经拉过闸并写明了原因。再覆盖成通用的"认领失败"会做两件坏事：
            # 抹掉更精确的根因，并把它降级成可自动解除——令牌无效要人换令牌，
            # 自动重试一万次也不会好。
            return False
        deps.storage.set_kill_switch(
            True, "could not confirm in-flight platform tasks; submissions paused "
                  "to avoid duplicates after a storage reset",
            source=ADOPTION_SOURCE)
        return False

    state = deps.storage.kill_switch_state()
    if state["on"] and state.get("source") == ADOPTION_SOURCE:
        deps.storage.set_kill_switch(False, "in-flight platform tasks confirmed", source="")
        logger.warning("platform state confirmed; releasing the adoption kill switch")
    return True


@dataclass
class Deps:
    settings: Settings
    storage: object
    client: object
    sources: list = field(default_factory=list)


def build_deps(settings: Settings, stop_event=None) -> Deps:
    """构造 storage/client/sources（spec §2）。

    悬赏源固定在前：discovery.run 对同一 model_id 的重复候选保留悬赏版本
    （见 discovery/base.py），来源顺序体现这一优先级意图。HuggingFace 源
    仅在 settings.hf_discovery_enabled 为真时纳入。
    """
    storage = SqliteStorage(settings.storage_path)
    sources = [ManualBountySource(settings.bounty_config_path)]
    if settings.modelscope_discovery_enabled:
        # 平台内网连不上 huggingface.co，ModelScope 是那里唯一可达的模型源。
        sources.append(ModelScopeSource(storage, limit=settings.hf_fetch_limit,
                                       task_types=settings.discovery_task_types,
                                       stop_event=stop_event))
    if settings.hf_discovery_enabled:
        # storage 注入：1h 节流时间戳必须落盘，否则崩溃重启循环会每次重启
        # 都打一遍上游（CLAUDE.md：禁止业务模块自建内存态）。
        sources.append(HuggingFaceSource(storage, limit=settings.hf_fetch_limit))
    return Deps(
        settings=settings,
        storage=storage,
        client=PlatformClient(settings.base_url, settings.xc_token),
        sources=sources,
    )


def tick(deps: Deps, stop_event: threading.Event) -> None:
    """按 spec §2 顺序执行一次完整调度：discovery → eligibility（逐候选）→
    config_gen + storage.insert_task → submitter.drain → monitor.poll → failure.handle。

    SKIP_UNCERTAIN 的候选不标记 processed，下个 tick 重试；其余判定结果
    （ENQUEUE 成功/失败/需人工、SKIP_DUPLICATE）均标记 processed。

    metrics 在 finally 里 flush + 清零：提前 return（停机）或抛异常的 tick 也必须
    留下一行 JSON，否则最值得看的那些 tick 恰好什么都不打。
    """
    s = deps
    try:
        _tick_body(s, stop_event)
    finally:
        try:
            metrics.flush_tick_summary(kill_switch=s.storage.kill_switch_state())
        except Exception:
            logger.exception("failed to flush tick metrics")
        metrics.reset()  # 逐 tick 计数：flush 后清零，日志流每行反映"这一 tick"而非累计值


def _tick_body(s: Deps, stop_event: threading.Event) -> None:
    # 上一轮没能确认平台状态时，每个 tick 再试一次并在成功后自动解闸——
    # 一次网络抖动不该让智能体永久停摆。
    state = s.storage.kill_switch_state()
    if state["on"] and state.get("source") == ADOPTION_SOURCE:
        _adopt_in_flight_tasks(s)
    metrics.incr("candidates_discovered", discovery.run(s.sources, s.storage))
    pending = s.storage.pending_candidates()
    if len(pending) > MAX_CANDIDATES_PER_TICK:
        logger.info("evaluating %d of %d pending candidates this tick",
                    MAX_CANDIDATES_PER_TICK, len(pending))
        pending = pending[:MAX_CANDIDATES_PER_TICK]
    for cand in pending:
        if stop_event.is_set():
            return
        decision = eligibility.evaluate(cand, s.storage, s.client)
        if decision.verdict == Verdict.SKIP_UNCERTAIN:
            metrics.incr("skipped_uncertain")
            _note_uncertain(s, cand)
            continue  # （未到上限时）不标记 processed，下个 tick 重试
        _clear_uncertain(s, cand)
        if decision.verdict == Verdict.ENQUEUE:
            target_gpu = decision.target_gpu  # eligibility 为这个候选选中的卡
            try:
                req = config_gen.build_request(cand, target_gpu, s.settings.strategy_id)
            except config_gen.UnresolvableCandidateError as e:
                _insert_needs_human(s, cand, target_gpu, decision, e)
                s.storage.mark_candidate_processed(cand.model_id)
                continue
            try:
                s.storage.insert_task(TaskRecord(
                    model_id=cand.model_id, target_gpu=target_gpu,
                    framework=req.framework, status=TaskStatus.QUEUED,
                    priority=decision.priority, model_url=cand.model_url,
                    task_type=req.task_type, config_params=req.config_params,
                    bounty_deadline=cand.bounty_deadline))
                metrics.incr("enqueued")
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
    monitor.poll(s.storage, s.client, s.settings, stop_event=stop_event)
    if stop_event.is_set():
        return
    failure.handle(s.storage, s.client, s.settings)


def _insert_needs_human(s: Deps, cand, target_gpu: str, decision,
                        error: config_gen.UnresolvableCandidateError) -> None:
    """无法自动组装提交参数的候选：落一条 NEEDS_HUMAN 记录（spec §4.4）。

    只打一行 warning 然后把候选标成 processed 等于让它凭空消失——悬赏候选尤其
    不能这样丢。记录里带上 model_url/model_id/target_gpu，人工可以直接接手。
    """
    metrics.incr(error.reason)
    logger.warning("%s: %s", error.reason, error)
    _needs_human_record(s, cand, target_gpu, decision.priority, error.framework)


def _needs_human_record(s: Deps, cand, target_gpu: str, priority, framework: str = "") -> None:
    try:
        s.storage.insert_task(TaskRecord(
            model_id=cand.model_id, target_gpu=target_gpu,
            framework=framework, status=TaskStatus.NEEDS_HUMAN,
            priority=priority, model_url=cand.model_url,
            task_type="", config_params="",
            bounty_deadline=cand.bounty_deadline))
    except DuplicateTaskError:
        pass  # 已有记录（含之前落的 NEEDS_HUMAN），无须重复插入


def _uncertain_key(model_id: str) -> str:
    return f"uncertain:{model_id}"


def _note_uncertain(s: Deps, cand) -> None:
    """记一次"平台查不出结果"，连续 MAX_UNCERTAIN_TICKS 次后放行队列。

    SKIP_UNCERTAIN 的候选不标 processed（宁漏勿重：查不清就不提交），但候选循环每个
    tick 只取前 MAX_CANDIDATES_PER_TICK 条、且 `WHERE processed = 0` 的扫描顺序稳定：
    只要有 20 个候选持续查不出结果，它们就会永远占满整个切片，后面所有候选被无限期
    饿死，而进程看起来一切正常（/health 200、tick 不报错）。

    连续 5 个 tick 都问不出结果，说明平台对这个模型持续不作答——那是人的问题，不是
    值得无限重试的问题：落 NEEDS_HUMAN 记录并标记 processed 让队列前进。记录本身也
    保证了后续 eligibility 对同一 (model_id, target_gpu) 直接 SKIP_DUPLICATE，不会
    因为放行而变成一次盲提交。
    """
    key = _uncertain_key(cand.model_id)
    streak = s.storage.get_counter(key) + 1
    s.storage.set_counter(key, streak)
    if streak < MAX_UNCERTAIN_TICKS:
        return
    priority = Priority.BOUNTY if cand.is_bounty else Priority.NEW_MODEL
    # 平台一直查不出结果，选卡也就无从谈起；挑一张占位卡把记录落下即可，
    # 目的只是让人能看到这个候选、并让它不再堵住队列。
    placeholder_gpu = config_gen.select_target_gpu(s.storage) or rules.KNOWN_GPUS[0]
    _needs_human_record(s, cand, placeholder_gpu, priority)
    s.storage.mark_candidate_processed(cand.model_id)
    s.storage.set_counter(key, 0)
    metrics.incr("uncertain_escalated")
    logger.error(
        "platform could not be queried for %s in %d consecutive ticks; marked NEEDS_HUMAN "
        "so the candidate queue can advance", cand.model_id, streak)


def _clear_uncertain(s: Deps, cand) -> None:
    """候选这一轮问出结果了：清零连续计数（只在非零时写，避免每个候选一次无谓写盘）。"""
    key = _uncertain_key(cand.model_id)
    if s.storage.get_counter(key):
        s.storage.set_counter(key, 0)


if __name__ == "__main__":
    main()
