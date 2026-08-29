"""失败分类与重试层（spec §4.7）。M5 实现。"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import yaml

from .models import FailureKind, TaskRecord, TaskStatus
from .platform_client import PlatformClient
from .settings import Settings
from .storage import Storage

logger = logging.getLogger(__name__)

# 日志关键词 → 失败类型（实现时依据真实日志样本补全）
_QUALITY_PATTERNS = ("judge", "quality check failed", "score below")

# 调参梯子边界：max_model_len 下限、tp 上限
_MIN_MODEL_LEN, _MAX_TP = 2048, 4

# 连续引擎失败熔断阈值（spec §6：连续失败告警——可能是本系统配置模板本身有问题，
# 需要人工介入而非无限重试）。成功任务由 Task 8 的 monitor.poll 清零该计数。
_STREAK_LIMIT = 5


def classify(log_text: str) -> FailureKind:
    """基于日志关键词分类。无法判定时按 ENGINE 处理（重试成本低于误拉黑）。"""
    text = (log_text or "").lower()
    if any(kw in text for kw in _QUALITY_PATTERNS):
        return FailureKind.QUALITY
    return FailureKind.ENGINE


def _set_flag(command: list, flag: str, value: str) -> None:
    if flag in command:
        command[command.index(flag) + 1] = value
    else:
        command.extend([flag, value])


def next_config(record: TaskRecord) -> str | None:
    """引擎失败的调参序列：解析 record.config_params（YAML，spec 附录 A.1.1），
    按 retry_count 递进调整——0: gpu-memory-utilization 提至 0.95；
    1: max_model_len 减半（下限 2048）；2: tp 翻倍（上限 4，sut 与 ref 保持一致）；
    ≥3: 序列耗尽返回 None（调用方拉黑）。

    每一档都基于 record.config_params 当前值累加调整（调用方在重试后把返回值写回
    record.config_params，故各档修改会在后续重试中保留）。

    顶层 max_model_len 与 sut_config/ref_config.gpu_num 按平台样例（spec 附录 A.1.1）
    是无引号整数；command 列表里的 --max-model-len/-tp 参数值是平台 CLI 参数，须保持
    字符串。

    tp 档（retry_count==2）若翻倍后与当前值相同（已达 _MAX_TP 上限，如 72B 模型起始
    tp=4），说明这一档不能真正改变配置——重新提交同一份已失败过的配置只会触发平台的
    重复提交监控，因此视为序列耗尽，返回 None（调用方拉黑）。"""
    cfg = yaml.safe_load(record.config_params)
    sut = cfg["sut_config"]["values"]["command"]
    ref = cfg["ref_config"]["values"]["command"]
    if record.retry_count == 0:
        _set_flag(sut, "--gpu-memory-utilization", "0.95")
    elif record.retry_count == 1:
        new_len = max(int(cfg.get("max_model_len", 4096)) // 2, _MIN_MODEL_LEN)
        cfg["max_model_len"] = new_len
        for cmd in (sut, ref):
            _set_flag(cmd, "--max-model-len", str(new_len))
    elif record.retry_count == 2:
        current = int(sut[sut.index("-tp") + 1]) if "-tp" in sut else 1
        new_tp = min(current * 2, _MAX_TP)
        if new_tp == current:
            return None  # 已达上限，翻倍无效——不重复提交同一份失败配置
        for section, cmd in (("sut_config", sut), ("ref_config", ref)):
            _set_flag(cmd, "-tp", str(new_tp))
            cfg[section]["gpu_num"] = new_tp
    else:
        return None
    return yaml.safe_dump(cfg, sort_keys=False)


def handle(storage: Storage, client: PlatformClient, settings: Settings,
          now: datetime | None = None) -> None:
    """处理 monitor 标记的失败任务：

    - ENGINE：next_config 调参后重新入队，retry_count+1；≥max_retries → BLACKLISTED
      （重试上限后必须拉黑，避免无限重试触发平台重复提交监控）；每处理一条引擎失败
      给 consecutive_engine_failures 计数 +1，达到 _STREAK_LIMIT 时 set_kill_switch
      （spec §6）；
    - QUALITY：NEEDS_HUMAN，不自动重试；
    - TIMEOUT：stop_tasks 批量释放资源；悬赏未过期且尚未重试过可重排队一次，否则
      ABANDONED。
    """
    now = now or datetime.now(timezone.utc)

    engine_failed = storage.tasks_by_status(TaskStatus.ENGINE_FAILED)
    if engine_failed:
        streak = storage.get_counter("consecutive_engine_failures") + len(engine_failed)
        storage.set_counter("consecutive_engine_failures", streak)
        if streak >= _STREAK_LIMIT:
            storage.set_kill_switch(
                True, f"{streak} consecutive engine failures; config template may be broken")
            logger.error("engine failure streak %d >= %d; kill switch ON", streak, _STREAK_LIMIT)
    for rec in engine_failed:
        new_cfg = next_config(rec) if rec.retry_count < settings.max_retries else None
        if new_cfg is None:
            rec.status = TaskStatus.BLACKLISTED
            logger.warning("blacklisting %s@%s after %d retries",
                           rec.model_id, rec.target_gpu, rec.retry_count)
        else:
            rec.config_params = new_cfg
            rec.retry_count += 1
            rec.status = TaskStatus.QUEUED
            rec.task_id = None
            rec.submit_time = None
        storage.update_task(rec)

    for rec in storage.tasks_by_status(TaskStatus.QUALITY_FAILED):
        rec.status = TaskStatus.NEEDS_HUMAN
        storage.update_task(rec)

    for rec in storage.tasks_by_status(TaskStatus.TIMEOUT):
        if rec.task_id is not None:
            try:
                client.stop_tasks([rec.task_id])
            except Exception:
                logger.exception("stop_tasks failed for %s", rec.task_id)
        if rec.bounty_deadline is not None and rec.bounty_deadline > now and rec.retry_count == 0:
            rec.status = TaskStatus.QUEUED
            rec.retry_count += 1
            rec.task_id = None
            rec.submit_time = None
        else:
            rec.status = TaskStatus.ABANDONED
        storage.update_task(rec)
