"""M1：storage 唯一键幂等 —— 同 (model_id, target_gpu) 重复 insert_task
必须抛 DuplicateTaskError（防重复提交最终防线，spec §3 不变式）。"""
import pytest


@pytest.mark.skip(reason="M1 未实现")
def test_duplicate_task_rejected():
    ...


@pytest.mark.skip(reason="M1 未实现")
def test_kill_switch_roundtrip():
    ...
