"""M4：tp_size 推导（<14B→1，14–70B→2，>70B→4）、taskType 兜底规则、
无法判定时返回 None 而非盲提。"""
import pytest


@pytest.mark.skip(reason="M4 未实现")
def test_tp_size_by_params():
    ...


@pytest.mark.skip(reason="M4 未实现")
def test_task_type_fallback_rules():
    ...
