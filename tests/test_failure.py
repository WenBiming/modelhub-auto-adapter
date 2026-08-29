"""M5：日志关键词分类（OOM/CUDA → ENGINE；judge → QUALITY）、
调参序列递进、重试上限拉黑。"""
import pytest


@pytest.mark.skip(reason="M5 未实现")
def test_classify_engine_vs_quality():
    ...


@pytest.mark.skip(reason="M5 未实现")
def test_retry_exhaustion_blacklists():
    ...
