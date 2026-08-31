"""M6 冒烟测试：SIGTERM 应在平台 30s 宽限期内使进程优雅退出（spec §4.9）。

子进程直接跑 `auto_adapter.main.main()`：MODELHUB_BASE_URL 指向不可达地址以保证
离线（tick 内的 HTTP 调用会超时/报错但被 tick 顶层 try/except 吞掉，见
main.run_loop），HF_DISCOVERY_ENABLED=false 避免首个 tick 触发真实 HuggingFace
网络请求。TICK_SECONDS=60 保证测试运行期间不会进入第二个 tick。
"""
import os
import signal
import subprocess
import sys
import time


def test_sigterm_exits_within_grace_period(tmp_path):
    env = os.environ | {
        "XC_TOKEN": "t", "STRATEGY_ID": "s",
        "MODELHUB_BASE_URL": "http://127.0.0.1:1",  # 打不通也不该崩（tick 容错）
        "STORAGE_PATH": str(tmp_path / "agent.db"),
        "TICK_SECONDS": "60",
        "HF_DISCOVERY_ENABLED": "false",
        "MODELSCOPE_DISCOVERY_ENABLED": "false",
    }
    proc = subprocess.Popen(
        [sys.executable, "-c", "from auto_adapter.main import main; main()"], env=env)
    try:
        time.sleep(2)
        assert proc.poll() is None, "process should be running"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) == 0  # 30s 限额内（实际应秒级）
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
