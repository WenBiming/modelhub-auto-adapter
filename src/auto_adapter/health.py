"""平台合规健康检查端点（spec §4.9）。在 daemon 线程中运行。

平台用 K8s livenessProbe 探测 /health，连续失败会重启 Pod，重启三次后标记为失败。
因此 **/health 必须先于配置校验起来**：配置缺失时崩溃退出会让平台只报"失败"而
拿不到任何诊断信息，运维要盯着重启循环猜原因。取而代之的做法是保持存活、把错误
原因放进 / 端点和每个 tick 的日志里（参见 main.main 的处理）。

/ 端点仿照官方 demo（xc_agent_platform_demo）：报告运行状态与配置是否就位，
不回显任何凭据本身。
"""
from __future__ import annotations

import threading

from flask import Flask

app = Flask(__name__)

# 由 main 在启动过程中更新；健康检查线程只读。
_state: dict = {"status": "starting", "config_error": None, "dry_run": None,
                "token_env": None}
_lock = threading.Lock()


def set_state(**kwargs) -> None:
    with _lock:
        _state.update(kwargs)


@app.route("/health")
def health():
    # 平台合规要求：存活即 200。配置错误不体现在这里（否则只会触发重启循环，
    # 反而看不到原因），而是通过 / 端点与日志暴露。
    return {"status": "ok"}, 200


@app.route("/")
def status():
    with _lock:
        return dict(_state), 200


def start_in_background(port: int = 8080) -> threading.Thread:
    thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, use_reloader=False),
        daemon=True,
    )
    thread.start()
    return thread
