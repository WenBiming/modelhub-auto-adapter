"""平台合规健康检查端点（spec §4.9）。在 daemon 线程中运行。"""
from __future__ import annotations

import threading

from flask import Flask

app = Flask(__name__)


@app.route("/health")
def health():
    return {"status": "ok"}, 200


def start_in_background(port: int = 8080) -> threading.Thread:
    thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, use_reloader=False),
        daemon=True,
    )
    thread.start()
    return thread
