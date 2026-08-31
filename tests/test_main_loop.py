import threading

from auto_adapter.main import run_loop


def test_run_loop_stops_on_event():
    stop = threading.Event()
    calls = []

    def tick():
        calls.append(1)
        if len(calls) >= 3:
            stop.set()

    run_loop(tick, stop, interval_seconds=0)
    assert len(calls) == 3


def test_run_loop_survives_tick_exception():
    stop = threading.Event()
    calls = []

    def tick():
        calls.append(1)
        if len(calls) >= 2:
            stop.set()
        raise RuntimeError("boom")

    run_loop(tick, stop, interval_seconds=0)
    assert len(calls) == 2


def test_health_starts_before_config_validation_and_survives_bad_config(monkeypatch):
    """平台用 livenessProbe 探 /health，连续失败重启 Pod，三次后标记失败。
    配置缺失时若在 /health 起来之前就崩，运维只能看到"失败"两个字。
    正确行为：先监听，再校验；校验失败保持存活并周期性打出原因。"""
    import threading as _t
    from unittest.mock import patch

    from auto_adapter import health, main as main_mod

    for k in ("XC_TOKEN", "EXTERNAL_SERVICE_TOKEN", "STRATEGY_ID"):
        monkeypatch.delenv(k, raising=False)

    order = []
    stop = _t.Event()

    def fake_health(port=8080):
        order.append("health")
        return None

    def fake_idle(stop_event, reason):
        order.append(("idle", reason))
        stop.set()

    with patch.object(health, "start_in_background", fake_health), \
         patch.object(main_mod, "_idle_until_stopped", fake_idle), \
         patch.object(main_mod.signal, "signal", lambda *a, **k: None):
        main_mod.main()

    assert order[0] == "health", "健康检查必须先于配置校验启动"
    assert order[1][0] == "idle"
    assert "EXTERNAL_SERVICE_TOKEN" in order[1][1]


def test_health_status_endpoint_reports_config_error():
    from auto_adapter import health
    health.set_state(status="misconfigured", config_error="missing X", dry_run=None)
    client = health.app.test_client()
    assert client.get("/health").status_code == 200  # 合规要求：存活即 200
    body = client.get("/").get_json()
    assert body["status"] == "misconfigured" and body["config_error"] == "missing X"
