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


def test_adoption_kill_switch_releases_itself_once_confirmed(tmp_path):
    """认领原先只在启动跑一次：网络抖一下就把智能体永久锁死，只能重启。
    "暂时不可知"必须能自愈，"出事了"才等人。"""
    from unittest.mock import Mock, patch

    from auto_adapter.main import ADOPTION_SOURCE, Deps, _adopt_in_flight_tasks
    from auto_adapter.settings import Settings
    from auto_adapter.storage.sqlite import SqliteStorage

    storage = SqliteStorage(str(tmp_path / "t.db"))
    deps = Deps(settings=Settings(xc_token="t", strategy_id="s"),
                storage=storage, client=Mock(), sources=[])

    with patch("auto_adapter.monitor.adopt_orphaned_platform_tasks", return_value=-1):
        assert _adopt_in_flight_tasks(deps) is False
    state = storage.kill_switch_state()
    assert state["on"] and state["source"] == ADOPTION_SOURCE

    with patch("auto_adapter.monitor.adopt_orphaned_platform_tasks", return_value=0):
        assert _adopt_in_flight_tasks(deps) is True
    assert storage.kill_switch() is False  # 自愈


def test_adoption_never_releases_a_switch_it_did_not_set(tmp_path):
    """凭据失效/任务疑似被清理拉的闸绝不能被认领成功顺手解除——那些要人看一眼。"""
    from unittest.mock import Mock, patch

    from auto_adapter.main import Deps, _adopt_in_flight_tasks
    from auto_adapter.settings import Settings
    from auto_adapter.storage.sqlite import SqliteStorage

    storage = SqliteStorage(str(tmp_path / "t.db"))
    storage.set_kill_switch(True, "task 42 vanished from platform", source="vanish")
    deps = Deps(settings=Settings(xc_token="t", strategy_id="s"),
                storage=storage, client=Mock(), sources=[])

    with patch("auto_adapter.monitor.adopt_orphaned_platform_tasks", return_value=3):
        _adopt_in_flight_tasks(deps)

    assert storage.kill_switch() is True
    assert storage.kill_switch_state()["source"] == "vanish"


def test_credential_failure_is_not_downgraded_to_auto_releasable(tmp_path):
    """凭据失效拉的闸必须保留其原因、且不可自动解除：令牌无效重试一万次也不会好，
    要人去换令牌。早先通用的"认领失败"会覆盖掉更精确的根因并把它降级。"""
    from unittest.mock import Mock, patch

    import requests

    from auto_adapter.main import ADOPTION_SOURCE, Deps, _adopt_in_flight_tasks
    from auto_adapter.settings import Settings
    from auto_adapter.storage.sqlite import SqliteStorage

    storage = SqliteStorage(str(tmp_path / "t.db"))
    deps = Deps(settings=Settings(xc_token="t", strategy_id="s"),
                storage=storage, client=Mock(), sources=[])

    err = requests.HTTPError("401 Client Error")
    err.response = Mock(status_code=401)
    with patch("auto_adapter.monitor.adopt_orphaned_platform_tasks", side_effect=err):
        assert _adopt_in_flight_tasks(deps) is False

    state = storage.kill_switch_state()
    assert state["on"] is True
    assert "credential" in state["reason"].lower()
    assert state["source"] != ADOPTION_SOURCE  # 不可自动解除
