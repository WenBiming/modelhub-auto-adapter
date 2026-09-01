import os
import pytest
from auto_adapter.settings import ConfigError, Settings


def test_tick_seconds_below_60_raises(monkeypatch):
    """TICK_SECONDS < 60 must be rejected to prevent rate-limit bypass."""
    monkeypatch.setenv("XC_TOKEN", "token")
    monkeypatch.setenv("STRATEGY_ID", "strat")
    monkeypatch.setenv("MODELHUB_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("TICK_SECONDS", "30")

    with pytest.raises(ConfigError, match="TICK_SECONDS must be >= 60"):
        Settings.from_env()


def test_tick_seconds_60_succeeds(monkeypatch):
    """TICK_SECONDS=60 is the minimum allowed value."""
    monkeypatch.setenv("XC_TOKEN", "token")
    monkeypatch.setenv("STRATEGY_ID", "strat")
    monkeypatch.setenv("MODELHUB_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("TICK_SECONDS", "60")

    settings = Settings.from_env()
    assert settings.tick_seconds == 60


def test_tick_seconds_above_60_succeeds(monkeypatch):
    """TICK_SECONDS > 60 is allowed (stricter rate limit)."""
    monkeypatch.setenv("XC_TOKEN", "token")
    monkeypatch.setenv("STRATEGY_ID", "strat")
    monkeypatch.setenv("MODELHUB_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("TICK_SECONDS", "120")

    settings = Settings.from_env()
    assert settings.tick_seconds == 120


def _base_env(monkeypatch):
    """只设平台真正注入的两个变量：凭据 + STRATEGY_ID。"""
    from auto_adapter import settings as _s
    for k in (*_s.TOKEN_ENV_VARS, "MODELHUB_BASE_URL", "DRY_RUN", "TICK_SECONDS"):
        monkeypatch.delenv(k, raising=False)
    # 兜底扫描会捡起宿主机上任何含 token 的变量，测试里要清干净
    for k in list(_s.token_env_candidates()):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("STRATEGY_ID", "strat")


def test_platform_token_env_var_is_accepted(monkeypatch):
    """平台注入的凭据变量名是 EXTERNAL_SERVICE_TOKEN（实测自官方 demo 仓库），
    不是 XC_TOKEN。早先只读 XC_TOKEN 会让容器一启动就 KeyError 崩溃——健康检查
    端口都来不及监听，平台重启三次后直接标记失败。"""
    _base_env(monkeypatch)
    monkeypatch.setenv("EXTERNAL_SERVICE_TOKEN", "platform-token")

    s = Settings.from_env()

    assert s.xc_token == "platform-token"
    assert s.base_url == "https://modelhub.org.cn"  # 平台不注入 base_url，须有默认值


def test_xc_token_still_works_for_local_runs(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("XC_TOKEN", "local-token")
    assert Settings.from_env().xc_token == "local-token"


def test_explicit_token_overrides_the_platform_injection(monkeypatch):
    """平台注入的 EXTERNAL_SERVICE_TOKEN 被开放平台 API 以 401 拒绝（线上实测），
    运维只能自配一个有效 xcToken。显式配置必须压过平台默认注入，否则那个无效令牌
    会一直盖掉运维配的这个，而症状只是一句 401。"""
    _base_env(monkeypatch)
    monkeypatch.setenv("EXTERNAL_SERVICE_TOKEN", "platform-token-that-401s")
    monkeypatch.setenv("XC_TOKEN", "operator-supplied-token")
    assert Settings.from_env().xc_token == "operator-supplied-token"


def test_missing_credential_raises_config_error(monkeypatch):
    """缺凭据必须是 ConfigError（可被 main 转成"存活但不工作"），不是裸 KeyError。"""
    _base_env(monkeypatch)
    with pytest.raises(ConfigError, match="missing platform credential"):
        Settings.from_env()


def test_missing_strategy_id_raises_config_error(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.delenv("STRATEGY_ID", raising=False)
    monkeypatch.setenv("EXTERNAL_SERVICE_TOKEN", "t")
    with pytest.raises(ConfigError, match="STRATEGY_ID"):
        Settings.from_env()


def test_dry_run_flag(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("EXTERNAL_SERVICE_TOKEN", "t")
    assert Settings.from_env().dry_run is False
    for truthy in ("1", "true", "TRUE", "yes"):
        monkeypatch.setenv("DRY_RUN", truthy)
        assert Settings.from_env().dry_run is True
    monkeypatch.setenv("DRY_RUN", "false")
    assert Settings.from_env().dry_run is False


def test_unrelated_token_variables_are_reported_but_never_used(monkeypatch):
    """安全约束：环境里任何含 token 字样的变量都可能是别的服务的凭据（CI 令牌之类）。
    把它当作 Xc-Token 发给 modelhub.org.cn 就是把无关凭据泄露给第三方。
    扫描结果只用于诊断，绝不自动使用。"""
    _base_env(monkeypatch)
    monkeypatch.setenv("SOME_OTHER_SERVICE_TOKEN", "not-ours")

    assert "SOME_OTHER_SERVICE_TOKEN" in Settings.token_env_candidates_snapshot()
    with pytest.raises(ConfigError) as exc:
        Settings.from_env()
    # 报出来让人看见，但没有拿去用
    assert "SOME_OTHER_SERVICE_TOKEN" in str(exc.value)


def test_platform_header_style_name_is_accepted(monkeypatch):
    """假设验证用：若平台注入的名字就是 API 文档里的请求头名 XcToken，直接可用。"""
    _base_env(monkeypatch)
    monkeypatch.setenv("XcToken", "platform-token")
    s = Settings.from_env()
    assert s.xc_token == "platform-token" and s.token_env_name == "XcToken"
