import os
import pytest
from auto_adapter.settings import Settings


def test_tick_seconds_below_60_raises(monkeypatch):
    """TICK_SECONDS < 60 must raise ValueError to prevent rate-limit bypass."""
    monkeypatch.setenv("XC_TOKEN", "token")
    monkeypatch.setenv("STRATEGY_ID", "strat")
    monkeypatch.setenv("MODELHUB_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("TICK_SECONDS", "30")

    with pytest.raises(ValueError, match="TICK_SECONDS must be >= 60"):
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
