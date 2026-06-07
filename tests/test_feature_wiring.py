"""Wiring tests: Gemini analyzer selection, config flags, monitor coalescer."""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# get_analyzer selection
# ---------------------------------------------------------------------------

class TestAnalyzerSelection:
    def test_gemini_when_only_gemini_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        from jobctl.analysis.base import get_analyzer
        from jobctl.analysis.gemini import GeminiAnalyzer
        assert isinstance(get_analyzer({}), GeminiAnalyzer)

    def test_deepseek_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "d-key")
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        from jobctl.analysis.base import get_analyzer
        from jobctl.analysis.deepseek import DeepSeekAnalyzer
        assert isinstance(get_analyzer({}), DeepSeekAnalyzer)

    def test_offline_when_no_keys(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from jobctl.analysis.base import get_analyzer
        from jobctl.analysis.offline import OfflineAnalyzer
        assert isinstance(get_analyzer({}), OfflineAnalyzer)


# ---------------------------------------------------------------------------
# config flags
# ---------------------------------------------------------------------------

class TestConfig:
    def test_gemini_key_and_notify_defaults(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        from jobctl.config import load_config
        cfg = load_config(
            cluster_yaml_path=str(tmp_path / "none.yaml"),
            jobctl_config_path=str(tmp_path / "none.toml"),
        )
        assert cfg.gemini_api_key == "g-key"
        assert cfg.notify_macos_enabled is True  # on by default
        assert cfg.notify_sound == ""

    def test_notify_settings_from_toml(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text(
            "[jobctl]\n"
            "notify_macos_enabled = false\n"
            'notify_sound = "Glass"\n'
            "notify_window_seconds = 30\n"
        )
        from jobctl.config import load_config
        cfg = load_config(
            cluster_yaml_path=str(tmp_path / "none.yaml"),
            jobctl_config_path=str(toml),
        )
        assert cfg.notify_macos_enabled is False
        assert cfg.notify_sound == "Glass"
        assert cfg.notify_window_seconds == 30.0


# ---------------------------------------------------------------------------
# Monitor coalescer wiring
# ---------------------------------------------------------------------------

def _make_monitor(config, tmp_path):
    from jobctl.monitor.monitor import Monitor
    from jobctl.db.store import Store
    from jobctl.analysis.offline import OfflineAnalyzer
    store = Store(str(tmp_path / "m.db"))
    store.init_schema()
    return Monitor(store=store, config=config, analyzer=OfflineAnalyzer(),
                   notifiers_factory=lambda run: [])


class TestMonitorCoalescer:
    def test_coalescer_created_when_enabled_on_mac(self, monkeypatch, tmp_path):
        import jobctl.notify.macos as macos
        monkeypatch.setattr(macos, "is_macos_available", lambda: True)
        m = _make_monitor({"notify_macos_enabled": True}, tmp_path)
        assert m._mac_coalescer is not None

    def test_no_coalescer_when_disabled(self, monkeypatch, tmp_path):
        import jobctl.notify.macos as macos
        monkeypatch.setattr(macos, "is_macos_available", lambda: True)
        m = _make_monitor({"notify_macos_enabled": False}, tmp_path)
        assert m._mac_coalescer is None

    def test_no_coalescer_off_mac(self, monkeypatch, tmp_path):
        import jobctl.notify.macos as macos
        monkeypatch.setattr(macos, "is_macos_available", lambda: False)
        m = _make_monitor({"notify_macos_enabled": True}, tmp_path)
        assert m._mac_coalescer is None
