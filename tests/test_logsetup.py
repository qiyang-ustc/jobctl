"""Logging is written to ~/.jobctl/<component>.log (the 'check the logs' fix)."""
from __future__ import annotations

import logging

from jobctl import logsetup


def test_configure_logging_writes_to_component_file(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBCTL_HOME", str(tmp_path))
    logsetup._CONFIGURED.clear()

    path = logsetup.configure_logging("cli")
    assert path == tmp_path / "cli.log"

    logging.getLogger("jobctl.test").warning("hello-from-test")
    for h in logging.getLogger().handlers:
        h.flush()

    assert path.exists()
    assert "hello-from-test" in path.read_text()


def test_configure_logging_is_idempotent_per_component(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBCTL_HOME", str(tmp_path))
    logsetup._CONFIGURED.clear()
    before = len(logging.getLogger().handlers)
    logsetup.configure_logging("daemon")
    logsetup.configure_logging("daemon")
    after = len(logging.getLogger().handlers)
    assert after == before + 1  # only one handler added despite two calls
