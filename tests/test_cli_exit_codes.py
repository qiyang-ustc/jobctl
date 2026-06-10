"""Exit-code discipline for the waiting paths (`run --wait`, `await`).

Exit code reflects whether the job *ran* (operational), NOT whether the
result is scientifically good — that lives in the observation card's
expectation_match. So `completed` is 0 regardless of usable/weak/bad, and
only operational failures (failed/cancelled/stuck/timeout) are non-zero.
Background submits are unaffected (no terminal state yet).
"""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from jobctl.cli import main as cli_module
from jobctl.cli.main import app, _exit_code_for, _WAIT_TIMEOUT_DEFAULT


@pytest.mark.parametrize("state,expected", [
    ("completed", 0),
    ("failed", 2),
    ("cancelled", 3),
    ("stuck", 4),
    ("timeout", 124),
    ("weird", 1),       # unknown terminal -> operational error
])
def test_exit_code_for_maps_state(state, expected):
    assert _exit_code_for(state) == expected


def test_wait_timeout_default_is_12_hours():
    assert _WAIT_TIMEOUT_DEFAULT == 12 * 60 * 60


class _FakeClient:
    """Minimal client whose await_run yields a chosen terminal state."""
    def __init__(self, state="completed", raise_timeout=False):
        self._state = state
        self._raise_timeout = raise_timeout
        self.timeout_passed = None
        self.poll_interval_passed = None

    def submit(self, **kwargs):
        return {"run_id": "run-fake", "state": "submitted"}

    def await_run(self, run_id, poll_interval=1.0, timeout=600.0):
        self.timeout_passed = timeout
        self.poll_interval_passed = poll_interval
        if self._raise_timeout:
            raise TimeoutError("nope")
        return {"run_id": run_id, "state": self._state, "observation_card": None}


@pytest.fixture
def fake(monkeypatch):
    def _install(**kw):
        client = _FakeClient(**kw)
        cli_module._OVERRIDE_CLIENT = client
        return client
    yield _install
    cli_module._OVERRIDE_CLIENT = None


@pytest.mark.parametrize("state,code", [
    ("completed", 0), ("failed", 2), ("cancelled", 3), ("stuck", 4), ("timeout", 124),
])
def test_run_wait_exit_code_matches_state(fake, state, code):
    fake(state=state)
    result = CliRunner().invoke(app, ["run", "--wait", "--json", "somejob"])
    assert result.exit_code == code, result.output


@pytest.mark.parametrize("state,code", [
    ("completed", 0), ("failed", 2), ("cancelled", 3), ("stuck", 4),
])
def test_await_exit_code_matches_state(fake, state, code):
    fake(state=state)
    result = CliRunner().invoke(app, ["await", "run-fake", "--json"])
    assert result.exit_code == code, result.output


def test_run_wait_client_timeout_exits_124(fake):
    fake(raise_timeout=True)
    result = CliRunner().invoke(app, ["run", "--wait", "somejob"])
    assert result.exit_code == 124


def test_run_wait_passes_12h_timeout_to_client(fake):
    client = fake(state="completed")
    CliRunner().invoke(app, ["run", "--wait", "somejob"])
    assert client.timeout_passed == 12 * 60 * 60
    assert client.poll_interval_passed == 1.0


def test_run_wait_respects_timeout_override(fake):
    client = fake(state="completed")
    CliRunner().invoke(app, ["run", "--wait", "--timeout", "30", "somejob"])
    assert client.timeout_passed == 30.0


def test_await_uses_default_poll_interval(fake):
    client = fake(state="completed")
    CliRunner().invoke(app, ["await", "run-fake"])
    assert client.poll_interval_passed == 1.0
