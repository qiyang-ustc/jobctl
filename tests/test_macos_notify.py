"""Tests for jobctl.notify.macos — macOS Notification Center support.

These tests must pass on ANY platform: every macOS-specific behaviour is
exercised by monkeypatching ``sys.platform``, ``shutil.which`` and
``subprocess.run`` inside the module under test.
"""
from __future__ import annotations

import subprocess

import jobctl.notify.macos as macos
from jobctl.notify.macos import (
    MacNotifyCoalescer,
    is_macos_available,
    notify_macos,
    summarize_terminal_events,
)

# Unicode marks the implementation uses in summaries.
_CHECK = "✅"
_WARN = "⚠️"
_CROSS = "❌"


# ---------------------------------------------------------------------------
# 1. off-darwin: silent no-op, no subprocess call
# ---------------------------------------------------------------------------

def test_off_darwin_is_unavailable_and_no_subprocess(monkeypatch):
    # Force a non-macOS platform.
    monkeypatch.setattr(macos.sys, "platform", "linux")

    # If subprocess.run is touched, fail loudly.
    def _boom(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("subprocess.run must not be called off-darwin")

    monkeypatch.setattr(macos.subprocess, "run", _boom)

    assert is_macos_available() is False
    assert notify_macos("hello", title="T") is False


# ---------------------------------------------------------------------------
# 2. argv build on darwin: injection-safe, message is a separate argv element
# ---------------------------------------------------------------------------

def test_notify_macos_argv_build_injection_safe(monkeypatch):
    monkeypatch.setattr(macos.sys, "platform", "darwin")
    monkeypatch.setattr(macos.shutil, "which", lambda name: "/usr/bin/osascript")

    captured = {}

    class _Proc:
        returncode = 0

    def _fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(macos.subprocess, "run", _fake_run)

    # A message that would break naive string interpolation: contains a double
    # quote, a backslash, and ordinary text.
    nasty = 'he said "hi" \\ weird'
    result = notify_macos(nasty, title="T", subtitle="S", sound="Glass")

    assert result is True
    cmd = captured["cmd"]
    assert cmd[0] == "/usr/bin/osascript"
    assert cmd[1] == "-e"
    assert "on run argv" in cmd[2]
    # The exact nasty message must appear as a SEPARATE argv element (proving it
    # was NOT interpolated into the AppleScript source at cmd[2]).
    assert cmd[3] == nasty
    assert nasty not in cmd[2]
    # subtitle and sound must also be present as argv elements.
    assert "S" in cmd
    assert "Glass" in cmd


# ---------------------------------------------------------------------------
# 3. OSError from subprocess.run -> False, no exception escapes
# ---------------------------------------------------------------------------

def test_notify_macos_oserror_returns_false(monkeypatch):
    monkeypatch.setattr(macos.sys, "platform", "darwin")
    monkeypatch.setattr(macos.shutil, "which", lambda name: "/usr/bin/osascript")

    def _raise(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(macos.subprocess, "run", _raise)

    assert notify_macos("hello") is False


def test_notify_macos_timeout_returns_false(monkeypatch):
    monkeypatch.setattr(macos.sys, "platform", "darwin")
    monkeypatch.setattr(macos.shutil, "which", lambda name: "/usr/bin/osascript")

    def _raise(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="osascript", timeout=7.0)

    monkeypatch.setattr(macos.subprocess, "run", _raise)

    assert notify_macos("hello") is False


# ---------------------------------------------------------------------------
# 4. summarize_terminal_events (pure)
# ---------------------------------------------------------------------------

def test_summarize_single_success():
    events = [{"title": "train-A", "state": "completed", "match": "usable"}]
    s = summarize_terminal_events(events)
    assert s["title"] == "jobctl"
    assert s["subtitle"] is None
    assert s["message"].startswith(_CHECK)
    assert "train-A" in s["message"]
    assert s["sound"] is None


def test_summarize_single_success_none_match():
    events = [{"title": "train-A", "state": "completed", "match": None}]
    s = summarize_terminal_events(events)
    assert s["message"].startswith(_CHECK)


def test_summarize_single_failure():
    events = [{"title": "train-B", "state": "failed", "match": None}]
    s = summarize_terminal_events(events)
    assert s["message"].startswith(_CROSS)
    assert "train-B" in s["message"]


def test_summarize_single_completed_inconclusive_is_warning():
    events = [{"title": "train-C", "state": "completed", "match": "inconclusive"}]
    s = summarize_terminal_events(events)
    assert s["message"].startswith(_WARN)
    assert "train-C" in s["message"]
    assert not s["message"].startswith(_CROSS)


def test_summarize_four_events_two_ok_one_warn_one_bad():
    events = [
        {"title": "job1", "state": "completed", "match": "usable"},
        {"title": "job2", "state": "completed", "match": "weak_signal"},
        {"title": "job3", "state": "completed", "match": "inconclusive"},
        {"title": "job4", "state": "failed", "match": "bad_signal"},
    ]
    s = summarize_terminal_events(events)
    assert "4 jobs finished" in s["message"]
    assert f"{_CHECK}2" in s["message"]
    assert f"{_WARN}1" in s["message"]
    assert f"{_CROSS}1" in s["message"]
    assert s["subtitle"] is not None
    for title in ("job1", "job2", "job3"):
        assert title in s["subtitle"]
    assert "job4" not in s["subtitle"]


def test_summarize_more_than_three_truncates_subtitle():
    events = [
        {"title": "job1", "state": "completed", "match": "usable"},
        {"title": "job2", "state": "completed", "match": "usable"},
        {"title": "job3", "state": "completed", "match": "usable"},
        {"title": "job4", "state": "failed", "match": "failed"},
    ]
    s = summarize_terminal_events(events)
    assert "4 jobs finished" in s["message"]
    assert s["subtitle"].endswith(" ...")
    assert "job4" not in s["subtitle"]


# ---------------------------------------------------------------------------
# 5. MacNotifyCoalescer._flush
# ---------------------------------------------------------------------------

def test_coalescer_flush_calls_notify_once():
    calls = []

    def fake_notify(message, *, title="jobctl", subtitle=None, sound=None):
        calls.append({"message": message, "title": title, "subtitle": subtitle, "sound": sound})
        return True

    c = MacNotifyCoalescer(window=15.0, notify_fn=fake_notify)
    c.add({"title": "job1", "state": "completed", "match": "usable"})
    c.add({"title": "job2", "state": "completed", "match": "usable"})
    c.add({"title": "job3", "state": "failed", "match": "bad_signal"})

    c._flush()

    assert len(calls) == 1
    assert "3 jobs finished" in calls[0]["message"]
    assert f"{_CROSS}1" in calls[0]["message"]


def test_coalescer_flush_empty_is_noop():
    calls = []

    def fake_notify(message, **kwargs):
        calls.append(message)
        return True

    c = MacNotifyCoalescer(notify_fn=fake_notify)
    c._flush()
    assert calls == []


def test_coalescer_flush_uses_configured_sound():
    calls = []

    def fake_notify(message, *, title="jobctl", subtitle=None, sound=None):
        calls.append(sound)
        return True

    c = MacNotifyCoalescer(window=15.0, sound="Glass", notify_fn=fake_notify)
    c.add({"title": "job1", "state": "completed", "match": "usable"})
    c._flush()
    assert calls == ["Glass"]
