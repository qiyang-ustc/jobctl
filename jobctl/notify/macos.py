"""macOS Notification Center support for jobctl.

This module posts native macOS banners via ``osascript`` (AppleScript's
``display notification``).  It is a silent no-op on any non-macOS platform or
when ``osascript`` is unavailable, so callers can use it unconditionally.

Public symbols
--------------
- ``is_macos_available()`` — feature-detection guard.
- ``notify_macos(message, *, title, subtitle, sound, timeout)`` — post one
  banner; INJECTION-SAFE (all user text passed as argv, never interpolated).
- ``summarize_terminal_events(events)`` — PURE function turning a list of
  terminal run events into a single banner payload.
- ``MacNotifyCoalescer`` — async time-window batcher so a burst of completions
  collapses into a single "series" notification.

Design notes
------------
The AppleScript handler is built as ``on run argv ... end run`` and every piece
of user-controlled text (message, title, subtitle, sound name) is passed as a
separate ``argv`` item.  This means a message containing quotes, backslashes,
or AppleScript syntax cannot break out of a string literal — there are no
string literals holding user text at all.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)

# Unicode marks used in summaries (kept as module constants for clarity).
_CHECK = "✅"  # ✅
_WARN = "⚠️"  # ⚠️
_CROSS = "❌"  # ❌

# Terminal states / matches that count as success vs failure.
_SUCCESS_MATCHES = (None, "usable", "weak_signal")
_FAILURE_STATES = ("failed", "cancelled", "timeout", "stuck")
_FAILURE_MATCHES = ("bad_signal", "failed")


# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------

def is_macos_available() -> bool:
    """Return True iff we are on macOS and ``osascript`` is on PATH."""
    return sys.platform == "darwin" and shutil.which("osascript") is not None


# ---------------------------------------------------------------------------
# notify_macos
# ---------------------------------------------------------------------------

def notify_macos(
    message: str,
    *,
    title: str = "jobctl",
    subtitle: str | None = None,
    sound: str | None = None,
    timeout: float = 7.0,
) -> bool:
    """Post a macOS Notification Center banner.

    Returns True iff the banner was posted.  Off macOS, with no ``osascript``,
    or on any error this is a silent no-op returning False.

    All user-supplied text (``message``, ``title``, ``subtitle``, ``sound``) is
    passed as separate ``argv`` items to an ``on run argv ... end run``
    AppleScript handler — never string-interpolated into the script — so the
    text cannot inject AppleScript.
    """
    if not is_macos_available():
        return False

    osascript_path = shutil.which("osascript")
    if osascript_path is None:  # pragma: no cover - covered by is_macos_available
        return False

    parts = ["display notification (item 1 of argv) with title (item 2 of argv)"]
    args = [message, title]
    idx = 3
    if subtitle is not None:
        parts.append(f"subtitle (item {idx} of argv)")
        args.append(subtitle)
        idx += 1
    if sound:
        parts.append(f"sound name (item {idx} of argv)")
        args.append(sound)
        idx += 1

    script = "on run argv" + chr(10) + "  " + " ".join(parts) + chr(10) + "end run"
    cmd = [osascript_path, "-e", script, *args]

    # When running as root (e.g. a daemon), notifications must be posted as the
    # console user; otherwise they silently go nowhere.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        try:
            console_uid = os.stat("/dev/console").st_uid
            cmd = ["launchctl", "asuser", str(console_uid), *cmd]
        except OSError:
            pass

    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("notify_macos failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# summarize_terminal_events (pure)
# ---------------------------------------------------------------------------

def _classify_event(event: dict) -> str:
    """Classify a single terminal event as ``ok``, ``warn`` or ``bad``."""
    state = event.get("state")
    match = event.get("match")
    if state == "completed" and match in _SUCCESS_MATCHES:
        return "ok"
    if state == "completed":
        return "warn"
    if state in _FAILURE_STATES or match in _FAILURE_MATCHES:
        return "bad"
    return "warn"


def summarize_terminal_events(events: list[dict]) -> dict:
    """Reduce a list of terminal run events to a single banner payload.

    PURE: no side effects, deterministic.

    Each event is ``{"title": str, "state": str, "match": str | None}``.
    Returns ``{"title": str, "subtitle": str | None, "message": str,
    "sound": str | None}``.  ``sound`` is always None here — the caller supplies
    it.
    """
    if len(events) == 1:
        e = events[0]
        cls = _classify_event(e)
        mark = {"ok": _CHECK, "warn": _WARN, "bad": _CROSS}[cls]
        return {
            "title": "jobctl",
            "subtitle": None,
            "message": f"{mark} {e['state']}: {e['title']}",
            "sound": None,
        }

    classes = [_classify_event(e) for e in events]
    ok = sum(1 for cls in classes if cls == "ok")
    warn = sum(1 for cls in classes if cls == "warn")
    bad = sum(1 for cls in classes if cls == "bad")
    titles = [e["title"] for e in events[:3]]
    subtitle = ", ".join(titles)
    if len(events) > 3:
        subtitle += " ..."
    return {
        "title": "jobctl",
        "subtitle": subtitle,
        "message": f"{len(events)} jobs finished — {_CHECK}{ok} {_WARN}{warn} {_CROSS}{bad}",
        "sound": None,
    }


# ---------------------------------------------------------------------------
# MacNotifyCoalescer
# ---------------------------------------------------------------------------

class MacNotifyCoalescer:
    """Batch a burst of terminal events into one "series" notification.

    ``add()`` records an event and (re)arms a timer.  After ``window`` seconds
    of quiescence the pending events are summarized and posted as a single
    banner.  ``_flush`` is kept separate from ``_flush_later`` so it can be
    unit-tested without an event loop.
    """

    def __init__(
        self,
        window: float = 15.0,
        sound: str | None = None,
        notify_fn=notify_macos,
    ) -> None:
        self.window = window
        self.sound = sound
        self._notify = notify_fn
        self._pending: list[dict] = []
        self._task = None

    def add(self, event: dict) -> None:
        """Append a terminal event and (re)arm the flush timer.

        Requires a running asyncio event loop (the monitor's). With no running
        loop the event is still recorded but no timer is armed — callers without
        a loop should drive ``_flush`` directly (that's what the unit tests do).
        """
        self._pending.append(event)
        if self._task is None or self._task.done():
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return  # no loop — recorded; flush must be driven manually
            self._task = loop.create_task(self._flush_later())

    async def _flush_later(self) -> None:
        import asyncio

        await asyncio.sleep(self.window)
        self._flush()

    def _flush(self) -> None:
        """Summarize and post all pending events as one banner."""
        if not self._pending:
            return
        events = self._pending
        self._pending = []
        s = summarize_terminal_events(events)
        self._notify(
            s["message"],
            title=s["title"],
            subtitle=s.get("subtitle"),
            sound=self.sound or s.get("sound"),
        )
