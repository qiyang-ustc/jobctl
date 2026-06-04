"""Notifier ABC + concrete implementations + get_notifiers() factory.

Notifiers are fire-and-forget: they log/POST the observation card when a run
reaches a terminal state.  Network errors are swallowed so a webhook failure
never blocks the monitor loop.

Included:
- LogNotifier   — always active; writes to Python logging
- WebhookNotifier — POST card as JSON to an arbitrary URL
- SlackNotifier   — POST Slack incoming-webhook message containing card summary
- CallbackNotifier — POST card to a per-run callback URL (e.g. the agent that
                     triggered the run)
- EmailNotifier   — interface stub (not implemented)
- GitHubNotifier  — interface stub (not implemented)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from jobctl.db.models import Run
    from jobctl.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class Notifier(ABC):
    """Base class for all notifiers."""

    @abstractmethod
    def notify(self, run: "Run", card: dict) -> None:
        """Send the observation card for *run*."""


# ---------------------------------------------------------------------------
# LogNotifier — always included
# ---------------------------------------------------------------------------

class LogNotifier(Notifier):
    """Writes a structured summary of the observation card to the Python logger."""

    def notify(self, run: "Run", card: dict) -> None:
        status = card.get("status", run.state.value if hasattr(run.state, "value") else str(run.state))
        match = card.get("expectation_match", "")
        interpretation = card.get("interpretation", "")
        logger.info(
            "run=%s  status=%s  match=%s  | %s",
            run.run_id,
            status,
            match,
            interpretation,
        )


# ---------------------------------------------------------------------------
# WebhookNotifier
# ---------------------------------------------------------------------------

class WebhookNotifier(Notifier):
    """POST the observation card as JSON to an arbitrary webhook URL."""

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        self.url = url
        self.timeout = timeout

    def notify(self, run: "Run", card: dict) -> None:
        try:
            httpx.post(self.url, json=card, timeout=self.timeout)
        except Exception as exc:
            logger.warning("WebhookNotifier failed for run=%s: %s", run.run_id, exc)


# ---------------------------------------------------------------------------
# SlackNotifier
# ---------------------------------------------------------------------------

class SlackNotifier(Notifier):
    """POST a Slack incoming-webhook message summarising the observation card."""

    def __init__(self, webhook_url: str, timeout: float = 10.0) -> None:
        self.webhook_url = webhook_url
        self.timeout = timeout

    def _build_text(self, run: "Run", card: dict) -> str:
        status = card.get("status", str(run.state))
        match = card.get("expectation_match", "")
        interp = card.get("interpretation", "")
        jobfile = card.get("jobfile", "")
        server = card.get("server", "")
        parts = [f"*run_id:* `{run.run_id}`", f"*status:* {status}"]
        if jobfile:
            parts.append(f"*jobfile:* {jobfile}")
        if server:
            parts.append(f"*server:* {server}")
        if match:
            parts.append(f"*match:* {match}")
        if interp:
            parts.append(f"_{interp}_")
        return "  |  ".join(parts)

    def notify(self, run: "Run", card: dict) -> None:
        payload = {"text": self._build_text(run, card)}
        try:
            httpx.post(self.webhook_url, json=payload, timeout=self.timeout)
        except Exception as exc:
            logger.warning("SlackNotifier failed for run=%s: %s", run.run_id, exc)


# ---------------------------------------------------------------------------
# CallbackNotifier
# ---------------------------------------------------------------------------

class CallbackNotifier(Notifier):
    """POST the observation card to a per-run callback URL.

    Intended for agents that submit a run with a callback URL so they can
    receive a structured result without polling.
    """

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        self.url = url
        self.timeout = timeout

    def notify(self, run: "Run", card: dict) -> None:
        try:
            httpx.post(self.url, json=card, timeout=self.timeout)
        except Exception as exc:
            logger.warning("CallbackNotifier failed for run=%s: %s", run.run_id, exc)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class EmailNotifier(Notifier):
    """Email notifier — interface stub; not implemented."""

    def notify(self, run: "Run", card: dict) -> None:  # noqa: D401
        logger.debug("EmailNotifier: stub — not implemented (run=%s)", run.run_id)


class GitHubNotifier(Notifier):
    """GitHub notifier — interface stub; not implemented."""

    def notify(self, run: "Run", card: dict) -> None:  # noqa: D401
        logger.debug("GitHubNotifier: stub — not implemented (run=%s)", run.run_id)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_notifiers(config: "Config", run: "Run") -> list[Notifier]:
    """Return the list of active notifiers for *config* and *run*.

    LogNotifier is always included.  Additional notifiers are added when the
    corresponding configuration keys are present or when the run carries a
    callback URL.

    The ``config`` object may have the following optional attributes:
    - ``notify_webhook_url`` (str): URL for WebhookNotifier
    - ``notify_slack_url`` (str):   Slack incoming-webhook URL for SlackNotifier

    The ``run`` object may have an optional ``_callback_url`` attribute (str)
    set by the API layer when the client provides a callback URL at submission
    time.
    """
    notifiers: list[Notifier] = [LogNotifier()]

    webhook_url: str | None = getattr(config, "notify_webhook_url", None)
    if webhook_url:
        notifiers.append(WebhookNotifier(url=webhook_url))

    slack_url: str | None = getattr(config, "notify_slack_url", None)
    if slack_url:
        notifiers.append(SlackNotifier(webhook_url=slack_url))

    callback_url: str | None = getattr(run, "_callback_url", None)
    if callback_url:
        notifiers.append(CallbackNotifier(url=callback_url))

    return notifiers
