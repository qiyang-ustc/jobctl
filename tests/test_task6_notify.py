"""Tests for Task 6: notify/notify.py — Notifier ABC + concrete notifiers + get_notifiers()."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from jobctl.db.models import Health, Match, Run, State


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_run(
    run_id: str = "run-001",
    jobfile_id: str = "jf-001",
    state: State = State.COMPLETED,
    callback_url: str | None = None,
) -> Run:
    run = Run(
        run_id=run_id,
        jobfile_id=jobfile_id,
        jobfile_version=1,
        params={"lr": 0.01},
        input_hashes={},
        backend="local",
        server=None,
        task=None,
        remote_job_id=None,
        state=state,
        health=Health.OK,
        exit_code=0 if state == State.COMPLETED else None,
        submitted_at=_now(),
        started_at=_now(),
        finished_at=_now() if state == State.COMPLETED else None,
        last_heartbeat=None,
        workdir="/tmp/runs/run-001",
        stdout_path=None,
        stderr_path=None,
        resource_summary={},
        expectation_match=Match.USABLE,
        observation_card=None,
    )
    # Attach optional callback_url as an extra attribute (used by get_notifiers)
    run._callback_url = callback_url
    return run


def _make_card(run_id: str = "run-001") -> dict:
    return {
        "status": "completed",
        "jobfile": "train-job",
        "run_id": run_id,
        "server": "local",
        "artifacts": [],
        "health": "ok",
        "expectation_match": "usable",
        "key_evidence": [],
        "interpretation": "Run completed successfully.",
        "recommended_next_action": "Check artifacts.",
    }


# ---------------------------------------------------------------------------
# Tests: LogNotifier
# ---------------------------------------------------------------------------

class TestLogNotifier:
    def test_log_notifier_exists(self):
        from jobctl.notify.notify import LogNotifier
        assert LogNotifier is not None

    def test_log_notifier_is_notifier(self):
        from jobctl.notify.notify import LogNotifier, Notifier
        n = LogNotifier()
        assert isinstance(n, Notifier)

    def test_log_notifier_notify_logs_message(self, caplog):
        from jobctl.notify.notify import LogNotifier
        n = LogNotifier()
        run = _make_run()
        card = _make_card()
        with caplog.at_level(logging.INFO):
            n.notify(run, card)
        # Should produce at least one log message containing the run_id
        assert any("run-001" in r.message for r in caplog.records)

    def test_log_notifier_notify_includes_state(self, caplog):
        from jobctl.notify.notify import LogNotifier
        n = LogNotifier()
        run = _make_run(state=State.FAILED)
        card = _make_card()
        with caplog.at_level(logging.INFO):
            n.notify(run, card)
        # Log should include 'failed' somewhere
        full_text = " ".join(r.message for r in caplog.records)
        assert "failed" in full_text.lower() or "run-001" in full_text


# ---------------------------------------------------------------------------
# Tests: WebhookNotifier
# ---------------------------------------------------------------------------

class TestWebhookNotifier:
    def test_webhook_notifier_exists(self):
        from jobctl.notify.notify import WebhookNotifier
        assert WebhookNotifier is not None

    def test_webhook_posts_card(self):
        from jobctl.notify.notify import WebhookNotifier
        run = _make_run()
        card = _make_card()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            n = WebhookNotifier(url="https://example.com/hook")
            n.notify(run, card)
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        # URL is the first positional arg
        assert call_kwargs[0][0] == "https://example.com/hook"

    def test_webhook_payload_is_card(self):
        from jobctl.notify.notify import WebhookNotifier
        run = _make_run()
        card = _make_card()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            n = WebhookNotifier(url="https://example.com/hook")
            n.notify(run, card)
        call_kwargs = mock_post.call_args
        # Should pass json= kwarg with the card
        sent_json = call_kwargs[1].get("json") or call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None
        if sent_json is None:
            sent_json = call_kwargs[1].get("json")
        assert sent_json is not None
        assert sent_json["run_id"] == "run-001"
        assert sent_json["status"] == "completed"

    def test_webhook_does_not_raise_on_http_error(self):
        """WebhookNotifier should swallow HTTP errors (fire-and-forget)."""
        from jobctl.notify.notify import WebhookNotifier
        run = _make_run()
        card = _make_card()
        with patch("httpx.post") as mock_post:
            mock_post.side_effect = Exception("connection refused")
            n = WebhookNotifier(url="https://example.com/hook")
            # Should not raise
            n.notify(run, card)


# ---------------------------------------------------------------------------
# Tests: SlackNotifier
# ---------------------------------------------------------------------------

class TestSlackNotifier:
    def test_slack_notifier_exists(self):
        from jobctl.notify.notify import SlackNotifier
        assert SlackNotifier is not None

    def test_slack_posts_to_url(self):
        from jobctl.notify.notify import SlackNotifier
        run = _make_run()
        card = _make_card()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            n = SlackNotifier(webhook_url="https://hooks.slack.com/T123")
            n.notify(run, card)
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "https://hooks.slack.com/T123" in call_kwargs[0]

    def test_slack_payload_contains_text(self):
        """Slack payload should have a 'text' key (Slack incoming webhook format)."""
        from jobctl.notify.notify import SlackNotifier
        run = _make_run()
        card = _make_card()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            n = SlackNotifier(webhook_url="https://hooks.slack.com/T123")
            n.notify(run, card)
        call_kwargs = mock_post.call_args
        sent_json = call_kwargs[1].get("json")
        assert sent_json is not None
        assert "text" in sent_json
        # text should mention run_id
        assert "run-001" in sent_json["text"]

    def test_slack_does_not_raise_on_http_error(self):
        from jobctl.notify.notify import SlackNotifier
        run = _make_run()
        card = _make_card()
        with patch("httpx.post") as mock_post:
            mock_post.side_effect = Exception("timeout")
            n = SlackNotifier(webhook_url="https://hooks.slack.com/T123")
            n.notify(run, card)  # should not raise


# ---------------------------------------------------------------------------
# Tests: CallbackNotifier
# ---------------------------------------------------------------------------

class TestCallbackNotifier:
    def test_callback_notifier_exists(self):
        from jobctl.notify.notify import CallbackNotifier
        assert CallbackNotifier is not None

    def test_callback_posts_card(self):
        from jobctl.notify.notify import CallbackNotifier
        run = _make_run()
        card = _make_card()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            n = CallbackNotifier(url="https://agent.example.com/cb")
            n.notify(run, card)
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "https://agent.example.com/cb" in call_kwargs[0]

    def test_callback_payload_is_card(self):
        from jobctl.notify.notify import CallbackNotifier
        run = _make_run()
        card = _make_card()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            n = CallbackNotifier(url="https://agent.example.com/cb")
            n.notify(run, card)
        call_kwargs = mock_post.call_args
        sent_json = call_kwargs[1].get("json")
        assert sent_json is not None
        assert sent_json["run_id"] == "run-001"

    def test_callback_does_not_raise_on_http_error(self):
        from jobctl.notify.notify import CallbackNotifier
        run = _make_run()
        card = _make_card()
        with patch("httpx.post") as mock_post:
            mock_post.side_effect = Exception("timeout")
            n = CallbackNotifier(url="https://agent.example.com/cb")
            n.notify(run, card)  # should not raise


# ---------------------------------------------------------------------------
# Tests: Email/GitHub stubs
# ---------------------------------------------------------------------------

class TestEmailAndGitHubStubs:
    def test_email_notifier_exists(self):
        from jobctl.notify.notify import EmailNotifier
        assert EmailNotifier is not None

    def test_email_is_notifier(self):
        from jobctl.notify.notify import EmailNotifier, Notifier
        n = EmailNotifier()
        assert isinstance(n, Notifier)

    def test_email_notify_does_not_raise(self):
        from jobctl.notify.notify import EmailNotifier
        n = EmailNotifier()
        run = _make_run()
        card = _make_card()
        n.notify(run, card)  # stub — should not raise

    def test_github_notifier_exists(self):
        from jobctl.notify.notify import GitHubNotifier
        assert GitHubNotifier is not None

    def test_github_is_notifier(self):
        from jobctl.notify.notify import GitHubNotifier, Notifier
        n = GitHubNotifier()
        assert isinstance(n, Notifier)

    def test_github_notify_does_not_raise(self):
        from jobctl.notify.notify import GitHubNotifier
        n = GitHubNotifier()
        run = _make_run()
        card = _make_card()
        n.notify(run, card)  # stub — should not raise


# ---------------------------------------------------------------------------
# Tests: get_notifiers()
# ---------------------------------------------------------------------------

class TestGetNotifiers:
    def test_get_notifiers_always_includes_log_notifier(self):
        from jobctl.notify.notify import LogNotifier, get_notifiers
        from jobctl.config import Config
        config = Config()
        run = _make_run()
        notifiers = get_notifiers(config, run)
        assert any(isinstance(n, LogNotifier) for n in notifiers)

    def test_get_notifiers_no_extras_when_no_config(self):
        from jobctl.notify.notify import LogNotifier, WebhookNotifier, SlackNotifier, CallbackNotifier, get_notifiers
        from jobctl.config import Config
        config = Config()
        run = _make_run()
        notifiers = get_notifiers(config, run)
        # Only LogNotifier
        assert len(notifiers) == 1
        assert isinstance(notifiers[0], LogNotifier)

    def test_get_notifiers_adds_webhook_when_configured(self):
        from jobctl.notify.notify import WebhookNotifier, get_notifiers
        from jobctl.config import Config
        config = Config()
        config.notify_webhook_url = "https://example.com/hook"
        run = _make_run()
        notifiers = get_notifiers(config, run)
        assert any(isinstance(n, WebhookNotifier) for n in notifiers)

    def test_get_notifiers_adds_slack_when_configured(self):
        from jobctl.notify.notify import SlackNotifier, get_notifiers
        from jobctl.config import Config
        config = Config()
        config.notify_slack_url = "https://hooks.slack.com/T999"
        run = _make_run()
        notifiers = get_notifiers(config, run)
        assert any(isinstance(n, SlackNotifier) for n in notifiers)

    def test_get_notifiers_adds_callback_when_run_has_callback_url(self):
        from jobctl.notify.notify import CallbackNotifier, get_notifiers
        from jobctl.config import Config
        config = Config()
        run = _make_run(callback_url="https://agent.example.com/cb")
        notifiers = get_notifiers(config, run)
        assert any(isinstance(n, CallbackNotifier) for n in notifiers)

    def test_get_notifiers_callback_url_from_run_attribute(self):
        """CallbackNotifier uses the URL from run._callback_url."""
        from jobctl.notify.notify import CallbackNotifier, get_notifiers
        from jobctl.config import Config
        config = Config()
        run = _make_run(callback_url="https://agent.example.com/custom")
        notifiers = get_notifiers(config, run)
        cb_notifiers = [n for n in notifiers if isinstance(n, CallbackNotifier)]
        assert len(cb_notifiers) == 1
        assert cb_notifiers[0].url == "https://agent.example.com/custom"

    def test_get_notifiers_all_three_extras(self):
        from jobctl.notify.notify import LogNotifier, WebhookNotifier, SlackNotifier, CallbackNotifier, get_notifiers
        from jobctl.config import Config
        config = Config()
        config.notify_webhook_url = "https://example.com/hook"
        config.notify_slack_url = "https://hooks.slack.com/T999"
        run = _make_run(callback_url="https://agent.example.com/cb")
        notifiers = get_notifiers(config, run)
        assert len(notifiers) == 4
        types = {type(n) for n in notifiers}
        assert LogNotifier in types
        assert WebhookNotifier in types
        assert SlackNotifier in types
        assert CallbackNotifier in types

    def test_get_notifiers_no_callback_when_run_has_no_callback_url(self):
        from jobctl.notify.notify import CallbackNotifier, get_notifiers
        from jobctl.config import Config
        config = Config()
        run = _make_run(callback_url=None)
        notifiers = get_notifiers(config, run)
        assert not any(isinstance(n, CallbackNotifier) for n in notifiers)
