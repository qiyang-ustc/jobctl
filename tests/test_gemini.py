"""Tests for the Gemini analyzer.

These tests must pass with NO network and NO GEMINI_API_KEY: every call to
``httpx.post`` is monkeypatched on ``jobctl.analysis.gemini.httpx.post``.
"""
from __future__ import annotations

import httpx
import pytest

from jobctl.analysis.gemini import GeminiAnalyzer


class _FakeResponse:
    """Minimal stand-in for an httpx.Response."""

    def __init__(self, text: str) -> None:
        self._text = text

    def raise_for_status(self) -> None:  # no-op
        return None

    def json(self) -> dict:
        return {
            "candidates": [
                {"content": {"parts": [{"text": self._text}]}}
            ]
        }


def _patch_post_returning(monkeypatch, text: str) -> None:
    def fake_post(*args, **kwargs):
        return _FakeResponse(text)

    monkeypatch.setattr("jobctl.analysis.gemini.httpx.post", fake_post)


def _patch_post_raising(monkeypatch, exc: Exception) -> None:
    def fake_post(*args, **kwargs):
        raise exc

    monkeypatch.setattr("jobctl.analysis.gemini.httpx.post", fake_post)


def test_analyze_run_parses_json(monkeypatch):
    _patch_post_returning(
        monkeypatch, '{"interpretation":"ok","recommended_next_action":"go"}'
    )
    a = GeminiAnalyzer(api_key="x")
    result = a.analyze_run({"state": "succeeded"})
    assert isinstance(result, dict)
    assert result["interpretation"] == "ok"
    assert result["recommended_next_action"] == "go"


def test_analyze_run_fallback_on_error(monkeypatch):
    _patch_post_raising(monkeypatch, httpx.ConnectError("boom"))
    a = GeminiAnalyzer(api_key="x")
    result = a.analyze_run({"state": "succeeded"})
    assert isinstance(result, dict)
    assert "interpretation" in result
    assert "recommended_next_action" in result


def test_summarize_log_returns_text(monkeypatch):
    _patch_post_returning(monkeypatch, "a concise summary")
    a = GeminiAnalyzer(api_key="x")
    assert a.summarize_log("line1\nline2") == "a concise summary"


def test_summarize_log_fallback_on_error(monkeypatch):
    _patch_post_raising(monkeypatch, httpx.ConnectError("boom"))
    a = GeminiAnalyzer(api_key="x")
    out = a.summarize_log("line1\nline2")
    assert isinstance(out, str)
    assert out  # non-empty fallback


def test_propose_criteria_fallback_on_error(monkeypatch):
    _patch_post_raising(monkeypatch, httpx.ConnectError("boom"))
    a = GeminiAnalyzer(api_key="x")
    crit = a.propose_criteria({"note": "be strict"}, [], {"name": "job"})
    assert isinstance(crit, list)
    assert crit  # non-empty
    for item in crit:
        for key in ("id", "text", "kind", "check", "status", "strength", "evidence_run_ids"):
            assert key in item


def test_construct_and_methods_callable():
    a = GeminiAnalyzer(api_key="x")
    for name in (
        "analyze_run",
        "summarize_log",
        "explain_bad_signal",
        "suggest_next_action",
        "propose_criteria",
        "summarize_failures",
    ):
        assert callable(getattr(a, name))
    assert "gemini-2.5-flash-lite" in a._endpoint
    assert ":generateContent" in a._endpoint
