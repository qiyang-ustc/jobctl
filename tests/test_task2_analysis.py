"""Tests for Task 2: analysis/base.py, analysis/offline.py, analysis/deepseek.py."""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helper: minimal facts dict used across tests
# ---------------------------------------------------------------------------

def make_facts(**overrides):
    base = {
        "run_id": "run-001",
        "jobfile": "train-job v1",
        "state": "completed",
        "exit_code": 0,
        "health": "ok",
        "expectation_match": "usable",
        "artifacts": [{"name": "output.csv", "type": "csv", "preview": "col1,col2\n1,2"}],
        "key_evidence": ["exit_code=0", "output.csv present"],
        "server": "local",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# get_analyzer selector
# ---------------------------------------------------------------------------

class TestGetAnalyzer:
    def test_returns_offline_when_no_key(self, monkeypatch):
        """get_analyzer returns OfflineAnalyzer when DEEPSEEK_API_KEY is not set."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from jobctl.analysis.base import get_analyzer
        from jobctl.analysis.offline import OfflineAnalyzer

        analyzer = get_analyzer({})
        assert isinstance(analyzer, OfflineAnalyzer)

    def test_returns_deepseek_when_key_present(self, monkeypatch):
        """get_analyzer returns DeepSeekAnalyzer when DEEPSEEK_API_KEY is set."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
        from jobctl.analysis.base import get_analyzer
        from jobctl.analysis.deepseek import DeepSeekAnalyzer

        analyzer = get_analyzer({})
        assert isinstance(analyzer, DeepSeekAnalyzer)

    def test_analyzer_is_analyzer_instance(self, monkeypatch):
        """Both paths return an Analyzer subclass."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from jobctl.analysis.base import get_analyzer, Analyzer

        analyzer = get_analyzer({})
        assert isinstance(analyzer, Analyzer)


# ---------------------------------------------------------------------------
# OfflineAnalyzer
# ---------------------------------------------------------------------------

class TestOfflineAnalyzer:
    def _get(self):
        from jobctl.analysis.offline import OfflineAnalyzer
        return OfflineAnalyzer()

    def test_analyze_run_returns_required_keys(self):
        """analyze_run returns dict with interpretation + recommended_next_action."""
        az = self._get()
        result = az.analyze_run(make_facts())
        assert "interpretation" in result
        assert "recommended_next_action" in result
        assert isinstance(result["interpretation"], str)
        assert isinstance(result["recommended_next_action"], str)

    def test_analyze_run_bad_exit_code_mentions_failure(self):
        """When exit_code != 0, interpretation mentions failure."""
        az = self._get()
        result = az.analyze_run(make_facts(exit_code=1, state="failed", expectation_match="failed"))
        interp = result["interpretation"].lower()
        assert "fail" in interp or "error" in interp or "non-zero" in interp

    def test_analyze_run_usable_is_positive(self):
        """When expectation_match=usable, interpretation is positive."""
        az = self._get()
        result = az.analyze_run(make_facts(expectation_match="usable"))
        interp = result["interpretation"].lower()
        # Should have some positive signal
        assert any(word in interp for word in ["success", "pass", "ok", "good", "complet", "usable", "met", "converge"])

    def test_analyze_run_bad_signal_mentions_issue(self):
        """bad_signal expectation_match leads to cautionary interpretation."""
        az = self._get()
        result = az.analyze_run(make_facts(expectation_match="bad_signal"))
        interp = result["interpretation"].lower()
        assert any(word in interp for word in ["bad", "fail", "issue", "problem", "warn", "signal", "check"])

    def test_analyze_run_deterministic(self):
        """Same facts produce the same output every time (no randomness)."""
        az = self._get()
        facts = make_facts()
        r1 = az.analyze_run(facts)
        r2 = az.analyze_run(facts)
        assert r1 == r2

    def test_summarize_log_returns_string(self):
        """summarize_log returns a non-empty string."""
        az = self._get()
        log_text = "Epoch 1: loss=0.5\nEpoch 2: loss=0.3\nDone.\n"
        summary = az.summarize_log(log_text)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summarize_log_empty(self):
        """summarize_log handles empty input gracefully."""
        az = self._get()
        summary = az.summarize_log("")
        assert isinstance(summary, str)

    def test_explain_bad_signal_returns_string(self):
        """explain_bad_signal returns a non-empty string."""
        az = self._get()
        result = az.explain_bad_signal(make_facts(expectation_match="bad_signal"))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_suggest_next_action_returns_string(self):
        """suggest_next_action returns a string."""
        az = self._get()
        result = az.suggest_next_action(make_facts(), history=[])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_suggest_next_action_with_history(self):
        """suggest_next_action incorporates history (no crash)."""
        az = self._get()
        history = [make_facts(run_id="run-000", expectation_match="failed")]
        result = az.suggest_next_action(make_facts(), history=history)
        assert isinstance(result, str)

    def test_propose_criteria_returns_list(self):
        """propose_criteria returns a list of dicts."""
        az = self._get()
        feedback = {"kind": "reject", "text": "NaN appeared in output"}
        jobfile = {"name": "train-job", "expectation": "no NaN in logs"}
        result = az.propose_criteria(feedback, history=[], jobfile=jobfile)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_propose_criteria_have_required_fields(self):
        """Each proposed criterion has id, text, kind, check, status, strength."""
        az = self._get()
        feedback = {"kind": "reject", "text": "loss went NaN"}
        jobfile = {"name": "train-job", "expectation": "loss should decrease"}
        criteria = az.propose_criteria(feedback, history=[], jobfile=jobfile)
        for c in criteria:
            assert "id" in c
            assert "text" in c
            assert "kind" in c
            assert "check" in c
            assert "status" in c
            assert "strength" in c

    def test_propose_criteria_status_proposed(self):
        """Proposed criteria start with status=proposed, strength=1."""
        az = self._get()
        feedback = {"kind": "accept", "text": "all good"}
        jobfile = {"name": "my-job"}
        criteria = az.propose_criteria(feedback, history=[], jobfile=jobfile)
        for c in criteria:
            assert c["status"] == "proposed"
            assert c["strength"] == 1

    def test_summarize_failures_returns_string(self):
        """summarize_failures returns a string summary."""
        az = self._get()
        history = [
            make_facts(run_id="run-001", expectation_match="failed", exit_code=1),
            make_facts(run_id="run-002", expectation_match="bad_signal", exit_code=0),
        ]
        result = az.summarize_failures(history)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_failures_empty_history(self):
        """summarize_failures handles empty history."""
        az = self._get()
        result = az.summarize_failures([])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# DeepSeekAnalyzer — mock the openai client, never hit network
# ---------------------------------------------------------------------------

def _make_mock_openai_response(content: str):
    """Build a minimal fake openai ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestDeepSeekAnalyzer:
    def _get(self, mock_client=None):
        from jobctl.analysis.deepseek import DeepSeekAnalyzer
        az = DeepSeekAnalyzer(api_key="sk-test")
        if mock_client is not None:
            az._client = mock_client
        return az

    def _mock_client_returning(self, content: str):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_mock_openai_response(content)
        return client

    def test_analyze_run_calls_openai_create(self):
        """analyze_run calls client.chat.completions.create with the model."""
        response_json = json.dumps({
            "interpretation": "Run completed successfully.",
            "key_evidence": ["exit_code=0"],
            "recommended_next_action": "Accept results.",
        })
        mock_client = self._mock_client_returning(response_json)
        az = self._get(mock_client)
        result = az.analyze_run(make_facts())
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        # model should be deepseek-chat
        assert call_kwargs.kwargs.get("model") == "deepseek-chat" or \
               (call_kwargs.args and "deepseek-chat" in str(call_kwargs.args))

    def test_analyze_run_returns_parsed_dict(self):
        """analyze_run parses JSON from the response into a dict."""
        response_json = json.dumps({
            "interpretation": "Everything passed.",
            "key_evidence": ["no NaN"],
            "recommended_next_action": "Use results.",
        })
        mock_client = self._mock_client_returning(response_json)
        az = self._get(mock_client)
        result = az.analyze_run(make_facts())
        assert "interpretation" in result
        assert result["interpretation"] == "Everything passed."
        assert "recommended_next_action" in result

    def test_analyze_run_sends_compact_facts(self):
        """analyze_run serializes facts into the prompt (not full python repr)."""
        response_json = json.dumps({"interpretation": "ok", "recommended_next_action": "continue"})
        mock_client = self._mock_client_returning(response_json)
        az = self._get(mock_client)
        facts = make_facts(run_id="run-xyz", state="completed")
        az.analyze_run(facts)

        # Inspect the prompt sent
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0] if call_args.args else []
        if not messages:
            # fallback to positional arg inspection
            all_kwargs = call_args.kwargs
            messages = all_kwargs.get("messages", [])
        # The message content should contain the facts (as json or similar)
        combined = " ".join(str(m) for m in messages)
        assert "run-xyz" in combined or "completed" in combined

    def test_analyze_run_no_raise_on_valid_facts(self):
        """analyze_run does not raise for valid fact inputs."""
        response_json = json.dumps({"interpretation": "fine", "recommended_next_action": "next"})
        mock_client = self._mock_client_returning(response_json)
        az = self._get(mock_client)
        # Should not raise
        result = az.analyze_run(make_facts())
        assert result is not None

    def test_summarize_log_calls_client(self):
        """summarize_log uses the client to summarize."""
        mock_client = self._mock_client_returning("Short summary of the log.")
        az = self._get(mock_client)
        result = az.summarize_log("Epoch 1: loss=1.0\nEpoch 2: loss=0.5\n")
        mock_client.chat.completions.create.assert_called_once()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_explain_bad_signal_calls_client(self):
        """explain_bad_signal uses the client."""
        mock_client = self._mock_client_returning("NaN detected in layer 3.")
        az = self._get(mock_client)
        result = az.explain_bad_signal(make_facts(expectation_match="bad_signal"))
        mock_client.chat.completions.create.assert_called_once()
        assert isinstance(result, str)

    def test_suggest_next_action_calls_client(self):
        """suggest_next_action uses the client."""
        mock_client = self._mock_client_returning("Reduce learning rate.")
        az = self._get(mock_client)
        result = az.suggest_next_action(make_facts(), history=[])
        mock_client.chat.completions.create.assert_called_once()
        assert isinstance(result, str)

    def test_propose_criteria_returns_list(self):
        """propose_criteria returns a list of criterion dicts (parsed from JSON array)."""
        criteria_json = json.dumps([
            {"id": "c-001", "text": "No NaN in log", "kind": "absence",
             "check": {"pattern": "NaN"}, "status": "proposed", "strength": 1,
             "evidence_run_ids": []}
        ])
        mock_client = self._mock_client_returning(criteria_json)
        az = self._get(mock_client)
        result = az.propose_criteria({"kind": "reject", "text": "NaN appeared"}, [], {})
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["kind"] == "absence"

    def test_summarize_failures_calls_client(self):
        """summarize_failures uses the client."""
        mock_client = self._mock_client_returning("Two failures: NaN and OOM.")
        az = self._get(mock_client)
        history = [make_facts(run_id="run-001", expectation_match="failed")]
        result = az.summarize_failures(history)
        mock_client.chat.completions.create.assert_called_once()
        assert isinstance(result, str)

    def test_deepseek_uses_correct_base_url(self):
        """DeepSeekAnalyzer initializes the openai client with the DeepSeek base URL."""
        import openai as openai_module
        with patch.object(openai_module, "OpenAI") as mock_openai_cls:
            from importlib import reload
            import jobctl.analysis.deepseek as ds_mod
            reload(ds_mod)
            ds_mod.DeepSeekAnalyzer(api_key="sk-test-key")
            mock_openai_cls.assert_called_once()
            call_kwargs = mock_openai_cls.call_args.kwargs
            assert "api_key" in call_kwargs
            assert call_kwargs["api_key"] == "sk-test-key"
            base_url = call_kwargs.get("base_url", "")
            assert "deepseek" in base_url.lower()

    def test_analyze_run_malformed_json_falls_back(self):
        """When response is not valid JSON, analyze_run returns a dict with interpretation."""
        mock_client = self._mock_client_returning("Sorry, I cannot process that request.")
        az = self._get(mock_client)
        result = az.analyze_run(make_facts())
        # Should still return a dict, not raise
        assert isinstance(result, dict)
        assert "interpretation" in result

    def test_deepseek_is_analyzer_subclass(self):
        """DeepSeekAnalyzer is a subclass of Analyzer ABC."""
        from jobctl.analysis.base import Analyzer
        from jobctl.analysis.deepseek import DeepSeekAnalyzer
        assert issubclass(DeepSeekAnalyzer, Analyzer)


# ---------------------------------------------------------------------------
# Analyzer ABC — cannot instantiate directly
# ---------------------------------------------------------------------------

class TestAnalyzerABC:
    def test_cannot_instantiate_directly(self):
        """Analyzer ABC cannot be instantiated directly."""
        from jobctl.analysis.base import Analyzer
        with pytest.raises(TypeError):
            Analyzer()

    def test_abc_defines_required_methods(self):
        """Analyzer ABC declares all required abstract methods."""
        from jobctl.analysis.base import Analyzer
        import inspect
        abstract_methods = {
            name for name, method in inspect.getmembers(Analyzer, predicate=inspect.isfunction)
            if getattr(method, "__isabstractmethod__", False)
        }
        required = {
            "analyze_run", "summarize_log", "explain_bad_signal",
            "suggest_next_action", "propose_criteria", "summarize_failures",
        }
        assert required.issubset(abstract_methods)
