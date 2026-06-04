"""DeepSeek analyzer — OpenAI-compatible API client.

Uses the OpenAI SDK pointed at https://api.deepseek.com with model deepseek-chat.
The API key is taken from the DEEPSEEK_API_KEY environment variable or passed
directly to the constructor.

All calls return plain dicts/strings.  On parse failure the methods fall back to
a minimal offline-style response rather than raising, so callers can always count
on getting valid output.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

import openai

from jobctl.analysis.base import Analyzer

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_MODEL = "deepseek-chat"

# System prompt hints — kept short to minimise tokens.
_SYSTEM_ANALYZE_RUN = (
    "You are a research-run analysis assistant. "
    "Given structured JSON facts about a completed job, respond ONLY with valid JSON "
    "containing keys: interpretation (str), key_evidence (list[str] optional), "
    "recommended_next_action (str)."
)

_SYSTEM_SUMMARIZE_LOG = (
    "You are a log summariser. Respond with a single concise paragraph summarising "
    "the key events in the log. No JSON."
)

_SYSTEM_EXPLAIN_BAD = (
    "You are a research-run diagnostician. Explain briefly why this run has bad_signal "
    "based on the provided facts. Respond with plain text, no JSON."
)

_SYSTEM_NEXT_ACTION = (
    "You are a research assistant. Suggest the single best next action given the run "
    "facts and history. Respond with plain text, no JSON."
)

_SYSTEM_PROPOSE_CRITERIA = (
    "You are a quality-criteria engineer. Given user feedback, run history, and a "
    "JobFile description, propose 1-3 machine-checkable expectation criteria. "
    "Respond ONLY with a JSON array of objects with keys: "
    "id, text, kind (numeric|presence|absence|pattern), check (dict), "
    "status='proposed', strength=1, evidence_run_ids=[]."
)

_SYSTEM_SUMMARIZE_FAILURES = (
    "You are a failure-analysis assistant. Summarise the common failure patterns "
    "across the provided run history. Respond with plain text, no JSON."
)


class DeepSeekAnalyzer(Analyzer):
    """Analyzer backed by the DeepSeek chat API (OpenAI-compatible)."""

    def __init__(self, api_key: str, base_url: str = _DEEPSEEK_BASE_URL) -> None:
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model = _MODEL

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat(self, system: str, user: str) -> str:
        """Call the API and return the raw response content string."""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    @staticmethod
    def _parse_json_or_fallback(text: str, fallback: dict | list) -> Any:
        """Try to parse JSON; return fallback on failure."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            # Try to extract the first JSON block from a mixed response
            start = text.find("{") if isinstance(fallback, dict) else text.find("[")
            end = text.rfind("}") + 1 if isinstance(fallback, dict) else text.rfind("]") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except (json.JSONDecodeError, ValueError):
                    pass
            return fallback

    # ------------------------------------------------------------------
    # Analyzer interface
    # ------------------------------------------------------------------

    def analyze_run(self, facts: dict) -> dict:
        """Call DeepSeek to interpret a completed run."""
        user_msg = json.dumps(facts, default=str)
        raw = self._chat(_SYSTEM_ANALYZE_RUN, user_msg)
        fallback = {
            "interpretation": raw if raw else "Analysis unavailable.",
            "recommended_next_action": "Review the run output manually.",
        }
        result = self._parse_json_or_fallback(raw, fallback)
        if not isinstance(result, dict):
            return fallback
        # Ensure required keys
        if "interpretation" not in result:
            result["interpretation"] = raw or "Analysis unavailable."
        if "recommended_next_action" not in result:
            result["recommended_next_action"] = "Review the run output manually."
        return result

    def summarize_log(self, text: str) -> str:
        """Call DeepSeek to summarise a log file."""
        # Truncate very long logs to avoid excessive tokens
        max_chars = 8000
        user_msg = text[:max_chars] if len(text) > max_chars else text
        return self._chat(_SYSTEM_SUMMARIZE_LOG, user_msg) or "(empty response)"

    def explain_bad_signal(self, facts: dict) -> str:
        """Call DeepSeek to explain why a run has bad_signal."""
        user_msg = json.dumps(facts, default=str)
        return self._chat(_SYSTEM_EXPLAIN_BAD, user_msg) or "Bad signal explanation unavailable."

    def suggest_next_action(self, facts: dict, history: list) -> str:
        """Call DeepSeek to suggest a next action."""
        payload = {"facts": facts, "history": history[-5:]}  # cap history to last 5
        user_msg = json.dumps(payload, default=str)
        return self._chat(_SYSTEM_NEXT_ACTION, user_msg) or "Next action unavailable."

    def propose_criteria(self, feedback: dict, history: list, jobfile: dict) -> list[dict]:
        """Call DeepSeek to propose expectation criteria."""
        payload = {
            "feedback": feedback,
            "history_summary": history[-3:],
            "jobfile": jobfile,
        }
        user_msg = json.dumps(payload, default=str)
        raw = self._chat(_SYSTEM_PROPOSE_CRITERIA, user_msg)
        fallback: list[dict] = [
            {
                "id": f"proposed-{uuid.uuid4().hex[:8]}",
                "text": "Run exits with code 0",
                "kind": "numeric",
                "check": {"source": "exit_code", "op": "eq", "value": 0},
                "status": "proposed",
                "strength": 1,
                "evidence_run_ids": [],
            }
        ]
        result = self._parse_json_or_fallback(raw, fallback)
        if not isinstance(result, list):
            return fallback
        # Ensure every criterion has mandatory keys
        cleaned = []
        for item in result:
            if not isinstance(item, dict):
                continue
            item.setdefault("id", f"proposed-{uuid.uuid4().hex[:8]}")
            item.setdefault("status", "proposed")
            item.setdefault("strength", 1)
            item.setdefault("evidence_run_ids", [])
            cleaned.append(item)
        return cleaned if cleaned else fallback

    def summarize_failures(self, history: list) -> str:
        """Call DeepSeek to summarise failure patterns."""
        user_msg = json.dumps({"history": history}, default=str)
        return self._chat(_SYSTEM_SUMMARIZE_FAILURES, user_msg) or "Failure summary unavailable."
