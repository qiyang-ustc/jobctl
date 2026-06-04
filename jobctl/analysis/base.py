"""Analysis layer: Analyzer ABC and get_analyzer() selector."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


class Analyzer(ABC):
    """Abstract base class for all analyzers.

    All methods receive/return plain Python dicts and strings so they can be
    used offline or backed by any cheap-model API.
    """

    @abstractmethod
    def analyze_run(self, facts: dict) -> dict:
        """Analyze a completed run and return a narrative dict.

        Args:
            facts: Structured facts about the run (state, exit_code,
                   expectation_match, artifacts, key_evidence, ...).

        Returns:
            dict with at minimum:
              - interpretation (str)
              - recommended_next_action (str)
              - key_evidence (list[str], optional)
        """

    @abstractmethod
    def summarize_log(self, text: str) -> str:
        """Return a concise plain-text summary of a log file."""

    @abstractmethod
    def explain_bad_signal(self, facts: dict) -> str:
        """Return a short explanation of why this run has bad_signal."""

    @abstractmethod
    def suggest_next_action(self, facts: dict, history: list) -> str:
        """Suggest the next action given current facts and run history."""

    @abstractmethod
    def propose_criteria(self, feedback: dict, history: list, jobfile: dict) -> list[dict]:
        """Propose new expectation criteria from user feedback and history.

        Returns:
            List of criterion dicts, each with:
              id, text, kind, check, status="proposed", strength=1,
              evidence_run_ids=[]
        """

    @abstractmethod
    def summarize_failures(self, history: list) -> str:
        """Return a summary of failures across a history of runs."""


def get_analyzer(config: dict) -> Analyzer:
    """Return the appropriate Analyzer based on environment.

    Returns DeepSeekAnalyzer if DEEPSEEK_API_KEY is set in the environment,
    otherwise returns OfflineAnalyzer (deterministic, no network).

    Args:
        config: jobctl config dict (currently unused but reserved for future
                per-config analyzer selection).

    Returns:
        An Analyzer instance.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if api_key:
        from jobctl.analysis.deepseek import DeepSeekAnalyzer
        return DeepSeekAnalyzer(api_key=api_key)
    from jobctl.analysis.offline import OfflineAnalyzer
    return OfflineAnalyzer()
