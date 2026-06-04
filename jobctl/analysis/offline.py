"""Offline (deterministic) analyzer — no network, zero token cost.

Used in all tests and as fallback when DEEPSEEK_API_KEY is not set.
All outputs are purely derived from the structured facts passed in.
"""
from __future__ import annotations

import uuid
from typing import Any

from jobctl.analysis.base import Analyzer


# ---------------------------------------------------------------------------
# Interpretation templates
# ---------------------------------------------------------------------------

_MATCH_INTERPRETATIONS: dict[str, str] = {
    "usable": "Run completed successfully; all expectation criteria were met.",
    "weak_signal": "Run completed with weak signal; some non-critical criteria were not met or near threshold.",
    "bad_signal": "Run produced bad signal; one or more critical criteria were violated. Check for NaN, errors, or threshold breaches.",
    "inconclusive": "Run result is inconclusive; some criteria could not be evaluated (missing or unparsable artifacts).",
    "failed": "Run failed; a non-zero exit code or backend error was detected.",
    None: "Run completed; no expectation contract was evaluated.",
}

_MATCH_NEXT_ACTIONS: dict[str, str] = {
    "usable": "Accept results and proceed to the next stage.",
    "weak_signal": "Inspect artifacts and consider re-running with adjusted parameters.",
    "bad_signal": "Investigate the cause of the bad signal before proceeding.",
    "inconclusive": "Check that all expected artifacts were produced and retry if needed.",
    "failed": "Diagnose the failure (check logs and exit code), fix the issue, and rerun.",
    None: "Review the run output and decide on next steps.",
}

_HEALTH_NOTES: dict[str, str] = {
    "ok": "",
    "weak": " Health is weak — resource pressure may be affecting results.",
    "no_heartbeat": " No heartbeat was received — the run may have silently died.",
    "resource_pressure": " Resource pressure was detected during the run.",
    "stuck": " The run appears to have been stuck (no log growth + no heartbeat).",
}


class OfflineAnalyzer(Analyzer):
    """Deterministic analyzer that produces templated output from structured facts.

    No network calls; no randomness; identical inputs produce identical outputs.
    """

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def analyze_run(self, facts: dict) -> dict:
        """Build interpretation and next-action from facts deterministically."""
        match = facts.get("expectation_match")
        exit_code = facts.get("exit_code")
        health = facts.get("health", "ok")
        state = facts.get("state", "")

        # Determine interpretation
        if match in _MATCH_INTERPRETATIONS:
            interp = _MATCH_INTERPRETATIONS[match]
        elif exit_code is not None and exit_code != 0:
            interp = _MATCH_INTERPRETATIONS["failed"]
        elif state == "failed":
            interp = _MATCH_INTERPRETATIONS["failed"]
        else:
            interp = _MATCH_INTERPRETATIONS[None]

        health_note = _HEALTH_NOTES.get(health, "")
        if health_note:
            interp = interp.rstrip(".") + "." + health_note

        # Determine next action
        if match in _MATCH_NEXT_ACTIONS:
            next_action = _MATCH_NEXT_ACTIONS[match]
        elif exit_code is not None and exit_code != 0:
            next_action = _MATCH_NEXT_ACTIONS["failed"]
        elif state == "failed":
            next_action = _MATCH_NEXT_ACTIONS["failed"]
        else:
            next_action = _MATCH_NEXT_ACTIONS[None]

        result: dict[str, Any] = {
            "interpretation": interp,
            "recommended_next_action": next_action,
        }

        # Include key_evidence if present
        key_evidence = facts.get("key_evidence")
        if key_evidence:
            result["key_evidence"] = key_evidence

        return result

    def summarize_log(self, text: str) -> str:
        """Return a short head/tail summary of the log text."""
        if not text:
            return "(empty log)"
        lines = text.splitlines()
        total = len(lines)
        if total <= 10:
            return "\n".join(lines)
        head = lines[:5]
        tail = lines[-5:]
        return "\n".join(head) + f"\n... ({total - 10} lines omitted) ...\n" + "\n".join(tail)

    def explain_bad_signal(self, facts: dict) -> str:
        """Produce a templated explanation of bad_signal from facts."""
        match = facts.get("expectation_match", "bad_signal")
        key_evidence = facts.get("key_evidence", [])
        artifacts = facts.get("artifacts", [])

        parts = [f"Bad signal detected (expectation_match={match})."]
        if key_evidence:
            parts.append("Key evidence: " + "; ".join(str(e) for e in key_evidence) + ".")
        if artifacts:
            art_names = [a.get("name", "?") for a in artifacts]
            parts.append("Artifacts present: " + ", ".join(art_names) + ".")
        parts.append(
            "Possible causes include NaN values in outputs, threshold violations, "
            "or missing expected artifacts."
        )
        return " ".join(parts)

    def suggest_next_action(self, facts: dict, history: list) -> str:
        """Suggest a next action based on current facts and run history."""
        match = facts.get("expectation_match")
        exit_code = facts.get("exit_code")
        state = facts.get("state", "")
        failure_count = sum(
            1 for h in history
            if h.get("expectation_match") in ("failed", "bad_signal")
            or h.get("exit_code", 0) not in (0, None)
        )

        if match == "usable":
            if failure_count > 0:
                return (
                    f"Recovered after {failure_count} prior failure(s). "
                    "Accept results and consider updating the expectation contract."
                )
            return "Accept results and proceed."

        if match == "bad_signal":
            return (
                "Investigate the bad signal: check logs for NaN, errors, or threshold violations. "
                "Fix the issue and rerun."
            )

        if match == "failed" or (exit_code is not None and exit_code != 0):
            if failure_count >= 2:
                return (
                    f"Repeated failures ({failure_count + 1} total). "
                    "Consider a different configuration or check system resources."
                )
            return "Fix the error causing the non-zero exit code and rerun."

        if match == "weak_signal":
            return "Review near-threshold metrics and consider tighter parameters or more iterations."

        if match == "inconclusive":
            return "Ensure all expected artifacts are produced; check paths and patterns in the JobFile."

        return "Review the run output and decide whether to accept, rerun, or adjust parameters."

    def propose_criteria(self, feedback: dict, history: list, jobfile: dict) -> list[dict]:
        """Propose expectation criteria from user feedback.

        Returns a minimal but useful list of criteria derived from the feedback text
        and jobfile expectation string.
        """
        criteria: list[dict] = []
        text = feedback.get("text", "")
        kind = feedback.get("kind", "note")
        jobfile_expectation = jobfile.get("expectation", "")

        # Always propose absence of NaN if feedback mentions it or if reject
        if kind == "reject" or "nan" in text.lower() or "nan" in jobfile_expectation.lower():
            criteria.append(self._make_criterion(
                text="No NaN values in any log or output",
                kind="absence",
                check={"pattern": "NaN", "targets": ["stdout", "stderr", "*.log"]},
            ))

        # Propose absence of Traceback if error is mentioned
        if kind == "reject" or "traceback" in text.lower() or "error" in text.lower():
            criteria.append(self._make_criterion(
                text="No Python Traceback in logs",
                kind="absence",
                check={"pattern": "Traceback", "targets": ["stdout", "stderr"]},
            ))

        # Propose presence criterion if feedback accepts something
        if kind == "accept" and jobfile_expectation:
            criteria.append(self._make_criterion(
                text=f"Expectation met: {jobfile_expectation[:120]}",
                kind="presence",
                check={"description": jobfile_expectation},
            ))

        # If no criteria derived, return a generic quality criterion
        if not criteria:
            criteria.append(self._make_criterion(
                text="Run exits with code 0",
                kind="numeric",
                check={"source": "exit_code", "op": "eq", "value": 0},
            ))

        return criteria

    def summarize_failures(self, history: list) -> str:
        """Summarize failure patterns across a history of runs."""
        if not history:
            return "No failure history available."

        total = len(history)
        failed = sum(
            1 for h in history
            if h.get("expectation_match") in ("failed", "bad_signal")
            or (h.get("exit_code") is not None and h.get("exit_code") != 0)
        )
        if failed == 0:
            return f"No failures detected across {total} run(s) in history."

        pct = int(100 * failed / total)
        lines = [f"{failed}/{total} run(s) ({pct}%) had failures or bad signal."]

        # Summarize match distribution
        match_counts: dict[str, int] = {}
        for h in history:
            m = h.get("expectation_match", "unknown")
            match_counts[m] = match_counts.get(m, 0) + 1
        for match_val, count in sorted(match_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {match_val}: {count} run(s)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_criterion(text: str, kind: str, check: dict) -> dict:
        return {
            "id": f"proposed-{uuid.uuid4().hex[:8]}",
            "text": text,
            "kind": kind,
            "check": check,
            "status": "proposed",
            "strength": 1,
            "evidence_run_ids": [],
        }
