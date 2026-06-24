"""Expectations engine: evaluate, default_contract, classify.

Public API:
    evaluate(contract, run, artifacts, stdout, stderr) -> tuple[Match, list[str], list[dict]]
    default_contract(jobfile) -> ExpectationContract
    classify(exit_code, criteria_results) -> Match   (internal, exposed for testing)
"""
from __future__ import annotations

import csv
import fnmatch
import json
import re
import uuid
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from jobctl.db.models import (
    Artifact,
    Criterion,
    ExpectationContract,
    JobFile,
    Match,
    Run,
    State,
)


NUMERIC_NAN_PATTERN = r"(?<![A-Za-z0-9_])[-+]?nan(?![A-Za-z0-9_])"


# ---------------------------------------------------------------------------
# Internal: per-criterion evaluation
# ---------------------------------------------------------------------------

def _regex_flags(flag_names: list[str] | str | None) -> int:
    if flag_names is None:
        return 0
    if isinstance(flag_names, str):
        flag_names = [flag_names]

    flags = 0
    for name in flag_names:
        normalized = str(name).lower().replace("-", "_")
        if normalized in {"ignorecase", "ignore_case", "i"}:
            flags |= re.IGNORECASE
        elif normalized in {"multiline", "multi_line", "m"}:
            flags |= re.MULTILINE
        elif normalized in {"dotall", "s"}:
            flags |= re.DOTALL
    return flags


def _absence_matcher(check: dict[str, Any]) -> tuple[str, Any] | tuple[None, str]:
    pattern = check.get("pattern", "")
    if not pattern:
        return None, "No pattern specified"

    if check.get("regex", False):
        try:
            compiled = re.compile(pattern, _regex_flags(check.get("flags")))
        except re.error as exc:
            return None, f"Invalid regex: {exc}"
        return check.get("label", pattern), compiled.search

    if check.get("case_sensitive", True):
        return check.get("label", pattern), lambda content: pattern in content

    lowered = str(pattern).lower()
    return check.get("label", pattern), lambda content: lowered in content.lower()


def _eval_absence(
    criterion: Criterion,
    artifacts: list[Artifact],
    stdout: str,
    stderr: str,
) -> tuple[bool | None, str]:
    """Evaluate an absence criterion (pattern must NOT appear).

    Returns (passed, detail):
        True  -> pattern absent (pass)
        False -> pattern found (fail)
        None  -> could not evaluate (inconclusive)
    """
    check = criterion.check
    targets: list[str] = check.get("targets", ["stdout", "stderr"])

    label, matcher = _absence_matcher(check)
    if label is None:
        return None, matcher

    found_in: list[str] = []

    for target in targets:
        if target == "stdout":
            if matcher(stdout):
                found_in.append("stdout")
        elif target == "stderr":
            if matcher(stderr):
                found_in.append("stderr")
        else:
            # It's a glob against artifact paths
            for art in artifacts:
                if fnmatch.fnmatch(Path(art.local_path).name, target):
                    try:
                        content = Path(art.local_path).read_text(errors="replace")
                        if matcher(content):
                            found_in.append(art.local_path)
                    except OSError:
                        pass

    if found_in:
        detail = f"Pattern '{label}' found in: {', '.join(found_in)}"
        return False, detail

    detail = f"Pattern '{label}' not found (pass)"
    return True, detail


def _eval_presence(
    criterion: Criterion,
    artifacts: list[Artifact],
) -> tuple[bool | None, str]:
    """Evaluate a presence criterion (at least one artifact matches glob).

    Returns (passed, detail).
        True  -> artifact found
        False -> no artifact found; this is inconclusive (can't verify artifact exists
                 without running the job; treated as missing evidence, not a hard failure)
        None  -> no glob specified (configuration error)
    """
    check = criterion.check
    glob = check.get("glob", "")
    if not glob:
        return None, "No glob specified"

    for art in artifacts:
        name = Path(art.local_path).name
        if fnmatch.fnmatch(name, glob):
            return True, f"Artifact found: {art.local_path}"

    # No matching artifact — inconclusive: the artifact was not produced or not indexed
    return False, f"No artifact matching glob '{glob}' found (inconclusive)"


def _compare(value: float, op: str, threshold: float) -> bool:
    """Apply comparison operator."""
    ops = {
        "lt": value < threshold,
        "le": value <= threshold,
        "gt": value > threshold,
        "ge": value >= threshold,
        "eq": abs(value - threshold) < 1e-10,
        "ne": abs(value - threshold) >= 1e-10,
    }
    return ops.get(op, False)


def _near_threshold(value: float, op: str, threshold: float, pct: float) -> bool:
    """Check if value is near the threshold (within pct%) but doesn't pass the strict check.

    Near-threshold zone: the value fails the strict check but is within pct% of
    the threshold value. This signals WEAK_SIGNAL rather than BAD_SIGNAL.
    """
    if _compare(value, op, threshold):
        return False  # It passes — not "near" in the failure sense

    if threshold == 0:
        margin = abs(value) * (pct / 100.0)
        if margin == 0:
            return False
        return abs(value - threshold) <= margin

    margin = abs(threshold) * (pct / 100.0)
    return abs(value - threshold) <= margin


def _extract_from_stdout_stderr(
    check: dict,
    stdout: str,
    stderr: str,
) -> float | None:
    """Extract a numeric value from stdout or stderr using a regex."""
    extract_re = check.get("extract", "")
    source = check.get("source", "stdout")
    if not extract_re:
        return None

    text = stdout if source == "stdout" else stderr
    m = re.search(extract_re, text)
    if m:
        try:
            return float(m.group(1))
        except (IndexError, ValueError):
            return None
    return None


def _extract_from_artifact(check: dict, artifacts: list[Artifact]) -> float | None:
    """Extract a numeric from a csv or json artifact."""
    source_glob = check.get("source", "")
    if not source_glob:
        return None

    for art in artifacts:
        name = Path(art.local_path).name
        if not fnmatch.fnmatch(name, source_glob):
            continue

        try:
            content = Path(art.local_path).read_text(errors="replace")
        except OSError:
            continue

        # CSV column extraction
        csv_column = check.get("csv_column")
        if csv_column:
            row_selector = check.get("csv_row", "last")
            try:
                reader = list(csv.DictReader(StringIO(content)))
                if not reader:
                    continue
                row = reader[-1] if row_selector == "last" else reader[0]
                return float(row[csv_column])
            except (KeyError, ValueError, csv.Error):
                continue

        # JSON key-path extraction
        json_path = check.get("json_path")
        if json_path:
            try:
                obj = json.loads(content)
                # Simple dot-notation path traversal
                parts = json_path.split(".")
                val = obj
                for part in parts:
                    val = val[part]
                return float(val)
            except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                continue

        # Regex on file content
        extract_re = check.get("extract")
        if extract_re:
            m = re.search(extract_re, content)
            if m:
                try:
                    return float(m.group(1))
                except (IndexError, ValueError):
                    continue

    return None


def _eval_numeric(
    criterion: Criterion,
    artifacts: list[Artifact],
    stdout: str,
    stderr: str,
) -> tuple[bool | None, bool, str]:
    """Evaluate a numeric criterion.

    Returns (passed, near_threshold, detail):
        passed=True  -> strict comparison passes
        passed=False -> strict comparison fails; near_threshold may be True
        passed=None  -> could not extract value (inconclusive)
        near_threshold=True -> value is within near_threshold_pct of the threshold
    """
    check = criterion.check
    op = check.get("op", "lt")
    threshold = float(check.get("value", 0))
    near_threshold_pct = check.get("near_threshold_pct", None)

    source = check.get("source", "stdout")

    # Try to extract value
    extracted: float | None = None

    if source in ("stdout", "stderr"):
        extracted = _extract_from_stdout_stderr(check, stdout, stderr)
    else:
        # Assume it's a glob for artifacts
        extracted = _extract_from_artifact(check, artifacts)
        # Fallback to stdout regex if no artifact matched
        if extracted is None and check.get("extract"):
            extracted = _extract_from_stdout_stderr(check, stdout, stderr)

    if extracted is None:
        return None, False, "Could not extract numeric value"

    strict_pass = _compare(extracted, op, threshold)
    near = False
    if not strict_pass and near_threshold_pct is not None:
        near = _near_threshold(extracted, op, threshold, near_threshold_pct)

    detail = f"extracted={extracted}, op={op}, threshold={threshold}, pass={strict_pass}"
    if near:
        detail += " (near threshold)"

    return strict_pass, near, detail


def _eval_pattern(
    criterion: Criterion,
    artifacts: list[Artifact],
    stdout: str,
    stderr: str,
) -> tuple[bool | None, str]:
    """Evaluate a pattern criterion (regex must match).

    Returns (passed, detail).
    None = inconclusive (no source to check or pattern error).
    """
    check = criterion.check
    pattern = check.get("pattern", "")
    targets: list[str] = check.get("targets", ["stdout"])
    if not pattern:
        return None, "No pattern specified"

    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return None, f"Invalid regex: {exc}"

    for target in targets:
        if target == "stdout":
            if compiled.search(stdout):
                return True, f"Pattern '{pattern}' matched in stdout"
        elif target == "stderr":
            if compiled.search(stderr):
                return True, f"Pattern '{pattern}' matched in stderr"
        else:
            for art in artifacts:
                if fnmatch.fnmatch(Path(art.local_path).name, target):
                    try:
                        content = Path(art.local_path).read_text(errors="replace")
                        if compiled.search(content):
                            return True, f"Pattern '{pattern}' matched in {art.local_path}"
                    except OSError:
                        pass

    return False, f"Pattern '{pattern}' not found in any target"


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def classify(
    exit_code: int | None,
    state: str,
    criterion_results: list[dict],
) -> Match:
    """Deterministic classification of a run based on exit code and criteria results.

    Classification matrix:
        failed       -> non-zero exit code OR state is 'failed'
        bad_signal   -> any confirmed absence criterion violated, OR confirmed hard numeric fails
        weak_signal  -> proposed (unconfirmed) criterion fails, OR numeric near threshold
        inconclusive -> any criterion is inconclusive (missing/unparsable) and none failed hard
        usable       -> all confirmed criteria pass (or no criteria)

    Inconclusive criteria:
        - presence: artifact not found (may not have been produced yet or not indexed)
        - numeric: could not extract value from source
        - pattern: pattern not found (may indicate run didn't reach that point)
        - Any criterion with passed=None (config error or no source)
    """
    # Terminal failure state
    if (exit_code is not None and exit_code != 0) or state in ("failed",):
        return Match.FAILED

    if not criterion_results:
        return Match.USABLE

    has_inconclusive = False
    has_weak = False
    has_bad = False

    for result in criterion_results:
        passed = result.get("passed")
        kind = result.get("kind", "")
        status = result.get("status", "confirmed")
        near = result.get("near_threshold", False)

        if passed is True:
            continue

        if passed is None:
            # Inconclusive (could not evaluate — configuration/extraction error)
            has_inconclusive = True
            continue

        # passed is False
        if kind == "absence" and status == "confirmed":
            # Absence violation is always BAD_SIGNAL when confirmed
            has_bad = True
        elif kind == "numeric" and status == "confirmed" and not near:
            # Hard numeric failure (confirmed, not near threshold)
            has_bad = True
        elif near:
            # Near threshold: downgrade to weak regardless of status
            has_weak = True
        elif status != "confirmed":
            # Proposed/unconfirmed failure -> weak signal
            has_weak = True
        elif kind in ("presence", "pattern"):
            # Confirmed presence/pattern failure: treat as inconclusive
            # (artifact not produced / pattern not seen — run may be incomplete)
            has_inconclusive = True
        else:
            # Other confirmed failure -> bad signal
            has_bad = True

    if has_bad:
        return Match.BAD_SIGNAL
    if has_weak:
        return Match.WEAK_SIGNAL
    if has_inconclusive:
        return Match.INCONCLUSIVE
    return Match.USABLE


# ---------------------------------------------------------------------------
# Public: evaluate
# ---------------------------------------------------------------------------

def evaluate(
    contract: ExpectationContract,
    run: Run,
    artifacts: list[Artifact],
    stdout: str,
    stderr: str,
) -> tuple[Match, list[str], list[dict]]:
    """Evaluate a run against an ExpectationContract.

    Args:
        contract: The ExpectationContract to evaluate against.
        run: The completed Run (must have exit_code / state).
        artifacts: List of indexed Artifacts for this run.
        stdout: Full stdout text.
        stderr: Full stderr text.

    Returns:
        (match, key_evidence, per_criterion) where:
            match          - Match enum value
            key_evidence   - list of evidence strings
            per_criterion  - list of {id, text, kind, status, passed, detail, near_threshold}
    """
    key_evidence: list[str] = []
    per_criterion: list[dict] = []

    # --- Nonzero exit / failed state ---
    state_str = run.state.value if hasattr(run.state, "value") else str(run.state)
    exit_code = run.exit_code

    if (exit_code is not None and exit_code != 0) or state_str == "failed":
        exit_str = (
            f"exit_code={exit_code}"
            if exit_code is not None and exit_code != 0
            else "state=failed"
        )
        key_evidence.append(f"Run {exit_str}")
        return Match.FAILED, key_evidence, per_criterion

    # --- Evaluate each criterion ---
    for crit in contract.criteria:
        result: dict[str, Any] = {
            "id": crit.id,
            "text": crit.text,
            "kind": crit.kind,
            "status": crit.status,
            "passed": None,
            "detail": "",
            "near_threshold": False,
        }

        if crit.kind == "absence":
            passed, detail = _eval_absence(crit, artifacts, stdout, stderr)
            result["passed"] = passed
            result["detail"] = detail
            if passed is False:
                key_evidence.append(detail)

        elif crit.kind == "presence":
            passed, detail = _eval_presence(crit, artifacts)
            result["passed"] = passed
            result["detail"] = detail
            if passed is True:
                key_evidence.append(detail)

        elif crit.kind == "numeric":
            passed, near, detail = _eval_numeric(crit, artifacts, stdout, stderr)
            result["passed"] = passed
            result["detail"] = detail
            result["near_threshold"] = near
            if passed is True:
                # Extract the value for evidence
                m = re.search(r"extracted=([^\s,]+)", detail)
                if m:
                    key_evidence.append(f"{crit.text}: extracted={m.group(1)}")
                else:
                    key_evidence.append(detail)

        elif crit.kind == "pattern":
            passed, detail = _eval_pattern(crit, artifacts, stdout, stderr)
            result["passed"] = passed
            result["detail"] = detail
            if passed is True:
                key_evidence.append(detail)

        per_criterion.append(result)

    # --- Classify overall match ---
    match = classify(exit_code, state_str, per_criterion)

    # --- Add key evidence for failed items ---
    for result in per_criterion:
        if result.get("passed") is False and result.get("detail") not in key_evidence:
            detail = result["detail"]
            if detail and detail not in key_evidence:
                key_evidence.append(detail)

    return match, key_evidence, per_criterion


# ---------------------------------------------------------------------------
# Public: default_contract
# ---------------------------------------------------------------------------

def default_contract(jobfile: JobFile) -> ExpectationContract:
    """Seed a default ExpectationContract for a JobFile.

    Includes:
    - Absence criteria for NaN, Traceback, CUDA error in stdout/stderr/logs.
    - Optional: a criterion seeded from the JobFile's expectation text.

    All criteria start as status='proposed'.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Standard absence criteria
    criteria: list[Criterion] = [
        Criterion(
            id=f"default-nan-{uuid.uuid4().hex[:8]}",
            text="No NaN values in stdout, stderr, or log files",
            kind="absence",
            check={
                "pattern": NUMERIC_NAN_PATTERN,
                "regex": True,
                "flags": ["ignorecase"],
                "label": "numeric NaN",
                "targets": ["stdout", "stderr", "*.log"],
            },
            status="proposed",
            strength=1,
            evidence_run_ids=[],
        ),
        Criterion(
            id=f"default-traceback-{uuid.uuid4().hex[:8]}",
            text="No Python Traceback in stdout or stderr",
            kind="absence",
            check={"pattern": "Traceback", "targets": ["stdout", "stderr"]},
            status="proposed",
            strength=1,
            evidence_run_ids=[],
        ),
        Criterion(
            id=f"default-cuda-{uuid.uuid4().hex[:8]}",
            text="No CUDA error in stdout or stderr",
            kind="absence",
            check={"pattern": "CUDA error", "targets": ["stdout", "stderr"]},
            status="proposed",
            strength=1,
            evidence_run_ids=[],
        ),
    ]

    # Infer source from expectation text if available
    expectation_text = getattr(jobfile, "_expectation_text", None)
    source = f"default; jobfile={jobfile.name}"

    # Check if there's a raw expectation embedded in the JobFile source
    if jobfile.source_path:
        try:
            src_path = Path(jobfile.source_path)
            if src_path.suffix.lower() in (".yaml", ".yml") and src_path.exists():
                import yaml
                raw = yaml.safe_load(src_path.read_text()) or {}
                expectation_text = raw.get("expectation", "")
        except Exception:
            expectation_text = None

    if expectation_text and expectation_text.strip():
        source = f"default; jobfile={jobfile.name}; expectation={expectation_text!r}"
        criteria.append(
            Criterion(
                id=f"default-expectation-{uuid.uuid4().hex[:8]}",
                text=f"Expectation from JobFile: {expectation_text[:200]}",
                kind="pattern",
                check={"description": expectation_text},
                status="proposed",
                strength=1,
                evidence_run_ids=[],
            )
        )

    contract_id = f"ec-default-{uuid.uuid4().hex[:12]}"
    return ExpectationContract(
        id=contract_id,
        jobfile_id=jobfile.id,
        version=1,
        criteria=criteria,
        source=source,
        created_at=now,
        updated_at=now,
    )
