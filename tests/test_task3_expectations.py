"""Tests for Task 3: expectations/contracts.py + expectations/distiller.py

TDD: Tests written first, then implementation.

Coverage:
- evaluate(): numeric (csv/json/log regex), presence (glob), absence, pattern
- Classification matrix: usable / weak_signal / bad_signal / inconclusive / failed
- Missing artifact -> inconclusive; nonzero exit -> failed
- default_contract(): seeds absence(NaN/Traceback/CUDA error) + manifest expectation
- distiller.propose(): uses fake analyzer -> persisted as proposed/strength=1
- distiller.confirm(): status->confirmed, strength+=1
"""
from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from jobctl.db.models import (
    Artifact,
    ArtifactType,
    Criterion,
    ExpectationContract,
    Feedback,
    Health,
    JobFile,
    Match,
    Run,
    State,
)
from jobctl.db.store import Store
from jobctl.expectations.contracts import (
    NUMERIC_NAN_PATTERN,
    default_contract,
    evaluate,
)
from jobctl.expectations.distiller import (
    confirm,
    propose,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_jobfile(name: str = "test-job", expectation: str = "") -> JobFile:
    return JobFile(
        id=f"jf-{uuid.uuid4().hex[:8]}",
        name=name,
        version=1,
        source_path="/tmp/test.py",
        command_template="python test.py",
        params_schema={},
        backend_prefs=[{"backend": "local"}],
        artifact_patterns=["*.csv", "*.log"],
        expectation_contract_id=None,
        content_hash="abc123",
        created_at=_now(),
    )


def _make_run(
    exit_code: int | None = 0,
    state: State = State.COMPLETED,
    workdir: str | None = None,
    stdout_path: str | None = None,
    stderr_path: str | None = None,
) -> Run:
    return Run(
        run_id=f"run-{uuid.uuid4().hex[:8]}",
        jobfile_id="jf-test",
        jobfile_version=1,
        params={},
        input_hashes={},
        backend="local",
        server=None,
        task=None,
        remote_job_id=None,
        state=state,
        health=Health.OK,
        exit_code=exit_code,
        submitted_at=_now(),
        started_at=_now(),
        finished_at=_now(),
        last_heartbeat=_now(),
        workdir=workdir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        resource_summary={},
        expectation_match=None,
        observation_card=None,
    )


def _make_artifact(local_path: str, atype: ArtifactType = ArtifactType.TEXT_LOG) -> Artifact:
    p = Path(local_path)
    return Artifact(
        id=f"art-{uuid.uuid4().hex[:8]}",
        run_id="run-test",
        remote_path=local_path,
        local_path=local_path,
        type=atype,
        size=p.stat().st_size if p.exists() else 0,
        checksum="sha256:abc",
        preview={},
        created_at=_now(),
    )


def _make_criterion(
    kind: str,
    check: dict,
    status: str = "confirmed",
    strength: int = 2,
    text: str = "test criterion",
) -> Criterion:
    return Criterion(
        id=f"crit-{uuid.uuid4().hex[:8]}",
        text=text,
        kind=kind,
        check=check,
        status=status,
        strength=strength,
        evidence_run_ids=[],
    )


def _make_contract(criteria: list[Criterion], jobfile_id: str = "jf-test") -> ExpectationContract:
    return ExpectationContract(
        id=f"ec-{uuid.uuid4().hex[:8]}",
        jobfile_id=jobfile_id,
        version=1,
        criteria=criteria,
        source="test",
        created_at=_now(),
        updated_at=_now(),
    )


@pytest.fixture
def tmp_store(tmp_path: Path) -> Store:
    s = Store(str(tmp_path / "test.db"))
    s.init_schema()
    return s


class FakeAnalyzer:
    """Fake analyzer that returns canned criteria for testing."""

    def propose_criteria(self, feedback: dict, history: list, jobfile: dict) -> list[dict]:
        return [
            {
                "id": "proposed-aabbccdd",
                "text": "No NaN values in logs",
                "kind": "absence",
                "check": {"pattern": "NaN", "targets": ["stdout"]},
                "status": "proposed",
                "strength": 1,
                "evidence_run_ids": [],
            },
            {
                "id": "proposed-11223344",
                "text": "Output CSV exists",
                "kind": "presence",
                "check": {"glob": "*.csv"},
                "status": "proposed",
                "strength": 1,
                "evidence_run_ids": [],
            },
        ]


# ===========================================================================
# evaluate() tests — exit code rules
# ===========================================================================

class TestEvaluateExitCode:

    def test_nonzero_exit_code_gives_failed(self, tmp_path: Path) -> None:
        """Non-zero exit code -> Match.FAILED regardless of criteria."""
        run = _make_run(exit_code=1, state=State.FAILED)
        contract = _make_contract([])
        match, evidence, per_criterion = evaluate(contract, run, [], "", "")
        assert match == Match.FAILED
        assert any("exit" in e.lower() for e in evidence)

    def test_zero_exit_code_no_criteria_usable(self, tmp_path: Path) -> None:
        """Zero exit code, no criteria -> usable (nothing to fail)."""
        run = _make_run(exit_code=0)
        contract = _make_contract([])
        match, evidence, per_criterion = evaluate(contract, run, [], "", "")
        assert match == Match.USABLE
        assert per_criterion == []

    def test_backend_failure_state_gives_failed(self) -> None:
        """FAILED state with None exit_code still -> Match.FAILED."""
        run = _make_run(exit_code=None, state=State.FAILED)
        contract = _make_contract([])
        match, evidence, per_criterion = evaluate(contract, run, [], "", "")
        assert match == Match.FAILED

    def test_backend_failure_with_zero_exit_code_reports_state_not_success(self) -> None:
        """Scheduler-level failures can have ExitCode=0:0; evidence must not imply success."""
        run = _make_run(exit_code=0, state=State.FAILED)
        contract = _make_contract([])
        match, evidence, per_criterion = evaluate(contract, run, [], "", "")
        assert match == Match.FAILED
        assert "Run state=failed" in evidence
        assert "Run exit_code=0" not in evidence


# ===========================================================================
# evaluate() tests — absence criteria
# ===========================================================================

class TestEvaluateAbsence:

    def test_absence_no_pattern_in_stdout_passes(self) -> None:
        """Pattern not found -> absence criterion passes."""
        crit = _make_criterion("absence", {"pattern": "NaN", "targets": ["stdout"]})
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(
            contract, run, [], "normal output\nall fine", ""
        )
        assert match == Match.USABLE
        assert per_criterion[0]["passed"] is True

    def test_absence_pattern_found_in_stdout_gives_bad_signal(self) -> None:
        """Pattern found in stdout -> absence fails -> BAD_SIGNAL."""
        crit = _make_criterion("absence", {"pattern": "NaN", "targets": ["stdout"]})
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(
            contract, run, [], "step 1\nvalue=NaN detected\nstep 2", ""
        )
        assert match == Match.BAD_SIGNAL
        assert per_criterion[0]["passed"] is False
        assert any("NaN" in e for e in evidence)

    def test_absence_cuda_error_in_stderr_gives_bad_signal(self, tmp_path: Path) -> None:
        """CUDA error in stderr -> absence fails."""
        crit = _make_criterion("absence", {"pattern": "CUDA error", "targets": ["stderr"]})
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(
            contract, run, [], "", "CUDA error: device not found"
        )
        assert match == Match.BAD_SIGNAL
        assert per_criterion[0]["passed"] is False

    def test_absence_traceback_in_file_artifact(self, tmp_path: Path) -> None:
        """Absence check against a log file artifact."""
        log_file = tmp_path / "output.log"
        log_file.write_text("Starting job...\nTraceback (most recent call last):\n  File ...\n")
        art = _make_artifact(str(log_file), ArtifactType.TEXT_LOG)
        crit = _make_criterion(
            "absence",
            {"pattern": "Traceback", "targets": ["*.log"]},
        )
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(contract, run, [art], "", "")
        assert match == Match.BAD_SIGNAL
        assert per_criterion[0]["passed"] is False

    def test_absence_no_match_in_file_passes(self, tmp_path: Path) -> None:
        """Absence check against log file with no pattern -> passes."""
        log_file = tmp_path / "output.log"
        log_file.write_text("step 1 done\nstep 2 done\nfinished ok\n")
        art = _make_artifact(str(log_file), ArtifactType.TEXT_LOG)
        crit = _make_criterion("absence", {"pattern": "NaN", "targets": ["*.log"]})
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(contract, run, [art], "", "")
        assert match == Match.USABLE
        assert per_criterion[0]["passed"] is True

    def test_absence_targets_stdout_and_stderr(self) -> None:
        """Absence checks both stdout and stderr."""
        crit = _make_criterion(
            "absence",
            {"pattern": "NaN", "targets": ["stdout", "stderr"]},
        )
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        # Pattern only in stderr
        match, evidence, per_criterion = evaluate(
            contract, run, [], "clean stdout", "warning NaN found in computation"
        )
        assert match == Match.BAD_SIGNAL
        assert per_criterion[0]["passed"] is False

    def test_numeric_nan_absence_ignores_package_names(self) -> None:
        """The default numeric-NaN check should not flag packages like NaNMath."""
        crit = _make_criterion(
            "absence",
            {
                "pattern": NUMERIC_NAN_PATTERN,
                "regex": True,
                "flags": ["ignorecase"],
                "label": "numeric NaN",
                "targets": ["stderr"],
            },
        )
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        stderr = "[77ba4419] + NaNMath v1.1.4\nPrecompiling project..."
        match, evidence, per_criterion = evaluate(contract, run, [], "", stderr)
        assert match == Match.USABLE
        assert evidence == []
        assert per_criterion[0]["passed"] is True

    @pytest.mark.parametrize(
        "stderr",
        [
            "loss=NaN at iteration 12",
            "residual: -nan",
            "gradient [+NaN]",
        ],
    )
    def test_numeric_nan_absence_flags_numeric_nan_tokens(self, stderr: str) -> None:
        """The default numeric-NaN check still flags standalone NaN tokens."""
        crit = _make_criterion(
            "absence",
            {
                "pattern": NUMERIC_NAN_PATTERN,
                "regex": True,
                "flags": ["ignorecase"],
                "label": "numeric NaN",
                "targets": ["stderr"],
            },
        )
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(contract, run, [], "", stderr)
        assert match == Match.BAD_SIGNAL
        assert per_criterion[0]["passed"] is False
        assert "numeric NaN" in per_criterion[0]["detail"]


# ===========================================================================
# evaluate() tests — presence criteria
# ===========================================================================

class TestEvaluatePresence:

    def test_presence_glob_artifact_found_passes(self, tmp_path: Path) -> None:
        """Glob matches existing artifact -> passes."""
        csv_file = tmp_path / "energy.csv"
        csv_file.write_text("a,b\n1,2\n")
        art = _make_artifact(str(csv_file), ArtifactType.CSV)
        crit = _make_criterion("presence", {"glob": "*.csv"})
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(contract, run, [art], "", "")
        assert match == Match.USABLE
        assert per_criterion[0]["passed"] is True

    def test_presence_no_artifact_gives_inconclusive(self) -> None:
        """Presence check with no artifacts -> inconclusive."""
        crit = _make_criterion("presence", {"glob": "*.csv"})
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(contract, run, [], "", "")
        assert match == Match.INCONCLUSIVE
        assert per_criterion[0]["passed"] is False

    def test_presence_wrong_type_artifact_still_matches_glob(self, tmp_path: Path) -> None:
        """Presence by glob matches on path, not type."""
        png_file = tmp_path / "plot.png"
        png_file.write_bytes(b"\x89PNG")
        art = _make_artifact(str(png_file), ArtifactType.IMAGE)
        crit = _make_criterion("presence", {"glob": "*.png"})
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(contract, run, [art], "", "")
        assert match == Match.USABLE
        assert per_criterion[0]["passed"] is True


# ===========================================================================
# evaluate() tests — numeric criteria
# ===========================================================================

class TestEvaluateNumeric:

    def test_numeric_regex_from_stdout_passes(self) -> None:
        """Extract float via regex from stdout, compare < threshold -> passes."""
        crit = _make_criterion(
            "numeric",
            {
                "source": "stdout",
                "extract": r"energy=(-?\d+\.\d+)",
                "op": "lt",
                "value": -0.60,
            },
        )
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(
            contract, run, [], "step 100: energy=-0.662 converged", ""
        )
        assert match == Match.USABLE
        assert per_criterion[0]["passed"] is True
        assert any("-0.662" in e for e in evidence)

    def test_numeric_regex_from_stdout_fails(self) -> None:
        """Numeric fails when value doesn't meet threshold -> BAD_SIGNAL."""
        crit = _make_criterion(
            "numeric",
            {
                "source": "stdout",
                "extract": r"energy=(-?\d+\.\d+)",
                "op": "lt",
                "value": -0.70,
            },
        )
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(
            contract, run, [], "step 100: energy=-0.662 converged", ""
        )
        assert match == Match.BAD_SIGNAL
        assert per_criterion[0]["passed"] is False

    def test_numeric_near_threshold_weak_signal(self) -> None:
        """Numeric near threshold (within 5%) -> WEAK_SIGNAL."""
        crit = _make_criterion(
            "numeric",
            {
                "source": "stdout",
                "extract": r"energy=(-?\d+\.\d+)",
                "op": "lt",
                "value": -0.67,  # target: < -0.67
                "near_threshold_pct": 5,  # 5% near-threshold zone
            },
        )
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        # -0.662 fails the strict threshold but is within 5% of -0.67
        match, evidence, per_criterion = evaluate(
            contract, run, [], "energy=-0.662", ""
        )
        assert match == Match.WEAK_SIGNAL
        assert per_criterion[0]["passed"] is False  # strict pass = False

    def test_numeric_regex_no_match_inconclusive(self) -> None:
        """Regex finds nothing -> inconclusive (can't evaluate); passed is None."""
        crit = _make_criterion(
            "numeric",
            {
                "source": "stdout",
                "extract": r"energy=(-?\d+\.\d+)",
                "op": "lt",
                "value": -0.60,
            },
        )
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(
            contract, run, [], "no numeric values here", ""
        )
        assert match == Match.INCONCLUSIVE
        # passed=None means "could not extract value" (inconclusive, not a hard failure)
        assert per_criterion[0]["passed"] is None

    def test_numeric_csv_column(self, tmp_path: Path) -> None:
        """Extract last value from CSV column, compare with threshold."""
        csv_file = tmp_path / "results.csv"
        csv_file.write_text("step,energy\n1,-0.50\n2,-0.61\n3,-0.67\n")
        art = _make_artifact(str(csv_file), ArtifactType.CSV)
        crit = _make_criterion(
            "numeric",
            {
                "source": "*.csv",
                "csv_column": "energy",
                "csv_row": "last",
                "op": "lt",
                "value": -0.60,
            },
        )
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(contract, run, [art], "", "")
        assert match == Match.USABLE
        assert per_criterion[0]["passed"] is True

    def test_numeric_csv_column_fail(self, tmp_path: Path) -> None:
        """CSV column extraction fails threshold -> BAD_SIGNAL."""
        csv_file = tmp_path / "results.csv"
        csv_file.write_text("step,energy\n1,-0.50\n2,-0.55\n")
        art = _make_artifact(str(csv_file), ArtifactType.CSV)
        crit = _make_criterion(
            "numeric",
            {
                "source": "*.csv",
                "csv_column": "energy",
                "csv_row": "last",
                "op": "lt",
                "value": -0.60,
            },
        )
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(contract, run, [art], "", "")
        assert match == Match.BAD_SIGNAL

    def test_numeric_json_jsonpath(self, tmp_path: Path) -> None:
        """Extract numeric value from JSON via key path."""
        json_file = tmp_path / "metrics.json"
        json_file.write_text('{"final": {"energy": -0.675, "iter": 100}}')
        art = _make_artifact(str(json_file), ArtifactType.JSON)
        crit = _make_criterion(
            "numeric",
            {
                "source": "*.json",
                "json_path": "final.energy",
                "op": "lt",
                "value": -0.60,
            },
        )
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(contract, run, [art], "", "")
        assert match == Match.USABLE
        assert per_criterion[0]["passed"] is True

    def test_numeric_eq_op(self) -> None:
        """Numeric 'eq' operator works."""
        crit = _make_criterion(
            "numeric",
            {
                "source": "stdout",
                "extract": r"code=(\d+)",
                "op": "eq",
                "value": 0,
            },
        )
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(
            contract, run, [], "exit code=0", ""
        )
        assert match == Match.USABLE
        assert per_criterion[0]["passed"] is True

    def test_numeric_gt_op(self) -> None:
        """Numeric 'gt' (greater than) operator."""
        crit = _make_criterion(
            "numeric",
            {
                "source": "stdout",
                "extract": r"acc=(\d+\.\d+)",
                "op": "gt",
                "value": 0.9,
            },
        )
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(
            contract, run, [], "final acc=0.95", ""
        )
        assert match == Match.USABLE
        assert per_criterion[0]["passed"] is True


# ===========================================================================
# evaluate() tests — pattern criteria
# ===========================================================================

class TestEvaluatePattern:

    def test_pattern_matches_stdout_passes(self) -> None:
        """Regex pattern matches stdout -> passes."""
        crit = _make_criterion(
            "pattern",
            {"pattern": r"CONVERGED", "targets": ["stdout"]},
        )
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(
            contract, run, [], "iter 100: CONVERGED\n", ""
        )
        assert match == Match.USABLE
        assert per_criterion[0]["passed"] is True

    def test_pattern_no_match_inconclusive(self) -> None:
        """Pattern does not match -> inconclusive."""
        crit = _make_criterion(
            "pattern",
            {"pattern": r"CONVERGED", "targets": ["stdout"]},
        )
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(
            contract, run, [], "iter 100: still running", ""
        )
        assert match == Match.INCONCLUSIVE
        assert per_criterion[0]["passed"] is False


# ===========================================================================
# evaluate() tests — classification matrix (mixed criteria)
# ===========================================================================

class TestEvaluateClassification:

    def test_all_confirmed_pass_usable(self) -> None:
        """All confirmed criteria pass -> USABLE."""
        crits = [
            _make_criterion("absence", {"pattern": "NaN", "targets": ["stdout"]}, status="confirmed"),
            _make_criterion("presence", {"glob": "*.log"}, status="confirmed"),
        ]
        log_file_path = "/tmp/fake.log"
        # Create artifact for presence
        artifact = Artifact(
            id="art-1", run_id="run-1", remote_path=log_file_path,
            local_path=log_file_path, type=ArtifactType.TEXT_LOG,
            size=10, checksum="sha256:x", preview={}, created_at=_now(),
        )
        contract = _make_contract(crits)
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(
            contract, run, [artifact], "clean output", ""
        )
        assert match == Match.USABLE

    def test_absence_failure_dominates_over_presence_pass(self) -> None:
        """BAD_SIGNAL from absence dominates even if presence passes."""
        # Create a real temp file for the presence artifact
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            f.write(b"log content")
            log_path = f.name
        try:
            art = _make_artifact(log_path, ArtifactType.TEXT_LOG)
            crits = [
                _make_criterion("absence", {"pattern": "NaN", "targets": ["stdout"]}),
                _make_criterion("presence", {"glob": "*.log"}),
            ]
            contract = _make_contract(crits)
            run = _make_run(exit_code=0)
            match, evidence, per_criterion = evaluate(
                contract, run, [art], "output NaN found", ""
            )
            assert match == Match.BAD_SIGNAL
        finally:
            os.unlink(log_path)

    def test_proposed_criteria_dont_affect_usable(self) -> None:
        """Proposed (unconfirmed) criteria that fail -> WEAK_SIGNAL not BAD_SIGNAL."""
        # A confirmed absence passes, a proposed numeric fails -> should be weak or usable
        crits = [
            _make_criterion("absence", {"pattern": "NaN", "targets": ["stdout"]}, status="confirmed"),
            _make_criterion(
                "numeric",
                {"source": "stdout", "extract": r"val=(-?\d+\.\d+)", "op": "lt", "value": -1.0},
                status="proposed",  # not confirmed
            ),
        ]
        contract = _make_contract(crits)
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(
            contract, run, [], "val=-0.5 computed", ""
        )
        # Proposed numeric fails but is not confirmed -> weak_signal
        assert match == Match.WEAK_SIGNAL

    def test_all_inconclusive_criteria(self) -> None:
        """All criteria inconclusive -> INCONCLUSIVE."""
        crits = [
            _make_criterion("presence", {"glob": "*.csv"}),  # no csv artifact
            _make_criterion(
                "numeric",
                {"source": "stdout", "extract": r"energy=(-?\d+\.\d+)", "op": "lt", "value": -0.5},
            ),  # no match in stdout
        ]
        contract = _make_contract(crits)
        run = _make_run(exit_code=0)
        match, evidence, per_criterion = evaluate(contract, run, [], "no data here", "")
        assert match == Match.INCONCLUSIVE

    def test_weak_signal_near_threshold(self) -> None:
        """Near-threshold numeric with no other failures -> WEAK_SIGNAL."""
        crit = _make_criterion(
            "numeric",
            {
                "source": "stdout",
                "extract": r"acc=(-?\d+\.\d+)",
                "op": "gt",
                "value": 0.95,
                "near_threshold_pct": 5,
            },
        )
        contract = _make_contract([crit])
        run = _make_run(exit_code=0)
        # acc=0.93 fails strict (< 0.95) but within 5% of 0.95
        match, evidence, per_criterion = evaluate(
            contract, run, [], "accuracy: acc=0.93 achieved", ""
        )
        assert match == Match.WEAK_SIGNAL


# ===========================================================================
# default_contract() tests
# ===========================================================================

class TestDefaultContract:

    def test_default_contract_returns_expectation_contract(self) -> None:
        """default_contract returns an ExpectationContract instance."""
        jf = _make_jobfile()
        contract = default_contract(jf)
        assert isinstance(contract, ExpectationContract)
        assert contract.jobfile_id == jf.id

    def test_default_contract_has_absence_nan(self) -> None:
        """Default contract has an absence criterion for NaN."""
        jf = _make_jobfile()
        contract = default_contract(jf)
        kinds = [c.kind for c in contract.criteria]
        texts = [c.text for c in contract.criteria]
        assert "absence" in kinds
        # At least one criterion mentions NaN
        assert any("NaN" in t or "nan" in t.lower() for t in texts)
        nan_criteria = [
            c for c in contract.criteria
            if "NaN" in c.text or "nan" in c.text.lower()
        ]
        assert nan_criteria[0].check["regex"] is True
        assert nan_criteria[0].check["label"] == "numeric NaN"

    def test_default_contract_has_absence_traceback(self) -> None:
        """Default contract has an absence criterion for Traceback."""
        jf = _make_jobfile()
        contract = default_contract(jf)
        texts = [c.text for c in contract.criteria]
        assert any("Traceback" in t or "traceback" in t.lower() for t in texts)

    def test_default_contract_has_absence_cuda_error(self) -> None:
        """Default contract has an absence criterion for CUDA error."""
        jf = _make_jobfile()
        contract = default_contract(jf)
        texts = [c.text for c in contract.criteria]
        combined = " ".join(texts).lower()
        assert "cuda" in combined

    def test_default_contract_includes_expectation_text(self, tmp_path: Path) -> None:
        """JobFile with expectation text -> a criterion is seeded from it."""
        # Write a real YAML manifest so default_contract can read the expectation field
        yaml_file = tmp_path / "ipeps.jobfile.yaml"
        yaml_file.write_text(
            "name: ipeps-opt\n"
            "command: julia {script}\n"
            "expectation: 'energy converges below -0.66/site; no NaNs'\n"
        )
        from jobctl.jobfile import load_jobfile
        jf = load_jobfile(str(yaml_file))
        contract = default_contract(jf)
        # Should have at least one criterion mentioning the expectation
        assert len(contract.criteria) >= 1
        # The expectation should appear somewhere (either as text or in source)
        assert "energy" in contract.source.lower() or any(
            "energy" in c.text.lower() for c in contract.criteria
        )

    def test_default_contract_criteria_are_proposed(self) -> None:
        """Default contract criteria start with status=proposed."""
        jf = _make_jobfile()
        contract = default_contract(jf)
        for c in contract.criteria:
            assert c.status == "proposed"

    def test_default_contract_has_unique_criterion_ids(self) -> None:
        """All criterion IDs in the default contract are unique."""
        jf = _make_jobfile()
        contract = default_contract(jf)
        ids = [c.id for c in contract.criteria]
        assert len(ids) == len(set(ids))

    def test_default_contract_no_expectation_still_has_absence(self) -> None:
        """Even without expectation text, default criteria include NaN/Traceback/CUDA."""
        jf = _make_jobfile(expectation="")
        contract = default_contract(jf)
        assert len(contract.criteria) >= 3  # NaN + Traceback + CUDA at minimum


# ===========================================================================
# distiller.propose() tests
# ===========================================================================

class TestDistillerPropose:

    def test_propose_returns_list_of_criteria(self, tmp_store: Store) -> None:
        """propose() with fake analyzer returns Criterion list."""
        jf = _make_jobfile()
        tmp_store.add_jobfile(jf)

        # Add a contract so distiller can save to it
        contract = default_contract(jf)
        tmp_store.save_contract(contract)
        # Link jobfile to contract
        tmp_store.bump_version(jf.id, jf.content_hash)  # ensure it's in the db

        run = _make_run()
        run.jobfile_id = jf.id
        tmp_store.add_run(run)

        feedback = {"kind": "reject", "text": "too many NaN values", "run_id": run.run_id}
        analyzer = FakeAnalyzer()

        criteria = propose(tmp_store, run, feedback, analyzer)
        assert isinstance(criteria, list)
        assert len(criteria) >= 1
        assert all(isinstance(c, Criterion) for c in criteria)

    def test_propose_criteria_status_proposed(self, tmp_store: Store) -> None:
        """Proposed criteria have status=proposed and strength=1."""
        jf = _make_jobfile()
        tmp_store.add_jobfile(jf)
        contract = default_contract(jf)
        tmp_store.save_contract(contract)

        run = _make_run()
        run.jobfile_id = jf.id
        tmp_store.add_run(run)

        feedback = {"kind": "reject", "text": "NaN found", "run_id": run.run_id}
        analyzer = FakeAnalyzer()

        criteria = propose(tmp_store, run, feedback, analyzer)
        for c in criteria:
            assert c.status == "proposed"
            assert c.strength == 1

    def test_propose_persists_criteria_to_store(self, tmp_store: Store) -> None:
        """After propose(), the contract in store has the new criteria."""
        jf = _make_jobfile()
        tmp_store.add_jobfile(jf)
        contract = default_contract(jf)
        tmp_store.save_contract(contract)

        run = _make_run()
        run.jobfile_id = jf.id
        tmp_store.add_run(run)

        feedback = {"kind": "reject", "text": "NaN found", "run_id": run.run_id}
        analyzer = FakeAnalyzer()

        proposed_criteria = propose(tmp_store, run, feedback, analyzer)
        proposed_ids = {c.id for c in proposed_criteria}

        # Load the contract back
        saved_contract = tmp_store.get_contract(jf.id)
        assert saved_contract is not None
        saved_ids = {c.id for c in saved_contract.criteria}

        # All proposed IDs should now be in the stored contract
        for cid in proposed_ids:
            assert cid in saved_ids

    def test_propose_no_existing_contract_creates_one(self, tmp_store: Store) -> None:
        """If no contract exists, propose() creates and saves a new one."""
        jf = _make_jobfile()
        tmp_store.add_jobfile(jf)
        # No contract saved

        run = _make_run()
        run.jobfile_id = jf.id
        tmp_store.add_run(run)

        feedback = {"kind": "note", "text": "looks ok", "run_id": run.run_id}
        analyzer = FakeAnalyzer()

        criteria = propose(tmp_store, run, feedback, analyzer)
        assert len(criteria) >= 1

        saved = tmp_store.get_contract(jf.id)
        assert saved is not None

    def test_propose_uses_analyzer_canned_output(self, tmp_store: Store) -> None:
        """Criteria returned by propose match exactly what the analyzer returned."""
        jf = _make_jobfile()
        tmp_store.add_jobfile(jf)
        contract = default_contract(jf)
        tmp_store.save_contract(contract)

        run = _make_run()
        run.jobfile_id = jf.id
        tmp_store.add_run(run)

        feedback = {"kind": "reject", "text": "bad output", "run_id": run.run_id}
        analyzer = FakeAnalyzer()

        criteria = propose(tmp_store, run, feedback, analyzer)
        texts = [c.text for c in criteria]
        assert "No NaN values in logs" in texts
        assert "Output CSV exists" in texts


# ===========================================================================
# distiller.confirm() tests
# ===========================================================================

class TestDistillerConfirm:

    def _setup_contract_with_criteria(self, store: Store) -> tuple[Store, str, str]:
        """Helper: create jobfile + contract with 1 proposed criterion."""
        jf = _make_jobfile()
        store.add_jobfile(jf)

        criterion_id = f"crit-{uuid.uuid4().hex[:8]}"
        crit = Criterion(
            id=criterion_id,
            text="No NaN in logs",
            kind="absence",
            check={"pattern": "NaN", "targets": ["stdout"]},
            status="proposed",
            strength=1,
            evidence_run_ids=[],
        )
        contract = ExpectationContract(
            id=f"ec-{uuid.uuid4().hex[:8]}",
            jobfile_id=jf.id,
            version=1,
            criteria=[crit],
            source="test",
            created_at=_now(),
            updated_at=_now(),
        )
        store.save_contract(contract)
        return store, jf.id, criterion_id

    def test_confirm_returns_criterion(self, tmp_store: Store) -> None:
        """confirm() returns the updated Criterion object."""
        store, jf_id, criterion_id = self._setup_contract_with_criteria(tmp_store)
        result = confirm(store, criterion_id)
        assert isinstance(result, Criterion)
        assert result.id == criterion_id

    def test_confirm_sets_status_confirmed(self, tmp_store: Store) -> None:
        """confirm() changes status from proposed to confirmed."""
        store, jf_id, criterion_id = self._setup_contract_with_criteria(tmp_store)
        result = confirm(store, criterion_id)
        assert result.status == "confirmed"

    def test_confirm_increments_strength(self, tmp_store: Store) -> None:
        """confirm() increments strength by 1 (1 -> 2)."""
        store, jf_id, criterion_id = self._setup_contract_with_criteria(tmp_store)
        result = confirm(store, criterion_id)
        assert result.strength == 2

    def test_confirm_persists_to_store(self, tmp_store: Store) -> None:
        """After confirm(), the contract in the store is updated."""
        store, jf_id, criterion_id = self._setup_contract_with_criteria(tmp_store)
        confirm(store, criterion_id)

        # Reload contract and check the criterion
        contract = store.get_contract(jf_id)
        assert contract is not None
        found = next((c for c in contract.criteria if c.id == criterion_id), None)
        assert found is not None
        assert found.status == "confirmed"
        assert found.strength == 2

    def test_confirm_not_found_raises(self, tmp_store: Store) -> None:
        """confirm() raises ValueError if criterion_id not found."""
        with pytest.raises((ValueError, KeyError, LookupError)):
            confirm(tmp_store, "nonexistent-criterion-id")

    def test_confirm_twice_increments_strength_further(self, tmp_store: Store) -> None:
        """Calling confirm() again increments strength to 3."""
        store, jf_id, criterion_id = self._setup_contract_with_criteria(tmp_store)
        confirm(store, criterion_id)
        result2 = confirm(store, criterion_id)
        assert result2.strength == 3
        assert result2.status == "confirmed"


# ===========================================================================
# Smoke import test
# ===========================================================================

def test_smoke_import() -> None:
    """contracts and distiller modules import without errors."""
    from jobctl.expectations import contracts, distiller  # noqa: F401
    assert hasattr(contracts, "evaluate")
    assert hasattr(contracts, "default_contract")
    assert hasattr(distiller, "propose")
    assert hasattr(distiller, "confirm")
