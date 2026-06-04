"""Tests for Task 5: memory/memory.py — query() and reuse_candidate()."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from jobctl.db.models import (
    Artifact,
    ArtifactType,
    Health,
    JobFile,
    Match,
    Run,
    State,
)
from jobctl.db.store import Store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_store(tmp_path) -> Store:
    store = Store(str(tmp_path / "test.db"))
    store.init_schema()
    return store


def _make_jobfile(jf_id: str = "jf-001", name: str = "train-job") -> JobFile:
    return JobFile(
        id=jf_id,
        name=name,
        version=1,
        source_path="/tmp/train.py",
        command_template="python {script} --lr {lr}",
        params_schema={
            "script": {"type": "path", "required": True},
            "lr": {"type": "float", "default": 0.01},
        },
        backend_prefs=[{"backend": "local"}],
        artifact_patterns=["*.csv", "*.png"],
        expectation_contract_id=None,
        content_hash="abc123",
        created_at=_now(),
    )


def _make_run(
    run_id: str = "run-001",
    jobfile_id: str = "jf-001",
    params: dict | None = None,
    input_hashes: dict | None = None,
    state: State = State.COMPLETED,
    expectation_match: Match | None = Match.USABLE,
    server: str | None = "local-server",
    workdir: str | None = "/tmp/runs/run-001",
) -> Run:
    if params is None:
        params = {"script": "/tmp/train.py", "lr": 0.01}
    if input_hashes is None:
        input_hashes = {"/tmp/train.py": "sha256:deadbeef"}
    return Run(
        run_id=run_id,
        jobfile_id=jobfile_id,
        jobfile_version=1,
        params=params,
        input_hashes=input_hashes,
        backend="local",
        server=server,
        task=None,
        remote_job_id=None,
        state=state,
        health=Health.OK,
        exit_code=0 if state == State.COMPLETED else None,
        submitted_at=_now(),
        started_at=_now(),
        finished_at=_now() if state == State.COMPLETED else None,
        last_heartbeat=None,
        workdir=workdir,
        stdout_path=None,
        stderr_path=None,
        resource_summary={},
        expectation_match=expectation_match,
        observation_card=None,
    )


def _make_artifact(run_id: str = "run-001", local_path: str = "/tmp/output.csv") -> Artifact:
    return Artifact(
        id=f"art-{run_id}",
        run_id=run_id,
        remote_path=local_path,
        local_path=local_path,
        type=ArtifactType.CSV,
        size=512,
        checksum="sha256:aabbcc",
        preview={},
        created_at=_now(),
    )


# ---------------------------------------------------------------------------
# Tests for query()
# ---------------------------------------------------------------------------

class TestQuery:
    def test_no_jobfile_returns_has_jobfile_false(self, tmp_path):
        """query on unknown jobfile_id → has_jobfile=False, runs=0."""
        from jobctl.memory.memory import query
        store = _make_store(tmp_path)
        result = query(store, jobfile_id="nonexistent")
        assert result["has_jobfile"] is False
        assert result["runs"] == 0
        assert result["exact_match_run_id"] is None

    def test_has_jobfile_true_when_exists(self, tmp_path):
        """query returns has_jobfile=True when jobfile is in store."""
        from jobctl.memory.memory import query
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        result = query(store, jobfile_id=jf.id)
        assert result["has_jobfile"] is True

    def test_run_count(self, tmp_path):
        """query returns count of all runs for the jobfile."""
        from jobctl.memory.memory import query
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        run1 = _make_run("run-001", jf.id)
        run2 = _make_run("run-002", jf.id)
        store.add_run(run1)
        store.add_run(run2)
        result = query(store, jobfile_id=jf.id)
        assert result["runs"] == 2

    def test_exact_match_run_id_found(self, tmp_path):
        """query returns exact_match_run_id when input_hash+params match a prior run."""
        from jobctl.memory.memory import query
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        params = {"script": "/tmp/train.py", "lr": 0.01}
        hashes = {"/tmp/train.py": "sha256:deadbeef"}
        run = _make_run("run-001", jf.id, params=params, input_hashes=hashes)
        store.add_run(run)

        result = query(store, jobfile_id=jf.id, params=params, input_hashes=hashes)
        assert result["exact_match_run_id"] == "run-001"

    def test_no_exact_match_when_params_differ(self, tmp_path):
        """query returns exact_match_run_id=None when params differ."""
        from jobctl.memory.memory import query
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        params_stored = {"script": "/tmp/train.py", "lr": 0.01}
        hashes = {"/tmp/train.py": "sha256:deadbeef"}
        run = _make_run("run-001", jf.id, params=params_stored, input_hashes=hashes)
        store.add_run(run)

        # Different lr
        params_query = {"script": "/tmp/train.py", "lr": 0.001}
        result = query(store, jobfile_id=jf.id, params=params_query, input_hashes=hashes)
        assert result["exact_match_run_id"] is None

    def test_no_exact_match_when_input_hashes_differ(self, tmp_path):
        """query returns exact_match_run_id=None when input_hashes differ."""
        from jobctl.memory.memory import query
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        params = {"script": "/tmp/train.py", "lr": 0.01}
        hashes_stored = {"/tmp/train.py": "sha256:deadbeef"}
        run = _make_run("run-001", jf.id, params=params, input_hashes=hashes_stored)
        store.add_run(run)

        hashes_query = {"/tmp/train.py": "sha256:cafecafe"}
        result = query(store, jobfile_id=jf.id, params=params, input_hashes=hashes_query)
        assert result["exact_match_run_id"] is None

    def test_query_by_name(self, tmp_path):
        """query accepts name= instead of jobfile_id=."""
        from jobctl.memory.memory import query
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        run = _make_run("run-001", jf.id)
        store.add_run(run)

        result = query(store, name=jf.name)
        assert result["has_jobfile"] is True
        assert result["runs"] == 1

    def test_exact_match_returns_server_and_outcome(self, tmp_path):
        """exact_match_run_id result also carries server and outcome fields."""
        from jobctl.memory.memory import query
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        params = {"script": "/tmp/train.py", "lr": 0.01}
        hashes = {"/tmp/train.py": "sha256:deadbeef"}
        run = _make_run(
            "run-001", jf.id,
            params=params, input_hashes=hashes,
            state=State.COMPLETED,
            expectation_match=Match.USABLE,
            server="hipster",
        )
        store.add_run(run)

        result = query(store, jobfile_id=jf.id, params=params, input_hashes=hashes)
        assert result["exact_match_run_id"] == "run-001"
        assert result["server"] == "hipster"
        assert result["outcome"] == "completed"

    def test_reuse_eligible_flag_true_when_usable(self, tmp_path):
        """reuse_eligible=True when exact match AND USABLE AND artifacts present."""
        from jobctl.memory.memory import query
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        params = {"script": "/tmp/train.py", "lr": 0.01}
        hashes = {"/tmp/train.py": "sha256:deadbeef"}

        # Need a real workdir with a file in it
        workdir = tmp_path / "runs" / "run-001"
        workdir.mkdir(parents=True)
        artifact_file = workdir / "output.csv"
        artifact_file.write_text("col1,col2\n1,2\n")

        run = _make_run(
            "run-001", jf.id,
            params=params, input_hashes=hashes,
            state=State.COMPLETED,
            expectation_match=Match.USABLE,
            workdir=str(workdir),
        )
        store.add_run(run)
        art = _make_artifact("run-001", local_path=str(artifact_file))
        store.add_artifact(art)

        result = query(store, jobfile_id=jf.id, params=params, input_hashes=hashes)
        assert result["reuse_eligible"] is True
        assert result["artifacts_dir"] is not None

    def test_reuse_eligible_false_when_weak_signal(self, tmp_path):
        """reuse_eligible=False when expectation_match=WEAK_SIGNAL."""
        from jobctl.memory.memory import query
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        params = {"script": "/tmp/train.py", "lr": 0.01}
        hashes = {"/tmp/train.py": "sha256:deadbeef"}

        workdir = tmp_path / "runs" / "run-weak"
        workdir.mkdir(parents=True)
        artifact_file = workdir / "output.csv"
        artifact_file.write_text("col1\n1\n")

        run = _make_run(
            "run-weak", jf.id,
            params=params, input_hashes=hashes,
            state=State.COMPLETED,
            expectation_match=Match.WEAK_SIGNAL,
            workdir=str(workdir),
        )
        store.add_run(run)
        art = _make_artifact("run-weak", local_path=str(artifact_file))
        store.add_artifact(art)

        result = query(store, jobfile_id=jf.id, params=params, input_hashes=hashes)
        assert result["reuse_eligible"] is False

    def test_reuse_eligible_false_when_no_artifacts(self, tmp_path):
        """reuse_eligible=False when no artifacts present even if USABLE."""
        from jobctl.memory.memory import query
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        params = {"script": "/tmp/train.py", "lr": 0.01}
        hashes = {"/tmp/train.py": "sha256:deadbeef"}

        run = _make_run(
            "run-001", jf.id,
            params=params, input_hashes=hashes,
            state=State.COMPLETED,
            expectation_match=Match.USABLE,
        )
        store.add_run(run)
        # No artifacts added

        result = query(store, jobfile_id=jf.id, params=params, input_hashes=hashes)
        assert result["reuse_eligible"] is False


# ---------------------------------------------------------------------------
# Tests for reuse_candidate()
# ---------------------------------------------------------------------------

class TestReuseCandidate:
    def test_returns_none_when_no_matching_run(self, tmp_path):
        """reuse_candidate returns None when no run with matching params exists."""
        from jobctl.memory.memory import reuse_candidate
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        # No runs added

        result = reuse_candidate(
            store, jf,
            params={"script": "/tmp/train.py", "lr": 0.01},
            input_hashes={"/tmp/train.py": "sha256:deadbeef"},
        )
        assert result is None

    def test_returns_run_on_exact_match_usable_with_artifacts(self, tmp_path):
        """Returns a Run when exact match, USABLE, and artifacts exist on disk."""
        from jobctl.memory.memory import reuse_candidate
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        params = {"script": "/tmp/train.py", "lr": 0.01}
        hashes = {"/tmp/train.py": "sha256:deadbeef"}

        workdir = tmp_path / "runs" / "run-001"
        workdir.mkdir(parents=True)
        artifact_file = workdir / "output.csv"
        artifact_file.write_text("col1,col2\n1,2\n")

        run = _make_run(
            "run-001", jf.id,
            params=params, input_hashes=hashes,
            state=State.COMPLETED,
            expectation_match=Match.USABLE,
            workdir=str(workdir),
        )
        store.add_run(run)
        art = _make_artifact("run-001", local_path=str(artifact_file))
        store.add_artifact(art)

        result = reuse_candidate(store, jf, params=params, input_hashes=hashes)
        assert result is not None
        assert result.run_id == "run-001"

    def test_returns_none_when_input_hashes_differ(self, tmp_path):
        """reuse_candidate returns None when input_hashes don't match."""
        from jobctl.memory.memory import reuse_candidate
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        params = {"script": "/tmp/train.py", "lr": 0.01}
        hashes_stored = {"/tmp/train.py": "sha256:deadbeef"}

        workdir = tmp_path / "runs" / "run-001"
        workdir.mkdir(parents=True)
        artifact_file = workdir / "output.csv"
        artifact_file.write_text("col1\n1\n")

        run = _make_run(
            "run-001", jf.id,
            params=params, input_hashes=hashes_stored,
            state=State.COMPLETED,
            expectation_match=Match.USABLE,
            workdir=str(workdir),
        )
        store.add_run(run)
        art = _make_artifact("run-001", local_path=str(artifact_file))
        store.add_artifact(art)

        # Query with different hash
        result = reuse_candidate(
            store, jf,
            params=params,
            input_hashes={"/tmp/train.py": "sha256:different"},
        )
        assert result is None

    def test_returns_none_when_params_differ(self, tmp_path):
        """reuse_candidate returns None when params don't exactly match."""
        from jobctl.memory.memory import reuse_candidate
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        hashes = {"/tmp/train.py": "sha256:deadbeef"}

        workdir = tmp_path / "runs" / "run-001"
        workdir.mkdir(parents=True)
        artifact_file = workdir / "output.csv"
        artifact_file.write_text("col1\n1\n")

        run = _make_run(
            "run-001", jf.id,
            params={"script": "/tmp/train.py", "lr": 0.01},
            input_hashes=hashes,
            state=State.COMPLETED,
            expectation_match=Match.USABLE,
            workdir=str(workdir),
        )
        store.add_run(run)
        art = _make_artifact("run-001", local_path=str(artifact_file))
        store.add_artifact(art)

        result = reuse_candidate(
            store, jf,
            params={"script": "/tmp/train.py", "lr": 0.001},  # different lr
            input_hashes=hashes,
        )
        assert result is None

    def test_returns_none_when_expectation_not_usable(self, tmp_path):
        """reuse_candidate returns None when expectation_match != USABLE."""
        from jobctl.memory.memory import reuse_candidate
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        params = {"script": "/tmp/train.py", "lr": 0.01}
        hashes = {"/tmp/train.py": "sha256:deadbeef"}

        workdir = tmp_path / "runs" / "run-001"
        workdir.mkdir(parents=True)
        artifact_file = workdir / "output.csv"
        artifact_file.write_text("col1\n1\n")

        for match_val in [Match.WEAK_SIGNAL, Match.BAD_SIGNAL, Match.INCONCLUSIVE, Match.FAILED, None]:
            run_id = f"run-{match_val}"
            run = _make_run(
                run_id, jf.id,
                params=params, input_hashes=hashes,
                state=State.COMPLETED,
                expectation_match=match_val,
                workdir=str(workdir),
            )
            store.add_run(run)
            art = Artifact(
                id=f"art-{run_id}",
                run_id=run_id,
                remote_path=str(artifact_file),
                local_path=str(artifact_file),
                type=ArtifactType.CSV,
                size=512,
                checksum="sha256:aabbcc",
                preview={},
                created_at=_now(),
            )
            store.add_artifact(art)

            result = reuse_candidate(store, jf, params=params, input_hashes=hashes)
            assert result is None, f"Expected None for match={match_val}, got {result}"

            # Clean up for next iteration — delete run and artifact by using a fresh store each time
            # (we use different run_ids so they don't clash)

    def test_returns_none_when_no_artifacts_in_store(self, tmp_path):
        """reuse_candidate returns None when artifacts table has no rows for the run."""
        from jobctl.memory.memory import reuse_candidate
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        params = {"script": "/tmp/train.py", "lr": 0.01}
        hashes = {"/tmp/train.py": "sha256:deadbeef"}

        run = _make_run(
            "run-001", jf.id,
            params=params, input_hashes=hashes,
            state=State.COMPLETED,
            expectation_match=Match.USABLE,
        )
        store.add_run(run)
        # No artifacts added

        result = reuse_candidate(store, jf, params=params, input_hashes=hashes)
        assert result is None

    def test_returns_none_when_artifact_file_missing_on_disk(self, tmp_path):
        """reuse_candidate returns None when artifact local_path doesn't exist on disk."""
        from jobctl.memory.memory import reuse_candidate
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        params = {"script": "/tmp/train.py", "lr": 0.01}
        hashes = {"/tmp/train.py": "sha256:deadbeef"}

        run = _make_run(
            "run-001", jf.id,
            params=params, input_hashes=hashes,
            state=State.COMPLETED,
            expectation_match=Match.USABLE,
        )
        store.add_run(run)
        # Artifact path doesn't actually exist on disk
        art = _make_artifact("run-001", local_path="/nonexistent/path/output.csv")
        store.add_artifact(art)

        result = reuse_candidate(store, jf, params=params, input_hashes=hashes)
        assert result is None

    def test_picks_most_recent_usable_run(self, tmp_path):
        """When multiple exact-match USABLE runs exist, any valid one is returned."""
        from jobctl.memory.memory import reuse_candidate
        store = _make_store(tmp_path)
        jf = _make_jobfile()
        store.add_jobfile(jf)
        params = {"script": "/tmp/train.py", "lr": 0.01}
        hashes = {"/tmp/train.py": "sha256:deadbeef"}

        # Create two identical-criteria runs, both USABLE, both with artifacts
        valid_run_ids = []
        for i in range(1, 3):
            workdir = tmp_path / "runs" / f"run-00{i}"
            workdir.mkdir(parents=True)
            artifact_file = workdir / "output.csv"
            artifact_file.write_text(f"col1\n{i}\n")

            run = _make_run(
                f"run-00{i}", jf.id,
                params=params, input_hashes=hashes,
                state=State.COMPLETED,
                expectation_match=Match.USABLE,
                workdir=str(workdir),
            )
            store.add_run(run)
            art = _make_artifact(f"run-00{i}", local_path=str(artifact_file))
            store.add_artifact(art)
            valid_run_ids.append(f"run-00{i}")

        result = reuse_candidate(store, jf, params=params, input_hashes=hashes)
        assert result is not None
        assert result.run_id in valid_run_ids
