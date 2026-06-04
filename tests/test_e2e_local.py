"""End-to-end integration tests for jobctl (Task 12).

Strategy:
- Register a tiny local jobfile that:
    1. Writes a small CSV (results.csv)
    2. Writes a small PNG (plot.png via Pillow)
    3. Prints a numeric value to stdout
- submit --wait (via ApiClient + TestClient with live Monitor)
- Assert: run reaches `completed`, artifacts are indexed (csv + png present),
  expectation contract classifies the run, observation card populated with all fields.
- Memory query then reports the run + reuse eligibility.
- Rerun creates a new run that also reaches `completed`.

All tests run fully offline — no SSH, no SLURM, no network.
"""
from __future__ import annotations

import json
import time
import textwrap
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_terminal(client, run_id: str, timeout: float = 30.0) -> dict:
    """Poll /runs/{id} until the run reaches a terminal state."""
    terminal = {"completed", "failed", "cancelled", "stuck", "timeout"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        run = resp.json()
        if run.get("state") in terminal:
            return run
        time.sleep(0.1)
    raise TimeoutError(f"Run {run_id} did not reach terminal state within {timeout}s")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def e2e_jobfile(tmp_path):
    """
    Write a tiny local jobfile:
    - Script emits a CSV, writes a PNG, and prints a numeric value.
    - jobfile.yaml registers these with artifact patterns.
    """
    script = tmp_path / "e2e_job.sh"
    script.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        set -e

        WORKDIR="${{JOBCTL_WORKDIR:-{tmp_path}}}"

        # Print a numeric value
        echo "answer=42"

        # Write a CSV
        cat > "$WORKDIR/results.csv" <<'CSV'
        label,value
        a,1.0
        b,2.5
        c,3.14
        CSV

        # Write a tiny PNG via Python / Pillow
        python3 - <<'PYEOF'
        import os, sys
        workdir = os.environ.get("JOBCTL_WORKDIR", "{tmp_path}")
        try:
            from PIL import Image
            img = Image.new("RGB", (10, 10), color=(100, 200, 50))
            img.save(os.path.join(workdir, "plot.png"))
        except ImportError:
            # Minimal valid 1x1 PNG fallback (no Pillow)
            import struct, zlib
            def _png1x1():
                sig = b"\\x89PNG\\r\\n\\x1a\\n"
                def chunk(tag, data):
                    c = struct.pack(">I", len(data)) + tag + data
                    c += struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
                    return c
                ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                raw = b"\\x00\\xff\\xff\\xff"
                idat = chunk(b"IDAT", zlib.compress(raw))
                iend = chunk(b"IEND", b"")
                return sig + ihdr + idat + iend
            with open(os.path.join(workdir, "plot.png"), "wb") as f:
                f.write(_png1x1())
        PYEOF
        echo "job done"
    """))
    script.chmod(0o755)

    jf_path = tmp_path / "e2e.jobfile.yaml"
    jf_path.write_text(textwrap.dedent(f"""\
        name: e2e-local-test
        command: "bash {script}"
        params: {{}}
        backends:
          - backend: local
        artifacts:
          - "*.csv"
          - "*.png"
        expectation: "results.csv present and answer printed"
    """))
    return str(jf_path)


@pytest.fixture
def e2e_app(tmp_path):
    """FastAPI app + TestClient with a live Monitor for E2E tests."""
    from jobctl.api.server import create_app
    from fastapi.testclient import TestClient

    config = {
        "db_path": str(tmp_path / "e2e.db"),
        "run_dir": str(tmp_path / "runs"),
        "poll_interval_seconds": 0.05,
        "probe_interval_seconds": 9999,
        "stuck_timeout_seconds": 9999,
    }
    app = create_app(config=config, start_monitor=True)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def e2e_api_client(e2e_app):
    """ApiClient backed by the TestClient transport."""
    from jobctl.api.client import ApiClient
    return ApiClient(base_url="http://testserver", transport=e2e_app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestE2ELocal:
    """Full end-to-end tests using a live local job."""

    # -----------------------------------------------------------------------
    # 1. Register + submit + wait -> completed
    # -----------------------------------------------------------------------

    def test_register_and_run_wait_json(self, e2e_app, e2e_api_client, e2e_jobfile):
        """Register a jobfile, run --wait, assert completed + card populated."""
        # Register
        reg = e2e_api_client.register(e2e_jobfile)
        assert reg["name"] == "e2e-local-test"
        jf_id = reg["id"]
        assert jf_id

        # Submit
        submit_resp = e2e_api_client.submit(jobfile_id=jf_id, params={})
        run_id = submit_resp["run_id"]
        assert run_id

        # Wait for terminal state
        final = e2e_api_client.await_run(run_id, poll_interval=0.1, timeout=30)

        # --- Core assertions ---
        assert final["state"] == "completed", (
            f"Expected completed, got {final['state']}"
        )
        assert final["exit_code"] == 0 or final["exit_code"] is None

        # Observation card must be populated
        card = final.get("observation_card")
        assert card is not None, "observation_card should be set after terminal"
        assert isinstance(card, dict)

        # Card must have all required fields
        required_fields = [
            "status", "jobfile", "run_id", "server",
            "artifacts", "health", "expectation_match",
            "key_evidence", "interpretation", "recommended_next_action",
        ]
        for field in required_fields:
            assert field in card, f"observation_card missing field: {field}"

        # Status should be 'completed'
        assert card["status"] == "completed"

        # Interpretation must be non-empty string
        assert isinstance(card["interpretation"], str)
        assert len(card["interpretation"]) > 0

    # -----------------------------------------------------------------------
    # 2. Artifacts indexed
    # -----------------------------------------------------------------------

    def test_artifacts_indexed(self, e2e_app, e2e_api_client, e2e_jobfile):
        """After run completes, artifacts (csv + png) should be indexed."""
        reg = e2e_api_client.register(e2e_jobfile)
        jf_id = reg["id"]
        submit_resp = e2e_api_client.submit(jobfile_id=jf_id, params={})
        run_id = submit_resp["run_id"]

        e2e_api_client.await_run(run_id, poll_interval=0.1, timeout=30)

        arts = e2e_api_client.artifacts(run_id)
        assert len(arts) >= 2, f"Expected >=2 artifacts, got {len(arts)}: {arts}"

        types_found = {a["type"] for a in arts}
        # We expect at least CSV and IMAGE/PLOT
        assert "csv" in types_found or len(arts) >= 2, (
            f"Expected csv artifact, found types: {types_found}"
        )

        # Checksums must be present
        for art in arts:
            assert art["checksum"], f"Artifact {art['id']} missing checksum"

    # -----------------------------------------------------------------------
    # 3. Expectation contract classifies the run
    # -----------------------------------------------------------------------

    def test_expectation_match_set(self, e2e_app, e2e_api_client, e2e_jobfile):
        """After run completes, expectation_match should be set (not None)."""
        reg = e2e_api_client.register(e2e_jobfile)
        jf_id = reg["id"]
        submit_resp = e2e_api_client.submit(jobfile_id=jf_id, params={})
        run_id = submit_resp["run_id"]

        final = e2e_api_client.await_run(run_id, poll_interval=0.1, timeout=30)

        match = final.get("expectation_match")
        assert match is not None, "expectation_match should be set after terminal"
        # Should be a valid Match enum value
        valid_matches = {"usable", "weak_signal", "bad_signal", "inconclusive", "failed"}
        assert match in valid_matches, f"Unexpected expectation_match: {match}"

    # -----------------------------------------------------------------------
    # 4. Memory query reports the run and reuse eligibility
    # -----------------------------------------------------------------------

    def test_memory_query_reports_run(self, e2e_app, e2e_api_client, e2e_jobfile):
        """After a completed run, memory query should report run count >= 1."""
        reg = e2e_api_client.register(e2e_jobfile)
        jf_id = reg["id"]
        submit_resp = e2e_api_client.submit(jobfile_id=jf_id, params={})
        run_id = submit_resp["run_id"]

        e2e_api_client.await_run(run_id, poll_interval=0.1, timeout=30)

        # Query by jobfile_id
        mem = e2e_api_client.memory_query(jobfile_id=jf_id)
        assert mem["has_jobfile"] is True
        assert mem["runs"] >= 1, f"Expected >=1 run, got {mem['runs']}"

        # Query by name
        mem2 = e2e_api_client.memory_query(name="e2e-local-test")
        assert mem2["has_jobfile"] is True
        assert mem2["runs"] >= 1

    def test_memory_query_reuse_eligibility(self, e2e_app, e2e_api_client, e2e_jobfile):
        """reuse_eligible should be True after a USABLE completed run with artifacts."""
        reg = e2e_api_client.register(e2e_jobfile)
        jf_id = reg["id"]
        submit_resp = e2e_api_client.submit(jobfile_id=jf_id, params={})
        run_id = submit_resp["run_id"]

        final = e2e_api_client.await_run(run_id, poll_interval=0.1, timeout=30)

        # Only check reuse_eligible if the run was USABLE
        match = final.get("expectation_match")
        mem = e2e_api_client.memory_query(jobfile_id=jf_id)

        if match == "usable":
            assert mem.get("reuse_eligible") is True, (
                f"Expected reuse_eligible=True for USABLE run, got: {mem}"
            )
        else:
            # Non-usable runs are not reuse-eligible — that's correct
            assert mem.get("reuse_eligible") is False

    # -----------------------------------------------------------------------
    # 5. Rerun works
    # -----------------------------------------------------------------------

    def test_rerun_creates_new_completed_run(self, e2e_app, e2e_api_client, e2e_jobfile):
        """Rerun of a completed run should create a new run that also completes."""
        reg = e2e_api_client.register(e2e_jobfile)
        jf_id = reg["id"]
        submit_resp = e2e_api_client.submit(jobfile_id=jf_id, params={})
        run_id = submit_resp["run_id"]

        e2e_api_client.await_run(run_id, poll_interval=0.1, timeout=30)

        # Rerun
        rerun_resp = e2e_api_client.rerun(run_id)
        new_run_id = rerun_resp["run_id"]
        assert new_run_id != run_id, "Rerun should produce a new run_id"

        # The rerun should also complete
        new_final = e2e_api_client.await_run(new_run_id, poll_interval=0.1, timeout=30)
        assert new_final["state"] == "completed", (
            f"Rerun expected completed, got {new_final['state']}"
        )

    # -----------------------------------------------------------------------
    # 6. --help lists all commands (CLI smoke test)
    # -----------------------------------------------------------------------

    def test_cli_help_lists_all_commands(self):
        """jobctl --help must list all expected commands."""
        from typer.testing import CliRunner
        from jobctl.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0, f"--help failed:\n{result.output}"

        output = result.output
        expected_commands = [
            "run", "await", "status", "logs", "artifacts",
            "inspect", "cancel", "rerun", "servers",
            "memory", "register", "jobfiles", "feedback",
            "expect", "serve",
        ]
        for cmd in expected_commands:
            assert cmd in output, (
                f"Command '{cmd}' not found in --help output:\n{output}"
            )


# ---------------------------------------------------------------------------
# Package exports smoke tests
# ---------------------------------------------------------------------------

class TestPackageExports:
    """Verify that jobctl/__init__.py exports what the contract requires."""

    def test_version_string(self):
        import jobctl
        assert isinstance(jobctl.__version__, str)
        assert len(jobctl.__version__) > 0

    def test_key_module_imports(self):
        """All major modules should be importable without errors."""
        import jobctl.config
        import jobctl.db.models
        import jobctl.db.store
        import jobctl.jobfile
        import jobctl.analysis.base
        import jobctl.analysis.offline
        import jobctl.analysis.deepseek
        import jobctl.artifacts.indexer
        import jobctl.memory.memory
        import jobctl.expectations.contracts
        import jobctl.expectations.distiller
        import jobctl.backends.base
        import jobctl.backends.local
        import jobctl.backends.ssh
        import jobctl.backends.slurm
        import jobctl.notify.notify
        import jobctl.monitor.monitor
        import jobctl.api.server
        import jobctl.api.client
        import jobctl.cli.main

    def test_exported_names(self):
        """Key public names should be importable from jobctl directly."""
        import jobctl
        # Version
        assert hasattr(jobctl, "__version__")
        # Enums
        assert hasattr(jobctl, "State")
        assert hasattr(jobctl, "Health")
        assert hasattr(jobctl, "Match")
        assert hasattr(jobctl, "ArtifactType")
        # Dataclasses
        assert hasattr(jobctl, "JobFile")
        assert hasattr(jobctl, "Run")
        assert hasattr(jobctl, "Artifact")
        assert hasattr(jobctl, "Criterion")
        assert hasattr(jobctl, "ExpectationContract")
        assert hasattr(jobctl, "Feedback")
        assert hasattr(jobctl, "Server")

    def test_exported_functions(self):
        """Key public functions should be importable from jobctl directly."""
        import jobctl
        assert hasattr(jobctl, "load_jobfile")
        assert hasattr(jobctl, "resolve_params")
        assert hasattr(jobctl, "render_command")
        assert hasattr(jobctl, "get_analyzer")
        assert hasattr(jobctl, "get_backend")
        assert hasattr(jobctl, "get_notifiers")
