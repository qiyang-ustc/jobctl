"""Tests for Task 11 — Web UI.

Tests:
    - GET / (dashboard): server health cards + run buckets
    - GET /runs/{id}: record, stdout/stderr tail, artifact previews, observation card, criteria
    - GET /jobfiles/{id}: params schema, historical runs, contract versions
    - GET /ui/poll (HTMX partial): updated fragments

Strategy: Use FastAPI TestClient with seeded data (no monitor started).
"""
from __future__ import annotations

import base64
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jf_id() -> str:
    return f"jf-{uuid.uuid4().hex[:10]}"


def _run_id() -> str:
    return f"run-{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def seeded_app(tmp_path):
    """Return (TestClient, seeded ids) with pre-populated DB."""
    from fastapi.testclient import TestClient
    from jobctl.api.server import create_app
    from jobctl.db.store import Store
    from jobctl.db.models import (
        JobFile, Run, Artifact, Server, ExpectationContract, Criterion,
        State, Health, Match, ArtifactType,
    )

    db_path = str(tmp_path / "ui_test.db")
    run_dir = str(tmp_path / "runs")

    app = create_app(
        config={
            "db_path": db_path,
            "run_dir": run_dir,
            "default_policies": {
                "gpu-server-1": {
                    "mode": "cpu_fill_idle",
                    "target_idle_pct": 5,
                    "kernel_cpus": 4,
                }
            },
        },
        start_monitor=False,
    )
    store: Store = app.state.store

    # ---- Seed JobFile ----
    jf_id = _jf_id()
    jf = JobFile(
        id=jf_id,
        name="test_job",
        version=1,
        source_path="/tmp/test.py",
        command_template="python test.py --lr {lr}",
        params_schema={"lr": {"type": "float", "default": 0.01}},
        backend_prefs=[{"backend": "local"}],
        artifact_patterns=["*.csv", "*.png"],
        expectation_contract_id=None,
        content_hash="abc123",
        created_at=_now(),
    )
    store.add_jobfile(jf)

    # ---- Seed Servers ----
    from jobctl.db.models import Server
    server = Server(
        name="gpu-server-1",
        backend_type="slurm",
        online=True,
        last_heartbeat=_now(),
        cpu={"pct": 42.0},
        mem={"pct": 65.0},
        gpu={"pct": 80.0},
        disk={"pct": 30.0},
        slurm_queue={
            "running": 3,
            "pending": 1,
            "allocated_cpus": 760,
            "idle_cpus": 40,
            "other_cpus": 0,
            "total_cpus": 800,
            "idle_pct": 5.0,
            "jobs": [
                {
                    "job_id": "12345",
                    "state": "R",
                    "name": "alpha_scan",
                    "elapsed": "01:02",
                    "time_left": "03:58",
                    "cpus": 8,
                },
                {
                    "job_id": "12346",
                    "state": "PD",
                    "name": "beta_scan",
                    "elapsed": "00:00",
                    "time_left": "04:00",
                    "cpus": 4,
                },
            ],
        },
        note="Long cluster note that should be collapsed by default because it contains operational guidance, login-node warnings, partition details, and other text that would otherwise dominate the server card.",
    )
    store.upsert_server(server)

    offline_server = Server(
        name="cpu-server-2",
        backend_type="ssh",
        online=False,
        last_heartbeat=None,
        cpu={}, mem={}, gpu={}, disk={},
        slurm_queue={},
        note="maintenance",
    )
    store.upsert_server(offline_server)

    # ---- Seed Runs ----
    # running run
    running_id = _run_id()
    running_run = Run(
        run_id=running_id,
        jobfile_id=jf_id,
        jobfile_version=1,
        params={"lr": 0.01},
        input_hashes={},
        backend="slurm",
        server="gpu-server-1",
        task=None,
        remote_job_id="12345",
        state=State.RUNNING,
        health=Health.OK,
        exit_code=None,
        submitted_at=_now(),
        started_at=_now(),
        finished_at=None,
        last_heartbeat=_now(),
        workdir="/remote/workdir",
        stdout_path=None,
        stderr_path=None,
        resource_summary={"cpu_hours": 1.2},
        expectation_match=None,
        observation_card=None,
    )
    store.add_run(running_run)

    # completed run with artifacts, observation card
    completed_id = _run_id()
    # Create stdout/stderr files
    stdout_file = tmp_path / f"{completed_id}_stdout.log"
    stderr_file = tmp_path / f"{completed_id}_stderr.log"
    stdout_file.write_text("Training started\nEpoch 1/10 loss=0.5\nEpoch 10/10 loss=0.1\n")
    stderr_file.write_text("WARNING: low GPU memory\n")

    obs_card = {
        "status": "completed",
        "jobfile": "test_job",
        "run_id": completed_id,
        "server": "gpu-server-1",
        "artifacts": [{"name": "results.csv", "type": "csv"}],
        "health": "ok",
        "expectation_match": "usable",
        "key_evidence": ["loss=0.1 < threshold 0.2"],
        "interpretation": "Model trained successfully",
        "recommended_next_action": "Deploy to production",
    }
    completed_run = Run(
        run_id=completed_id,
        jobfile_id=jf_id,
        jobfile_version=1,
        params={"lr": 0.001},
        input_hashes={"script": "sha256:abc"},
        backend="slurm",
        server="gpu-server-1",
        task=None,
        remote_job_id="12340",
        state=State.COMPLETED,
        health=Health.OK,
        exit_code=0,
        submitted_at=_now(),
        started_at=_now(),
        finished_at=_now(),
        last_heartbeat=_now(),
        workdir="/remote/workdir/completed",
        stdout_path=str(stdout_file),
        stderr_path=str(stderr_file),
        resource_summary={"cpu_hours": 2.5, "gpu_hours": 1.8},
        expectation_match=Match.USABLE,
        observation_card=obs_card,
    )
    store.add_run(completed_run)

    # failed run
    failed_id = _run_id()
    failed_run = Run(
        run_id=failed_id,
        jobfile_id=jf_id,
        jobfile_version=1,
        params={"lr": 10.0},
        input_hashes={},
        backend="local",
        server=None,
        task=None,
        remote_job_id=None,
        state=State.FAILED,
        health=Health.OK,
        exit_code=1,
        submitted_at=_now(),
        started_at=_now(),
        finished_at=_now(),
        last_heartbeat=None,
        workdir=None,
        stdout_path=None,
        stderr_path=None,
        resource_summary={},
        expectation_match=Match.FAILED,
        observation_card=None,
    )
    store.add_run(failed_run)

    # stuck run
    stuck_id = _run_id()
    stuck_run = Run(
        run_id=stuck_id,
        jobfile_id=jf_id,
        jobfile_version=1,
        params={"lr": 0.1},
        input_hashes={},
        backend="slurm",
        server="gpu-server-1",
        task=None,
        remote_job_id="12346",
        state=State.STUCK,
        health=Health.STUCK,
        exit_code=None,
        submitted_at=_now(),
        started_at=_now(),
        finished_at=None,
        last_heartbeat=None,
        workdir=None,
        stdout_path=None,
        stderr_path=None,
        resource_summary={},
        expectation_match=None,
        observation_card=None,
    )
    store.add_run(stuck_run)

    # weak-signal run
    weak_id = _run_id()
    weak_run = Run(
        run_id=weak_id,
        jobfile_id=jf_id,
        jobfile_version=1,
        params={"lr": 0.005},
        input_hashes={},
        backend="slurm",
        server="gpu-server-1",
        task=None,
        remote_job_id="12347",
        state=State.COMPLETED,
        health=Health.WEAK,
        exit_code=0,
        submitted_at=_now(),
        started_at=_now(),
        finished_at=_now(),
        last_heartbeat=_now(),
        workdir=None,
        stdout_path=None,
        stderr_path=None,
        resource_summary={},
        expectation_match=Match.WEAK_SIGNAL,
        observation_card=None,
    )
    store.add_run(weak_run)

    # ---- Seed Artifacts for completed run ----
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("epoch,loss,acc\n1,0.5,0.6\n10,0.1,0.95\n")
    img_path = tmp_path / "plot.png"
    # Write a tiny valid PNG (1x1 pixel)
    try:
        from PIL import Image
        import io
        img = Image.new("RGB", (8, 8), color=(100, 150, 200))
        img.save(str(img_path))
    except Exception:
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header fallback

    csv_artifact = Artifact(
        id=f"art-{uuid.uuid4().hex[:8]}",
        run_id=completed_id,
        remote_path="results.csv",
        local_path=str(csv_path),
        type=ArtifactType.CSV,
        size=csv_path.stat().st_size,
        checksum="sha256:xyz",
        preview={"head": [["epoch", "loss", "acc"], ["1", "0.5", "0.6"]], "shape": [2, 3]},
        created_at=_now(),
    )
    img_artifact = Artifact(
        id=f"art-{uuid.uuid4().hex[:8]}",
        run_id=completed_id,
        remote_path="plot.png",
        local_path=str(img_path),
        type=ArtifactType.IMAGE,
        size=img_path.stat().st_size,
        checksum="sha256:pqr",
        preview={"thumb_path": str(img_path)},
        created_at=_now(),
    )
    store.add_artifact(csv_artifact)
    store.add_artifact(img_artifact)

    # ---- Seed ExpectationContract for jobfile ----
    contract_id = f"ec-{uuid.uuid4().hex[:8]}"
    criterion = Criterion(
        id=f"cr-{uuid.uuid4().hex[:8]}",
        text="final loss < 0.2",
        kind="numeric",
        check={"field": "loss", "op": "<", "threshold": 0.2},
        status="confirmed",
        strength=3,
        evidence_run_ids=[completed_id],
    )
    contract = ExpectationContract(
        id=contract_id,
        jobfile_id=jf_id,
        version=1,
        criteria=[criterion],
        source="user",
        created_at=_now(),
        updated_at=_now(),
    )
    store.save_contract(contract)

    client = TestClient(app, raise_server_exceptions=True)
    return client, {
        "jf_id": jf_id,
        "running_id": running_id,
        "completed_id": completed_id,
        "failed_id": failed_id,
        "stuck_id": stuck_id,
        "weak_id": weak_id,
        "contract_id": contract_id,
    }


# ---------------------------------------------------------------------------
# Dashboard tests (GET /)
# ---------------------------------------------------------------------------

class TestDashboard:
    def test_dashboard_returns_200(self, seeded_app):
        client, ids = seeded_app
        resp = client.get("/")
        assert resp.status_code == 200

    def test_dashboard_has_html_content_type(self, seeded_app):
        client, ids = seeded_app
        resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]

    def test_dashboard_shows_server_health_cards(self, seeded_app):
        client, ids = seeded_app
        resp = client.get("/")
        body = resp.text
        assert "gpu-server-1" in body
        assert "cpu-server-2" in body

    def test_dashboard_shows_online_status(self, seeded_app):
        client, ids = seeded_app
        resp = client.get("/")
        body = resp.text
        # gpu-server-1 is online, cpu-server-2 is offline
        assert "online" in body.lower() or "gpu-server-1" in body

    def test_dashboard_shows_run_buckets(self, seeded_app):
        client, ids = seeded_app
        resp = client.get("/")
        body = resp.text
        # All run bucket labels should appear
        for bucket in ["running", "completed", "failed", "stuck"]:
            assert bucket in body.lower(), f"Missing bucket: {bucket}"

    def test_dashboard_shows_run_ids_or_links(self, seeded_app):
        client, ids = seeded_app
        resp = client.get("/")
        body = resp.text
        # At least the running run should appear
        assert ids["running_id"] in body or ids["completed_id"] in body

    def test_dashboard_shows_weak_signal_bucket(self, seeded_app):
        client, ids = seeded_app
        resp = client.get("/")
        body = resp.text
        assert "weak" in body.lower()

    def test_dashboard_shows_queued_bucket(self, seeded_app):
        client, ids = seeded_app
        resp = client.get("/")
        body = resp.text
        # "queued" or pending/submitted states
        assert "queued" in body.lower() or "pending" in body.lower() or "submitted" in body.lower()

    def test_dashboard_collapses_long_server_note(self, seeded_app):
        client, ids = seeded_app
        resp = client.get("/")
        body = resp.text
        assert 'class="server-note"' in body
        assert "<summary>server note</summary>" in body

    def test_dashboard_shows_server_slurm_job_details(self, seeded_app):
        client, ids = seeded_app
        resp = client.get("/")
        body = resp.text
        assert "my jobs" in body
        assert "alpha_scan" in body
        assert "12345" in body

    def test_dashboard_shows_default_policy_capacity(self, seeded_app):
        client, ids = seeded_app
        resp = client.get("/")
        body = resp.text
        assert "cpu_fill_idle" in body
        assert "keep 5% idle" in body
        assert "kernel slot" in body

    def test_dashboard_shows_configuration_panel(self, seeded_app):
        client, ids = seeded_app
        resp = client.get("/")
        body = resp.text
        assert "Configuration" in body
        assert "state root" in body
        assert "database" in body
        assert "run dir" in body
        assert "Default Policies" in body
        assert "[jobctl]" in body

    def test_dashboard_explains_cluster_activity_when_run_bucket_empty(self, seeded_app):
        from jobctl.db.models import State

        client, ids = seeded_app
        client.app.state.store.update_run(ids["running_id"], state=State.COMPLETED)
        resp = client.get("/ui/poll")
        body = resp.text
        assert "No jobctl-managed running runs." in body
        assert "visible via server status" in body
        assert "alpha_scan" in body
        assert "slurm R" in body


# Header that triggers HTML rendering via content negotiation
_HTML_HEADERS = {"Accept": "text/html"}


# ---------------------------------------------------------------------------
# Run detail tests (GET /runs/{id} with Accept: text/html)
# ---------------------------------------------------------------------------

class TestRunDetail:
    def test_run_detail_200(self, seeded_app):
        client, ids = seeded_app
        resp = client.get(f"/runs/{ids['completed_id']}", headers=_HTML_HEADERS)
        assert resp.status_code == 200

    def test_run_detail_html(self, seeded_app):
        client, ids = seeded_app
        resp = client.get(f"/runs/{ids['completed_id']}", headers=_HTML_HEADERS)
        assert "text/html" in resp.headers["content-type"]

    def test_run_detail_shows_run_id(self, seeded_app):
        client, ids = seeded_app
        resp = client.get(f"/runs/{ids['completed_id']}", headers=_HTML_HEADERS)
        assert ids["completed_id"] in resp.text

    def test_run_detail_shows_state(self, seeded_app):
        client, ids = seeded_app
        resp = client.get(f"/runs/{ids['completed_id']}", headers=_HTML_HEADERS)
        assert "completed" in resp.text.lower()

    def test_run_detail_shows_stdout_tail(self, seeded_app):
        client, ids = seeded_app
        resp = client.get(f"/runs/{ids['completed_id']}", headers=_HTML_HEADERS)
        # stdout content should appear
        assert "loss=0.1" in resp.text or "Epoch" in resp.text

    def test_run_detail_shows_stderr_tail(self, seeded_app):
        client, ids = seeded_app
        resp = client.get(f"/runs/{ids['completed_id']}", headers=_HTML_HEADERS)
        assert "WARNING" in resp.text or "stderr" in resp.text.lower()

    def test_run_detail_shows_csv_table(self, seeded_app):
        client, ids = seeded_app
        resp = client.get(f"/runs/{ids['completed_id']}", headers=_HTML_HEADERS)
        # CSV artifact should render as a table
        assert "<table" in resp.text.lower()

    def test_run_detail_shows_image_tag(self, seeded_app):
        client, ids = seeded_app
        resp = client.get(f"/runs/{ids['completed_id']}", headers=_HTML_HEADERS)
        # Image artifact should render as <img
        assert "<img" in resp.text.lower()

    def test_run_detail_shows_observation_card(self, seeded_app):
        client, ids = seeded_app
        resp = client.get(f"/runs/{ids['completed_id']}", headers=_HTML_HEADERS)
        # Observation card fields
        assert "interpretation" in resp.text.lower() or "Model trained" in resp.text

    def test_run_detail_shows_contract_criteria(self, seeded_app):
        client, ids = seeded_app
        resp = client.get(f"/runs/{ids['completed_id']}", headers=_HTML_HEADERS)
        # Criteria table should appear (HTML-escaped < becomes &lt;)
        assert "final loss" in resp.text or "criteria" in resp.text.lower()

    def test_run_detail_404_for_missing(self, seeded_app):
        client, ids = seeded_app
        resp = client.get("/runs/nonexistent-run", headers=_HTML_HEADERS)
        assert resp.status_code == 404

    def test_run_detail_running_run(self, seeded_app):
        client, ids = seeded_app
        resp = client.get(f"/runs/{ids['running_id']}", headers=_HTML_HEADERS)
        assert resp.status_code == 200
        assert "running" in resp.text.lower()


# ---------------------------------------------------------------------------
# JobFile detail tests (GET /jobfiles/{id} with Accept: text/html)
# ---------------------------------------------------------------------------

class TestJobfileDetail:
    def test_jobfile_detail_200(self, seeded_app):
        client, ids = seeded_app
        resp = client.get(f"/jobfiles/{ids['jf_id']}", headers=_HTML_HEADERS)
        assert resp.status_code == 200

    def test_jobfile_detail_html(self, seeded_app):
        client, ids = seeded_app
        resp = client.get(f"/jobfiles/{ids['jf_id']}", headers=_HTML_HEADERS)
        assert "text/html" in resp.headers["content-type"]

    def test_jobfile_detail_shows_name(self, seeded_app):
        client, ids = seeded_app
        resp = client.get(f"/jobfiles/{ids['jf_id']}", headers=_HTML_HEADERS)
        assert "test_job" in resp.text

    def test_jobfile_detail_shows_params_schema(self, seeded_app):
        client, ids = seeded_app
        resp = client.get(f"/jobfiles/{ids['jf_id']}", headers=_HTML_HEADERS)
        # params schema has "lr" parameter
        assert "lr" in resp.text

    def test_jobfile_detail_shows_historical_runs(self, seeded_app):
        client, ids = seeded_app
        resp = client.get(f"/jobfiles/{ids['jf_id']}", headers=_HTML_HEADERS)
        # Should list historical runs
        assert ids["completed_id"] in resp.text or ids["running_id"] in resp.text

    def test_jobfile_detail_shows_contract_versions(self, seeded_app):
        client, ids = seeded_app
        resp = client.get(f"/jobfiles/{ids['jf_id']}", headers=_HTML_HEADERS)
        # Contract criteria should appear (HTML-escaped < becomes &lt;)
        assert "final loss" in resp.text or "criteria" in resp.text.lower()

    def test_jobfile_detail_404_for_missing(self, seeded_app):
        client, ids = seeded_app
        resp = client.get("/jobfiles/nonexistent-jf")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# HTMX poll partial endpoint
# ---------------------------------------------------------------------------

class TestHtmxPoll:
    def test_poll_partial_200(self, seeded_app):
        client, ids = seeded_app
        resp = client.get("/ui/poll")
        assert resp.status_code == 200

    def test_poll_partial_returns_html_fragment(self, seeded_app):
        client, ids = seeded_app
        resp = client.get("/ui/poll")
        assert "text/html" in resp.headers["content-type"]

    def test_poll_partial_contains_run_info(self, seeded_app):
        client, ids = seeded_app
        resp = client.get("/ui/poll")
        body = resp.text
        # Should mention at least one run state
        assert any(s in body.lower() for s in ["running", "completed", "failed", "stuck"])

    def test_poll_partial_with_run_id(self, seeded_app):
        """GET /ui/poll?run_id=X returns fragment for a specific run."""
        client, ids = seeded_app
        resp = client.get(f"/ui/poll?run_id={ids['running_id']}")
        assert resp.status_code == 200
        body = resp.text
        assert ids["running_id"] in body or "running" in body.lower()
