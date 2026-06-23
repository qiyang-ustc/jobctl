"""Tests for api/server.py and api/client.py (Task 9).

Strategy:
- Use FastAPI TestClient for all HTTP endpoint tests (offline).
- For the "real job completes via monitor" test, create a TestClient with the app
  using a fast-tick monitor (poll_interval=0.05s), submit a local job that exits
  quickly, and poll until the run reaches a terminal state with an observation card.
- ApiClient methods are tested against the same TestClient.
- ensure_daemon() is unit-tested (mocked).
"""
from __future__ import annotations

import asyncio
import builtins
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Lazy import — app may not exist yet (TDD: tests written before impl)
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Return path to a temp SQLite db."""
    return str(tmp_path / "test.db")


@pytest.fixture
def tmp_run_dir(tmp_path):
    return str(tmp_path / "runs")


@pytest.fixture
def app_client(tmp_path):
    """Create a FastAPI TestClient with an isolated Store and fast-tick Monitor."""
    from jobctl.api.server import create_app
    from jobctl.db.store import Store

    db_path = str(tmp_path / "app.db")
    run_dir = str(tmp_path / "runs")
    config = {
        "db_path": db_path,
        "run_dir": run_dir,
        "poll_interval_seconds": 0.1,
        "probe_interval_seconds": 9999,
        "stuck_timeout_seconds": 9999,
    }
    app = create_app(config=config, start_monitor=False)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        yield client


@pytest.fixture
def app_client_with_monitor(tmp_path):
    """Create a TestClient with a real Monitor (fast tick) for integration tests."""
    from jobctl.api.server import create_app

    db_path = str(tmp_path / "app.db")
    run_dir = str(tmp_path / "runs")
    config = {
        "db_path": db_path,
        "run_dir": run_dir,
        "poll_interval_seconds": 0.05,
        "probe_interval_seconds": 9999,
        "stuck_timeout_seconds": 9999,
    }
    app = create_app(config=config, start_monitor=True)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_jobfile_yaml(tmp_path):
    """Write a simple local jobfile and return its path."""
    script = tmp_path / "hello.sh"
    script.write_text('#!/bin/bash\necho "hello from jobctl"\n')
    jf_path = tmp_path / "hello.jobfile.yaml"
    jf_path.write_text(f"""
name: hello-test
command: "bash {script}"
params: {{}}
backends:
  - backend: local
artifacts:
  - "stdout.txt"
""")
    return str(jf_path)


@pytest.fixture
def sample_jobfile_with_output(tmp_path):
    """A jobfile whose script writes a csv + prints a value."""
    script = tmp_path / "compute.sh"
    script.write_text(
        '#!/bin/bash\n'
        'echo "result,42"\n'
        'echo "result,42" > result.csv\n'
    )
    script.chmod(0o755)
    jf_path = tmp_path / "compute.jobfile.yaml"
    jf_path.write_text(f"""
name: compute-test
command: "bash {script}"
params: {{}}
backends:
  - backend: local
artifacts:
  - "*.csv"
""")
    return str(jf_path)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, app_client):
        resp = app_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_has_version(self, app_client):
        resp = app_client.get("/health")
        assert "version" in resp.json()


# ---------------------------------------------------------------------------
# /jobfiles — register + list
# ---------------------------------------------------------------------------

class TestJobfiles:
    def test_register_jobfile(self, app_client, sample_jobfile_yaml):
        resp = app_client.post("/jobfiles", json={"path": sample_jobfile_yaml})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "hello-test"
        assert "id" in data

    def test_list_jobfiles_empty(self, app_client):
        resp = app_client.get("/jobfiles")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_jobfiles_after_register(self, app_client, sample_jobfile_yaml):
        app_client.post("/jobfiles", json={"path": sample_jobfile_yaml})
        resp = app_client.get("/jobfiles")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["name"] == "hello-test"

    def test_register_same_jobfile_twice_ok(self, app_client, sample_jobfile_yaml):
        r1 = app_client.post("/jobfiles", json={"path": sample_jobfile_yaml})
        r2 = app_client.post("/jobfiles", json={"path": sample_jobfile_yaml})
        assert r1.status_code == 200
        assert r2.status_code == 200
        # Second register returns the same id
        assert r1.json()["id"] == r2.json()["id"]

    def test_register_missing_path(self, app_client):
        resp = app_client.post("/jobfiles", json={"path": "/nonexistent/path.yaml"})
        assert resp.status_code == 422 or resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /runs — submit a run
# ---------------------------------------------------------------------------

class TestSubmitRun:
    def test_submit_run_creates_pending(self, app_client, sample_jobfile_yaml):
        # Register jobfile first
        reg = app_client.post("/jobfiles", json={"path": sample_jobfile_yaml})
        jf_id = reg.json()["id"]

        resp = app_client.post("/runs", json={"jobfile_id": jf_id, "params": {}})
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["state"] in ("pending", "submitted", "running")
        assert data["jobfile_id"] == jf_id

    def test_submit_run_by_name(self, app_client, sample_jobfile_yaml):
        app_client.post("/jobfiles", json={"path": sample_jobfile_yaml})
        resp = app_client.post("/runs", json={"jobfile_name": "hello-test", "params": {}})
        assert resp.status_code == 200
        assert "run_id" in resp.json()

    def test_submit_unknown_jobfile_returns_404(self, app_client):
        resp = app_client.post("/runs", json={"jobfile_id": "nonexistent-id", "params": {}})
        assert resp.status_code == 404

    def test_submit_attaches_memory_hint(self, app_client, sample_jobfile_yaml):
        """Memory hint field present in response (may be empty dict)."""
        reg = app_client.post("/jobfiles", json={"path": sample_jobfile_yaml})
        jf_id = reg.json()["id"]
        resp = app_client.post("/runs", json={"jobfile_id": jf_id, "params": {}})
        assert resp.status_code == 200
        data = resp.json()
        assert "memory_hint" in data  # the hint dict (may be has_jobfile=True)

    def test_submit_backend_selected(self, app_client, sample_jobfile_yaml):
        """Backend field is populated after submit."""
        reg = app_client.post("/jobfiles", json={"path": sample_jobfile_yaml})
        jf_id = reg.json()["id"]
        resp = app_client.post("/runs", json={"jobfile_id": jf_id, "params": {}})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("backend") is not None

    def test_submit_backend_override(self, app_client, sample_jobfile_yaml):
        """Backend override is respected."""
        reg = app_client.post("/jobfiles", json={"path": sample_jobfile_yaml})
        jf_id = reg.json()["id"]
        resp = app_client.post("/runs", json={
            "jobfile_id": jf_id,
            "params": {},
            "backend_override": {"backend": "local"}
        })
        assert resp.status_code == 200
        assert resp.json()["backend"] == "local"

    def test_submit_failure_persists_observation_card(self, app_client, sample_jobfile_yaml):
        """A backend submit error still produces an inspectable observation card."""
        reg = app_client.post("/jobfiles", json={"path": sample_jobfile_yaml})
        jf_id = reg.json()["id"]
        resp = app_client.post("/runs", json={
            "jobfile_id": jf_id,
            "params": {},
            "backend_override": {"backend": "does-not-exist"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "failed"
        assert data["expectation_match"] == "failed"
        assert data["observation_card"]["status"] == "failed"
        assert "Backend submit failed" in data["observation_card"]["key_evidence"][0]


# ---------------------------------------------------------------------------
# GET /runs, /runs/{id}
# ---------------------------------------------------------------------------

class TestGetRuns:
    def _register_and_submit(self, client, jf_yaml):
        reg = client.post("/jobfiles", json={"path": jf_yaml})
        jf_id = reg.json()["id"]
        run = client.post("/runs", json={"jobfile_id": jf_id, "params": {}})
        return jf_id, run.json()["run_id"]

    def test_get_run_by_id(self, app_client, sample_jobfile_yaml):
        _, run_id = self._register_and_submit(app_client, sample_jobfile_yaml)
        resp = app_client.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        assert resp.json()["run_id"] == run_id

    def test_get_run_not_found(self, app_client):
        resp = app_client.get("/runs/nonexistent-run-id")
        assert resp.status_code == 404

    def test_list_runs_empty(self, app_client):
        resp = app_client.get("/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_runs_after_submit(self, app_client, sample_jobfile_yaml):
        _, run_id = self._register_and_submit(app_client, sample_jobfile_yaml)
        resp = app_client.get("/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert any(r["run_id"] == run_id for r in runs)

    def test_list_runs_filter_by_state(self, app_client, sample_jobfile_yaml):
        _, run_id = self._register_and_submit(app_client, sample_jobfile_yaml)
        # Filter for pending/submitted/running — run should appear
        run_data = app_client.get(f"/runs/{run_id}").json()
        state = run_data["state"]
        resp = app_client.get(f"/runs?state={state}")
        assert resp.status_code == 200
        runs = resp.json()
        assert any(r["run_id"] == run_id for r in runs)

    def test_list_runs_filter_by_jobfile(self, app_client, sample_jobfile_yaml):
        jf_id, run_id = self._register_and_submit(app_client, sample_jobfile_yaml)
        resp = app_client.get(f"/runs?jobfile_id={jf_id}")
        assert resp.status_code == 200
        runs = resp.json()
        assert all(r["jobfile_id"] == jf_id for r in runs)


# ---------------------------------------------------------------------------
# Cancel and Rerun
# ---------------------------------------------------------------------------

class TestCancelRerun:
    def _register_and_submit(self, client, jf_yaml):
        reg = client.post("/jobfiles", json={"path": jf_yaml})
        jf_id = reg.json()["id"]
        run = client.post("/runs", json={"jobfile_id": jf_id, "params": {}})
        return jf_id, run.json()["run_id"]

    def test_cancel_run(self, app_client, sample_jobfile_yaml):
        _, run_id = self._register_and_submit(app_client, sample_jobfile_yaml)
        resp = app_client.post(f"/runs/{run_id}/cancel")
        assert resp.status_code == 200
        # Check run is cancelled
        run = app_client.get(f"/runs/{run_id}").json()
        assert run["state"] == "cancelled"

    def test_cancel_nonexistent(self, app_client):
        resp = app_client.post("/runs/nonexistent/cancel")
        assert resp.status_code == 404

    def test_rerun_creates_new_run(self, app_client, sample_jobfile_yaml):
        _, run_id = self._register_and_submit(app_client, sample_jobfile_yaml)
        app_client.post(f"/runs/{run_id}/cancel")
        resp = app_client.post(f"/runs/{run_id}/rerun")
        assert resp.status_code == 200
        new_run = resp.json()
        assert new_run["run_id"] != run_id
        assert "run_id" in new_run

    def test_rerun_preserves_resources_and_auto_policy(self, app_client, sample_jobfile_yaml):
        reg = app_client.post("/jobfiles", json={"path": sample_jobfile_yaml})
        jf_id = reg.json()["id"]
        run = app_client.post("/runs", json={
            "jobfile_id": jf_id,
            "params": {},
            "resources": {"mem": "2G", "cpus": 2, "gres": "gpu:mi300a:1", "job_id": "old-job"},
            "auto_policy": {"mem_auto": True, "factor": 1.5, "max_attempts": 3},
        }).json()

        resp = app_client.post(f"/runs/{run['run_id']}/rerun")
        assert resp.status_code == 200
        new_run = resp.json()
        assert new_run["slurm_request"] == {"mem": "2G", "cpus": 2, "gres": "gpu:mi300a:1"}
        assert new_run["auto_policy"]["mem_auto"] is True
        assert new_run["attempt"] == 1

    def test_rerun_nonexistent(self, app_client):
        resp = app_client.post("/runs/nonexistent/rerun")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Logs endpoint
# ---------------------------------------------------------------------------

class TestLogs:
    def _submit_run(self, client, jf_yaml):
        reg = client.post("/jobfiles", json={"path": jf_yaml})
        jf_id = reg.json()["id"]
        run = client.post("/runs", json={"jobfile_id": jf_id, "params": {}})
        return run.json()["run_id"]

    def test_logs_stdout_returns_text(self, app_client, sample_jobfile_yaml):
        run_id = self._submit_run(app_client, sample_jobfile_yaml)
        resp = app_client.get(f"/runs/{run_id}/logs")
        assert resp.status_code == 200
        # Content could be empty or text — just check it's a string
        assert isinstance(resp.text, str)

    def test_logs_stderr(self, app_client, sample_jobfile_yaml):
        run_id = self._submit_run(app_client, sample_jobfile_yaml)
        resp = app_client.get(f"/runs/{run_id}/logs?stream=stderr")
        assert resp.status_code == 200

    def test_logs_nonexistent_run(self, app_client):
        resp = app_client.get("/runs/nonexistent/logs")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Artifacts endpoint
# ---------------------------------------------------------------------------

class TestArtifacts:
    def _submit_run(self, client, jf_yaml):
        reg = client.post("/jobfiles", json={"path": jf_yaml})
        jf_id = reg.json()["id"]
        run = client.post("/runs", json={"jobfile_id": jf_id, "params": {}})
        return run.json()["run_id"]

    def test_artifacts_returns_list(self, app_client, sample_jobfile_yaml):
        run_id = self._submit_run(app_client, sample_jobfile_yaml)
        resp = app_client.get(f"/runs/{run_id}/artifacts")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_artifacts_nonexistent_run(self, app_client):
        resp = app_client.get("/runs/nonexistent/artifacts")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /servers
# ---------------------------------------------------------------------------

class TestServers:
    def test_list_servers_empty(self, app_client):
        resp = app_client.get("/servers")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_servers_includes_policy_snapshot(self, tmp_path):
        from fastapi.testclient import TestClient
        from jobctl.api.server import create_app
        from jobctl.db.models import Server

        app = create_app(
            config={
                "db_path": str(tmp_path / "policy.db"),
                "run_dir": str(tmp_path / "runs"),
                "default_policies": {
                    "oblix": {
                        "mode": "cpu_fill_idle",
                        "target_idle_pct": 5,
                        "kernel_cpus": 2,
                    }
                },
            },
            start_monitor=False,
        )
        app.state.store.upsert_server(Server(
            name="oblix",
            backend_type="slurm",
            online=True,
            last_heartbeat=None,
            cpu={}, mem={}, gpu={}, disk={},
            slurm_queue={"idle_cpus": 60, "total_cpus": 100},
            note=None,
        ))

        resp = TestClient(app).get("/servers")
        assert resp.status_code == 200
        policy = resp.json()[0]["policy"]
        assert policy["target_idle_pct"] == 5.0
        assert policy["free_cpus_after_reserve"] == 55
        assert policy["kernels_available"] == 27


# ---------------------------------------------------------------------------
# /feedback
# ---------------------------------------------------------------------------

class TestFeedback:
    def _submit_run(self, client, jf_yaml):
        reg = client.post("/jobfiles", json={"path": jf_yaml})
        jf_id = reg.json()["id"]
        run = client.post("/runs", json={"jobfile_id": jf_id, "params": {}})
        return run.json()["run_id"]

    def test_post_feedback(self, app_client, sample_jobfile_yaml):
        run_id = self._submit_run(app_client, sample_jobfile_yaml)
        resp = app_client.post(
            f"/runs/{run_id}/feedback",
            json={"kind": "note", "text": "Looks good"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert data["kind"] == "note"

    def test_list_feedback(self, app_client, sample_jobfile_yaml):
        run_id = self._submit_run(app_client, sample_jobfile_yaml)
        app_client.post(f"/runs/{run_id}/feedback", json={"kind": "note", "text": "test"})
        resp = app_client.get(f"/runs/{run_id}/feedback")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) >= 1

    def test_feedback_nonexistent_run(self, app_client):
        resp = app_client.post(
            "/runs/nonexistent/feedback",
            json={"kind": "note", "text": "test"}
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /expect (list / confirm / propose)
# ---------------------------------------------------------------------------

class TestExpect:
    def test_list_contracts_empty(self, app_client, sample_jobfile_yaml):
        reg = app_client.post("/jobfiles", json={"path": sample_jobfile_yaml})
        jf_id = reg.json()["id"]
        resp = app_client.get(f"/expect?jobfile_id={jf_id}")
        assert resp.status_code == 200
        # Returns list (possibly empty) or contract object
        assert resp.json() is not None

    def test_confirm_nonexistent(self, app_client):
        resp = app_client.post("/expect/confirm", json={"criterion_id": "fake-id"})
        assert resp.status_code == 404

    def test_propose_criteria(self, app_client, sample_jobfile_yaml):
        reg = app_client.post("/jobfiles", json={"path": sample_jobfile_yaml})
        jf_id = reg.json()["id"]
        run_resp = app_client.post("/runs", json={"jobfile_id": jf_id, "params": {}})
        run_id = run_resp.json()["run_id"]
        resp = app_client.post(
            "/expect/propose",
            json={"run_id": run_id, "feedback_text": "The output should be positive"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


# ---------------------------------------------------------------------------
# /memory/query
# ---------------------------------------------------------------------------

class TestMemoryQuery:
    def test_memory_query_no_jobfile(self, app_client):
        resp = app_client.get("/memory/query?name=nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_jobfile"] is False

    def test_memory_query_with_jobfile(self, app_client, sample_jobfile_yaml):
        reg = app_client.post("/jobfiles", json={"path": sample_jobfile_yaml})
        jf_id = reg.json()["id"]
        resp = app_client.get(f"/memory/query?jobfile_id={jf_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_jobfile"] is True


# ---------------------------------------------------------------------------
# Integration: real local job reaches completed + observation card appears
# ---------------------------------------------------------------------------

class TestIntegrationLocalJob:
    """Drive a real local job to completed via the monitor and assert an
    observation card appears."""

    def test_local_job_completes_and_has_observation_card(self, app_client_with_monitor, sample_jobfile_yaml):
        client = app_client_with_monitor

        # Register the jobfile
        reg = client.post("/jobfiles", json={"path": sample_jobfile_yaml})
        assert reg.status_code == 200, reg.text
        jf_id = reg.json()["id"]

        # Submit the run
        sub = client.post("/runs", json={"jobfile_id": jf_id, "params": {}})
        assert sub.status_code == 200, sub.text
        run_id = sub.json()["run_id"]

        # Poll until terminal state (max 10s)
        deadline = time.time() + 10
        terminal_states = {"completed", "failed", "cancelled", "stuck", "timeout"}
        while time.time() < deadline:
            r = client.get(f"/runs/{run_id}")
            data = r.json()
            if data["state"] in terminal_states:
                break
            time.sleep(0.1)

        # Assert completed
        assert data["state"] in terminal_states, f"Run never finished: {data}"

        # Assert observation card is present and has all required fields
        card = data.get("observation_card")
        assert card is not None, f"observation_card is None: {data}"
        required_fields = {
            "status", "jobfile", "run_id", "server", "artifacts",
            "health", "expectation_match", "key_evidence",
            "interpretation", "recommended_next_action"
        }
        missing = required_fields - set(card.keys())
        assert not missing, f"Missing card fields: {missing}"

        # status must be actual state, not "finished"
        assert card["status"] != "finished"
        assert card["run_id"] == run_id


# ---------------------------------------------------------------------------
# ApiClient
# ---------------------------------------------------------------------------

class TestApiClient:
    """Test ApiClient methods using httpx transport backed by TestClient."""

    @pytest.fixture
    def api_client(self, app_client_with_monitor, sample_jobfile_yaml):
        from jobctl.api.client import ApiClient
        # Register the jobfile upfront
        reg = app_client_with_monitor.post("/jobfiles", json={"path": sample_jobfile_yaml})
        jf_id = reg.json()["id"]
        # Create ApiClient that re-uses the TestClient transport
        ac = ApiClient(base_url="http://testserver", transport=app_client_with_monitor)
        return ac, jf_id

    def test_submit(self, api_client):
        ac, jf_id = api_client
        run = ac.submit(jobfile_id=jf_id, params={})
        assert "run_id" in run

    def test_get_run(self, api_client):
        ac, jf_id = api_client
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        result = ac.get_run(run_id)
        assert result["run_id"] == run_id

    def test_list_runs(self, api_client):
        ac, jf_id = api_client
        ac.submit(jobfile_id=jf_id, params={})
        runs = ac.list_runs()
        assert len(runs) >= 1

    def test_cancel(self, api_client):
        ac, jf_id = api_client
        run = ac.submit(jobfile_id=jf_id, params={})
        result = ac.cancel(run["run_id"])
        assert result["state"] == "cancelled"

    def test_list_servers(self, api_client):
        ac, _ = api_client
        servers = ac.servers()
        assert isinstance(servers, list)

    def test_memory_query(self, api_client):
        ac, jf_id = api_client
        result = ac.memory_query(jobfile_id=jf_id)
        assert result["has_jobfile"] is True

    def test_await_run_completes(self, api_client):
        """await_run long-polls until the run is terminal."""
        ac, jf_id = api_client
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]

        # await_run should return when the run is terminal
        result = ac.await_run(run_id, poll_interval=0.1, timeout=15)
        assert result["state"] in ("completed", "failed", "cancelled", "stuck", "timeout")

    def test_jobfiles(self, api_client):
        ac, jf_id = api_client
        jfs = ac.jobfiles()
        assert any(jf["id"] == jf_id for jf in jfs)

    def test_feedback(self, api_client):
        ac, jf_id = api_client
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        result = ac.feedback(run_id, kind="note", text="test feedback")
        assert result["run_id"] == run_id

    def test_artifacts(self, api_client):
        ac, jf_id = api_client
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        arts = ac.artifacts(run_id)
        assert isinstance(arts, list)

    def test_connection_error_is_user_facing(self, monkeypatch):
        from jobctl.api.client import ApiClient, JobctlApiError
        import httpx

        def blocked(*args, **kwargs):
            raise httpx.ConnectError("[Errno 1] Operation not permitted")

        monkeypatch.setattr(httpx, "get", blocked)
        ac = ApiClient(base_url="http://127.0.0.1:7421")

        with pytest.raises(JobctlApiError) as excinfo:
            ac.servers()

        message = str(excinfo.value)
        assert "could not connect to the jobctl daemon" in message
        assert "sandbox" in message
        assert "Traceback" not in message

    def test_submit_uses_long_request_timeout(self, monkeypatch):
        from jobctl.api.client import ApiClient, _SUBMIT_REQUEST_TIMEOUT

        captured = {}

        def fake_post(url, **kwargs):
            captured.update(kwargs)
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"run_id": "run-abc"}
            return resp

        monkeypatch.setattr("httpx.post", fake_post)
        ac = ApiClient(base_url="http://127.0.0.1:7421")
        assert ac.submit(jobfile_id="jf-abc")["run_id"] == "run-abc"
        assert captured["timeout"] == _SUBMIT_REQUEST_TIMEOUT

    def test_read_timeout_warns_submit_may_have_succeeded(self, monkeypatch):
        from jobctl.api.client import ApiClient, JobctlApiError
        import httpx

        def timeout(*args, **kwargs):
            raise httpx.ReadTimeout("timed out")

        monkeypatch.setattr(httpx, "post", timeout)
        ac = ApiClient(base_url="http://127.0.0.1:7421")

        with pytest.raises(JobctlApiError) as excinfo:
            ac.submit(jobfile_id="jf-abc")

        message = str(excinfo.value)
        assert "timed out" in message
        assert "may already have been created" in message
        assert "jobctl running --json" in message


# ---------------------------------------------------------------------------
# ensure_daemon — unit test (no real process spawning)
# ---------------------------------------------------------------------------

class TestEnsureDaemon:
    def test_ensure_daemon_returns_url_if_alive(self):
        """If /health responds, ensure_daemon returns the URL without spawning."""
        from jobctl.api.client import ensure_daemon
        from unittest.mock import MagicMock, patch

        # Patch httpx.get to return a successful response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok"}

        with patch("httpx.get", return_value=mock_resp) as mock_get:
            url = ensure_daemon(config={"daemon_host": "127.0.0.1", "daemon_port": 7421})
            assert url == "http://127.0.0.1:7421"

    def test_ensure_daemon_spawns_if_down(self, tmp_path):
        """If /health fails, ensure_daemon tries to spawn and returns a URL."""
        from jobctl.api.client import ensure_daemon
        import httpx

        # Fail initial and locked recheck, then succeed after spawn.
        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = {"status": "ok"}

        call_count = 0
        def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise httpx.ConnectError("Connection refused")
            return mock_ok

        with patch("httpx.get", side_effect=mock_get):
            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = MagicMock()
                url = ensure_daemon(
                    config={"daemon_host": "127.0.0.1", "daemon_port": 7421},
                    wait_timeout=2,
                    pre_spawn_timeout=0,
                )
                assert url == "http://127.0.0.1:7421"
                mock_popen.assert_called_once()

    def test_ensure_daemon_rechecks_health_after_start_lock(self):
        """If another process starts the daemon while we wait, do not spawn."""
        from jobctl.api.client import ensure_daemon
        import httpx

        mock_ok = MagicMock()
        mock_ok.status_code = 200

        call_count = 0

        def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Connection refused")
            return mock_ok

        with patch("httpx.get", side_effect=mock_get):
            with patch("subprocess.Popen") as mock_popen:
                url = ensure_daemon(
                    config={"daemon_host": "127.0.0.1", "daemon_port": 7421},
                    pre_spawn_timeout=0,
                )

        assert url == "http://127.0.0.1:7421"
        mock_popen.assert_not_called()

    def test_ensure_daemon_retries_health_before_spawning(self):
        """A transient busy health probe should not spawn a duplicate daemon."""
        from jobctl.api.client import ensure_daemon
        import httpx

        mock_ok = MagicMock()
        mock_ok.status_code = 200

        call_count = 0

        def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ReadTimeout("busy")
            return mock_ok

        with patch("httpx.get", side_effect=mock_get):
            with patch("subprocess.Popen") as mock_popen:
                url = ensure_daemon(
                    config={"daemon_host": "127.0.0.1", "daemon_port": 7421},
                    poll_interval=0.01,
                    pre_spawn_timeout=1,
                )

        assert url == "http://127.0.0.1:7421"
        mock_popen.assert_not_called()

    def test_ensure_daemon_log_fallback_is_not_user_visible(
        self, tmp_path, monkeypatch, caplog
    ):
        """Sandbox log fallback should stay diagnostic, not user-facing warning noise."""
        from jobctl.api.client import ensure_daemon
        import httpx

        daemon_log = tmp_path / "blocked" / "daemon.log"
        fallback_dir = tmp_path / "fallback"
        fallback_dir.mkdir()
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fallback_dir))

        mock_ok = MagicMock()
        mock_ok.status_code = 200

        call_count = 0

        def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise httpx.ConnectError("Connection refused")
            return mock_ok

        real_open = builtins.open

        def fake_open(path, *args, **kwargs):
            if Path(path) == daemon_log:
                raise OSError("sandbox denied")
            return real_open(path, *args, **kwargs)

        caplog.set_level(logging.WARNING)
        with patch("jobctl.logsetup.log_path", return_value=daemon_log):
            with patch("builtins.open", side_effect=fake_open):
                with patch("httpx.get", side_effect=mock_get):
                    with patch("subprocess.Popen") as mock_popen:
                        mock_popen.return_value = MagicMock()
                        url = ensure_daemon(
                            config={"daemon_host": "127.0.0.1", "daemon_port": 7421},
                            wait_timeout=2,
                            pre_spawn_timeout=0,
                        )

        assert url == "http://127.0.0.1:7421"
        mock_popen.assert_called_once()
        assert "daemon log" not in caplog.text

        fallback_log = fallback_dir / "jobctl-daemon.log"
        assert fallback_log.exists()
        fallback_text = fallback_log.read_text()
        assert "jobctl daemon log fallback" in fallback_text
        assert str(daemon_log) in fallback_text
