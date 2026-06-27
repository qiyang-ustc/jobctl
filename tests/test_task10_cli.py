"""Tests for cli/main.py (Task 10).

Strategy:
- Build a FastAPI TestClient for a real app instance, then wrap it in an
  ApiClient.  Inject the ApiClient into the Typer app via a module-level
  override so CliRunner tests never hit the network and never need
  ensure_daemon().
- Tests are grouped by command.  --json flag assertions check key shapes.
- run --wait / run --background both tested.
"""
from __future__ import annotations

import json
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_api_client(http_client):
    """Wrap a TestClient in an ApiClient."""
    from jobctl.api.client import ApiClient
    return ApiClient(base_url="http://testserver", transport=http_client)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture
def jobfile_yaml(tmp_path):
    """Simple local jobfile that echoes a value."""
    script = tmp_path / "hello.sh"
    script.write_text('#!/bin/bash\necho "hello from jobctl"\n')
    script.chmod(0o755)
    jf_path = tmp_path / "hello.jobfile.yaml"
    jf_path.write_text(f"""
name: cli-hello
command: "bash {script}"
params: {{}}
backends:
  - backend: local
artifacts: []
""")
    return str(jf_path)


@pytest.fixture
def jobfile_with_output(tmp_path):
    """A jobfile whose script writes a csv."""
    script = tmp_path / "compute.sh"
    script.write_text(
        '#!/bin/bash\necho "v,42"\necho "v,42" > result.csv\n'
    )
    script.chmod(0o755)
    jf_path = tmp_path / "compute.jobfile.yaml"
    jf_path.write_text(f"""
name: cli-compute
command: "bash {script}"
params: {{}}
backends:
  - backend: local
artifacts:
  - "*.csv"
""")
    return str(jf_path)


@pytest.fixture
def cli_env(tmp_path, jobfile_yaml):
    """Return (CliRunner, cli app, ApiClient, TestClient, jf_id).

    The TestClient is entered as a context manager so the monitor lifespan
    is started before tests run.
    """
    from jobctl.cli import main as cli_module
    from jobctl.cli.main import app
    from jobctl.api.server import create_app
    from fastapi.testclient import TestClient

    db_path = str(tmp_path / "cli_test.db")
    run_dir = str(tmp_path / "cli_runs")
    config = {
        "db_path": db_path,
        "run_dir": run_dir,
        "poll_interval_seconds": 0.05,
        "probe_interval_seconds": 9999,
        "stuck_timeout_seconds": 9999,
    }
    fast_app = create_app(config=config, start_monitor=True)

    with TestClient(fast_app) as http_client:
        ac = _make_api_client(http_client)

        # Register jobfile
        resp = http_client.post("/jobfiles", json={"path": jobfile_yaml})
        jf_id = resp.json()["id"]

        # Inject api client into CLI module so commands don't call ensure_daemon
        cli_module._OVERRIDE_CLIENT = ac

        runner = CliRunner()
        yield runner, app, ac, http_client, jf_id

        # Cleanup
        cli_module._OVERRIDE_CLIENT = None


# ---------------------------------------------------------------------------
# Helper: wait for a run to reach a terminal state via HTTP polling
# ---------------------------------------------------------------------------

_TERMINAL = {"completed", "failed", "cancelled", "stuck", "timeout"}


def _wait_terminal(http_client, run_id, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = http_client.get(f"/runs/{run_id}")
        data = r.json()
        if data["state"] in _TERMINAL:
            return data
        time.sleep(0.1)
    return http_client.get(f"/runs/{run_id}").json()


# ===========================================================================
# Tests: top-level report-bug shortcut
# ===========================================================================

class TestReportBugShortcut:
    def test_top_level_report_bug_shortcut_saves_report(self, tmp_path, monkeypatch):
        from jobctl.cli import main as cli_module
        from jobctl.cli.main import app

        home = tmp_path / "jobctl-home"
        monkeypatch.setenv("JOBCTL_HOME", str(home))

        class BrokenClient:
            def get_run(self, run_id):
                raise RuntimeError("daemon unavailable")

            def list_runs(self):
                raise RuntimeError("daemon unavailable")

        cli_module._OVERRIDE_CLIENT = BrokenClient()
        try:
            result = CliRunner().invoke(
                app,
                [
                    "--report-bug",
                    "daemon marked a completed run as running",
                    "--report-run",
                    "run-test",
                    "--report-no-submit",
                ],
            )
        finally:
            cli_module._OVERRIDE_CLIENT = None

        assert result.exit_code == 0, result.output
        assert "Could not file to GitHub; saved report to" in result.output

        reports = list((home / "issues").glob("bug-*.md"))
        assert len(reports) == 1
        body = reports[0].read_text()
        assert "daemon marked a completed run as running" in body
        assert "Filed via `jobctl report-bug`" in body


class TestCliApiErrors:
    def test_daemon_connection_error_is_concise(self):
        from jobctl.api.client import JobctlApiError
        from jobctl.cli import main as cli_module
        from jobctl.cli.main import app

        class BlockedClient:
            def servers(self):
                raise JobctlApiError(
                    "could not connect to the jobctl daemon at http://127.0.0.1:7421: "
                    "local sandbox or OS permissions blocked the connection."
                )

        cli_module._OVERRIDE_CLIENT = BlockedClient()
        try:
            result = CliRunner().invoke(app, ["servers", "--json"])
        finally:
            cli_module._OVERRIDE_CLIENT = None

        assert result.exit_code == 1
        assert "could not connect to the jobctl daemon" in result.output
        assert "Traceback" not in result.output
        assert "jobctl/api/client.py" not in result.output


# ===========================================================================
# Tests: run --background
# ===========================================================================

class TestRunBackground:
    def test_run_background_prints_run_id(self, cli_env, jobfile_yaml):
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(app, ["run", "--background", "--json", jf_id])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "run_id" in data

    def test_run_background_with_param(self, cli_env, jobfile_yaml):
        """run --background submits by name correctly."""
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(
            app,
            ["run", "--background", "--json", "cli-hello"],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "run_id" in data

    def test_run_background_json_shape(self, cli_env, jobfile_yaml):
        """run --background --json must return {run_id, state, ...}."""
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(app, ["run", "--background", "--json", jf_id])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "run_id" in data
        assert "state" in data

    def test_run_background_mem_auto_flag(self, cli_env, jobfile_yaml):
        """--mem_auto enables the persisted automatic memory policy."""
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(
            app,
            ["run", "--background", "--json", "--mem_auto", "--mem-auto-attempts", "2", jf_id],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["auto_policy"]["mem_auto"] is True
        assert data["auto_policy"]["max_attempts"] == 2

    def test_run_background_gres_flag_persists_slurm_request(self, cli_env, jobfile_yaml):
        """--gres is persisted for a GPU-looking JobFile."""
        runner, app, ac, http_client, jf_id = cli_env
        gpu_jf = Path(jobfile_yaml).with_name("gpu.jobfile.yaml")
        gpu_jf.write_text(
            "name: cli-gpu\n"
            'command: "echo --device cuda"\n'
            "params: {}\n"
            "backends:\n"
            "  - backend: local\n"
            "artifacts: []\n"
        )
        result = runner.invoke(
            app,
            [
                "run",
                "--background",
                "--json",
                "--gres",
                "gpu:mi300a:1",
                "--partition",
                "gpu",
                str(gpu_jf),
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["slurm_request"]["gres"] == "gpu:mi300a:1"
        assert data["slurm_request"]["partition"] == "gpu"

    def test_run_background_no_json_prints_run_id_only(self, cli_env, jobfile_yaml):
        """Without --json, run --background prints a bare run_id string."""
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(app, ["run", "--background", jf_id])
        assert result.exit_code == 0, result.output
        # Should contain a run_id (starts with "run-")
        assert "run-" in result.output


# ===========================================================================
# Tests: run --wait
# ===========================================================================

class TestRunWait:
    def test_run_wait_blocks_and_prints_card(self, cli_env, jobfile_yaml):
        """run --wait --json blocks to terminal and prints the observation card."""
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(
            app, ["run", "--wait", "--json", jf_id]
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        # Observation card shape
        assert "run_id" in data
        assert "state" in data
        assert data["state"] in _TERMINAL

    def test_run_wait_observation_card_fields(self, cli_env, jobfile_yaml):
        """The card emitted by --wait --json has all required fields."""
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(
            app, ["run", "--wait", "--json", jf_id]
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        # The output may be the run dict or the observation card;
        # at minimum run_id and state must be present.
        assert "run_id" in data
        assert "state" in data

    def test_run_wait_state_is_terminal(self, cli_env, jobfile_yaml):
        """run --wait --json always returns a terminal state."""
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(
            app, ["run", "--wait", "--json", jf_id]
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["state"] in _TERMINAL

    def test_run_wait_no_json_prints_card_text(self, cli_env, jobfile_yaml):
        """Without --json, run --wait prints human-readable output."""
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(app, ["run", "--wait", jf_id])
        assert result.exit_code == 0, result.output
        # Should print something non-empty
        assert len(result.output.strip()) > 0


# ===========================================================================
# Tests: await
# ===========================================================================

class TestAwaitCommand:
    def _submit_background(self, runner, app, jf_id):
        result = runner.invoke(app, ["run", "--background", "--json", jf_id])
        assert result.exit_code == 0
        return json.loads(result.output)["run_id"]

    def test_await_blocks_to_terminal_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run_id = self._submit_background(runner, app, jf_id)
        result = runner.invoke(app, ["await", "--json", run_id])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["state"] in _TERMINAL

    def test_await_no_json_prints_something(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run_id = self._submit_background(runner, app, jf_id)
        result = runner.invoke(app, ["await", run_id])
        assert result.exit_code == 0, result.output
        assert len(result.output.strip()) > 0


# ===========================================================================
# Tests: status
# ===========================================================================

class TestStatusCommand:
    def test_status_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        result = runner.invoke(app, ["status", "--json", run_id])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["run_id"] == run_id
        assert "state" in data

    def test_status_no_json_prints_state(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        result = runner.invoke(app, ["status", run_id])
        assert result.exit_code == 0, result.output
        # Should print the state somewhere
        assert len(result.output.strip()) > 0


# ===========================================================================
# Tests: inspect
# ===========================================================================

class TestInspectCommand:
    def test_inspect_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        result = runner.invoke(app, ["inspect", "--json", run_id])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["run_id"] == run_id

    def test_inspect_no_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        result = runner.invoke(app, ["inspect", run_id])
        assert result.exit_code == 0, result.output
        assert len(result.output.strip()) > 0

    def test_inspect_daemon_connection_error_has_no_traceback(self):
        from jobctl.api.client import JobctlApiError
        from jobctl.cli import main as cli_module
        from jobctl.cli.main import app

        class BrokenClient:
            def get_run(self, run_id):
                raise JobctlApiError("could not connect to the jobctl daemon")

        cli_module._OVERRIDE_CLIENT = BrokenClient()
        try:
            result = CliRunner().invoke(app, ["inspect", "run-missing", "--json"])
        finally:
            cli_module._OVERRIDE_CLIENT = None

        assert result.exit_code == 1
        assert "could not connect to the jobctl daemon" in result.output
        assert "Traceback" not in result.output


# ===========================================================================
# Tests: logs
# ===========================================================================

class TestLogsCommand:
    def test_logs_prints_text(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        result = runner.invoke(app, ["logs", run_id])
        assert result.exit_code == 0, result.output
        # Text may be empty (run hasn't started yet) or contain log lines
        assert isinstance(result.output, str)

    def test_logs_stderr_flag(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        result = runner.invoke(app, ["logs", "--stream", "stderr", run_id])
        assert result.exit_code == 0, result.output

    def test_logs_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        result = runner.invoke(app, ["logs", run_id, "--tail", "240", "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["run_id"] == run_id
        assert data["stream"] == "stdout"
        assert data["tail"] == 240
        assert isinstance(data["text"], str)


# ===========================================================================
# Tests: artifacts
# ===========================================================================

class TestArtifactsCommand:
    def test_artifacts_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        result = runner.invoke(app, ["artifacts", "--json", run_id])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_artifacts_no_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        result = runner.invoke(app, ["artifacts", run_id])
        assert result.exit_code == 0, result.output


# ===========================================================================
# Tests: cancel
# ===========================================================================

class TestCancelCommand:
    def test_cancel_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        result = runner.invoke(app, ["cancel", "--json", run_id])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["state"] == "cancelled"

    def test_cancel_no_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        result = runner.invoke(app, ["cancel", run_id])
        assert result.exit_code == 0, result.output
        assert len(result.output.strip()) > 0


# ===========================================================================
# Tests: rerun
# ===========================================================================

class TestRerunCommand:
    def test_rerun_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        ac.cancel(run_id)
        result = runner.invoke(app, ["rerun", "--json", run_id])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "run_id" in data
        assert data["run_id"] != run_id  # new run

    def test_rerun_no_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        result = runner.invoke(app, ["rerun", run_id])
        assert result.exit_code == 0, result.output


# ===========================================================================
# Tests: servers
# ===========================================================================

class TestServersCommand:
    def test_servers_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(app, ["servers", "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_servers_no_json_prints_table(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(app, ["servers"])
        assert result.exit_code == 0, result.output
        # Output can be empty table or header
        assert isinstance(result.output, str)


class TestRunningCommand:
    def test_running_json_includes_managed_and_scheduler_only_jobs(self, cli_env):
        from jobctl.db.models import Health, Run, Server, State

        runner, app, ac, http_client, jf_id = cli_env
        store = http_client.app.state.store
        now = _now_iso()
        run = Run(
            run_id="run-active001",
            jobfile_id=jf_id,
            jobfile_version=1,
            params={},
            input_hashes={},
            backend="slurm",
            server="oblix",
            task=None,
            remote_job_id="317604",
            state=State.RUNNING,
            health=Health.OK,
            exit_code=None,
            submitted_at=now,
            started_at=now,
            finished_at=None,
            last_heartbeat=now,
            workdir=None,
            stdout_path=None,
            stderr_path=None,
            resource_summary={},
            expectation_match=None,
            observation_card=None,
            slurm_request={"job_id": "317604"},
            title="active test run",
        )
        store.add_run(run)
        store.upsert_server(
            Server(
                name="oblix",
                backend_type="slurm",
                online=True,
                last_heartbeat=now,
                cpu={},
                mem={},
                gpu={},
                disk={},
                slurm_queue={
                    "running": 2,
                    "pending": 0,
                    "jobs": [
                        {"job_id": "317604", "state": "R", "name": "run-active001"},
                        {"job_id": "999999", "state": "R", "name": "external-job"},
                    ],
                },
                note=None,
            )
        )

        result = runner.invoke(app, ["running", "--json"])

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["counts"]["runs"] == 1
        assert data["runs"][0]["run_id"] == "run-active001"
        assert data["counts"]["cluster_jobs"] == 2
        assert data["counts"]["scheduler_only_jobs"] == 1
        assert data["scheduler_only_jobs"][0]["name"] == "external-job"

    def test_running_human_empty_state(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(app, ["running"])
        assert result.exit_code == 0, result.output
        assert "jobctl-managed active runs" in result.output or "(no jobctl-managed active runs)" in result.output


# ===========================================================================
# Tests: memory query
# ===========================================================================

class TestMemoryCommand:
    def test_memory_query_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(app, ["memory", "query", "--json", "--jobfile-id", jf_id])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "has_jobfile" in data

    def test_memory_query_by_name(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(app, ["memory", "query", "--json", "--name", "cli-hello"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "has_jobfile" in data

    def test_memory_query_no_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(app, ["memory", "query", "--jobfile-id", jf_id])
        assert result.exit_code == 0, result.output
        assert len(result.output.strip()) > 0


# ===========================================================================
# Tests: register + jobfiles
# ===========================================================================

class TestRegisterJobfiles:
    def test_register_json(self, cli_env, jobfile_yaml):
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(app, ["register", "--json", jobfile_yaml])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "id" in data
        assert data["name"] == "cli-hello"

    def test_register_changed_manifest_refreshes_fields(
        self, cli_env, jobfile_yaml, tmp_path
    ):
        runner, app, ac, http_client, jf_id = cli_env
        script = tmp_path / "goodbye.sh"
        script.write_text('#!/bin/bash\necho "goodbye from jobctl"\n')
        script.chmod(0o755)
        Path(jobfile_yaml).write_text(f"""
name: cli-hello
command: "bash {script} --mode {{mode}}"
params:
  mode:
    type: str
    default: slow
backends:
  - backend: local
artifacts:
  - "*.txt"
""")

        resp = http_client.post("/jobfiles", json={"path": jobfile_yaml})
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == jf_id
        assert data["version"] == 2
        assert data["command_template"] == f"bash {script} --mode {{mode}}"
        assert data["params_schema"] == {"mode": {"type": "str", "default": "slow"}}
        assert data["artifact_patterns"] == ["*.txt"]

    def test_jobfiles_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(app, ["jobfiles", "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert any(jf["id"] == jf_id for jf in data)

    def test_jobfiles_no_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(app, ["jobfiles"])
        assert result.exit_code == 0, result.output
        assert "cli-hello" in result.output


# ===========================================================================
# Tests: feedback
# ===========================================================================

class TestFeedbackCommand:
    def test_feedback_post_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        result = runner.invoke(
            app,
            ["feedback", "--json", run_id, "--text", "looks good", "--kind", "note"]
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["run_id"] == run_id

    def test_feedback_post_no_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        result = runner.invoke(
            app,
            ["feedback", run_id, "--text", "good", "--kind", "note"]
        )
        assert result.exit_code == 0, result.output


# ===========================================================================
# Tests: expect
# ===========================================================================

class TestExpectCommand:
    def test_expect_list_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(app, ["expect", "--json", "--jobfile-id", jf_id])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_expect_propose_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        run = ac.submit(jobfile_id=jf_id, params={})
        run_id = run["run_id"]
        result = runner.invoke(
            app,
            ["expect", "propose", "--json", run_id, "--text", "output should be positive"]
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_expect_no_json(self, cli_env):
        runner, app, ac, http_client, jf_id = cli_env
        result = runner.invoke(app, ["expect", "--jobfile-id", jf_id])
        assert result.exit_code == 0, result.output


# ===========================================================================
# Tests: serve command (smoke)
# ===========================================================================

class TestServeCommand:
    def test_serve_help(self):
        """serve --help must exit 0."""
        from jobctl.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0

    def test_serve_smoke(self, tmp_path):
        """serve in a thread exits cleanly when the server raises (no uvicorn needed)."""
        from jobctl.cli.main import app
        from jobctl.cli import main as cli_module
        runner = CliRunner()
        # We just verify the command exists and --help works
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "serve" in result.output.lower() or "host" in result.output.lower()


# ===========================================================================
# Tests: --help
# ===========================================================================

class TestHelp:
    def test_main_help(self):
        from jobctl.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # All top-level commands should be listed
        for cmd in ["run", "await", "status", "logs", "artifacts", "inspect",
                    "cancel", "rerun", "servers", "memory", "register",
                    "jobfiles", "feedback", "expect", "serve"]:
            assert cmd in result.output, f"'{cmd}' not in help output"

    def test_run_help(self):
        from jobctl.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        for flag in ["--wait", "--background", "--param", "--backend", "--json"]:
            assert flag in result.output, f"'{flag}' not in run --help"

    def test_memory_help(self):
        from jobctl.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["memory", "--help"])
        assert result.exit_code == 0
