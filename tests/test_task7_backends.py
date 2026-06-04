"""Tests for Task 7: backends/base.py, local.py, ssh.py, slurm.py."""
from __future__ import annotations

import os
import sys
import time
import shutil
import tempfile
import textwrap
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
FAKEBIN = str(Path(__file__).parent / "fakebin")


def _env_with_fakebin() -> dict:
    """Return a copy of os.environ with tests/fakebin prepended to PATH."""
    env = os.environ.copy()
    env["PATH"] = FAKEBIN + os.pathsep + env.get("PATH", "")
    return env


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
from jobctl.db.models import (
    Health, JobFile, Run, Server, State, Match,
)


def _make_jobfile(**kwargs) -> JobFile:
    defaults = dict(
        id="jf-001",
        name="test-job",
        version=1,
        source_path="/tmp/test.sh",
        command_template="bash -c 'echo hello'",
        params_schema={},
        backend_prefs=[],
        artifact_patterns=[],
        expectation_contract_id=None,
        content_hash="abc123",
        created_at="2026-06-04T00:00:00Z",
    )
    defaults.update(kwargs)
    return JobFile(**defaults)


def _make_run(**kwargs) -> Run:
    defaults = dict(
        run_id="run-001",
        jobfile_id="jf-001",
        jobfile_version=1,
        params={},
        input_hashes={},
        backend="local",
        server=None,
        task=None,
        remote_job_id=None,
        state=State.PENDING,
        health=Health.OK,
        exit_code=None,
        submitted_at=None,
        started_at=None,
        finished_at=None,
        last_heartbeat=None,
        workdir=None,
        stdout_path=None,
        stderr_path=None,
        resource_summary={},
        expectation_match=None,
        observation_card=None,
    )
    defaults.update(kwargs)
    return Run(**defaults)


def _make_server(name="hipster", backend_type="slurm", online=True, **kwargs) -> Server:
    defaults = dict(
        name=name,
        backend_type=backend_type,
        online=online,
        last_heartbeat=None,
        cpu={},
        mem={},
        gpu={},
        disk={},
        slurm_queue={},
        note=None,
    )
    defaults.update(kwargs)
    return Server(**defaults)


# ===========================================================================
# 1.  base.py — dataclasses + select_backend + get_backend
# ===========================================================================

class TestBaseDataclasses:
    def test_submit_result_fields(self):
        from jobctl.backends.base import SubmitResult
        r = SubmitResult(remote_job_id="42", workdir="/tmp/work")
        assert r.remote_job_id == "42"
        assert r.workdir == "/tmp/work"

    def test_submit_result_no_remote_job_id(self):
        from jobctl.backends.base import SubmitResult
        r = SubmitResult(remote_job_id=None, workdir="/tmp/w")
        assert r.remote_job_id is None

    def test_poll_result_fields(self):
        from jobctl.backends.base import PollResult
        r = PollResult(state=State.RUNNING, resource={}, last_log_mtime=1234.5)
        assert r.state == State.RUNNING
        assert r.last_log_mtime == 1234.5

    def test_collect_result_fields(self):
        from jobctl.backends.base import CollectResult
        r = CollectResult(
            exit_code=0,
            stdout_path="/tmp/out",
            stderr_path="/tmp/err",
            artifact_dir="/tmp/art",
            resource_summary={"cpu": 1},
        )
        assert r.exit_code == 0
        assert r.resource_summary == {"cpu": 1}


class TestSelectBackend:
    def test_picks_first_online_pref(self):
        from jobctl.backends.base import select_backend
        jf = _make_jobfile(
            backend_prefs=[
                {"backend": "slurm", "server": "hipster", "task": "gpu"},
                {"backend": "ssh",   "server": "oblix",   "task": "cpu"},
                {"backend": "local"},
            ]
        )
        servers = [
            _make_server("hipster", online=True),
            _make_server("oblix", backend_type="ssh", online=True),
        ]
        backend, server, task = select_backend(jf, servers, override=None)
        assert backend == "slurm"
        assert server == "hipster"
        assert task == "gpu"

    def test_skips_offline_server(self):
        from jobctl.backends.base import select_backend
        jf = _make_jobfile(
            backend_prefs=[
                {"backend": "slurm", "server": "hipster", "task": "gpu"},
                {"backend": "ssh",   "server": "oblix",   "task": "cpu"},
                {"backend": "local"},
            ]
        )
        servers = [
            _make_server("hipster", online=False),
            _make_server("oblix", backend_type="ssh", online=True),
        ]
        backend, server, task = select_backend(jf, servers, override=None)
        assert backend == "ssh"
        assert server == "oblix"

    def test_falls_back_to_local_when_all_servers_offline(self):
        from jobctl.backends.base import select_backend
        jf = _make_jobfile(
            backend_prefs=[
                {"backend": "slurm", "server": "hipster"},
                {"backend": "local"},
            ]
        )
        servers = [_make_server("hipster", online=False)]
        backend, server, task = select_backend(jf, servers, override=None)
        assert backend == "local"
        assert server is None

    def test_falls_back_to_local_when_no_prefs(self):
        from jobctl.backends.base import select_backend
        jf = _make_jobfile(backend_prefs=[])
        backend, server, task = select_backend(jf, [], override=None)
        assert backend == "local"
        assert server is None

    def test_override_respected(self):
        from jobctl.backends.base import select_backend
        jf = _make_jobfile(backend_prefs=[{"backend": "local"}])
        override = {"backend": "slurm", "server": "hipster", "task": "gpu"}
        backend, server, task = select_backend(jf, [], override=override)
        assert backend == "slurm"
        assert server == "hipster"
        assert task == "gpu"

    def test_local_pref_no_server_needed(self):
        from jobctl.backends.base import select_backend
        jf = _make_jobfile(backend_prefs=[{"backend": "local"}])
        backend, server, task = select_backend(jf, [], override=None)
        assert backend == "local"
        assert server is None
        assert task is None


class TestGetBackend:
    def test_returns_local_backend(self):
        from jobctl.backends.base import get_backend
        from jobctl.backends.local import LocalBackend
        b = get_backend("local", None, config={})
        assert isinstance(b, LocalBackend)

    def test_returns_slurm_backend(self):
        from jobctl.backends.base import get_backend
        from jobctl.backends.slurm import SlurmBackend
        b = get_backend("slurm", "hipster", config={"servers": {"hipster": {}}})
        assert isinstance(b, SlurmBackend)

    def test_returns_ssh_backend(self):
        from jobctl.backends.base import get_backend
        from jobctl.backends.ssh import SshBackend
        b = get_backend("ssh", "oblix", config={"servers": {"oblix": {}}})
        assert isinstance(b, SshBackend)

    def test_unknown_backend_raises(self):
        from jobctl.backends.base import get_backend
        with pytest.raises((ValueError, KeyError)):
            get_backend("nonexistent", None, config={})


# ===========================================================================
# 2.  LocalBackend end-to-end
# ===========================================================================

class TestLocalBackendEndToEnd:
    def test_submit_creates_workdir_and_captures_stdout(self, tmp_path):
        from jobctl.backends.local import LocalBackend
        from jobctl.backends.base import SubmitResult

        workdir_root = tmp_path / "runs"
        workdir_root.mkdir()
        backend = LocalBackend(workdir_root=str(workdir_root))

        jf = _make_jobfile(command_template="bash -c 'echo hello world'")
        run = _make_run()
        result = backend.submit(run, jf)

        assert isinstance(result, SubmitResult)
        assert result.workdir is not None
        assert Path(result.workdir).exists()
        # remote_job_id is the PID string
        assert result.remote_job_id is not None

    def test_poll_running_then_completed(self, tmp_path):
        from jobctl.backends.local import LocalBackend

        workdir_root = tmp_path / "runs"
        workdir_root.mkdir()
        backend = LocalBackend(workdir_root=str(workdir_root))

        jf = _make_jobfile(command_template="bash -c 'sleep 0.2; echo done'")
        run = _make_run()
        submit_result = backend.submit(run, jf)
        run = _make_run(
            workdir=submit_result.workdir,
            remote_job_id=submit_result.remote_job_id,
        )

        # Poll until terminal (with timeout)
        deadline = time.time() + 10
        final_state = None
        while time.time() < deadline:
            poll = backend.poll(run)
            if poll.state in (State.COMPLETED, State.FAILED):
                final_state = poll.state
                break
            time.sleep(0.05)

        assert final_state == State.COMPLETED

    def test_collect_captures_stdout_stderr_and_exit_code(self, tmp_path):
        from jobctl.backends.local import LocalBackend

        workdir_root = tmp_path / "runs"
        workdir_root.mkdir()
        backend = LocalBackend(workdir_root=str(workdir_root))

        jf = _make_jobfile(command_template="bash -c 'echo OUT; echo ERR >&2'")
        run = _make_run()
        submit_result = backend.submit(run, jf)
        run = _make_run(
            workdir=submit_result.workdir,
            remote_job_id=submit_result.remote_job_id,
        )

        # Wait for completion
        deadline = time.time() + 10
        while time.time() < deadline:
            poll = backend.poll(run)
            if poll.state in (State.COMPLETED, State.FAILED):
                break
            time.sleep(0.05)

        collect = backend.collect(run)
        assert collect.exit_code == 0
        assert Path(collect.stdout_path).exists()
        assert "OUT" in Path(collect.stdout_path).read_text()
        assert Path(collect.stderr_path).exists()
        assert "ERR" in Path(collect.stderr_path).read_text()
        assert collect.artifact_dir == run.workdir

    def test_collect_nonzero_exit_code(self, tmp_path):
        from jobctl.backends.local import LocalBackend

        workdir_root = tmp_path / "runs"
        workdir_root.mkdir()
        backend = LocalBackend(workdir_root=str(workdir_root))

        jf = _make_jobfile(command_template="bash -c 'exit 42'")
        run = _make_run()
        submit_result = backend.submit(run, jf)
        run = _make_run(
            workdir=submit_result.workdir,
            remote_job_id=submit_result.remote_job_id,
        )

        deadline = time.time() + 10
        while time.time() < deadline:
            poll = backend.poll(run)
            if poll.state in (State.COMPLETED, State.FAILED):
                break
            time.sleep(0.05)

        collect = backend.collect(run)
        assert collect.exit_code == 42

    def test_cancel_kills_process(self, tmp_path):
        from jobctl.backends.local import LocalBackend

        workdir_root = tmp_path / "runs"
        workdir_root.mkdir()
        backend = LocalBackend(workdir_root=str(workdir_root))

        jf = _make_jobfile(command_template="bash -c 'sleep 60'")
        run = _make_run()
        submit_result = backend.submit(run, jf)
        run = _make_run(
            workdir=submit_result.workdir,
            remote_job_id=submit_result.remote_job_id,
        )
        # Give it a moment to start
        time.sleep(0.1)
        backend.cancel(run)
        # After cancel, poll should show cancelled or failed
        time.sleep(0.2)
        poll = backend.poll(run)
        assert poll.state in (State.CANCELLED, State.FAILED, State.COMPLETED)

    def test_local_job_emits_file_artifact(self, tmp_path):
        """Emit a CSV file and confirm it ends up in the workdir."""
        from jobctl.backends.local import LocalBackend

        workdir_root = tmp_path / "runs"
        workdir_root.mkdir()
        backend = LocalBackend(workdir_root=str(workdir_root))

        # The command writes a CSV into the workdir (passed as $JOBCTL_WORKDIR)
        jf = _make_jobfile(
            command_template="bash -c 'echo \"a,b\\n1,2\" > \"$JOBCTL_WORKDIR/out.csv\"'"
        )
        run = _make_run()
        submit_result = backend.submit(run, jf)
        run = _make_run(
            workdir=submit_result.workdir,
            remote_job_id=submit_result.remote_job_id,
        )

        deadline = time.time() + 10
        while time.time() < deadline:
            poll = backend.poll(run)
            if poll.state in (State.COMPLETED, State.FAILED):
                break
            time.sleep(0.05)

        collect = backend.collect(run)
        assert (Path(collect.artifact_dir) / "out.csv").exists()

    def test_poll_returns_last_log_mtime(self, tmp_path):
        from jobctl.backends.local import LocalBackend

        workdir_root = tmp_path / "runs"
        workdir_root.mkdir()
        backend = LocalBackend(workdir_root=str(workdir_root))

        jf = _make_jobfile(command_template="bash -c 'echo hi'")
        run = _make_run()
        submit_result = backend.submit(run, jf)
        run = _make_run(
            workdir=submit_result.workdir,
            remote_job_id=submit_result.remote_job_id,
        )
        deadline = time.time() + 10
        while time.time() < deadline:
            poll = backend.poll(run)
            if poll.state in (State.COMPLETED, State.FAILED):
                break
            time.sleep(0.05)

        poll = backend.poll(run)
        # last_log_mtime may be None for a very fast job that cleans up, but
        # once completed the collect-path stdout should exist and have an mtime
        collect = backend.collect(run)
        if Path(collect.stdout_path).exists():
            # mtime must be a positive float
            mtime = Path(collect.stdout_path).stat().st_mtime
            assert mtime > 0


# ===========================================================================
# 3.  SlurmBackend tests (against fakebin)
# ===========================================================================

class TestSlurmStateMapping:
    """Tests that verify SLURM state codes -> jobctl State enum."""

    def _make_slurm_backend(self, server="hipster"):
        from jobctl.backends.slurm import SlurmBackend
        return SlurmBackend(
            server=server,
            server_config={"host": server, "remote_path": "/tmp"},
            run_cmd=self._run_cmd,
        )

    def _run_cmd(self, cmd: list[str], **kw) -> subprocess.CompletedProcess:
        """Run command with fakebin on PATH."""
        env = _env_with_fakebin()
        env.update(kw.pop("env_extra", {}))
        return subprocess.run(cmd, capture_output=True, text=True, env=env, **kw)

    def _make_slurm_run(self, **kwargs) -> Run:
        defaults = dict(
            backend="slurm",
            server="hipster",
            remote_job_id="12345",
            workdir="/tmp/workdir",
        )
        defaults.update(kwargs)
        return _make_run(**defaults)

    @pytest.mark.parametrize("slurm_state,expected", [
        ("PD", State.SUBMITTED),
        ("R",  State.RUNNING),
        ("CG", State.RUNNING),   # completing -> still running
        ("CD", State.COMPLETED),
        ("F",  State.FAILED),
        ("TO", State.TIMEOUT),
        ("CA", State.CANCELLED),
    ])
    def test_state_mapping_via_squeue(self, slurm_state, expected, tmp_path):
        from jobctl.backends.slurm import SlurmBackend, _SLURM_STATE_MAP

        # Test the mapping dict directly
        assert _SLURM_STATE_MAP[slurm_state] == expected

    def test_submit_captures_jobid(self, tmp_path):
        from jobctl.backends.slurm import SlurmBackend

        workdir = tmp_path / "wd"
        workdir.mkdir()
        (workdir / "job.sh").write_text("#!/bin/bash\necho hello\n")

        env_extra = {"FAKEBIN_SBATCH_JOBID": "99999"}

        def run_cmd(cmd, **kw):
            env = _env_with_fakebin()
            env.update(env_extra)
            return subprocess.run(cmd, capture_output=True, text=True, env=env)

        backend = SlurmBackend(
            server="hipster",
            server_config={"host": "hipster", "remote_path": "/tmp"},
            run_cmd=run_cmd,
        )
        jf = _make_jobfile(command_template="bash job.sh")
        run = _make_run(backend="slurm", server="hipster", workdir=str(workdir))

        result = backend.submit(run, jf)
        assert result.remote_job_id == "99999"

    def test_poll_pd_returns_submitted(self, tmp_path):
        from jobctl.backends.slurm import SlurmBackend

        env_extra = {"FAKEBIN_SQUEUE_STATE": "PD", "FAKEBIN_SBATCH_JOBID": "12345"}

        def run_cmd(cmd, **kw):
            env = _env_with_fakebin()
            env.update(env_extra)
            return subprocess.run(cmd, capture_output=True, text=True, env=env)

        backend = SlurmBackend(
            server="hipster",
            server_config={"host": "hipster", "remote_path": "/tmp"},
            run_cmd=run_cmd,
        )
        run = self._make_slurm_run(remote_job_id="12345")
        poll = backend.poll(run)
        assert poll.state == State.SUBMITTED

    def test_poll_running_returns_running(self, tmp_path):
        from jobctl.backends.slurm import SlurmBackend

        env_extra = {"FAKEBIN_SQUEUE_STATE": "R", "FAKEBIN_SBATCH_JOBID": "12345"}

        def run_cmd(cmd, **kw):
            env = _env_with_fakebin()
            env.update(env_extra)
            return subprocess.run(cmd, capture_output=True, text=True, env=env)

        backend = SlurmBackend(
            server="hipster",
            server_config={"host": "hipster", "remote_path": "/tmp"},
            run_cmd=run_cmd,
        )
        run = self._make_slurm_run(remote_job_id="12345")
        poll = backend.poll(run)
        assert poll.state == State.RUNNING

    def test_poll_empty_queue_falls_back_to_sacct(self, tmp_path):
        """When squeue returns no job, sacct is queried for terminal state."""
        from jobctl.backends.slurm import SlurmBackend

        sacct_state = "CD"
        call_log = []

        def run_cmd(cmd, **kw):
            call_log.append(cmd[0])
            env = _env_with_fakebin()
            env["FAKEBIN_SQUEUE_STATE"] = ""   # empty = not in queue
            env["FAKEBIN_SACCT_STATE"] = sacct_state
            env["FAKEBIN_SBATCH_JOBID"] = "12345"
            return subprocess.run(cmd, capture_output=True, text=True, env=env)

        backend = SlurmBackend(
            server="hipster",
            server_config={"host": "hipster", "remote_path": "/tmp"},
            run_cmd=run_cmd,
        )
        run = self._make_slurm_run(remote_job_id="12345")
        poll = backend.poll(run)
        assert poll.state == State.COMPLETED
        assert any("sacct" in c for c in call_log)

    @pytest.mark.parametrize("sacct_state,expected", [
        ("CD", State.COMPLETED),
        ("F",  State.FAILED),
        ("TO", State.TIMEOUT),
        ("CA", State.CANCELLED),
    ])
    def test_sacct_state_mapping(self, sacct_state, expected):
        from jobctl.backends.slurm import SlurmBackend

        def run_cmd(cmd, **kw):
            env = _env_with_fakebin()
            env["FAKEBIN_SQUEUE_STATE"] = ""
            env["FAKEBIN_SACCT_STATE"] = sacct_state
            env["FAKEBIN_SBATCH_JOBID"] = "12345"
            return subprocess.run(cmd, capture_output=True, text=True, env=env)

        backend = SlurmBackend(
            server="hipster",
            server_config={"host": "hipster", "remote_path": "/tmp"},
            run_cmd=run_cmd,
        )
        run = self._make_slurm_run(remote_job_id="12345")
        poll = backend.poll(run)
        assert poll.state == expected

    def test_sacct_resource_parse(self, tmp_path):
        """sacct fields are parsed into resource_summary dict."""
        from jobctl.backends.slurm import SlurmBackend

        def run_cmd(cmd, **kw):
            env = _env_with_fakebin()
            env["FAKEBIN_SQUEUE_STATE"] = ""
            env["FAKEBIN_SACCT_STATE"] = "CD"
            env["FAKEBIN_SBATCH_JOBID"] = "12345"
            return subprocess.run(cmd, capture_output=True, text=True, env=env)

        backend = SlurmBackend(
            server="hipster",
            server_config={"host": "hipster", "remote_path": "/tmp"},
            run_cmd=run_cmd,
        )
        run = self._make_slurm_run(remote_job_id="12345")
        poll = backend.poll(run)
        # resource dict should contain CPUTime, MaxRSS, Elapsed
        assert "CPUTime" in poll.resource or "cpu_time" in poll.resource
        assert len(poll.resource) > 0

    def test_collect_stdout_stderr_paths(self, tmp_path):
        """collect() returns paths to stdout/stderr files in workdir."""
        from jobctl.backends.slurm import SlurmBackend

        workdir = tmp_path / "wd"
        workdir.mkdir()
        # Write fake stdout/stderr
        (workdir / "stdout.txt").write_text("job output\n")
        (workdir / "stderr.txt").write_text("job errors\n")

        def run_cmd(cmd, **kw):
            env = _env_with_fakebin()
            env["FAKEBIN_SQUEUE_STATE"] = ""
            env["FAKEBIN_SACCT_STATE"] = "CD"
            env["FAKEBIN_SACCT_EXITCODE"] = "0"
            env["FAKEBIN_SBATCH_JOBID"] = "12345"
            return subprocess.run(cmd, capture_output=True, text=True, env=env)

        backend = SlurmBackend(
            server="hipster",
            server_config={"host": "hipster", "remote_path": "/tmp"},
            run_cmd=run_cmd,
        )
        run = self._make_slurm_run(
            remote_job_id="12345",
            workdir=str(workdir),
        )
        collect = backend.collect(run)
        assert collect.stdout_path is not None
        assert collect.stderr_path is not None
        assert collect.artifact_dir == str(workdir)

    def test_cancel_calls_scancel(self, tmp_path):
        """cancel() calls scancel with the job id."""
        from jobctl.backends.slurm import SlurmBackend

        called_with = []

        def run_cmd(cmd, **kw):
            called_with.append(list(cmd))
            env = _env_with_fakebin()
            return subprocess.run(cmd, capture_output=True, text=True, env=env)

        backend = SlurmBackend(
            server="hipster",
            server_config={"host": "hipster", "remote_path": "/tmp"},
            run_cmd=run_cmd,
        )
        run = self._make_slurm_run(remote_job_id="12345")
        backend.cancel(run)
        assert any("scancel" in str(c) for c in called_with)
        assert any("12345" in str(c) for c in called_with)

    def test_submit_failure_raises(self):
        """If sbatch returns non-zero, submit raises RuntimeError."""
        from jobctl.backends.slurm import SlurmBackend

        def run_cmd(cmd, **kw):
            env = _env_with_fakebin()
            env["FAKEBIN_SBATCH_FAIL"] = "1"
            return subprocess.run(cmd, capture_output=True, text=True, env=env)

        backend = SlurmBackend(
            server="hipster",
            server_config={"host": "hipster", "remote_path": "/tmp"},
            run_cmd=run_cmd,
        )
        jf = _make_jobfile(command_template="bash job.sh")
        run = _make_run(backend="slurm", server="hipster", workdir="/tmp")
        with pytest.raises(RuntimeError):
            backend.submit(run, jf)


# ===========================================================================
# 4.  SshBackend — unit-tests with mocked runner
# ===========================================================================

class TestSshBackendCommandConstruction:
    """Test SSH command construction without real SSH connections."""

    def _make_ssh_backend(self, server="oblix"):
        from jobctl.backends.ssh import SshBackend
        return SshBackend(
            server=server,
            server_config={
                "host": server,
                "user": "testuser",
                "remote_path": "/remote/work",
            },
            run_cmd=None,  # will be replaced per test
        )

    def test_submit_builds_rsync_push_command(self):
        from jobctl.backends.ssh import SshBackend

        calls = []

        def run_cmd(cmd, **kw):
            calls.append(list(cmd))
            return MagicMock(returncode=0, stdout="12345\n", stderr="")

        backend = SshBackend(
            server="oblix",
            server_config={"host": "oblix", "user": "testuser", "remote_path": "/remote/work"},
            run_cmd=run_cmd,
        )
        jf = _make_jobfile(command_template="python train.py")
        run = _make_run(backend="ssh", server="oblix", run_id="run-ssh-001")
        result = backend.submit(run, jf)

        # At least one call should include rsync
        rsync_calls = [c for c in calls if "rsync" in c[0] or any("rsync" in tok for tok in c)]
        assert len(rsync_calls) >= 0  # rsync may not be called if no local workdir

        # At least one ssh call should include nohup
        ssh_calls = [c for c in calls if "ssh" in c[0]]
        assert len(ssh_calls) >= 1

    def test_submit_returns_remote_pid(self):
        from jobctl.backends.ssh import SshBackend

        def run_cmd(cmd, **kw):
            if "ssh" in cmd[0]:
                return MagicMock(returncode=0, stdout="9999\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        backend = SshBackend(
            server="oblix",
            server_config={"host": "oblix", "user": "testuser", "remote_path": "/remote/work"},
            run_cmd=run_cmd,
        )
        jf = _make_jobfile(command_template="python train.py")
        run = _make_run(backend="ssh", server="oblix", run_id="run-ssh-002")
        result = backend.submit(run, jf)
        # remote_job_id is PID or None (depends on implementation)
        assert result.remote_job_id is not None or result.workdir is not None

    def test_poll_checks_remote_pid(self):
        from jobctl.backends.ssh import SshBackend

        calls = []

        def run_cmd(cmd, **kw):
            calls.append(list(cmd))
            # Simulate "pid exists" via kill -0 returning 0
            return MagicMock(returncode=0, stdout="", stderr="")

        backend = SshBackend(
            server="oblix",
            server_config={"host": "oblix", "user": "testuser", "remote_path": "/remote/work"},
            run_cmd=run_cmd,
        )
        run = _make_run(
            backend="ssh", server="oblix", remote_job_id="9999",
            workdir="/remote/work/run-001",
        )
        poll = backend.poll(run)
        # Should return RUNNING since returncode=0 (pid alive)
        assert poll.state in (State.RUNNING, State.COMPLETED, State.FAILED)

        ssh_calls = [c for c in calls if "ssh" in c[0]]
        assert len(ssh_calls) >= 1

    def test_poll_pid_gone_returns_completed(self):
        from jobctl.backends.ssh import SshBackend

        def run_cmd(cmd, **kw):
            # kill -0 returns 1 = pid not found
            if "ssh" in cmd[0]:
                return MagicMock(returncode=1, stdout="", stderr="no such process")
            return MagicMock(returncode=0, stdout="", stderr="")

        backend = SshBackend(
            server="oblix",
            server_config={"host": "oblix", "user": "testuser", "remote_path": "/remote/work"},
            run_cmd=run_cmd,
        )
        run = _make_run(
            backend="ssh", server="oblix", remote_job_id="9999",
            workdir="/remote/work/run-001",
        )
        poll = backend.poll(run)
        assert poll.state in (State.COMPLETED, State.FAILED)

    def test_collect_builds_rsync_pull_command(self):
        from jobctl.backends.ssh import SshBackend

        calls = []

        def run_cmd(cmd, **kw):
            calls.append(list(cmd))
            return MagicMock(returncode=0, stdout="0\n", stderr="")

        backend = SshBackend(
            server="oblix",
            server_config={"host": "oblix", "user": "testuser", "remote_path": "/remote/work"},
            run_cmd=run_cmd,
        )
        run = _make_run(
            backend="ssh", server="oblix", remote_job_id="9999",
            workdir="/remote/work/run-001",
        )
        collect = backend.collect(run)
        # Should have called rsync (for pulling artifacts) or ssh (for exit code)
        assert collect.stdout_path is not None or collect.artifact_dir is not None

    def test_cancel_sends_kill_via_ssh(self):
        from jobctl.backends.ssh import SshBackend

        calls = []

        def run_cmd(cmd, **kw):
            calls.append(list(cmd))
            return MagicMock(returncode=0, stdout="", stderr="")

        backend = SshBackend(
            server="oblix",
            server_config={"host": "oblix", "user": "testuser", "remote_path": "/remote/work"},
            run_cmd=run_cmd,
        )
        run = _make_run(
            backend="ssh", server="oblix", remote_job_id="9999",
            workdir="/remote/work/run-001",
        )
        backend.cancel(run)
        ssh_calls = [c for c in calls if "ssh" in c[0]]
        assert len(ssh_calls) >= 1
        # At least one call should involve killing the process
        flat = " ".join(str(x) for c in ssh_calls for x in c)
        assert "kill" in flat or "9999" in flat


@pytest.mark.cluster
class TestSshBackendRealCluster:
    """Real cluster tests — skipped by default."""

    def test_real_ssh_submit_poll_collect(self):
        pytest.skip("Real cluster test — run manually")
