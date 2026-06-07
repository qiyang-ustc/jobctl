"""SshBackend: runs jobs on a remote host via SSH + rsync."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from jobctl.backends.base import Backend, CollectResult, PollResult, SubmitResult, resolved_command
from jobctl.db.models import Health, State

if TYPE_CHECKING:
    from jobctl.db.models import JobFile, Run


def _default_run_cmd(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _resolve_remote_path(raw: str, jobfile_name: str = "runs") -> str:
    """Expand ~ and substitute {project} with jobfile name (slug)."""
    slug = jobfile_name.replace(" ", "_").replace("/", "_") or "runs"
    path = raw.format(project=slug)
    # Tilde in remote paths is interpreted by the remote shell, leave as-is
    # unless it would confuse rsync (rsync handles ~/... fine via SSH)
    return path


class SshBackend(Backend):
    """Backend that runs jobs on a remote host via SSH.

    Lifecycle:
    1. ``submit``:
       - (Optional) rsync local workdir to remote.
       - Launch command via ``nohup … & echo $!`` to get the remote PID.
       - Write a pidfile on the remote.
    2. ``poll``:
       - SSH ``kill -0 <pid>`` — zero exit means process is alive.
    3. ``collect``:
       - Rsync artifact dir back to a local mirror.
       - Read exit code from a remote exit-code file.
    4. ``cancel``:
       - SSH ``kill <pid>``.
    """

    name = "ssh"

    def __init__(
        self,
        server: str,
        server_config: dict,
        run_cmd: Callable | None = None,
    ) -> None:
        self._server = server
        self._host = server_config.get("host", server)
        self._user = server_config.get("user")
        self._remote_path_template = server_config.get("remote_path", f"/tmp/jobctl/{server}")
        self._local_run_dir = server_config.get("run_dir") or str(
            Path(os.environ.get("JOBCTL_HOME", str(Path.home() / ".jobctl"))) / "runs"
        )
        self._run_cmd = run_cmd or _default_run_cmd

    def _remote_path(self, jobfile_name: str = "runs") -> str:
        return _resolve_remote_path(self._remote_path_template, jobfile_name)

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    def submit(self, run: "Run", jobfile: "JobFile") -> SubmitResult:
        """Launch job on remote host via nohup; return PID as remote_job_id."""
        base = self._remote_path(jobfile.name if jobfile else "runs")
        remote_workdir = f"{base}/{run.run_id}"
        stdout_path = f"{remote_workdir}/stdout.txt"
        stderr_path = f"{remote_workdir}/stderr.txt"
        exit_code_path = f"{remote_workdir}/exit_code.txt"
        pid_file = f"{remote_workdir}/pid.txt"

        # Create remote workdir
        self._ssh(f"mkdir -p {remote_workdir}")

        # Build the remote command:
        # nohup bash -c '... ; echo $? > exit_code.txt' > stdout.txt 2> stderr.txt &
        # echo $! > pid.txt
        # cd into workdir so relative paths in the command (e.g. results.csv) land there
        inner = f"cd {remote_workdir} && ({resolved_command(run, jobfile)}); echo $? > {exit_code_path}"
        remote_cmd = (
            f"nohup bash -c {_shell_quote(inner)} "
            f"> {stdout_path} 2> {stderr_path} & "
            f"echo $! | tee {pid_file}"
        )
        result = self._ssh(remote_cmd)

        pid: str | None = None
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if line.isdigit():
                pid = line
                break

        return SubmitResult(remote_job_id=pid, workdir=remote_workdir)

    def poll(self, run: "Run") -> PollResult:
        """Check if the remote process is alive via kill -0.

        A unique marker disambiguates "process alive/dead" (a successful SSH
        whose payload tells us the truth) from "SSH itself failed" (unreachable
        → reachable=False, the monitor leaves the run untouched). Previously an
        SSH failure was misread as COMPLETED.
        """
        pid = run.remote_job_id
        if not pid:
            return PollResult(state=State.FAILED, resource={})

        keep = run.state if isinstance(run.state, State) else State.RUNNING
        result = self._ssh(f"kill -0 {pid} 2>/dev/null; echo ALIVE:$?")
        out = result.stdout.strip()

        if "ALIVE:0" in out:
            return PollResult(state=State.RUNNING, resource={})
        if "ALIVE:1" in out:
            # Process is gone — it finished (exit code resolved in collect()).
            return PollResult(state=State.COMPLETED, resource={})
        # No marker → the SSH command itself failed: unreachable, not finished.
        return PollResult(state=keep, resource={}, reachable=False)

    def collect(self, run: "Run") -> CollectResult:
        """Rsync artifacts back and return paths + exit code."""
        remote_workdir = run.workdir or f"{self._remote_path()}/{run.run_id}"

        # Local mirror directory
        local_mirror = str(Path(self._local_run_dir) / run.run_id)
        Path(local_mirror).mkdir(parents=True, exist_ok=True)

        # rsync pull: remote -> local
        # Use server SSH alias directly (honours ~/.ssh/config)
        remote_spec = f"{self._host}:{remote_workdir}/"
        rsync_cmd = ["rsync", "-az", remote_spec, local_mirror + "/"]
        self._run_cmd(rsync_cmd)

        # Read exit code from local mirror
        exit_code_file = Path(local_mirror) / "exit_code.txt"
        exit_code: int | None = None
        if exit_code_file.exists():
            try:
                exit_code = int(exit_code_file.read_text().strip())
            except (ValueError, OSError):
                exit_code = None
        else:
            # Try reading from remote
            result = self._ssh(f"cat {remote_workdir}/exit_code.txt 2>/dev/null || echo ''")
            try:
                exit_code = int(result.stdout.strip())
            except ValueError:
                exit_code = None

        return CollectResult(
            exit_code=exit_code,
            stdout_path=str(Path(local_mirror) / "stdout.txt"),
            stderr_path=str(Path(local_mirror) / "stderr.txt"),
            artifact_dir=local_mirror,
            resource_summary={},
        )

    def cancel(self, run: "Run") -> None:
        """Kill the remote process by PID."""
        pid = run.remote_job_id
        if not pid:
            return
        self._ssh(f"kill {pid} 2>/dev/null || true")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ssh(self, remote_cmd: str) -> subprocess.CompletedProcess:
        """Run a command on the remote host via SSH (fast-fail on unreachable)."""
        user_prefix = f"{self._user}@" if self._user else ""
        cmd = [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            f"{user_prefix}{self._host}", remote_cmd,
        ]
        try:
            return self._run_cmd(cmd, timeout=30)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(cmd, returncode=255, stdout="", stderr="ssh timed out")


def _shell_quote(s: str) -> str:
    """Simple single-quote shell escaping."""
    return "'" + s.replace("'", "'\\''") + "'"
