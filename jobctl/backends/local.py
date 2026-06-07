"""LocalBackend: runs jobs as local subprocesses in a temp workdir."""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from jobctl.backends.base import Backend, CollectResult, PollResult, SubmitResult, resolved_command
from jobctl.db.models import Health, State

if TYPE_CHECKING:
    from jobctl.db.models import JobFile, Run


class LocalBackend(Backend):
    """Runs jobs as local subprocesses.

    Each job gets its own subdirectory under *workdir_root*.  The subprocess
    runs with ``JOBCTL_WORKDIR`` injected so scripts can write artifacts to it.
    stdout/stderr are captured to ``stdout.txt`` / ``stderr.txt`` in that dir.
    A ``pid.txt`` file is written immediately after spawning so ``poll()`` can
    check liveness without storing state in memory.
    """

    name = "local"

    def __init__(self, workdir_root: str | None = None) -> None:
        if workdir_root is None:
            workdir_root = str(Path(os.environ.get("JOBCTL_HOME", str(Path.home() / ".jobctl"))) / "runs")
        self._workdir_root = workdir_root
        Path(self._workdir_root).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    def submit(self, run: "Run", jobfile: "JobFile") -> SubmitResult:
        """Spawn the job as a subprocess; return its PID and workdir."""
        workdir = Path(self._workdir_root) / run.run_id
        workdir.mkdir(parents=True, exist_ok=True)

        stdout_path = workdir / "stdout.txt"
        stderr_path = workdir / "stderr.txt"
        exit_code_path = workdir / "exit_code.txt"

        env = os.environ.copy()
        env["JOBCTL_WORKDIR"] = str(workdir)
        env["JOBCTL_RUN_ID"] = run.run_id

        # Wrap the command so we capture the exit code to a file
        wrapper = (
            f"({resolved_command(run, jobfile)}); echo $? > {exit_code_path}"
        )

        proc = subprocess.Popen(
            ["bash", "-c", wrapper],
            stdout=open(stdout_path, "w"),
            stderr=open(stderr_path, "w"),
            cwd=str(workdir),
            env=env,
            start_new_session=True,  # detach from our process group
        )

        pid_path = workdir / "pid.txt"
        pid_path.write_text(str(proc.pid))

        return SubmitResult(remote_job_id=str(proc.pid), workdir=str(workdir))

    def poll(self, run: "Run") -> PollResult:
        """Check if the subprocess is still running.

        A job is considered complete when the ``exit_code.txt`` sentinel file
        appears (written by the bash wrapper as its very last action).  Before
        that file exists we use the PID to decide running vs pending; a zombie
        process (kill -0 succeeds but state == Z) is treated as finished.
        """
        workdir = Path(run.workdir) if run.workdir else None
        if workdir is None:
            return PollResult(state=State.FAILED, resource={}, last_log_mtime=None)

        exit_code_path = workdir / "exit_code.txt"

        # Determine last log mtime
        last_log_mtime: float | None = None
        stdout_path = workdir / "stdout.txt"
        if stdout_path.exists():
            try:
                last_log_mtime = stdout_path.stat().st_mtime
            except OSError:
                pass

        # The exit_code.txt file is written by the bash wrapper as its last
        # action.  Its presence is the definitive "done" signal.
        if exit_code_path.exists():
            try:
                code = int(exit_code_path.read_text().strip())
            except (ValueError, OSError):
                code = 1
            state = State.COMPLETED if code == 0 else State.FAILED
            return PollResult(state=state, resource={"exit_code": code}, last_log_mtime=last_log_mtime)

        # No exit_code.txt yet — check PID
        pid = self._read_pid(workdir)
        if pid is not None:
            alive = self._pid_alive(pid)
            if alive and not self._pid_is_zombie(pid):
                return PollResult(state=State.RUNNING, resource={}, last_log_mtime=last_log_mtime)
            # Zombie or dead — give the wrapper a moment to write exit_code.txt
            time.sleep(0.05)
            if exit_code_path.exists():
                try:
                    code = int(exit_code_path.read_text().strip())
                except (ValueError, OSError):
                    code = 1
                state = State.COMPLETED if code == 0 else State.FAILED
                return PollResult(state=state, resource={"exit_code": code}, last_log_mtime=last_log_mtime)
            # Still no file — process ended without writing exit code
            return PollResult(state=State.COMPLETED, resource={}, last_log_mtime=last_log_mtime)

        # No PID file — assume not yet started or already finished
        return PollResult(state=State.RUNNING, resource={}, last_log_mtime=last_log_mtime)

    def collect(self, run: "Run") -> CollectResult:
        """Return paths to stdout/stderr and the exit code."""
        workdir = Path(run.workdir) if run.workdir else Path(".")
        stdout_path = workdir / "stdout.txt"
        stderr_path = workdir / "stderr.txt"
        exit_code_path = workdir / "exit_code.txt"

        exit_code: int | None = None
        if exit_code_path.exists():
            try:
                exit_code = int(exit_code_path.read_text().strip())
            except (ValueError, OSError):
                exit_code = None

        return CollectResult(
            exit_code=exit_code,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            artifact_dir=str(workdir),
            resource_summary={},
        )

    def cancel(self, run: "Run") -> None:
        """Kill the subprocess by PID (kills the whole process group)."""
        workdir = Path(run.workdir) if run.workdir else None
        if workdir is None:
            return
        pid = self._read_pid(workdir)
        if pid is None:
            return
        # Try to kill the whole process group first (most effective for children)
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        # Also directly kill the PID
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        # Write a sentinel exit code so poll() sees it as done
        if workdir:
            try:
                (workdir / "exit_code.txt").write_text("-1")
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_pid(workdir: Path) -> int | None:
        pid_path = workdir / "pid.txt"
        if not pid_path.exists():
            return None
        try:
            return int(pid_path.read_text().strip())
        except (ValueError, OSError):
            return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    @staticmethod
    def _pid_is_zombie(pid: int) -> bool:
        """Return True if the process is a zombie (Z state)."""
        try:
            stat_file = Path(f"/proc/{pid}/status")
            if stat_file.exists():
                for line in stat_file.read_text().splitlines():
                    if line.startswith("State:"):
                        return "Z" in line
        except OSError:
            pass
        # On macOS use ps
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "stat="],
                capture_output=True, text=True, timeout=2,
            )
            return "Z" in result.stdout
        except Exception:
            return False
