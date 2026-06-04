"""SlurmBackend: submits jobs via sbatch; polls via squeue/sacct; collects via sacct."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from jobctl.backends.base import Backend, CollectResult, PollResult, SubmitResult
from jobctl.db.models import Health, State

if TYPE_CHECKING:
    from jobctl.db.models import JobFile, Run


# ---------------------------------------------------------------------------
# SLURM state code -> jobctl State mapping
# ---------------------------------------------------------------------------

_SLURM_STATE_MAP: dict[str, State] = {
    "PD":  State.SUBMITTED,   # PenDing
    "R":   State.RUNNING,     # Running
    "CG":  State.RUNNING,     # CompletinG (finishing, still running)
    "CD":  State.COMPLETED,   # CompleteD
    "F":   State.FAILED,      # Failed
    "TO":  State.TIMEOUT,     # TimeOut
    "CA":  State.CANCELLED,   # CAncelled
    "BF":  State.FAILED,      # Boot Failure
    "DL":  State.FAILED,      # DeadLine
    "NF":  State.FAILED,      # Node Failure
    "OOM": State.FAILED,      # Out Of Memory
    "PR":  State.FAILED,      # PReempted
    "RV":  State.FAILED,      # ReVoked
    "SE":  State.FAILED,      # Special Exit
    "ST":  State.CANCELLED,   # SToped
    "S":   State.RUNNING,     # Suspended
    "CF":  State.SUBMITTED,   # ConFiguring
    "RD":  State.SUBMITTED,   # waiting for ReservD resources
}


def _default_run_cmd(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Default implementation: run locally (for SSH-wrapped calls or local tests)."""
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


class SlurmBackend(Backend):
    """Backend that submits jobs to a SLURM cluster.

    In production the commands are SSH-wrapped (``ssh <server> <cmd>``).
    For unit tests a custom *run_cmd* callable is injected so that fakebin
    scripts can be found on the PATH without real SSH.
    """

    name = "slurm"

    def __init__(
        self,
        server: str,
        server_config: dict,
        run_cmd: Callable | None = None,
    ) -> None:
        self._server = server
        self._server_config = server_config
        self._host = server_config.get("host", server)
        self._user = server_config.get("user")
        self._remote_path = server_config.get("remote_path", f"/tmp/jobctl/{server}")
        self._run_cmd = run_cmd or self._ssh_run_cmd

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    def submit(self, run: "Run", jobfile: "JobFile") -> SubmitResult:
        """Write a minimal job script and run sbatch; capture job ID."""
        workdir = run.workdir or self._remote_path
        # Build the sbatch script inline
        script_content = "#!/bin/bash\n"
        script_content += f"#SBATCH --job-name={run.run_id}\n"
        script_content += f"#SBATCH --output={workdir}/stdout.txt\n"
        script_content += f"#SBATCH --error={workdir}/stderr.txt\n"
        script_content += "\n"
        script_content += jobfile.command_template + "\n"

        # Write script to a temp file (local; for ssh this would be rsync'd)
        import tempfile as _tmpfile
        with _tmpfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False, prefix="jobctl_"
        ) as f:
            f.write(script_content)
            script_path = f.name

        try:
            result = self._run_cmd(["sbatch", script_path])
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

        if result.returncode != 0:
            raise RuntimeError(
                f"sbatch failed (rc={result.returncode}): {result.stderr.strip()}"
            )

        job_id = self._parse_sbatch_output(result.stdout)
        if job_id is None:
            raise RuntimeError(
                f"Could not parse job ID from sbatch output: {result.stdout!r}"
            )

        return SubmitResult(remote_job_id=job_id, workdir=workdir)

    def poll(self, run: "Run") -> PollResult:
        """Query squeue; if not found, fall back to sacct for terminal state."""
        job_id = run.remote_job_id
        if not job_id:
            return PollResult(state=State.FAILED, resource={})

        # --- squeue ---
        result = self._run_cmd(["squeue", f"--job={job_id}", "--format=%i|%T|%R", "--noheader"])
        squeue_out = result.stdout.strip()

        # Filter to lines matching our job ID (skip header lines)
        state_code: str | None = None
        for line in squeue_out.splitlines():
            parts = line.split("|")
            if not parts:
                continue
            cell0 = parts[0].strip()
            # Skip header lines (JOBID is not a numeric job id)
            if not cell0.isdigit():
                continue
            if cell0 == job_id:
                if len(parts) >= 2:
                    state_code = parts[1].strip()
                break

        if state_code is not None:
            mapped = _SLURM_STATE_MAP.get(state_code, State.RUNNING)
            return PollResult(state=mapped, resource={})

        # Job not in squeue — use sacct for terminal state + resources
        return self._poll_via_sacct(job_id)

    def collect(self, run: "Run") -> CollectResult:
        """Collect terminal job info: exit code from sacct, stdout/stderr paths."""
        job_id = run.remote_job_id or ""
        workdir = run.workdir or "."

        # Get exit code from sacct
        exit_code: int | None = None
        resource_summary: dict = {}
        if job_id:
            sacct_result = self._run_cmd([
                "sacct",
                f"--jobs={job_id}",
                "--format=JobID,State,ExitCode,CPUTime,MaxRSS,Elapsed",
                "--noheader",
                "--parsable2",
            ])
            exit_code, resource_summary = self._parse_sacct(sacct_result.stdout, job_id)

        stdout_path = str(Path(workdir) / "stdout.txt")
        stderr_path = str(Path(workdir) / "stderr.txt")

        return CollectResult(
            exit_code=exit_code,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            artifact_dir=workdir,
            resource_summary=resource_summary,
        )

    def cancel(self, run: "Run") -> None:
        """Cancel a SLURM job via scancel."""
        job_id = run.remote_job_id
        if not job_id:
            return
        self._run_cmd(["scancel", job_id])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ssh_run_cmd(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        """Wrap a command to run on the remote server via SSH."""
        remote_cmd = " ".join(cmd)
        user_prefix = f"{self._user}@" if self._user else ""
        ssh_cmd = ["ssh", f"{user_prefix}{self._host}", remote_cmd]
        return subprocess.run(ssh_cmd, capture_output=True, text=True, **kwargs)

    @staticmethod
    def _parse_sbatch_output(text: str) -> str | None:
        """Extract job ID from 'Submitted batch job 12345' line."""
        m = re.search(r"Submitted batch job\s+(\d+)", text)
        return m.group(1) if m else None

    def _poll_via_sacct(self, job_id: str) -> PollResult:
        """Query sacct for job state and resources."""
        result = self._run_cmd([
            "sacct",
            f"--jobs={job_id}",
            "--format=JobID,State,ExitCode,CPUTime,MaxRSS,Elapsed",
            "--noheader",
            "--parsable2",
        ])
        _, resource = self._parse_sacct(result.stdout, job_id)
        state_str = resource.get("State", "")
        # Normalise: "COMPLETED" -> "CD", "FAILED" -> "F", etc.
        state = self._sacct_state_to_state(state_str)
        return PollResult(state=state, resource=resource)

    @staticmethod
    def _sacct_state_to_state(sacct_state: str) -> State:
        """Map sacct State column value to jobctl State enum."""
        s = sacct_state.upper().strip()
        # sacct returns full names; squeue uses short codes
        mapping = {
            "COMPLETED":  State.COMPLETED,
            "CD":         State.COMPLETED,
            "FAILED":     State.FAILED,
            "F":          State.FAILED,
            "TIMEOUT":    State.TIMEOUT,
            "TO":         State.TIMEOUT,
            "CANCELLED":  State.CANCELLED,
            "CA":         State.CANCELLED,
            "RUNNING":    State.RUNNING,
            "R":          State.RUNNING,
            "PENDING":    State.SUBMITTED,
            "PD":         State.SUBMITTED,
            "NODE_FAIL":  State.FAILED,
            "NF":         State.FAILED,
            "OUT_OF_MEMORY": State.FAILED,
            "OOM":        State.FAILED,
        }
        # Handle "CANCELLED by 0" style values
        if s.startswith("CANCELLED"):
            return State.CANCELLED
        return mapping.get(s, State.FAILED)

    @staticmethod
    def _parse_sacct(text: str, job_id: str) -> tuple[int | None, dict]:
        """Parse sacct --parsable2 output.

        Returns (exit_code, resource_dict).  resource_dict contains raw sacct
        fields (CPUTime, MaxRSS, Elapsed, State, ExitCode).
        """
        # Try header-less parsable2 first (pipe-delimited, no trailing |)
        # Lines look like: 12345|COMPLETED|0:0|00:01:00|128000K|00:00:30
        lines = [l for l in text.strip().splitlines() if l.strip()]
        if not lines:
            return None, {}

        # Check if first line is a header
        header = None
        data_lines = lines
        if lines and not lines[0][0].isdigit():
            # First line is header
            header = [h.strip() for h in lines[0].split("|")]
            data_lines = lines[1:]

        if header is None:
            # Default column order from our sacct call
            header = ["JobID", "State", "ExitCode", "CPUTime", "MaxRSS", "Elapsed"]

        resource: dict = {}
        exit_code: int | None = None

        for line in data_lines:
            parts = [p.strip() for p in line.split("|")]
            if not parts:
                continue
            # Filter to the main job entry (not sub-step entries like 12345.batch)
            job_field = parts[0] if parts else ""
            if job_field and job_field != job_id and not job_field.startswith(job_id + "."):
                continue

            row = dict(zip(header, parts))
            resource.update(row)

            # Parse exit code: "0:0" format -> first part
            exit_str = row.get("ExitCode", "")
            if exit_str:
                try:
                    exit_code = int(exit_str.split(":")[0])
                except (ValueError, IndexError):
                    pass

        return exit_code, resource
