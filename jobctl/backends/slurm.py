"""SlurmBackend: submits jobs via sbatch; polls via squeue/sacct; collects via rsync."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from jobctl.backends.base import Backend, CollectResult, PollResult, SubmitResult, resolved_command
from jobctl.db.models import Health, State

if TYPE_CHECKING:
    from jobctl.db.models import JobFile, Run

logger = logging.getLogger(__name__)

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


def _resolve_remote_path(raw: str, jobfile_name: str = "runs") -> str:
    """Substitute {project} with jobfile name slug."""
    slug = jobfile_name.replace(" ", "_").replace("/", "_") or "runs"
    return raw.format(project=slug)


class SlurmBackend(Backend):
    """Backend that submits jobs to a SLURM cluster via SSH.

    Lifecycle:
    1. ``submit``:
       - Create remote workdir via SSH.
       - Write sbatch script on remote via SSH stdin (cat > file).
       - Run ``sbatch <remote_script>`` via SSH; capture job ID.
    2. ``poll``:
       - SSH ``squeue``; fall back to ``sacct`` when not in queue.
    3. ``collect``:
       - Rsync remote workdir to a local mirror (best-effort; non-fatal on failure).
       - Parse exit code from sacct.
    4. ``cancel``:
       - SSH ``scancel <job_id>``.

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
        self._remote_path_template = server_config.get("remote_path", f"/tmp/jobctl/{server}")
        # run_cmd is used for SLURM commands (sbatch, squeue, sacct, scancel).
        # Defaults to ssh-wrapped execution; tests inject a fakebin runner.
        self._run_cmd = run_cmd or self._default_ssh_run_cmd

    def _remote_path(self, jobfile_name: str = "runs") -> str:
        return _resolve_remote_path(self._remote_path_template, jobfile_name)

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    def submit(self, run: "Run", jobfile: "JobFile") -> SubmitResult:
        """Write a job script on the remote host and run sbatch; capture job ID."""
        base = self._remote_path(jobfile.name if jobfile else "runs")
        workdir = f"{base}/{run.run_id}"

        # Create remote workdir via SSH (uses _run_cmd so tests can intercept)
        self._run_cmd(["mkdir", "-p", workdir])

        # Expand ~ to absolute path on remote so SLURM #SBATCH directives work
        # (SLURM does not expand ~ in --output/--error paths)
        abs_result = self._run_cmd(["echo", workdir])
        abs_workdir = abs_result.stdout.strip() if abs_result.stdout.strip() else workdir
        workdir = abs_workdir

        remote_script = f"{workdir}/job.sh"

        # Build sbatch directives
        directives = self._build_directives(run, jobfile, workdir)

        script_content = "#!/bin/bash\n"
        for d in directives:
            script_content += f"#SBATCH {d}\n"
        script_content += "\n"
        # cd into workdir so relative paths (like results.csv) land there
        script_content += f"cd {workdir}\n"
        script_content += resolved_command(run, jobfile) + "\n"

        # Write script to remote via SSH stdin (avoids local temp file issues)
        # _write_remote_file uses _run_cmd so tests can intercept
        self._write_remote_file(remote_script, script_content)

        # Submit
        result = self._run_cmd(["sbatch", remote_script])

        if result.returncode != 0:
            raise RuntimeError(
                f"sbatch failed (rc={result.returncode}): {result.stderr.strip()}"
            )

        job_id = self._parse_sbatch_output(result.stdout)
        if job_id is None:
            raise RuntimeError(
                f"Could not parse job ID from sbatch output: {result.stdout!r}"
            )

        # Record the resolved submission request (for the run-detail panel).
        slurm_request = self._resource_request(run)
        slurm_request["job_id"] = job_id
        return SubmitResult(remote_job_id=job_id, workdir=workdir, slurm_request=slurm_request)

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
        """Collect results: rsync remote workdir to local mirror, parse sacct."""
        job_id = run.remote_job_id or ""
        remote_workdir = run.workdir or "."

        # Local mirror: ~/.jobctl/runs/<run_id>/
        local_mirror = str(Path.home() / ".jobctl" / "runs" / run.run_id)
        Path(local_mirror).mkdir(parents=True, exist_ok=True)

        # Rsync remote workdir -> local mirror (best-effort, non-fatal on error)
        # Use raw subprocess (not _run_cmd) since rsync is not a SLURM command
        # and tests don't need to intercept it.
        remote_spec = f"{self._host}:{remote_workdir}/"
        rsync_result = subprocess.run(
            ["rsync", "-az", remote_spec, local_mirror + "/"],
            capture_output=True, text=True,
        )
        if rsync_result.returncode != 0:
            logger.warning(
                "collect: rsync from %s failed (rc=%d): %s",
                remote_spec, rsync_result.returncode, rsync_result.stderr.strip()
            )
            # Fall back to remote_workdir for artifact discovery
            # (only useful if workdir happens to be locally accessible)
            artifact_dir = remote_workdir
        else:
            artifact_dir = local_mirror

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

        return CollectResult(
            exit_code=exit_code,
            stdout_path=str(Path(artifact_dir) / "stdout.txt"),
            stderr_path=str(Path(artifact_dir) / "stderr.txt"),
            artifact_dir=artifact_dir,
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

    def _resource_request(self, run: "Run") -> dict:
        """Resolve the SLURM resource request for *run*.

        Precedence: per-run override (``run.slurm_request``) > per-server config
        (``self._server_config``) > built-in fallback. ``account`` and
        ``partition`` are OMITTED when neither override nor server config sets
        them — different clusters use different (or no) accounts/partitions, so
        forcing one server's values onto another is wrong (the old bug: oblix,
        which uses partition ``lln`` and no account, got hipster's
        ``--account=linuxusers --partition=capacity``).
        """
        cfg = self._server_config
        ov = getattr(run, "slurm_request", None) or {}

        req: dict = {}
        account = ov.get("account", cfg.get("account"))
        if account:
            req["account"] = account
        partition = ov.get("partition", cfg.get("partition"))
        if partition:
            req["partition"] = partition
        req["time"] = ov.get("time", cfg.get("time", "00:05:00"))
        req["mem"] = ov.get("mem", cfg.get("mem", "1G"))
        # accept either 'cpus' (CLI/override) or 'cpus_per_task' (server config)
        req["cpus"] = ov.get("cpus", ov.get("cpus_per_task", cfg.get("cpus_per_task", 1)))
        return req

    def _build_directives(self, run: "Run", jobfile: "JobFile", workdir: str) -> list[str]:
        """Build #SBATCH directive lines from the resolved resource request."""
        req = self._resource_request(run)
        directives = [
            f"--job-name={run.run_id}",
            f"--output={workdir}/stdout.txt",
            f"--error={workdir}/stderr.txt",
        ]
        if req.get("account"):
            directives.append(f"--account={req['account']}")
        if req.get("partition"):
            directives.append(f"--partition={req['partition']}")
        directives.append(f"--time={req['time']}")
        directives.append(f"--mem={req['mem']}")
        directives.append(f"--cpus-per-task={req['cpus']}")
        return directives

    def _write_remote_file(self, remote_path: str, content: str) -> None:
        """Write *content* to *remote_path* on the remote by piping via run_cmd.

        Uses self._run_cmd so tests can intercept.  The command is a shell
        one-liner: ``cat > <path>``.
        """
        # We pass a special sentinel command so tests can recognise it.
        # The actual content is passed via the ``input`` kwarg.
        user_prefix = f"{self._user}@" if self._user else ""
        host_arg = f"{user_prefix}{self._host}"

        # Build the ssh command list (compatible with the _run_cmd signature)
        # We use subprocess.run directly here since we need stdin=
        result = subprocess.run(
            ["ssh", host_arg, f"cat > {remote_path}"],
            input=content,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(
                "_write_remote_file: failed to write %s (rc=%d): %s",
                remote_path, result.returncode, result.stderr.strip()
            )

    def _default_ssh_run_cmd(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
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
