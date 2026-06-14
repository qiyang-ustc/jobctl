"""Bug-report assembly + GitHub issue submission.

Lets an agent (or a human) file a jobctl bug straight from the CLI:

    jobctl report-bug "monitor marked my running job stuck" --run run-abc123

It bundles diagnostics (version, platform, log tails, the run record, recent
failed/stuck runs) and opens a GitHub issue on the jobctl repo so the
maintainer sees it. Falls back to a local file when `gh` is unavailable.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "qiyang-ustc/jobctl"

# Fields of a run record worth including in a report, in display order.
_RUN_FIELDS = (
    "run_id", "state", "health", "backend", "server", "remote_job_id",
    "exit_code", "submitted_at", "started_at", "finished_at", "workdir",
    "slurm_request", "parent_run_id", "attempt", "auto_policy",
)


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:]) or "(empty)"
    except OSError:
        return "(none)"


def build_report(
    description: str,
    *,
    version: str = "?",
    run: dict | None = None,
    recent: list[dict] | None = None,
    log_dir: Path | None = None,
) -> str:
    """Assemble a Markdown bug report body."""
    if log_dir is None:
        from jobctl import logsetup
        log_dir = logsetup.log_dir()
    log_dir = Path(log_dir)

    out: list[str] = [f"## What happened\n\n{description}\n", "## Environment"]
    out.append(f"- jobctl version: {version}")
    out.append(f"- platform: {platform.platform()}")
    out.append(f"- python: {sys.version.split()[0]}")
    out.append("")

    if run:
        out.append("## Run")
        for k in _RUN_FIELDS:
            out.append(f"- {k}: {run.get(k)}")
        out.append("")

    if recent:
        out.append("## Recent failed/stuck runs")
        for r in recent:
            out.append(
                f"- {r.get('run_id')} {r.get('state')} "
                f"{r.get('backend')}/{r.get('server')} exit={r.get('exit_code')}"
            )
        out.append("")

    out.append("## daemon.log (tail)\n```\n" + _tail(log_dir / "daemon.log") + "\n```")
    out.append("## cli.log (tail)\n```\n" + _tail(log_dir / "cli.log") + "\n```")
    out.append("\n_Filed via `jobctl report-bug`._")
    return "\n".join(out)


def _gh_runner(repo: str, title: str, body: str) -> str:
    proc = subprocess.run(
        ["gh", "issue", "create", "--repo", repo, "--title", title, "--body", body],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "gh issue create failed")
    return proc.stdout.strip()


def submit_issue(title: str, body: str, *, repo: str = REPO, runner=None) -> str | None:
    """File a GitHub issue. Returns the issue URL, or None if it couldn't.

    *runner* is injectable for testing; the default shells out to `gh`.
    """
    use_default = runner is None
    runner = runner or _gh_runner
    if use_default and shutil.which("gh") is None:
        return None
    try:
        return runner(repo, title, body)
    except Exception:
        return None
