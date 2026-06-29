"""Bug-report assembly plus optional GitHub issue submission.

Lets an agent (or a human) create a local jobctl bug report from the CLI:

    jobctl --report-bug "monitor marked my running job stuck" --report-run run-abc123
    jobctl report-bug "monitor marked my running job stuck" --run run-abc123

It bundles diagnostics (version, platform, log tails, the run record, recent
failed/stuck runs) into a local Markdown report for the current user to review.
Uploading to GitHub is explicit opt-in via the CLI submit flags.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
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
        try:
            from jobctl import logsetup
            log_dir = logsetup.log_dir()
        except OSError:
            log_dir = Path(tempfile.gettempdir()) / "jobctl"
    log_dir = Path(log_dir)

    out: list[str] = [
        "## Privacy note\n\n"
        "This report is generated locally for the current user to review. "
        "It may include local jobctl log tails; jobctl does not upload it "
        "unless the user explicitly passes the submit flag.\n",
        f"## What happened\n\n{description}\n",
        "## Environment",
    ]
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
    out.append("\n_Generated locally via `jobctl report-bug`._")
    return "\n".join(out)


def save_local_report(title: str, body: str, *, log_root: Path | None = None) -> Path:
    """Save a bug report locally, falling back to temp when state root is denied."""
    candidates: list[Path] = []
    if log_root is not None:
        candidates.append(Path(log_root) / "issues")
    else:
        try:
            from jobctl import logsetup
            candidates.append(logsetup.log_dir() / "issues")
        except OSError:
            pass
    candidates.append(Path(tempfile.gettempdir()) / "jobctl-issues")

    last_error: OSError | None = None
    for issues_dir in candidates:
        try:
            issues_dir.mkdir(parents=True, exist_ok=True)
            path = issues_dir / f"bug-{datetime.now().strftime('%Y%m%dT%H%M%S')}.md"
            path.write_text(f"# {title}\n\n{body}\n")
            return path
        except OSError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise OSError("no local report paths available")


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
