"""jobctl CLI — Typer app.

Commands:
    run         Submit a job (--wait blocks to terminal; --background returns run_id).
    await       Block on a run_id until terminal state.
    status      Print the current state of a run.
    logs        Tail stdout or stderr for a run.
    artifacts   List artifacts for a run.
    inspect     Print the full run record.
    cancel      Cancel a run.
    rerun       Create a copy of a run and re-submit it.
    servers     List server health rows.
    memory      Sub-group: memory query.
    register    Register a JobFile path with the daemon.
    jobfiles    List all registered JobFiles.
    feedback    Post user feedback for a run.
    expect      Sub-group: list / propose / confirm expectation contracts.
    serve       Start the jobctl FastAPI daemon (uvicorn).

All commands that produce structured output support --json to emit compact JSON.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Annotated, Optional

import typer

# ---------------------------------------------------------------------------
# Module-level override for testing (injected by test fixtures).
# ---------------------------------------------------------------------------
_OVERRIDE_CLIENT = None  # type: ignore


def _get_client():
    """Return the ApiClient to use. Test fixtures may override this."""
    if _OVERRIDE_CLIENT is not None:
        return _OVERRIDE_CLIENT

    from jobctl.api.client import ApiClient, ensure_daemon
    base_url = ensure_daemon()
    return ApiClient(base_url=base_url)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_json(obj):
    typer.echo(json.dumps(obj, indent=None, default=str))


def _print_pretty(obj):
    """Pretty-print a dict/list to stdout."""
    if isinstance(obj, (dict, list)):
        typer.echo(json.dumps(obj, indent=2, default=str))
    else:
        typer.echo(str(obj))


def _print_table(rows: list[dict], columns: list[str]):
    """Print rows as a simple ASCII table."""
    if not rows:
        typer.echo("(no entries)")
        return
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            val = str(row.get(col, ""))
            widths[col] = max(widths[col], len(val))
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    separator = "  ".join("-" * widths[col] for col in columns)
    typer.echo(header)
    typer.echo(separator)
    for row in rows:
        typer.echo("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="jobctl — JobFile-native research run gateway.",
    no_args_is_help=True,
)

# Sub-groups
memory_app = typer.Typer(help="Run-memory queries.", no_args_is_help=True)
expect_app = typer.Typer(help="Expectation contracts.", no_args_is_help=True)

app.add_typer(memory_app, name="memory")
app.add_typer(expect_app, name="expect")


@app.callback()
def _main(ctx: typer.Context):
    """Log every jobctl invocation so all calls leave a trail in ~/.jobctl/cli.log."""
    # Skip the heavy log setup for `serve` (it configures its own 'daemon' log).
    if ctx.invoked_subcommand == "serve":
        return
    try:
        from jobctl.logsetup import configure_logging
        import logging as _logging
        configure_logging("cli")
        _logging.getLogger("jobctl.cli").info("invoke: %s", " ".join(sys.argv[1:]) or "(no args)")
    except Exception:
        pass  # logging must never break a command


_TERMINAL_STATES = {"completed", "failed", "cancelled", "stuck", "timeout"}

# How long `run --wait` / `await` block before giving up (12 hours) — long
# enough for real cluster jobs; override with --timeout.
_WAIT_TIMEOUT_DEFAULT = 12 * 60 * 60

# Terminal state -> process exit code (for the waiting paths). Exit code means
# "did the job RUN to completion" (operational); the observation card's
# expectation_match means "is the result GOOD" (scientific). So `completed` is
# 0 regardless of usable/weak/bad, and `jobctl run --wait && next` only
# short-circuits on an operational failure.
_EXIT_FOR_STATE = {
    "completed": 0,
    "failed": 2,
    "cancelled": 3,
    "stuck": 4,
    "timeout": 124,   # job hit its time limit (matches timeout(1)'s convention)
}


def _exit_code_for(state) -> int:
    """Map a terminal run state to a process exit code (unknown -> 1)."""
    return _EXIT_FOR_STATE.get(str(state), 1)

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@app.command()
def run(
    jobfile_ref: Annotated[str, typer.Argument(help="JobFile path (manifest or bare script), id, or name")],
    wait: Annotated[bool, typer.Option("--wait/--background", help="Block until terminal")] = False,
    param: Annotated[Optional[list[str]], typer.Option("--param", "-p", help="key=value param")] = None,
    backend: Annotated[Optional[str], typer.Option("--backend", help="Backend override (local/ssh/slurm)")] = None,
    server: Annotated[Optional[str], typer.Option("--server", help="Server name override")] = None,
    reuse: Annotated[bool, typer.Option("--reuse/--no-reuse", help="Skip if reuse-eligible")] = False,
    callback: Annotated[Optional[str], typer.Option("--callback", help="Callback URL")] = None,
    mem: Annotated[Optional[str], typer.Option("--mem", help="SLURM memory (e.g. 100M, 4G)")] = None,
    cpus: Annotated[Optional[int], typer.Option("--cpus", help="SLURM cpus-per-task")] = None,
    time_limit: Annotated[Optional[str], typer.Option("--time", help="SLURM time limit (e.g. 00:11:00)")] = None,
    partition: Annotated[Optional[str], typer.Option("--partition", help="SLURM partition")] = None,
    account: Annotated[Optional[str], typer.Option("--account", help="SLURM account")] = None,
    timeout: Annotated[float, typer.Option("--timeout", help="Seconds --wait blocks before giving up")] = _WAIT_TIMEOUT_DEFAULT,
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """Submit a job. --wait blocks until done and prints the observation card.
    --background (default) returns immediately and prints {run_id}.

    SLURM resource flags (--mem/--cpus/--time/--partition/--account) override
    per-server config for slurm submits and are shown in the run-detail panel."""
    client = _get_client()

    # Parse params
    params: dict = {}
    if param:
        for kv in param:
            if "=" in kv:
                k, v = kv.split("=", 1)
                params[k.strip()] = v.strip()
            else:
                typer.echo(f"Warning: skipping malformed param '{kv}' (expected key=value)", err=True)

    # Build backend override
    backend_override: dict | None = None
    if backend:
        backend_override = {"backend": backend}
        if server:
            backend_override["server"] = server

    # Resolve the ref. If it points to an existing file, it is a JobFile path
    # (manifest or bare script): register it first to obtain its id, then submit
    # by id. Otherwise treat it as the name or id of an already-registered
    # JobFile. This makes `jobctl run path/to/job.yaml` work like a local command.
    # SLURM resource overrides (only non-None keys; partition/account also
    # selectable here without a --backend flag).
    resources: dict = {}
    if mem is not None:
        resources["mem"] = mem
    if cpus is not None:
        resources["cpus"] = cpus
    if time_limit is not None:
        resources["time"] = time_limit
    if partition is not None:
        resources["partition"] = partition
    if account is not None:
        resources["account"] = account

    submit_kwargs: dict = {
        "params": params,
        "backend_override": backend_override,
        "reuse": reuse,
        "callback_url": callback,
        "resources": resources or None,
    }
    if os.path.isfile(jobfile_ref):
        try:
            # Absolutize the jobfile path: the daemon/backend run in a different
            # working directory than this CLI invocation.
            jf = client.register(os.path.abspath(jobfile_ref))
            # Absolutize path-typed params for the same reason — backends execute
            # in their own workdir, so a relative script path would not be found.
            schema = jf.get("params_schema") or {}
            for k, v in list(params.items()):
                spec = schema.get(k) or {}
                if isinstance(v, str) and (spec.get("type") == "path" or os.path.isfile(v)):
                    params[k] = os.path.abspath(v)
            submit_kwargs["jobfile_id"] = jf["id"]
            result = client.submit(**submit_kwargs)
        except RuntimeError as exc:
            typer.echo(f"Error submitting run: {exc}", err=True)
            raise typer.Exit(1)
    else:
        # Not a file — try as name, then fall back to id.
        try:
            result = client.submit(jobfile_name=jobfile_ref, **submit_kwargs)
        except RuntimeError as exc:
            try:
                result = client.submit(jobfile_id=jobfile_ref, **submit_kwargs)
            except Exception:
                typer.echo(f"Error submitting run: {exc}", err=True)
                raise typer.Exit(1)

    run_id = result["run_id"]

    if wait:
        # Block until terminal, then print the final run dict (with observation card)
        try:
            final = client.await_run(run_id, poll_interval=0.5, timeout=timeout)
        except TimeoutError:
            typer.echo(f"Timeout waiting for run {run_id} (waited {timeout:g}s)", err=True)
            raise typer.Exit(124)

        # Build output: prefer observation_card if available, else full run dict
        card = final.get("observation_card")
        output_obj = card if (card and isinstance(card, dict)) else final
        # Always include run_id and state at top level
        if isinstance(output_obj, dict):
            output_obj = dict(output_obj)
            output_obj.setdefault("run_id", final.get("run_id"))
            output_obj.setdefault("state", final.get("state"))

        if json_out:
            _print_json(output_obj)
        else:
            # Human-readable card
            typer.echo(f"Run {run_id} — {final.get('state', '?').upper()}")
            if card:
                typer.echo(f"  interpretation: {card.get('interpretation', '')}")
                typer.echo(f"  health:         {card.get('health', '')}")
                typer.echo(f"  match:          {card.get('expectation_match', '')}")
                typer.echo(f"  next action:    {card.get('recommended_next_action', '')}")
                arts = card.get("artifacts", [])
                if arts:
                    typer.echo(f"  artifacts ({len(arts)}):")
                    for a in arts:
                        typer.echo(f"    - {a.get('name', a)}")
            else:
                typer.echo(f"  backend: {final.get('backend')}  server: {final.get('server')}")

        # Exit code reflects whether the job RAN (operational), not result quality.
        raise typer.Exit(_exit_code_for(final.get("state")))
    else:
        # Background mode — print {run_id} immediately
        if json_out:
            _print_json(result)
        else:
            typer.echo(run_id)


# ---------------------------------------------------------------------------
# await
# ---------------------------------------------------------------------------

@app.command(name="await")
def await_cmd(
    run_id: Annotated[str, typer.Argument(help="Run ID to wait for")],
    timeout: Annotated[float, typer.Option("--timeout", help="Seconds to wait before giving up")] = _WAIT_TIMEOUT_DEFAULT,
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """Block until a run reaches a terminal state, then print the result.

    Exit code: 0 completed · 2 failed · 3 cancelled · 4 stuck · 124 timeout."""
    client = _get_client()
    try:
        final = client.await_run(run_id, poll_interval=0.2, timeout=timeout)
    except TimeoutError:
        typer.echo(f"Timeout waiting for run {run_id} (waited {timeout:g}s)", err=True)
        raise typer.Exit(124)

    card = final.get("observation_card")
    output_obj = card if (card and isinstance(card, dict)) else final
    if isinstance(output_obj, dict):
        output_obj = dict(output_obj)
        output_obj.setdefault("run_id", final.get("run_id"))
        output_obj.setdefault("state", final.get("state"))

    if json_out:
        _print_json(output_obj)
    else:
        typer.echo(f"Run {run_id} — {final.get('state', '?').upper()}")
        if card:
            typer.echo(json.dumps(card, indent=2, default=str))
        else:
            _print_pretty(final)

    raise typer.Exit(_exit_code_for(final.get("state")))


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@app.command()
def status(
    run_id: Annotated[str, typer.Argument(help="Run ID")],
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """Print the current state of a run."""
    client = _get_client()
    run = client.get_run(run_id)

    if json_out:
        _print_json(run)
    else:
        state = run.get("state", "unknown")
        health = run.get("health", "")
        typer.echo(f"run_id:  {run_id}")
        typer.echo(f"state:   {state}")
        typer.echo(f"health:  {health}")
        typer.echo(f"backend: {run.get('backend')}  server: {run.get('server')}")


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------

@app.command()
def logs(
    run_id: Annotated[str, typer.Argument(help="Run ID")],
    stream: Annotated[str, typer.Option("--stream", help="stdout or stderr")] = "stdout",
    tail: Annotated[int, typer.Option("--tail", help="Number of lines to tail")] = 200,
):
    """Tail stdout or stderr for a run."""
    client = _get_client()
    text = client.logs(run_id, stream=stream, tail=tail)
    typer.echo(text, nl=False)


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------

@app.command()
def artifacts(
    run_id: Annotated[str, typer.Argument(help="Run ID")],
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """List artifacts for a run."""
    client = _get_client()
    arts = client.artifacts(run_id)

    if json_out:
        _print_json(arts)
    else:
        if not arts:
            typer.echo("(no artifacts)")
        else:
            _print_table(arts, ["id", "type", "size", "local_path"])


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

@app.command()
def inspect(
    run_id: Annotated[str, typer.Argument(help="Run ID")],
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """Print the full run record including observation card."""
    client = _get_client()
    run = client.get_run(run_id)

    if json_out:
        _print_json(run)
    else:
        _print_pretty(run)


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------

@app.command()
def cancel(
    run_id: Annotated[str, typer.Argument(help="Run ID")],
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """Cancel a run."""
    client = _get_client()
    result = client.cancel(run_id)

    if json_out:
        _print_json(result)
    else:
        typer.echo(f"Cancelled run {run_id} (state: {result.get('state')})")


# ---------------------------------------------------------------------------
# rerun
# ---------------------------------------------------------------------------

@app.command()
def rerun(
    run_id: Annotated[str, typer.Argument(help="Run ID to rerun")],
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """Create a copy of a run with the same params and re-submit it."""
    client = _get_client()
    result = client.rerun(run_id)

    if json_out:
        _print_json(result)
    else:
        new_id = result.get("run_id")
        typer.echo(f"Created rerun: {new_id} (state: {result.get('state')})")


# ---------------------------------------------------------------------------
# servers
# ---------------------------------------------------------------------------

@app.command()
def servers(
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """List server health rows."""
    client = _get_client()
    srv_list = client.servers()

    if json_out:
        _print_json(srv_list)
    else:
        if not srv_list:
            typer.echo("(no servers registered)")
        else:
            _print_table(srv_list, ["name", "backend_type", "online", "note"])


# ---------------------------------------------------------------------------
# report-bug
# ---------------------------------------------------------------------------

@app.command(name="report-bug")
def report_bug(
    description: Annotated[str, typer.Argument(help="What went wrong, in one line")],
    run_id: Annotated[Optional[str], typer.Option("--run", help="Related run id")] = None,
    submit: Annotated[bool, typer.Option("--submit/--no-submit", help="File a GitHub issue")] = True,
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """Report a jobctl bug: bundles diagnostics (version, log tails, the run
    record, recent failures) and opens a GitHub issue on the jobctl repo so the
    maintainer can fix it. Falls back to a local file if GitHub is unreachable."""
    from datetime import datetime
    from jobctl import report as _report
    from jobctl.logsetup import log_dir

    try:
        from importlib.metadata import version as _pkg_version
        version = _pkg_version("jobctl")
    except Exception:
        version = "0.1.0"

    # Best-effort diagnostics from the daemon (never fatal).
    run = None
    recent = None
    try:
        client = _get_client()
        if run_id:
            run = client.get_run(run_id)
        runs = client.list_runs()
        recent = [r for r in runs if r.get("state") in ("failed", "stuck", "timeout")][-10:]
    except Exception as exc:
        typer.echo(f"(note: could not gather daemon diagnostics: {exc})", err=True)

    body = _report.build_report(description, version=version, run=run, recent=recent)
    title = f"[bug] {description[:70]}"

    url = _report.submit_issue(title, body) if submit else None
    if url:
        result = {"submitted": True, "issue_url": url}
    else:
        issues_dir = log_dir() / "issues"
        issues_dir.mkdir(parents=True, exist_ok=True)
        path = issues_dir / f"bug-{datetime.now().strftime('%Y%m%dT%H%M%S')}.md"
        path.write_text(f"# {title}\n\n{body}\n")
        result = {
            "submitted": False,
            "saved_to": str(path),
            "manual": f"gh issue create --repo {_report.REPO} --title {title!r} --body-file {path}",
        }

    if json_out:
        _print_json(result)
    elif url:
        typer.echo(f"Filed issue: {url}")
    else:
        typer.echo(f"Could not file to GitHub; saved report to {result['saved_to']}")
        typer.echo(f"  file it manually: {result['manual']}")


# ---------------------------------------------------------------------------
# gc
# ---------------------------------------------------------------------------

@app.command()
def gc(
    dry_run: Annotated[bool, typer.Option("--dry-run", help="List orphans without deleting")] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """Remove orphan run directories in ~/.jobctl/runs that have no DB record.

    A directory is only removed when its name isn't a known run_id, so runs the
    daemon still tracks are never touched."""
    from jobctl import gc as _gc
    from jobctl.logsetup import log_dir

    client = _get_client()
    known = {r["run_id"] for r in client.list_runs()}
    runs_dir = log_dir() / "runs"
    orphans, removed = _gc.gc_runs(str(runs_dir), known, dry_run=dry_run)

    result = {
        "runs_dir": str(runs_dir),
        "orphans_found": len(orphans),
        "removed": len(removed),
        "dry_run": dry_run,
    }
    if json_out:
        _print_json(result)
    elif dry_run:
        typer.echo(f"{len(orphans)} orphan run dir(s) would be removed from {runs_dir} (dry-run).")
    else:
        typer.echo(f"Removed {len(removed)} orphan run dir(s) from {runs_dir} (of {len(orphans)} found).")


# ---------------------------------------------------------------------------
# memory sub-group
# ---------------------------------------------------------------------------

@memory_app.command(name="query")
def memory_query(
    jobfile_id: Annotated[Optional[str], typer.Option("--jobfile-id", help="Filter by JobFile id")] = None,
    name: Annotated[Optional[str], typer.Option("--name", help="Filter by JobFile name")] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """Query run memory for a jobfile."""
    client = _get_client()
    result = client.memory_query(jobfile_id=jobfile_id, name=name)

    if json_out:
        _print_json(result)
    else:
        _print_pretty(result)


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------

@app.command()
def register(
    path: Annotated[str, typer.Argument(help="Path to .jobfile.yaml or bare script")],
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """Register a JobFile with the daemon."""
    client = _get_client()
    result = client.register(path)

    if json_out:
        _print_json(result)
    else:
        typer.echo(f"Registered: {result.get('name')} (id: {result.get('id')}, version: {result.get('version')})")


# ---------------------------------------------------------------------------
# jobfiles
# ---------------------------------------------------------------------------

@app.command()
def jobfiles(
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """List all registered JobFiles."""
    client = _get_client()
    jf_list = client.jobfiles()

    if json_out:
        _print_json(jf_list)
    else:
        if not jf_list:
            typer.echo("(no jobfiles registered)")
        else:
            _print_table(jf_list, ["id", "name", "version", "backend_prefs"])


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------

@app.command()
def feedback(
    run_id: Annotated[str, typer.Argument(help="Run ID")],
    text: Annotated[str, typer.Option("--text", help="Feedback text")] = "",
    kind: Annotated[str, typer.Option("--kind", help="Feedback kind (note/good/bad)")] = "note",
    accept: Annotated[bool, typer.Option("--accept", help="Mark the run's result as accepted (kind=good)")] = False,
    reject: Annotated[bool, typer.Option("--reject", help="Mark the run's result as rejected (kind=bad)")] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """Post user feedback for a run."""
    if accept and reject:
        typer.echo("Error: --accept and --reject are mutually exclusive", err=True)
        raise typer.Exit(1)
    if accept:
        kind = "good"
    elif reject:
        kind = "bad"
    client = _get_client()
    result = client.feedback(run_id, kind=kind, text=text)

    if json_out:
        _print_json(result)
    else:
        typer.echo(f"Feedback recorded (id: {result.get('id')})")


# ---------------------------------------------------------------------------
# expect sub-group
# ---------------------------------------------------------------------------

@expect_app.callback(invoke_without_command=True)
def expect_main(
    ctx: typer.Context,
    jobfile_id: Annotated[Optional[str], typer.Option("--jobfile-id", help="Filter by JobFile id")] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """List expectation contracts, or use sub-commands propose/confirm."""
    if ctx.invoked_subcommand is not None:
        return
    client = _get_client()
    result = client.expect(jobfile_id=jobfile_id)

    if json_out:
        _print_json(result)
    else:
        if not result:
            typer.echo("(no contracts)")
        else:
            _print_pretty(result)


@expect_app.command(name="list")
def expect_list(
    jobfile: Annotated[Optional[str], typer.Argument(help="JobFile id or name (optional; all if omitted)")] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """List expectation contracts (optionally filtered by JobFile id or name)."""
    client = _get_client()
    jobfile_id = jobfile
    if jobfile:
        try:
            for jf in client.jobfiles():
                if jf.get("id") == jobfile or jf.get("name") == jobfile:
                    jobfile_id = jf["id"]
                    break
        except Exception:
            pass
    result = client.expect(jobfile_id=jobfile_id)

    if json_out:
        _print_json(result)
    else:
        if not result:
            typer.echo("(no contracts)")
        else:
            _print_pretty(result)


@expect_app.command(name="propose")
def expect_propose(
    run_id: Annotated[str, typer.Argument(help="Run ID to propose criteria from")],
    text: Annotated[str, typer.Option("--text", help="Feedback text to base criteria on")] = "",
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """Propose new expectation criteria from a run + feedback text."""
    client = _get_client()
    result = client.propose_criteria(run_id=run_id, feedback_text=text)

    if json_out:
        _print_json(result)
    else:
        if not result:
            typer.echo("(no criteria proposed)")
        else:
            _print_pretty(result)


@expect_app.command(name="confirm")
def expect_confirm(
    criterion_id: Annotated[str, typer.Argument(help="Criterion ID to confirm")],
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """Confirm an expectation criterion (status -> confirmed, strength +1)."""
    client = _get_client()
    result = client.confirm_criterion(criterion_id)

    if json_out:
        _print_json(result)
    else:
        typer.echo(f"Confirmed criterion {criterion_id} (status: {result.get('status')}, strength: {result.get('strength')})")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Bind host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Bind port")] = 7421,
    reload: Annotated[bool, typer.Option("--reload/--no-reload", help="Hot-reload (dev)")] = False,
    db_path: Annotated[Optional[str], typer.Option("--db-path", help="SQLite DB path")] = None,
):
    """Start the jobctl FastAPI daemon (uvicorn)."""
    import uvicorn
    from jobctl.api.server import create_app
    from jobctl.config import load_config as _load_config
    from jobctl.logsetup import configure_logging

    log_file = configure_logging("daemon")
    import logging as _logging
    _logging.getLogger("jobctl").info("daemon starting on %s:%s (log: %s)", host, port, log_file)

    # Load cluster.yaml so server configs (remote_path, account, partition, etc.)
    # are available to backends at runtime.
    _cfg = _load_config()
    config: dict = {"servers": _cfg.servers}
    if _cfg.db_path:
        config["db_path"] = _cfg.db_path
    if _cfg.run_dir:
        config["run_dir"] = _cfg.run_dir
    if db_path:
        config["db_path"] = db_path

    application = create_app(config=config, start_monitor=True)
    uvicorn.run(application, host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
