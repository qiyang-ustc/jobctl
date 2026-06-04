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

_TERMINAL_STATES = {"completed", "failed", "cancelled", "stuck", "timeout"}

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@app.command()
def run(
    jobfile_ref: Annotated[str, typer.Argument(help="JobFile id or name")],
    wait: Annotated[bool, typer.Option("--wait/--background", help="Block until terminal")] = False,
    param: Annotated[Optional[list[str]], typer.Option("--param", "-p", help="key=value param")] = None,
    backend: Annotated[Optional[str], typer.Option("--backend", help="Backend override (local/ssh/slurm)")] = None,
    server: Annotated[Optional[str], typer.Option("--server", help="Server name override")] = None,
    reuse: Annotated[bool, typer.Option("--reuse/--no-reuse", help="Skip if reuse-eligible")] = False,
    callback: Annotated[Optional[str], typer.Option("--callback", help="Callback URL")] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """Submit a job. --wait blocks until done and prints the observation card.
    --background (default) returns immediately and prints {run_id}."""
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

    # Determine if the ref is an id or a name
    submit_kwargs: dict = {
        "params": params,
        "backend_override": backend_override,
    }
    # Try as ID first; fall back to name
    submit_kwargs["jobfile_name"] = jobfile_ref

    try:
        result = client.submit(**submit_kwargs)
    except RuntimeError as exc:
        # Maybe it's an ID, not a name
        try:
            submit_kwargs2 = dict(submit_kwargs)
            del submit_kwargs2["jobfile_name"]
            submit_kwargs2["jobfile_id"] = jobfile_ref
            result = client.submit(**submit_kwargs2)
        except Exception:
            typer.echo(f"Error submitting run: {exc}", err=True)
            raise typer.Exit(1)

    run_id = result["run_id"]

    if wait:
        # Block until terminal, then print the final run dict (with observation card)
        try:
            final = client.await_run(run_id, poll_interval=0.5, timeout=300)
        except TimeoutError:
            typer.echo(f"Timeout waiting for run {run_id}", err=True)
            raise typer.Exit(1)

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
    timeout: Annotated[float, typer.Option("--timeout", help="Seconds to wait")] = 300.0,
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """Block until a run reaches a terminal state, then print the result."""
    client = _get_client()
    try:
        final = client.await_run(run_id, poll_interval=0.2, timeout=timeout)
    except TimeoutError:
        typer.echo(f"Timeout waiting for run {run_id}", err=True)
        raise typer.Exit(1)

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
    json_out: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """Post user feedback for a run."""
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

    config: dict = {}
    if db_path:
        config["db_path"] = db_path

    application = create_app(config=config, start_monitor=True)
    uvicorn.run(application, host=host, port=port, reload=reload)
