"""FastAPI daemon: REST endpoints + monitor startup + Web UI routes.

Endpoints:
    GET  /health
    POST /jobfiles              — register a JobFile from a path
    GET  /jobfiles              — list registered JobFiles
    POST /runs                  — submit a run (attaches memory hint, selects backend)
    GET  /runs                  — list runs (filter: state, jobfile_id)
    GET  /runs/{id}             — get a run
    POST /runs/{id}/cancel      — cancel a run
    POST /runs/{id}/rerun       — rerun (copy params -> new run)
    GET  /runs/{id}/logs        — tail stdout or stderr
    GET  /runs/{id}/artifacts   — list artifacts for a run
    GET  /servers               — list server health rows
    POST /runs/{id}/feedback    — submit user feedback for a run
    GET  /runs/{id}/feedback    — list feedback for a run
    GET  /expect                — list expectation contracts (?jobfile_id=)
    POST /expect/confirm        — confirm a criterion
    POST /expect/propose        — propose new criteria from feedback
    GET  /memory/query          — query run memory

    Web UI (HTML):
    GET  /                          — dashboard (server cards + run buckets)
    GET  /runs/{id}                 — run detail (logs, artifacts, observation card, criteria)
    GET  /jobfiles/{id}             — jobfile detail (schema, runs, contract)
    GET  /ui/poll                   — HTMX poll partial
    GET  /ui/artifact-thumb         — serve artifact image thumbnail
    /static/*                       — static assets

Usage:
    app = create_app(config={...}, start_monitor=True)
    # or
    uvicorn.run("jobctl.api.server:create_app", factory=True, ...)
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import jobctl

logger = logging.getLogger(__name__)

_TERMINAL_STATES = {"completed", "failed", "cancelled", "stuck", "timeout"}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config: dict | None = None, start_monitor: bool = True) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config:        Configuration dict.  Defaults to empty dict (uses Store
                       defaults + env).
        start_monitor: If True, start the Monitor asyncio loop on startup and
                       stop it on shutdown.

    Returns:
        Configured FastAPI application.
    """
    if config is None:
        config = {}

    # Resolve DB and run paths
    db_path = config.get("db_path") or os.path.expanduser("~/.jobctl/jobctl.db")
    run_dir = config.get("run_dir") or os.path.expanduser("~/.jobctl/runs")

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    from jobctl.db.store import Store
    from jobctl.analysis.base import get_analyzer
    from jobctl.notify.notify import get_notifiers
    from jobctl.monitor.monitor import Monitor

    store = Store(db_path)
    store.init_schema()

    analyzer = get_analyzer(config)

    def notifiers_factory(run):
        return get_notifiers(config, run)

    monitor_config = {
        **config,
        "poll_interval_seconds": config.get("poll_interval_seconds", 10.0),
        "probe_interval_seconds": config.get("probe_interval_seconds", 60.0),
        "stuck_timeout_seconds": config.get("stuck_timeout_seconds", 600.0),
    }
    monitor = Monitor(
        store=store,
        config=monitor_config,
        analyzer=analyzer,
        notifiers_factory=notifiers_factory,
    )

    # Attach a real SSH prober when servers are configured (production). Tests
    # construct Monitor directly with their own prober, or run with no servers /
    # a long probe interval, so they keep the default no-op prober.
    _servers_cfg = config.get("servers") or {}
    if _servers_cfg:
        from jobctl.monitor.prober import SshProber
        monitor._prober = SshProber(_servers_cfg)

    if start_monitor:
        @asynccontextmanager
        async def _lifespan(application: FastAPI):
            stop_event = asyncio.Event()
            task = asyncio.create_task(monitor.run_loop(stop_event))
            logger.info("Monitor started")
            try:
                yield
            finally:
                stop_event.set()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                logger.info("Monitor stopped")

        app = FastAPI(title="jobctl", version=jobctl.__version__, lifespan=_lifespan)
    else:
        app = FastAPI(title="jobctl", version=jobctl.__version__)

    # Store shared state on app
    app.state.store = store
    app.state.config = config
    app.state.analyzer = analyzer
    app.state.monitor = monitor
    app.state.run_dir = run_dir

    # ------------------------------------------------------------------
    # Mount static files and set up Jinja2 templates
    # ------------------------------------------------------------------
    _UI_DIR = Path(__file__).parent.parent / "ui"
    _STATIC_DIR = _UI_DIR / "static"
    _TEMPLATES_DIR = _UI_DIR / "templates"

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    # Expose the basename filter in templates
    templates.env.filters["basename"] = lambda p: Path(p).name if p else ""
    templates.env.filters["urlencode"] = lambda s: str(s).replace("/", "%2F") if s else ""
    templates.env.filters["enumval"] = lambda v: getattr(v, "value", v)
    app.state.templates = templates
    app.state.version = jobctl.__version__

    # ------------------------------------------------------------------
    # Register routes
    # ------------------------------------------------------------------
    _register_routes(app)

    return app


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def _register_routes(app: FastAPI) -> None:
    """Attach all API routes to *app*."""

    def _store(request: Request):
        return request.app.state.store

    def _config(request: Request):
        return request.app.state.config

    def _analyzer(request: Request):
        return request.app.state.analyzer

    def _monitor(request: Request):
        return request.app.state.monitor

    # ------------------------------------------------------------------
    # /health
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": jobctl.__version__}

    # ------------------------------------------------------------------
    # /jobfiles
    # ------------------------------------------------------------------

    @app.post("/jobfiles")
    async def register_jobfile(body: dict, request: Request):
        """Register a JobFile from a file path."""
        from jobctl.jobfile import load_jobfile, content_hash as calc_hash

        path = body.get("path", "")
        if not path:
            raise HTTPException(status_code=422, detail="'path' is required")

        if not Path(path).exists():
            raise HTTPException(status_code=400, detail=f"File not found: {path}")

        try:
            jf = load_jobfile(path)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        store = _store(request)

        # Check if already registered by name
        existing = store.get_jobfile_by_name(jf.name)
        if existing is not None:
            # Re-register: check hash. If same, return existing; if different, bump version.
            if existing.content_hash == jf.content_hash:
                return _jobfile_to_dict(existing)
            # Update the existing record's version + hash
            store.bump_version(existing.id, jf.content_hash)
            updated = store.get_jobfile(existing.id)
            return _jobfile_to_dict(updated)

        # Seed and persist a default expectation contract, linked to the jobfile,
        # so the expectation layer is active from first registration: /expect and
        # the UI show it, runs classify against it, and the distiller has a base
        # contract to merge proposed criteria into.
        from jobctl.expectations.contracts import default_contract
        contract = default_contract(jf)
        jf.expectation_contract_id = contract.id
        store.add_jobfile(jf)
        store.save_contract(contract)
        return _jobfile_to_dict(jf)

    @app.get("/jobfiles")
    async def list_jobfiles(request: Request):
        store = _store(request)
        return [_jobfile_to_dict(jf) for jf in store.list_jobfiles()]

    # ------------------------------------------------------------------
    # /runs
    # ------------------------------------------------------------------

    @app.post("/runs")
    async def submit_run(body: dict, request: Request):
        """Submit a new run.

        Body:
            jobfile_id (str, optional)
            jobfile_name (str, optional)
            params (dict, optional)
            backend_override (dict, optional) — {backend, server, task}
            title (str, optional) — human-readable label: what this run is for
            note (str, optional) — freeform description
            tags (list[str], optional) — classification / loose grouping
        """
        from jobctl.jobfile import resolve_params, render_command, input_hashes as calc_input_hashes
        from jobctl.backends.base import select_backend, get_backend
        from jobctl.memory.memory import query as mem_query
        from jobctl.db.models import Run, State, Health

        store = _store(request)
        cfg = _config(request)

        # Resolve jobfile
        jf = None
        if body.get("jobfile_id"):
            jf = store.get_jobfile(body["jobfile_id"])
        elif body.get("jobfile_name"):
            jf = store.get_jobfile_by_name(body["jobfile_name"])

        if jf is None:
            raise HTTPException(status_code=404, detail="JobFile not found")

        # Resolve params
        params = body.get("params") or {}
        try:
            resolved_params = resolve_params(jf, params)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        # Compute input hashes
        try:
            ihashes = calc_input_hashes(jf, resolved_params)
        except Exception:
            ihashes = {}

        # Memory hint
        memory_hint = mem_query(
            store,
            jobfile_id=jf.id,
            params=resolved_params,
            input_hashes=ihashes,
        )

        # Reuse short-circuit: if requested and an exact prior usable run exists,
        # return it instead of launching a new run.
        if body.get("reuse") and memory_hint.get("reuse_eligible") and memory_hint.get("exact_match_run_id"):
            prior = store.get_run(memory_hint["exact_match_run_id"])
            if prior is not None:
                result = _run_to_dict(prior, jf.name)
                result["memory_hint"] = memory_hint
                result["reused"] = True
                return result

        # Select backend
        servers = store.list_servers()
        backend_override = body.get("backend_override")
        backend_name, server_name, task_name = select_backend(
            jf, servers, override=backend_override
        )

        now = datetime.now(timezone.utc).isoformat()
        run_id = f"run-{uuid.uuid4().hex[:12]}"

        run = Run(
            run_id=run_id,
            jobfile_id=jf.id,
            jobfile_version=jf.version,
            params=resolved_params,
            input_hashes=ihashes,
            backend=backend_name,
            server=server_name,
            task=task_name,
            remote_job_id=None,
            state=State.PENDING,
            health=Health.OK,
            exit_code=None,
            submitted_at=now,
            started_at=None,
            finished_at=None,
            last_heartbeat=None,
            workdir=None,
            stdout_path=None,
            stderr_path=None,
            resource_summary={},
            expectation_match=None,
            observation_card=None,
            slurm_request=body.get("resources") or None,
            title=(body.get("title") or None),
            note=(body.get("note") or None),
            tags=(body.get("tags") or None),
        )
        store.add_run(run)

        # Submit to backend immediately
        try:
            backend = get_backend(backend_name, server_name, cfg)
            submit_result = backend.submit(run, jf)
            update_fields = dict(
                state=State.SUBMITTED,
                remote_job_id=submit_result.remote_job_id,
                workdir=submit_result.workdir,
            )
            # Persist the resolved SLURM request (with job_id) for the panel.
            if submit_result.slurm_request is not None:
                update_fields["slurm_request"] = submit_result.slurm_request
            store.update_run(run_id, **update_fields)
            # Update in-memory monitor's backend cache for this run
            monitor = _monitor(request)
            monitor._backends[run_id] = backend
            # Register a per-run callback URL (in-memory, mirrors the backend
            # cache) so the monitor can POST the card on terminal state.
            callback_url = body.get("callback_url")
            if callback_url:
                monitor._callback_urls[run_id] = callback_url
        except Exception as exc:
            logger.exception("Backend submit failed for run=%s: %s", run_id, exc)
            from jobctl.db.models import State
            store.update_run(run_id, state=State.FAILED)

        run = store.get_run(run_id)
        result = _run_to_dict(run, jf.name)
        result["memory_hint"] = memory_hint
        return result

    @app.get("/runs")
    async def list_runs(
        request: Request,
        state: str | None = Query(default=None),
        jobfile_id: str | None = Query(default=None),
    ):
        from jobctl.db.models import State as StateEnum
        store = _store(request)
        state_filter = StateEnum(state) if state else None
        runs = store.list_runs(state=state_filter, jobfile_id=jobfile_id)
        name_cache: dict[str, str | None] = {}
        for r in runs:
            if r.jobfile_id not in name_cache:
                jf = store.get_jobfile(r.jobfile_id)
                name_cache[r.jobfile_id] = jf.name if jf else None
        return [_run_to_dict(r, name_cache.get(r.jobfile_id)) for r in runs]

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str, request: Request):
        store = _store(request)
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        # Content negotiation: HTML for browsers, JSON for API clients
        accept = request.headers.get("accept", "")
        if "text/html" in accept and "application/json" not in accept:
            return await _ui_run_detail_impl(run_id, request)

        return _run_dict_with_name(store, run)

    @app.patch("/runs/{run_id}")
    async def patch_run(run_id: str, body: dict, request: Request):
        """Edit a run's human-readable identity (title / note / tags).

        Body may contain any subset of {title, note, tags}. Empty string / empty
        list clears the field. Other run fields are immutable here.
        """
        store = _store(request)
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        fields: dict[str, Any] = {}
        if "title" in body:
            fields["title"] = body["title"] or None
        if "note" in body:
            fields["note"] = body["note"] or None
        if "tags" in body:
            fields["tags"] = body["tags"] or None
        if fields:
            store.update_run(run_id, **fields)
        return _run_dict_with_name(store, store.get_run(run_id))

    @app.post("/runs/{run_id}/cancel")
    async def cancel_run(run_id: str, request: Request):
        from jobctl.db.models import State
        from jobctl.backends.base import get_backend

        store = _store(request)
        cfg = _config(request)
        monitor = _monitor(request)

        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        # Try backend cancel
        try:
            backend = (
                monitor._backends.get(run_id)
                or get_backend(run.backend or "local", run.server, cfg)
            )
            backend.cancel(run)
        except Exception as exc:
            logger.warning("cancel_run: backend cancel failed: %s", exc)

        store.update_run(run_id, state=State.CANCELLED)
        run = store.get_run(run_id)
        return _run_dict_with_name(store, run)

    @app.post("/runs/{run_id}/rerun")
    async def rerun(run_id: str, request: Request):
        from jobctl.jobfile import render_command, input_hashes as calc_input_hashes
        from jobctl.backends.base import select_backend, get_backend
        from jobctl.db.models import Run, State, Health
        from jobctl.memory.memory import query as mem_query

        store = _store(request)
        cfg = _config(request)

        orig = store.get_run(run_id)
        if orig is None:
            raise HTTPException(status_code=404, detail="Run not found")

        jf = store.get_jobfile(orig.jobfile_id)
        if jf is None:
            raise HTTPException(status_code=404, detail="JobFile not found")

        now = datetime.now(timezone.utc).isoformat()
        new_run_id = f"run-{uuid.uuid4().hex[:12]}"

        servers = store.list_servers()
        backend_name, server_name, task_name = select_backend(jf, servers, override=None)

        new_run = Run(
            run_id=new_run_id,
            jobfile_id=jf.id,
            jobfile_version=jf.version,
            params=orig.params,
            input_hashes=orig.input_hashes,
            backend=backend_name,
            server=server_name,
            task=task_name,
            remote_job_id=None,
            state=State.PENDING,
            health=Health.OK,
            exit_code=None,
            submitted_at=now,
            started_at=None,
            finished_at=None,
            last_heartbeat=None,
            workdir=None,
            stdout_path=None,
            stderr_path=None,
            resource_summary={},
            expectation_match=None,
            observation_card=None,
            title=orig.title,
            note=orig.note,
            tags=orig.tags,
        )
        store.add_run(new_run)

        try:
            backend = get_backend(backend_name, server_name, cfg)
            submit_result = backend.submit(new_run, jf)
            store.update_run(
                new_run_id,
                state=State.SUBMITTED,
                remote_job_id=submit_result.remote_job_id,
                workdir=submit_result.workdir,
            )
            monitor = _monitor(request)
            monitor._backends[new_run_id] = backend
        except Exception as exc:
            logger.exception("rerun: backend submit failed: %s", exc)
            store.update_run(new_run_id, state=State.FAILED)

        new_run = store.get_run(new_run_id)
        return _run_dict_with_name(store, new_run)

    @app.get("/runs/{run_id}/logs", response_class=PlainTextResponse)
    async def get_logs(
        run_id: str,
        request: Request,
        stream: str = Query(default="stdout"),
        tail: int = Query(default=200),
    ):
        store = _store(request)
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        path = run.stdout_path if stream == "stdout" else run.stderr_path

        if path is None or not Path(path).exists():
            return PlainTextResponse("")

        try:
            text = Path(path).read_text(errors="replace")
            # Return last `tail` lines
            lines = text.splitlines()
            return PlainTextResponse("\n".join(lines[-tail:]))
        except OSError:
            return PlainTextResponse("")

    @app.get("/runs/{run_id}/artifacts")
    async def list_artifacts(run_id: str, request: Request):
        store = _store(request)
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        arts = store.list_artifacts(run_id)
        return [_artifact_to_dict(a) for a in arts]

    # ------------------------------------------------------------------
    # /servers
    # ------------------------------------------------------------------

    @app.get("/servers")
    async def list_servers(request: Request):
        store = _store(request)
        return [_server_to_dict(s) for s in store.list_servers()]

    # ------------------------------------------------------------------
    # /feedback
    # ------------------------------------------------------------------

    @app.post("/runs/{run_id}/feedback")
    async def post_feedback(run_id: str, body: dict, request: Request):
        from jobctl.db.models import Feedback

        store = _store(request)
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        now = datetime.now(timezone.utc).isoformat()
        fb = Feedback(
            id=f"fb-{uuid.uuid4().hex[:12]}",
            run_id=run_id,
            kind=body.get("kind", "note"),
            text=body.get("text", ""),
            created_at=now,
        )
        store.add_feedback(fb)
        return _feedback_to_dict(fb)

    @app.get("/runs/{run_id}/feedback")
    async def get_feedback(run_id: str, request: Request):
        store = _store(request)
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return [_feedback_to_dict(fb) for fb in store.list_feedback(run_id)]

    # ------------------------------------------------------------------
    # /expect
    # ------------------------------------------------------------------

    @app.get("/expect")
    async def list_contracts(
        request: Request,
        jobfile_id: str | None = Query(default=None),
    ):
        store = _store(request)
        if jobfile_id:
            contract = store.get_contract(jobfile_id)
            if contract is None:
                return []
            return [_contract_to_dict(contract)]
        # No filter — return all contracts for all jobfiles
        jfs = store.list_jobfiles()
        result = []
        for jf in jfs:
            c = store.get_contract(jf.id)
            if c is not None:
                result.append(_contract_to_dict(c))
        return result

    @app.post("/expect/confirm")
    async def confirm_criterion(body: dict, request: Request):
        from jobctl.expectations.distiller import confirm

        store = _store(request)
        crit_id = body.get("criterion_id", "")
        try:
            criterion = confirm(store, crit_id)
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="Criterion not found")
        return {
            "id": criterion.id,
            "text": criterion.text,
            "status": criterion.status,
            "strength": criterion.strength,
        }

    @app.post("/expect/propose")
    async def propose_criteria(body: dict, request: Request):
        from jobctl.expectations.distiller import propose
        from jobctl.db.models import Feedback

        store = _store(request)
        analyzer = _analyzer(request)

        run_id = body.get("run_id", "")
        feedback_text = body.get("feedback_text", "")

        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        # Build a minimal feedback object for the distiller
        now = datetime.now(timezone.utc).isoformat()
        fb = Feedback(
            id=f"fb-{uuid.uuid4().hex[:12]}",
            run_id=run_id,
            kind="propose",
            text=feedback_text,
            created_at=now,
        )

        try:
            criteria = propose(store, run, fb, analyzer)
        except Exception as exc:
            logger.warning("propose_criteria failed: %s", exc)
            criteria = []

        return [
            {
                "id": c.id,
                "text": c.text,
                "kind": c.kind,
                "status": c.status,
                "strength": c.strength,
            }
            for c in criteria
        ]

    # ------------------------------------------------------------------
    # /memory/query
    # ------------------------------------------------------------------

    @app.get("/memory/query")
    async def memory_query(
        request: Request,
        jobfile_id: str | None = Query(default=None),
        name: str | None = Query(default=None),
    ):
        from jobctl.memory.memory import query

        store = _store(request)
        return query(store, jobfile_id=jobfile_id, name=name)

    # ------------------------------------------------------------------
    # Web UI routes
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def ui_dashboard(request: Request):
        """Dashboard: server health cards + run buckets."""
        store = _store(request)
        templates = request.app.state.templates

        servers = store.list_servers()
        buckets = _build_ui_buckets(store)

        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "servers": servers,
                "buckets": buckets,
                "version": request.app.state.version,
                "page": "dashboard",
            },
        )

    async def _ui_run_detail_impl(run_id: str, request: Request) -> HTMLResponse:
        """Render the run detail HTML page (helper, called with content negotiation)."""
        store = _store(request)
        templates = request.app.state.templates

        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        jf = store.get_jobfile(run.jobfile_id)
        jobfile_name = jf.name if jf else run.jobfile_id

        # Read log tails
        stdout_tail = ""
        stderr_tail = ""
        if run.stdout_path and Path(run.stdout_path).exists():
            try:
                lines = Path(run.stdout_path).read_text(errors="replace").splitlines()
                stdout_tail = "\n".join(lines[-100:])
            except OSError:
                pass
        if run.stderr_path and Path(run.stderr_path).exists():
            try:
                lines = Path(run.stderr_path).read_text(errors="replace").splitlines()
                stderr_tail = "\n".join(lines[-100:])
            except OSError:
                pass

        artifacts = store.list_artifacts(run_id)

        # Contract and per-criterion evaluation
        contract = store.get_contract(run.jobfile_id) if run.jobfile_id else None

        # Build per-criterion map from observation card if present
        per_criterion_map: dict[str, Any] = {}
        per_criterion: list[dict] = []
        if run.observation_card and contract:
            card_criteria = run.observation_card.get("per_criterion", [])
            for pc in card_criteria:
                per_criterion.append(pc)
                if "id" in pc:
                    per_criterion_map[pc["id"]] = pc

        # Normalize state/health/match for template use
        run_state = run.state.value if hasattr(run.state, "value") else run.state
        run_health = run.health.value if hasattr(run.health, "value") else run.health
        run_match = (
            run.expectation_match.value
            if hasattr(run.expectation_match, "value")
            else run.expectation_match
        )

        # Build a plain dict for the template so attribute access is easy
        run_dict = {
            "run_id": run.run_id,
            "jobfile_id": run.jobfile_id,
            "state": run_state,
            "health": run_health,
            "expectation_match": run_match,
            "params": run.params,
            "resource_summary": run.resource_summary,
            "submitted_at": run.submitted_at,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "backend": run.backend,
            "server": run.server,
            "observation_card": run.observation_card,
            "slurm_request": getattr(run, "slurm_request", None),
            "title": getattr(run, "title", None),
            "note": getattr(run, "note", None),
            "tags": getattr(run, "tags", None),
            "display_title": _display_title(run, jobfile_name),
        }

        # Normalize artifact types for template
        artifact_dicts = []
        for art in artifacts:
            artifact_dicts.append({
                "id": art.id,
                "run_id": art.run_id,
                "remote_path": art.remote_path,
                "local_path": art.local_path,
                "type": art.type.value if hasattr(art.type, "value") else art.type,
                "size": art.size,
                "checksum": art.checksum,
                "preview": art.preview,
                "created_at": art.created_at,
            })

        return templates.TemplateResponse(
            request,
            "run.html",
            {
                "run": _DictObj(run_dict),
                "jobfile_name": jobfile_name,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "artifacts": [_DictObj(a) for a in artifact_dicts],
                "contract": contract,
                "per_criterion": per_criterion,
                "per_criterion_map": per_criterion_map,
                "version": request.app.state.version,
            },
        )

    @app.get("/jobfiles/{jobfile_id}", response_class=HTMLResponse)
    async def ui_jobfile_detail(jobfile_id: str, request: Request):
        """JobFile detail: params schema, historical runs, contract versions."""
        store = _store(request)
        templates = request.app.state.templates

        jf = store.get_jobfile(jobfile_id)
        if jf is None:
            raise HTTPException(status_code=404, detail="JobFile not found")

        runs = store.list_runs(jobfile_id=jobfile_id)
        contract = store.get_contract(jobfile_id)

        # Normalize run state/match for template
        run_objs = []
        for run in runs:
            run_objs.append(_DictObj({
                "run_id": run.run_id,
                "state": run.state.value if hasattr(run.state, "value") else run.state,
                "health": run.health.value if hasattr(run.health, "value") else run.health,
                "expectation_match": (
                    run.expectation_match.value
                    if hasattr(run.expectation_match, "value")
                    else run.expectation_match
                ),
                "params": run.params,
                "backend": run.backend,
                "server": run.server,
                "submitted_at": run.submitted_at,
                "finished_at": run.finished_at,
            }))

        return templates.TemplateResponse(
            request,
            "jobfile.html",
            {
                "jobfile": jf,
                "runs": run_objs,
                "contract": contract,
                "version": request.app.state.version,
            },
        )

    @app.get("/ui/poll", response_class=HTMLResponse)
    async def ui_poll(
        request: Request,
        run_id: str | None = Query(default=None),
        section: str | None = Query(default=None),
    ):
        """HTMX poll partial — returns updated fragments."""
        store = _store(request)
        templates = request.app.state.templates

        if run_id:
            # Return a run-state fragment for a specific run
            run = store.get_run(run_id)
            if run is None:
                return HTMLResponse(content=f'<span class="text-muted">Run {run_id} not found</span>')

            run_state = run.state.value if hasattr(run.state, "value") else run.state
            run_health = run.health.value if hasattr(run.health, "value") else run.health
            run_match = (
                run.expectation_match.value
                if hasattr(run.expectation_match, "value")
                else run.expectation_match
            )
            html = (
                f'<div class="flex-gap">'
                f'<h1>Run <span class="mono">{run.run_id}</span></h1>'
                f'<span class="badge badge-{run_state}">{run_state}</span>'
                f'<span class="badge badge-{run_health}">{run_health}</span>'
                f'</div>'
            )
            return HTMLResponse(content=html)

        # Default: return run buckets fragment
        buckets = _build_ui_buckets(store)

        return templates.TemplateResponse(
            request,
            "partials/buckets.html",
            {
                "buckets": buckets,
                "version": request.app.state.version,
            },
        )

    @app.get("/ui/artifact-thumb")
    async def ui_artifact_thumb(
        request: Request,
        path: str = Query(default=""),
    ):
        """Serve artifact image by local path (base64-inlined or raw)."""
        if not path:
            raise HTTPException(status_code=400, detail="path required")

        fp = Path(path)
        if not fp.exists() or not fp.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")

        suffix = fp.suffix.lower()
        media_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".webp": "image/webp",
        }
        media_type = media_map.get(suffix, "application/octet-stream")

        try:
            data = fp.read_bytes()
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return Response(content=data, media_type=media_type)


# ---------------------------------------------------------------------------
# Helper: dict-backed object for Jinja2 attribute access
# ---------------------------------------------------------------------------

class _DictObj:
    """Wrap a dict so Jinja2 can use attribute-style access (obj.key)."""
    def __init__(self, d: dict) -> None:
        self.__dict__.update(d)

    def __getattr__(self, item: str):
        return None  # graceful default for missing keys


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _jobfile_to_dict(jf) -> dict:
    return {
        "id": jf.id,
        "name": jf.name,
        "version": jf.version,
        "source_path": jf.source_path,
        "command_template": jf.command_template,
        "params_schema": jf.params_schema,
        "backend_prefs": jf.backend_prefs,
        "artifact_patterns": jf.artifact_patterns,
        "expectation_contract_id": jf.expectation_contract_id,
        "content_hash": jf.content_hash,
        "created_at": jf.created_at,
    }


def _run_to_dict(run, jobfile_name: str | None = None) -> dict:
    return {
        "run_id": run.run_id,
        "jobfile_id": run.jobfile_id,
        "jobfile_version": run.jobfile_version,
        "params": run.params,
        "input_hashes": run.input_hashes,
        "backend": run.backend,
        "server": run.server,
        "task": run.task,
        "remote_job_id": run.remote_job_id,
        "state": run.state.value if hasattr(run.state, "value") else run.state,
        "health": run.health.value if hasattr(run.health, "value") else run.health,
        "exit_code": run.exit_code,
        "submitted_at": run.submitted_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "last_heartbeat": run.last_heartbeat,
        "workdir": run.workdir,
        "stdout_path": run.stdout_path,
        "stderr_path": run.stderr_path,
        "resource_summary": run.resource_summary,
        "expectation_match": (
            run.expectation_match.value
            if hasattr(run.expectation_match, "value")
            else run.expectation_match
        ),
        "observation_card": run.observation_card,
        "slurm_request": getattr(run, "slurm_request", None),
        "title": getattr(run, "title", None),
        "note": getattr(run, "note", None),
        "tags": getattr(run, "tags", None),
        "display_title": _display_title(run, jobfile_name),
    }


def _run_dict_with_name(store, run) -> dict:
    """``_run_to_dict`` with the JobFile name resolved so ``display_title`` reads
    as "<jobfile> · <params>" instead of falling back to the jobfile_id hash."""
    if run is None:
        return _run_to_dict(run)
    jf = store.get_jobfile(run.jobfile_id)
    return _run_to_dict(run, jf.name if jf else None)


def _display_title(run, jobfile_name: str | None = None) -> str:
    """Human-readable label for a run.

    Uses the explicit title when set; otherwise derives "<jobfile> · <≤3 key
    params>" so a run is never shown as just an opaque hash. ``jobfile_name`` is
    the resolved JobFile name when available (the UI passes it); falls back to
    the jobfile_id.
    """
    title = getattr(run, "title", None)
    if title:
        return title
    base = jobfile_name or getattr(run, "jobfile_id", "") or "run"
    params = getattr(run, "params", None) or {}
    if params:
        keys = list(params.keys())[:3]
        ptxt = ", ".join(f"{k}={params[k]}" for k in keys)
        return f"{base} · {ptxt}"
    return base


class _Buckets:
    """Lightweight attribute bag for the 6 dashboard run buckets."""
    running: list
    queued: list
    stuck: list
    weak: list
    completed: list
    failed: list


def _build_ui_buckets(store) -> "_Buckets":
    """Augment every run with jobfile_name + display_title and classify into the
    6 dashboard buckets.

    Shared by the dashboard page (GET /) and the HTMX poll partial (GET /ui/poll)
    so the bucket classification can never drift out of sync between them.
    """
    all_runs = store.list_runs()
    jf_cache: dict[str, str] = {}
    for run in all_runs:
        if run.jobfile_id not in jf_cache:
            jf = store.get_jobfile(run.jobfile_id)
            jf_cache[run.jobfile_id] = jf.name if jf else run.jobfile_id
    for run in all_runs:
        name = jf_cache.get(run.jobfile_id, "")
        run.__dict__["jobfile_name"] = name
        run.__dict__["display_title"] = _display_title(run, name)
        run.__dict__["expectation_match"] = (
            run.expectation_match.value
            if hasattr(run.expectation_match, "value")
            else run.expectation_match
        )
    b = _Buckets()
    b.running   = [r for r in all_runs if r.state.value in ("running", "submitted")]
    b.queued    = [r for r in all_runs if r.state.value == "pending"]
    b.stuck     = [r for r in all_runs if r.state.value == "stuck"]
    b.weak      = [r for r in all_runs if r.expectation_match == "weak_signal"]
    b.completed = [r for r in all_runs if r.state.value == "completed" and r.expectation_match != "weak_signal"]
    b.failed    = [r for r in all_runs if r.state.value in ("failed", "cancelled", "timeout")]
    return b


def _artifact_to_dict(art) -> dict:
    return {
        "id": art.id,
        "run_id": art.run_id,
        "remote_path": art.remote_path,
        "local_path": art.local_path,
        "type": art.type.value if hasattr(art.type, "value") else art.type,
        "size": art.size,
        "checksum": art.checksum,
        "preview": art.preview,
        "created_at": art.created_at,
    }


def _server_to_dict(s) -> dict:
    return {
        "name": s.name,
        "backend_type": s.backend_type,
        "online": s.online,
        "last_heartbeat": s.last_heartbeat,
        "cpu": s.cpu,
        "mem": s.mem,
        "gpu": s.gpu,
        "disk": s.disk,
        "slurm_queue": s.slurm_queue,
        "note": s.note,
    }


def _feedback_to_dict(fb) -> dict:
    return {
        "id": fb.id,
        "run_id": fb.run_id,
        "kind": fb.kind,
        "text": fb.text,
        "created_at": fb.created_at,
    }


def _contract_to_dict(c) -> dict:
    return {
        "id": c.id,
        "jobfile_id": c.jobfile_id,
        "version": c.version,
        "criteria": [
            {
                "id": cr.id,
                "text": cr.text,
                "kind": cr.kind,
                "check": cr.check,
                "status": cr.status,
                "strength": cr.strength,
                "evidence_run_ids": cr.evidence_run_ids,
            }
            for cr in c.criteria
        ],
        "source": c.source,
        "created_at": c.created_at,
        "updated_at": c.updated_at,
    }
