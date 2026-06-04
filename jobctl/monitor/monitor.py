"""Monitor loop + build_observation_card.

The Monitor class:
- run_loop(stop_event): asyncio loop that drives probe_servers + poll_runs
  on a configurable tick interval.
- probe_servers(): checks server health via _prober (injectable) and persists
  Server rows.
- poll_runs(): polls all active (SUBMITTED / RUNNING) runs via their backends;
  updates state; calls on_terminal when a terminal state is reached.
- on_terminal(run): runs the terminal pipeline:
    1. collect() results
    2. index_run() artifacts
    3. evaluate() against expectation contract
    4. build_observation_card()
    5. notify() all notifiers
    6. persist card + expectation_match

Stuck detection:
    A RUNNING run is declared STUCK when BOTH conditions hold:
    - last_log_mtime is stale (or None) — no log growth for > stuck_timeout_seconds
    - last_heartbeat is stale (or None) — no heartbeat for > stuck_timeout_seconds

build_observation_card:
    Always returns all required fields:
    {status, jobfile, run_id, server, artifacts:[{name,type,preview}],
     health, expectation_match, key_evidence, interpretation,
     recommended_next_action}
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from jobctl.db.models import Health, Match, Run, Server, State

if TYPE_CHECKING:
    from jobctl.analysis.base import Analyzer
    from jobctl.artifacts.indexer import Artifact
    from jobctl.backends.base import Backend
    from jobctl.db.models import JobFile
    from jobctl.db.store import Store
    from jobctl.notify.notify import Notifier

logger = logging.getLogger(__name__)

# Default thresholds
_DEFAULT_STUCK_TIMEOUT = 600       # seconds without log update + heartbeat
_DEFAULT_POLL_INTERVAL = 10.0      # seconds between poll cycles
_DEFAULT_PROBE_INTERVAL = 60.0     # seconds between server probes

# Terminal states — no further polling needed
_TERMINAL_STATES = {
    State.COMPLETED,
    State.FAILED,
    State.CANCELLED,
    State.TIMEOUT,
    State.STUCK,
}

# Active (non-terminal, non-pending) states that we poll
_POLLABLE_STATES = {State.SUBMITTED, State.RUNNING}


# ---------------------------------------------------------------------------
# build_observation_card — module-level function (also used by other layers)
# ---------------------------------------------------------------------------

def build_observation_card(
    run: Run,
    jobfile: "JobFile",
    artifacts: "list[Artifact]",
    match: Match | None,
    key_evidence: list[str],
    health: Health,
    analyzer: "Analyzer",
) -> dict:
    """Build a complete observation card for a terminal run.

    Returns a dict with ALL required fields:
        status, jobfile, run_id, server, artifacts, health,
        expectation_match, key_evidence, interpretation, recommended_next_action

    Never returns just "finished" — status is always the real state name.
    """
    # Resolve match value
    match_str: str | None = None
    if match is not None:
        match_str = match.value if isinstance(match, Match) else str(match)

    # Resolve health value
    health_str = health.value if isinstance(health, Health) else str(health)

    # Resolve state/status — use actual state name, never "finished"
    state = run.state
    status = state.value if isinstance(state, State) else str(state)

    # Build artifacts list for the card
    artifact_entries = []
    for art in artifacts:
        name = Path(art.local_path).name if art.local_path else Path(art.remote_path).name
        atype = art.type.value if hasattr(art.type, "value") else str(art.type)
        artifact_entries.append({
            "name": name,
            "type": atype,
            "preview": art.preview or {},
            "local_path": art.local_path,
            "remote_path": art.remote_path,
            "checksum": art.checksum,
            "size": art.size,
        })

    # Build facts for analyzer
    facts: dict = {
        "state": status,
        "exit_code": run.exit_code,
        "expectation_match": match_str,
        "health": health_str,
        "key_evidence": key_evidence,
        "artifacts": [{"name": e["name"], "type": e["type"]} for e in artifact_entries],
        "run_id": run.run_id,
        "server": run.server,
        "jobfile": jobfile.name if jobfile else None,
        "resource_summary": run.resource_summary or {},
    }

    # Get analysis from analyzer
    try:
        analysis = analyzer.analyze_run(facts)
    except Exception as exc:
        logger.warning("Analyzer.analyze_run failed: %s", exc)
        analysis = {
            "interpretation": f"Run {status}.",
            "recommended_next_action": "Review run output.",
        }

    interpretation = analysis.get("interpretation", f"Run {status}.")
    recommended_next_action = analysis.get(
        "recommended_next_action", "Review run output."
    )

    card = {
        "status": status,
        "jobfile": jobfile.name if jobfile else None,
        "run_id": run.run_id,
        "server": run.server,
        "artifacts": artifact_entries,
        "health": health_str,
        "expectation_match": match_str,
        "key_evidence": list(key_evidence),
        "interpretation": interpretation,
        "recommended_next_action": recommended_next_action,
    }
    return card


# ---------------------------------------------------------------------------
# Default Prober (no-op — real probing is done by api/server.py health checks)
# ---------------------------------------------------------------------------

class _NullProber:
    """Prober that always returns None (server unknown/offline)."""

    def probe(self, name: str) -> Server | None:
        return None


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class Monitor:
    """Asyncio monitor loop that manages run lifecycle.

    Attributes:
        store: The Store (SQLite repository).
        config: Raw config dict from jobctl.config.
        analyzer: Analyzer instance for interpretation.
        notifiers_factory: Callable(run) -> list[Notifier].
        _prober: Injectable prober with probe(name) -> Server|None method.
        _backends: Dict mapping run_id -> Backend instance.
                   Injected in tests; in production the Monitor resolves backends
                   from config on first access.
    """

    def __init__(
        self,
        store: "Store",
        config: dict,
        analyzer: "Analyzer",
        notifiers_factory: Callable[["Run"], "list[Notifier]"],
    ) -> None:
        self.store = store
        self.config = config
        self.analyzer = analyzer
        self.notifiers_factory = notifiers_factory

        # Injectable: prober used by probe_servers
        self._prober = _NullProber()

        # Injectable: run_id -> Backend (for testing; production resolves dynamically)
        self._backends: dict[str, "Backend"] = {}

        # run_id -> per-run callback URL, set by the API at submit time so the
        # terminal pipeline can POST the observation card to it.
        self._callback_urls: dict[str, str] = {}

        # Timing thresholds
        self._stuck_timeout = float(
            config.get("stuck_timeout_seconds", _DEFAULT_STUCK_TIMEOUT)
        )
        self._poll_interval = float(
            config.get("poll_interval_seconds", _DEFAULT_POLL_INTERVAL)
        )
        self._probe_interval = float(
            config.get("probe_interval_seconds", _DEFAULT_PROBE_INTERVAL)
        )

        # Track when we last probed servers
        self._last_probe_time: float = 0.0

    # ------------------------------------------------------------------
    # run_loop
    # ------------------------------------------------------------------

    async def run_loop(self, stop_event: asyncio.Event) -> None:
        """Main asyncio monitor loop.

        Runs until stop_event is set.  On each tick:
        1. Optionally probe servers (once per probe_interval).
        2. Poll all active runs.
        """
        while not stop_event.is_set():
            now = time.time()

            # Probe servers periodically
            if now - self._last_probe_time >= self._probe_interval:
                try:
                    await self.probe_servers()
                except Exception as exc:
                    logger.exception("probe_servers failed: %s", exc)
                self._last_probe_time = time.time()

            # Poll active runs
            try:
                await self.poll_runs()
            except Exception as exc:
                logger.exception("poll_runs failed: %s", exc)

            # Wait for next tick (or stop)
            try:
                await asyncio.wait_for(
                    asyncio.shield(stop_event.wait()),
                    timeout=self._poll_interval,
                )
            except asyncio.TimeoutError:
                pass  # Normal — time to poll again

    # ------------------------------------------------------------------
    # probe_servers
    # ------------------------------------------------------------------

    async def probe_servers(self) -> None:
        """Probe all configured servers and update their DB rows.

        Uses self._prober.probe(name) to get a Server snapshot.  If the prober
        returns None (unreachable), the server is marked offline.
        """
        servers_config: dict = self.config.get("servers", {})
        now_iso = datetime.now(timezone.utc).isoformat()
        names = list(servers_config)

        # Probe all servers concurrently, off the event loop (SSH is blocking).
        async def _probe(name: str) -> "Server | None":
            try:
                return await asyncio.to_thread(self._prober.probe, name)
            except Exception as exc:
                logger.warning("Prober failed for server %s: %s", name, exc)
                return None

        snapshots = await asyncio.gather(*[_probe(n) for n in names]) if names else []

        for name, snapshot in zip(names, snapshots):
            if snapshot is not None:
                # Server is reachable; persist updated snapshot
                server = Server(
                    name=name,
                    backend_type=snapshot.backend_type or servers_config[name].get("backend", "ssh"),
                    online=True,
                    last_heartbeat=now_iso,
                    cpu=snapshot.cpu or {},
                    mem=snapshot.mem or {},
                    gpu=snapshot.gpu or {},
                    disk=snapshot.disk or {},
                    slurm_queue=snapshot.slurm_queue or {},
                    note=snapshot.note,
                )
            else:
                # Server unreachable — mark offline, preserve existing resource info
                existing = self.store.get_server(name)
                if existing is not None:
                    server = Server(
                        name=name,
                        backend_type=existing.backend_type,
                        online=False,
                        last_heartbeat=existing.last_heartbeat,
                        cpu=existing.cpu,
                        mem=existing.mem,
                        gpu=existing.gpu,
                        disk=existing.disk,
                        slurm_queue=existing.slurm_queue,
                        note="unreachable",
                    )
                else:
                    server = Server(
                        name=name,
                        backend_type=servers_config[name].get("backend", "ssh"),
                        online=False,
                        last_heartbeat=None,
                        cpu={}, mem={}, gpu={}, disk={}, slurm_queue={},
                        note="unreachable",
                    )

            self.store.upsert_server(server)

    # ------------------------------------------------------------------
    # poll_runs
    # ------------------------------------------------------------------

    async def poll_runs(self) -> None:
        """Poll all SUBMITTED/RUNNING runs; update state; call on_terminal."""
        # Fetch runs in pollable states
        runs: list[Run] = []
        for state in _POLLABLE_STATES:
            runs.extend(self.store.list_runs(state=state))

        for run in runs:
            try:
                await self._poll_one(run)
            except Exception as exc:
                logger.exception(
                    "poll_runs: error polling run=%s: %s", run.run_id, exc
                )

    async def _poll_one(self, run: Run) -> None:
        """Poll a single run and transition state as needed."""
        backend = self._get_backend_for_run(run)
        if backend is None:
            logger.warning("No backend for run=%s; skipping poll", run.run_id)
            return

        try:
            poll = backend.poll(run)
        except Exception as exc:
            logger.warning("Backend.poll failed for run=%s: %s", run.run_id, exc)
            return

        new_state = poll.state
        now_iso = datetime.now(timezone.utc).isoformat()

        # --- Stuck detection ---
        if run.state == State.RUNNING or new_state == State.RUNNING:
            if self._is_stuck(run, poll):
                new_state = State.STUCK

        # --- Update state if changed ---
        if new_state != run.state:
            updates: dict = {"state": new_state}

            if new_state == State.RUNNING and run.started_at is None:
                updates["started_at"] = now_iso

            if new_state == State.STUCK:
                updates["health"] = Health.STUCK

            self.store.update_run(run.run_id, **updates)

            # Refresh run from DB
            run = self.store.get_run(run.run_id)

        # --- Update heartbeat for RUNNING runs ---
        if new_state == State.RUNNING:
            self.store.update_run(run.run_id, last_heartbeat=now_iso)

        # --- Terminal handling ---
        if new_state in _TERMINAL_STATES:
            run = self.store.get_run(run.run_id)
            await self.on_terminal(run)

    def _is_stuck(self, run: Run, poll: "from jobctl.backends.base import PollResult") -> bool:
        """Determine if a RUNNING run is stuck.

        Stuck = log has not grown for > stuck_timeout_seconds
                AND heartbeat has not been updated for > stuck_timeout_seconds.

        When no log mtime or heartbeat is available yet (e.g. first poll of
        an SSH/SLURM job before stdout is locally mirrored), we fall back to
        submitted_at / started_at so that a brand-new run is never immediately
        declared stuck.
        """
        now = time.time()
        threshold = self._stuck_timeout

        # Check last_log_mtime
        log_stale = False
        last_mtime = poll.last_log_mtime
        if last_mtime is None:
            # Try stdout_path mtime as fallback
            if run.stdout_path:
                try:
                    last_mtime = Path(run.stdout_path).stat().st_mtime
                except OSError:
                    last_mtime = None
        if last_mtime is None:
            # No log mtime available; use started_at / submitted_at as proxy
            ref_ts: str | None = run.started_at or run.submitted_at
            if ref_ts is not None:
                try:
                    ref_dt = datetime.fromisoformat(ref_ts)
                    if (now - ref_dt.timestamp()) > threshold:
                        log_stale = True
                    # else: job only just started, not stale yet
                except (ValueError, OSError):
                    log_stale = True  # unparseable timestamp -> assume stale
            # If ref_ts is also None, the run is brand-new; do not mark log stale
        elif (now - last_mtime) > threshold:
            log_stale = True

        # Check heartbeat
        hb_stale = False
        last_hb = run.last_heartbeat
        if last_hb is None:
            # No heartbeat yet; use started_at / submitted_at as proxy
            ref_ts = run.started_at or run.submitted_at
            if ref_ts is not None:
                try:
                    ref_dt = datetime.fromisoformat(ref_ts)
                    if (now - ref_dt.timestamp()) > threshold:
                        hb_stale = True
                    # else: job only just started, not stale yet
                except (ValueError, OSError):
                    hb_stale = True
            # If ref_ts is None, brand-new run; do not mark hb stale
        else:
            try:
                hb_dt = datetime.fromisoformat(last_hb)
                hb_age = now - hb_dt.timestamp()
                if hb_age > threshold:
                    hb_stale = True
            except (ValueError, OSError):
                hb_stale = True

        return log_stale and hb_stale

    # ------------------------------------------------------------------
    # on_terminal
    # ------------------------------------------------------------------

    async def on_terminal(self, run: Run) -> None:
        """Run the terminal pipeline for a just-completed (or failed/stuck) run.

        1. collect() results from the backend (skip if already collected or stuck)
        2. index_run() artifacts
        3. evaluate() against expectation contract (or default_contract)
        4. build_observation_card()
        5. notify() all notifiers
        6. persist card + expectation_match to DB
        """
        from jobctl.artifacts.indexer import index_run
        from jobctl.expectations.contracts import evaluate, default_contract

        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Get the JobFile
        jobfile = self.store.get_jobfile(run.jobfile_id)

        # 2. Collect results (only for non-stuck terminal states where backend has data)
        if run.state not in (State.STUCK,) and run.exit_code is None:
            backend = self._get_backend_for_run(run)
            if backend is not None:
                try:
                    collect = backend.collect(run)
                    updates: dict = {
                        "exit_code": collect.exit_code,
                        "finished_at": now_iso,
                    }
                    if collect.stdout_path:
                        updates["stdout_path"] = collect.stdout_path
                    if collect.stderr_path:
                        updates["stderr_path"] = collect.stderr_path
                    if collect.artifact_dir:
                        # Always update workdir to the local mirror returned by
                        # collect() so that the artifact indexer (which uses
                        # run.workdir to glob) finds the locally-rsync'd files.
                        updates["workdir"] = collect.artifact_dir
                    if collect.resource_summary:
                        updates["resource_summary"] = collect.resource_summary
                    self.store.update_run(run.run_id, **updates)
                    run = self.store.get_run(run.run_id)
                except Exception as exc:
                    logger.warning(
                        "on_terminal: collect failed for run=%s: %s", run.run_id, exc
                    )
        elif run.state not in (State.STUCK,):
            # Update finished_at if not set
            if run.finished_at is None:
                self.store.update_run(run.run_id, finished_at=now_iso)
                run = self.store.get_run(run.run_id)

        # 3. Index artifacts
        artifacts = []
        if jobfile is not None:
            try:
                artifacts = index_run(self.store, run, jobfile)
            except Exception as exc:
                logger.warning(
                    "on_terminal: index_run failed for run=%s: %s", run.run_id, exc
                )

        # 4. Read stdout / stderr for evaluation
        stdout = ""
        stderr = ""
        if run.stdout_path:
            try:
                stdout = Path(run.stdout_path).read_text(errors="replace")
            except OSError:
                pass
        if run.stderr_path:
            try:
                stderr = Path(run.stderr_path).read_text(errors="replace")
            except OSError:
                pass

        # 5. Evaluate against contract
        match: Match | None = None
        key_evidence: list[str] = []
        per_crit: list[dict] = []
        health = run.health if isinstance(run.health, Health) else Health.OK

        if run.state == State.STUCK:
            match = Match.INCONCLUSIVE
            key_evidence = ["Run was detected as stuck (no log growth + no heartbeat)"]
            health = Health.STUCK
        else:
            if jobfile is not None:
                contract = None
                if jobfile.expectation_contract_id:
                    contract = self.store.get_contract_by_id(
                        jobfile.expectation_contract_id
                    )
                if contract is None:
                    contract = self.store.get_contract(jobfile.id)
                if contract is None:
                    contract = default_contract(jobfile)

                try:
                    match, key_evidence, per_crit = evaluate(
                        contract, run, artifacts, stdout, stderr
                    )
                except Exception as exc:
                    logger.warning(
                        "on_terminal: evaluate failed for run=%s: %s", run.run_id, exc
                    )
                    match = Match.INCONCLUSIVE
                    key_evidence = [f"Evaluation error: {exc}"]
            else:
                # No jobfile in DB; use exit code as simple proxy
                if run.exit_code is not None and run.exit_code != 0:
                    match = Match.FAILED
                    key_evidence = [f"exit_code={run.exit_code}"]
                else:
                    match = Match.USABLE

        # 6. Build observation card
        if jobfile is None:
            # Minimal synthetic jobfile for card building
            from jobctl.db.models import JobFile as JF
            import uuid as _uuid
            jobfile = JF(
                id=run.jobfile_id,
                name=run.jobfile_id,
                version=1,
                source_path=None,
                command_template="",
                params_schema={},
                backend_prefs=[],
                artifact_patterns=[],
                expectation_contract_id=None,
                content_hash="",
                created_at=now_iso,
            )

        card = build_observation_card(
            run=run,
            jobfile=jobfile,
            artifacts=artifacts,
            match=match,
            key_evidence=key_evidence,
            health=health,
            analyzer=self.analyzer,
        )

        # Carry per-criterion results in the card so the UI can show PASS/FAIL.
        if isinstance(card, dict):
            card["per_criterion"] = per_crit

        # 7. Persist card + expectation_match
        self.store.update_run(
            run.run_id,
            observation_card=card,
            expectation_match=match,
            health=health,
        )

        # 8. Notify — reattach any per-run callback URL so get_notifiers picks it up.
        cb = self._callback_urls.get(run.run_id)
        if cb:
            try:
                run._callback_url = cb
            except Exception:
                pass
        notifiers = self.notifiers_factory(run)
        for notifier in notifiers:
            try:
                notifier.notify(run, card)
            except Exception as exc:
                logger.warning(
                    "on_terminal: notifier %s failed for run=%s: %s",
                    type(notifier).__name__,
                    run.run_id,
                    exc,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_backend_for_run(self, run: Run) -> "Backend | None":
        """Return the Backend for a run.

        In tests, _backends is pre-populated with run_id -> Backend.
        In production, this method would call get_backend() from backends.base.
        """
        backend = self._backends.get(run.run_id)
        if backend is not None:
            return backend

        # Production fallback: resolve from config
        try:
            from jobctl.backends.base import get_backend
            return get_backend(
                backend=run.backend or "local",
                server=run.server,
                config=self.config,
            )
        except Exception as exc:
            logger.warning(
                "_get_backend_for_run: failed to resolve backend for run=%s: %s",
                run.run_id,
                exc,
            )
            return None
