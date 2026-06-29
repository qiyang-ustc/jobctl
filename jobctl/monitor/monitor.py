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
import sqlite3
import time
import uuid
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

# States we keep polling. STUCK is included so a run that was flagged stuck
# (often just a transient connectivity blip) gets RECONCILED once the cluster
# is reachable again — squeue/sacct will resolve it to its true state.
_POLLABLE_STATES = {State.SUBMITTED, State.RUNNING, State.STUCK}


def _exception_summary(exc: BaseException) -> str:
    message = str(exc)
    return f"{exc.__class__.__name__}: {message}" if message else exc.__class__.__name__


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
        # Hard cap on a single backend.poll() call so a hung SSH can never
        # block the monitor's event loop (backstop above the SSH timeout).
        self._poll_call_timeout = float(
            config.get("poll_call_timeout_seconds", 45.0)
        )

        # Track when we last probed servers
        self._last_probe_time: float = 0.0

        # macOS desktop notifications. A burst of terminal transitions is
        # coalesced into ONE "N jobs finished" banner (the "series" signal).
        # Disabled unless explicitly enabled AND we're on a mac with osascript —
        # a silent no-op everywhere else (incl. tests / CI).
        self._mac_coalescer = None
        if config.get("notify_macos_enabled", False):
            try:
                from jobctl.notify.macos import MacNotifyCoalescer, is_macos_available
                if is_macos_available():
                    self._mac_coalescer = MacNotifyCoalescer(
                        window=float(config.get("notify_window_seconds", 15.0)),
                        sound=(config.get("notify_sound") or None),
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("macOS notifier init failed: %s", exc)
                self._mac_coalescer = None

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
        """Poll all pollable runs and transition state.

        Runs are grouped by (backend, server) and polled in ONE batch call per
        group (backend.poll_many) — so a sweep of N SLURM jobs costs one
        `squeue` per cycle, not N SSH connections (the storm that throttled the
        login node and triggered the false-stuck cascade).
        """
        runs: list[Run] = []
        for state in _POLLABLE_STATES:
            runs.extend(self.store.list_runs(state=state))
        if not runs:
            return

        # Group by (backend, server)
        groups: dict[tuple, list[Run]] = {}
        for run in runs:
            groups.setdefault((run.backend or "local", run.server), []).append(run)

        for (_bname, _server), group in groups.items():
            backend = self._get_backend_for_run(group[0])
            if backend is None:
                for run in group:
                    logger.warning("No backend for run=%s; skipping poll", run.run_id)
                continue
            try:
                results = await asyncio.wait_for(
                    asyncio.to_thread(backend.poll_many, group),
                    timeout=self._poll_call_timeout,
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning(
                    "poll_many failed/timed out for %s/%s: %s",
                    _bname,
                    _server,
                    _exception_summary(exc),
                )
                for run in group:
                    self.store.update_run(run.run_id, health=Health.NO_HEARTBEAT)
                continue
            for run in group:
                poll = results.get(run.run_id)
                if poll is None:
                    self.store.update_run(run.run_id, health=Health.NO_HEARTBEAT)
                    continue
                try:
                    await self._apply_poll(run, poll)
                except Exception as exc:
                    logger.exception("apply_poll error run=%s: %s", run.run_id, exc)

    async def _poll_one(self, run: Run) -> None:
        """Poll a single run (fetch + apply). Kept for direct callers/tests."""
        backend = self._get_backend_for_run(run)
        if backend is None:
            logger.warning("No backend for run=%s; skipping poll", run.run_id)
            return
        try:
            poll = await asyncio.wait_for(
                asyncio.to_thread(backend.poll, run),
                timeout=self._poll_call_timeout,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning(
                "poll failed/timed out for run=%s: %s",
                run.run_id,
                _exception_summary(exc),
            )
            self.store.update_run(run.run_id, health=Health.NO_HEARTBEAT)
            return
        await self._apply_poll(run, poll)

    async def _apply_poll(self, run: Run, poll) -> None:
        """Apply a fetched PollResult to *run*: transition state as needed."""
        prev_state = run.state
        now_iso = datetime.now(timezone.utc).isoformat()

        # --- Unreachable: backend could not determine the job's state ---
        # Mark it as "no heartbeat" but DO NOT change run state — "I can't see
        # the job" is not "the job failed/is stuck". It will reconcile on a
        # later successful poll.
        if not getattr(poll, "reachable", True):
            logger.warning(
                "run=%s: cluster unreachable, keeping state=%s (health=no_heartbeat)",
                run.run_id, getattr(prev_state, "value", prev_state),
            )
            self.store.update_run(run.run_id, health=Health.NO_HEARTBEAT)
            return

        new_state = poll.state

        # --- Stuck detection (only when the job is still RUNNING) ---
        if prev_state == State.RUNNING or new_state == State.RUNNING:
            if self._is_stuck(run, poll):
                new_state = State.STUCK

        # --- Update state if changed ---
        if new_state != prev_state:
            updates: dict = {"state": new_state}
            if new_state == State.RUNNING and run.started_at is None:
                updates["started_at"] = now_iso
            if new_state == State.STUCK:
                updates["health"] = Health.STUCK
            elif new_state == State.RUNNING:
                updates["health"] = Health.OK  # recovered from a blip
            self.store.update_run(run.run_id, **updates)
            run = self.store.get_run(run.run_id)

        # --- Runtime GPU OOM check for logs already visible locally. SLURM CPU
        # OOM usually appears as a terminal scheduler state; GPU library OOM can
        # show up in stderr before the process exits, so stop it at poll time.
        if new_state == State.RUNNING and await self._check_running_gpu_oom(run):
            return

        # --- Update heartbeat (+ clear stale health) for RUNNING runs ---
        if new_state == State.RUNNING:
            self.store.update_run(run.run_id, last_heartbeat=now_iso, health=Health.OK)

        # --- Terminal handling: only fire on a NEW transition into terminal ---
        if new_state in _TERMINAL_STATES and new_state != prev_state:
            run = self.store.get_run(run.run_id)
            await self.on_terminal(run)

    def _is_stuck(self, run: Run, poll: "from jobctl.backends.base import PollResult") -> bool:
        """Stuck = the job is still RUNNING but its log has gone quiet too long.

        We require REAL evidence of a stalled log: a concrete ``last_log_mtime``
        (or an existing local stdout file) that hasn't advanced for
        > stuck_timeout_seconds. We deliberately do NOT:
          - use jobctl's own heartbeat (that reflects OUR poll cadence, not the
            job — a connectivity blip would falsely trip it; that was the bug), or
          - guess from started_at when no log exists (that falsely flagged every
            quiet long-running job, e.g. a SLURM job whose stdout is only
            mirrored on completion).
        No log evidence -> not stuck. The scheduler still saying RUNNING wins.
        """
        now = time.time()
        last_mtime = poll.last_log_mtime
        if last_mtime is None and run.stdout_path:
            try:
                last_mtime = Path(run.stdout_path).stat().st_mtime
            except OSError:
                last_mtime = None
        if last_mtime is None:
            return False  # no evidence of a stall -> trust the scheduler
        return (now - last_mtime) > self._stuck_timeout

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
        if health == Health.NO_HEARTBEAT:
            # A reachable terminal poll proves the job did not silently vanish;
            # keep no_heartbeat as a transient connectivity warning only.
            health = Health.OK

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

        from jobctl.memory.mem_auto import classify_oom

        oom_diagnosis = classify_oom(run.resource_summary, stdout, stderr)
        auto_retry_info = None
        if oom_diagnosis.is_oom:
            for evidence in oom_diagnosis.evidence:
                if evidence not in key_evidence:
                    key_evidence.append(evidence)
            health = Health.RESOURCE_PRESSURE
            if match in (None, Match.USABLE, Match.WEAK_SIGNAL, Match.INCONCLUSIVE):
                match = Match.FAILED

            if oom_diagnosis.kind == "cpu":
                if jobfile is not None:
                    auto_retry_info = await self._maybe_submit_mem_auto_retry(
                        run, jobfile, oom_diagnosis
                    )
                    if auto_retry_info and auto_retry_info.get("submitted"):
                        key_evidence.append(
                            "CPU OOM detected; mem_auto submitted retry "
                            f"{auto_retry_info['run_id']} with mem "
                            f"{auto_retry_info['old_mem']} -> {auto_retry_info['new_mem']}"
                        )
                    elif auto_retry_info:
                        key_evidence.append(
                            "CPU OOM detected; mem_auto did not submit a retry "
                            f"({auto_retry_info.get('reason', 'unknown')})"
                        )
                    else:
                        key_evidence.append("CPU OOM detected; mem_auto is not enabled")
                else:
                    key_evidence.append("CPU OOM detected; no JobFile found for automatic retry")
            elif oom_diagnosis.kind == "gpu":
                key_evidence.append("GPU OOM detected; CPU mem_auto retry is disabled")

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
            if oom_diagnosis.is_oom:
                card["oom"] = oom_diagnosis.to_dict()
            if auto_retry_info:
                card["auto_retry"] = auto_retry_info
                if auto_retry_info.get("submitted"):
                    card["recommended_next_action"] = (
                        "Monitor auto-submitted retry "
                        f"{auto_retry_info['run_id']} (mem "
                        f"{auto_retry_info['old_mem']} -> {auto_retry_info['new_mem']})."
                    )
            elif oom_diagnosis.kind == "cpu":
                card["recommended_next_action"] = (
                    "CPU OOM detected. Rerun with a larger --mem or enable "
                    "--mem-auto so jobctl can submit a conservative retry."
                )
            elif oom_diagnosis.kind == "gpu":
                card["recommended_next_action"] = (
                    "GPU OOM detected. Reduce GPU memory use or request a larger GPU; "
                    "no CPU memory retry was submitted."
                )

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

        # 9. Feed the macOS desktop-notification coalescer (best-effort).
        if self._mac_coalescer is not None:
            try:
                label = getattr(run, "title", None) or jobfile.name or run.run_id
                self._mac_coalescer.add({
                    "title": label,
                    "state": run.state.value if hasattr(run.state, "value") else run.state,
                    "match": match.value if hasattr(match, "value") else match,
                })
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("macOS notify enqueue failed for run=%s: %s", run.run_id, exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _maybe_submit_mem_auto_retry(self, run: Run, jobfile: "JobFile", diagnosis) -> dict | None:
        """Submit one CPU-OOM retry when the run's auto policy allows it."""
        policy = dict(getattr(run, "auto_policy", None) or {})
        if not policy.get("mem_auto"):
            return None

        attempt = int(getattr(run, "attempt", 1) or 1)
        max_attempts = int(policy.get("max_attempts", 3) or 3)
        if attempt >= max_attempts:
            return {
                "submitted": False,
                "reason": "max_attempts_reached",
                "attempt": attempt,
                "max_attempts": max_attempts,
            }

        existing = self.store.list_runs(parent_run_id=run.run_id)
        if existing:
            child = existing[0]
            return {
                "submitted": False,
                "reason": "retry_already_exists",
                "run_id": child.run_id,
                "attempt": getattr(child, "attempt", attempt + 1),
            }

        from jobctl.memory.mem_auto import next_mem_request

        request = dict(getattr(run, "slurm_request", None) or {})
        request.pop("job_id", None)
        old_mem = request.get("mem")
        new_mem = next_mem_request(
            old_mem,
            run.resource_summary or {},
            factor=float(policy.get("factor", 1.5) or 1.5),
            cap=policy.get("max"),
        )
        if not new_mem:
            return {
                "submitted": False,
                "reason": "no_larger_mem_available",
                "old_mem": old_mem,
                "cap": policy.get("max"),
            }

        request["mem"] = new_mem
        child_id = f"run-{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        tags = list(getattr(run, "tags", None) or [])
        if "mem-auto" not in tags:
            tags.append("mem-auto")

        child = Run(
            run_id=child_id,
            jobfile_id=run.jobfile_id,
            jobfile_version=jobfile.version,
            params=dict(run.params or {}),
            input_hashes=dict(run.input_hashes or {}),
            backend=run.backend,
            server=run.server,
            task=run.task,
            remote_job_id=None,
            state=State.PENDING,
            health=Health.OK,
            exit_code=None,
            submitted_at=now_iso,
            started_at=None,
            finished_at=None,
            last_heartbeat=None,
            workdir=None,
            stdout_path=None,
            stderr_path=None,
            resource_summary={},
            expectation_match=None,
            observation_card=None,
            slurm_request=request,
            title=getattr(run, "title", None),
            note=getattr(run, "note", None),
            tags=tags,
            parent_run_id=run.run_id,
            attempt=attempt + 1,
            auto_policy=policy,
        )
        try:
            self.store.add_run(child)
        except sqlite3.IntegrityError:
            existing = self.store.list_runs(parent_run_id=run.run_id)
            if existing:
                existing_child = existing[0]
                return {
                    "submitted": False,
                    "reason": "retry_already_exists",
                    "run_id": existing_child.run_id,
                    "attempt": getattr(existing_child, "attempt", attempt + 1),
                }
            raise

        backend = self._get_backend_for_run(run)
        if backend is None:
            self.store.update_run(child_id, state=State.FAILED, health=Health.WEAK)
            return {
                "submitted": False,
                "reason": "backend_unavailable",
                "run_id": child_id,
                "old_mem": old_mem,
                "new_mem": new_mem,
            }

        try:
            submit_result = await asyncio.to_thread(backend.submit, child, jobfile)
            update_fields = {
                "state": State.SUBMITTED,
                "remote_job_id": submit_result.remote_job_id,
                "workdir": submit_result.workdir,
            }
            if submit_result.slurm_request is not None:
                update_fields["slurm_request"] = submit_result.slurm_request
            self.store.update_run(child_id, **update_fields)
            self._backends[child_id] = backend
            callback = self._callback_urls.get(run.run_id)
            if callback:
                self._callback_urls[child_id] = callback
            return {
                "submitted": True,
                "run_id": child_id,
                "parent_run_id": run.run_id,
                "attempt": attempt + 1,
                "max_attempts": max_attempts,
                "old_mem": old_mem,
                "new_mem": new_mem,
                "oom_kind": diagnosis.kind,
            }
        except Exception as exc:
            logger.exception("mem_auto retry submit failed for parent=%s: %s", run.run_id, exc)
            finished_at = datetime.now(timezone.utc).isoformat()
            failed_child = self.store.get_run(child_id) or child
            card = build_observation_card(
                run=failed_child,
                jobfile=jobfile,
                artifacts=[],
                match=Match.FAILED,
                key_evidence=[f"mem_auto retry submit failed: {exc}"],
                health=Health.WEAK,
                analyzer=self.analyzer,
            )
            self.store.update_run(
                child_id,
                state=State.FAILED,
                health=Health.WEAK,
                exit_code=1,
                finished_at=finished_at,
                expectation_match=Match.FAILED,
                observation_card=card,
            )
            return {
                "submitted": False,
                "reason": "submit_failed",
                "run_id": child_id,
                "old_mem": old_mem,
                "new_mem": new_mem,
                "error": str(exc),
            }

    async def _check_running_gpu_oom(self, run: Run) -> bool:
        """Cancel a still-running job if local logs already show GPU OOM."""
        from jobctl.memory.mem_auto import classify_oom

        stdout, stderr = self._read_visible_logs(run)
        diagnosis = classify_oom({}, stdout, stderr)
        if diagnosis.kind != "gpu":
            return False

        backend = self._get_backend_for_run(run)
        if backend is not None:
            try:
                await asyncio.to_thread(backend.cancel, run)
            except Exception as exc:
                logger.warning("GPU OOM cancel failed for run=%s: %s", run.run_id, exc)

        self.store.update_run(
            run.run_id,
            state=State.FAILED,
            health=Health.RESOURCE_PRESSURE,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        terminal = self.store.get_run(run.run_id)
        if terminal is not None:
            await self.on_terminal(terminal)
        return True

    @staticmethod
    def _read_visible_logs(run: Run) -> tuple[str, str]:
        """Read stdout/stderr when they are already local; missing paths are ok."""
        def _read(path: str | None) -> str:
            if not path:
                return ""
            try:
                return Path(path).read_text(errors="replace")
            except OSError:
                return ""

        stdout_path = run.stdout_path
        stderr_path = run.stderr_path
        if not stdout_path and run.workdir:
            stdout_path = str(Path(run.workdir) / "stdout.txt")
        if not stderr_path and run.workdir:
            stderr_path = str(Path(run.workdir) / "stderr.txt")
        return _read(stdout_path), _read(stderr_path)

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
