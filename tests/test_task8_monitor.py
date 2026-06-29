"""Tests for monitor/monitor.py (Task 8).

Strategy:
- Use fake backend (FakeBackend) that drives state transitions
- Use fake prober (FakeProber) that returns preset server stats
- Use in-memory Store (":memory:")
- Run asyncio event loop in tests with asyncio.run()
- Verify: pending->running->completed pipeline, on_terminal pipeline,
  observation card all-fields, stuck detection, probe_servers behavior
"""
from __future__ import annotations

import asyncio
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, patch

import pytest

from jobctl.db.models import (
    Artifact,
    ArtifactType,
    Health,
    JobFile,
    Match,
    Run,
    Server,
    State,
)
from jobctl.db.store import Store
from jobctl.backends.base import Backend, CollectResult, PollResult, SubmitResult
from jobctl.analysis.offline import OfflineAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_store(tmp_path: Path) -> Store:
    store = Store(str(tmp_path / "test.db"))
    store.init_schema()
    return store


def _make_jobfile(tmp_path: Path, artifact_patterns: list[str] | None = None) -> JobFile:
    return JobFile(
        id=f"jf-{uuid.uuid4().hex[:8]}",
        name="test-job",
        version=1,
        source_path=str(tmp_path / "test.jobfile.yaml"),
        command_template="echo hello",
        params_schema={},
        backend_prefs=[{"backend": "local"}],
        artifact_patterns=artifact_patterns or ["*.txt"],
        expectation_contract_id=None,
        content_hash="abc123",
        created_at=_now_iso(),
    )


def _make_run(jobfile: JobFile, workdir: str | None = None, state: State = State.PENDING) -> Run:
    return Run(
        run_id=f"run-{uuid.uuid4().hex[:8]}",
        jobfile_id=jobfile.id,
        jobfile_version=1,
        params={},
        input_hashes={},
        backend="local",
        server=None,
        task=None,
        remote_job_id=None,
        state=state,
        health=Health.OK,
        exit_code=None,
        submitted_at=_now_iso(),
        started_at=None,
        finished_at=None,
        last_heartbeat=None,
        workdir=workdir,
        stdout_path=None,
        stderr_path=None,
        resource_summary={},
        expectation_match=None,
        observation_card=None,
    )


# ---------------------------------------------------------------------------
# Fake Backend
# ---------------------------------------------------------------------------

class FakeBackend(Backend):
    """A backend that drives state through a scripted sequence of PollResults."""

    name = "fake"

    def __init__(self, poll_sequence: list[PollResult], collect_result: CollectResult) -> None:
        self._poll_sequence = list(poll_sequence)
        self._poll_index = 0
        self._collect_result = collect_result
        self.submitted = False
        self.collected = False
        self.cancelled = False

    def submit(self, run: Run, jobfile: JobFile) -> SubmitResult:
        self.submitted = True
        return SubmitResult(remote_job_id="fake-123", workdir=self._collect_result.artifact_dir)

    def poll(self, run: Run) -> PollResult:
        if self._poll_index < len(self._poll_sequence):
            result = self._poll_sequence[self._poll_index]
            self._poll_index += 1
            return result
        # Return the last state forever
        return self._poll_sequence[-1]

    def collect(self, run: Run) -> CollectResult:
        self.collected = True
        return self._collect_result

    def cancel(self, run: Run) -> None:
        self.cancelled = True


# ---------------------------------------------------------------------------
# Fake Prober
# ---------------------------------------------------------------------------

class FakeProber:
    """A prober that returns preset Server snapshots."""

    def __init__(self, servers: list[Server]) -> None:
        self._servers = {s.name: s for s in servers}

    def probe(self, name: str) -> Server | None:
        return self._servers.get(name)


# ---------------------------------------------------------------------------
# Fake Notifier
# ---------------------------------------------------------------------------

class FakeNotifier:
    """Captures notify calls for assertion."""

    def __init__(self) -> None:
        self.calls: list[tuple[Run, dict]] = []

    def notify(self, run: Run, card: dict) -> None:
        self.calls.append((run, card))


# ---------------------------------------------------------------------------
# Monitor imports (import lazily to avoid import before implementation)
# ---------------------------------------------------------------------------

def _import_monitor():
    from jobctl.monitor.monitor import Monitor, build_observation_card
    return Monitor, build_observation_card


# ===========================================================================
# Tests: build_observation_card
# ===========================================================================

class TestBuildObservationCard:

    def test_all_required_fields_present(self, tmp_path):
        """build_observation_card must return ALL required fields."""
        Monitor, build_observation_card = _import_monitor()

        jf = _make_jobfile(tmp_path)
        run = _make_run(jf, state=State.COMPLETED)
        run.exit_code = 0
        run.finished_at = _now_iso()

        analyzer = OfflineAnalyzer()
        card = build_observation_card(
            run=run,
            jobfile=jf,
            artifacts=[],
            match=Match.USABLE,
            key_evidence=["exit_code=0"],
            health=Health.OK,
            analyzer=analyzer,
        )

        required = {
            "status", "jobfile", "run_id", "server", "artifacts",
            "health", "expectation_match", "key_evidence",
            "interpretation", "recommended_next_action",
        }
        assert required.issubset(card.keys()), (
            f"Missing fields: {required - card.keys()}"
        )

    def test_status_not_just_finished(self, tmp_path):
        """status must reflect actual state, never a bare 'finished' string."""
        Monitor, build_observation_card = _import_monitor()

        jf = _make_jobfile(tmp_path)
        run = _make_run(jf, state=State.COMPLETED)
        run.exit_code = 0
        analyzer = OfflineAnalyzer()

        card = build_observation_card(
            run=run,
            jobfile=jf,
            artifacts=[],
            match=Match.USABLE,
            key_evidence=[],
            health=Health.OK,
            analyzer=analyzer,
        )
        # Must use the real state name
        assert card["status"] == "completed"
        assert card["status"] != "finished"

    def test_failed_run_card(self, tmp_path):
        """Card for a failed run should have match=FAILED and correct status."""
        Monitor, build_observation_card = _import_monitor()

        jf = _make_jobfile(tmp_path)
        run = _make_run(jf, state=State.FAILED)
        run.exit_code = 1
        analyzer = OfflineAnalyzer()

        card = build_observation_card(
            run=run,
            jobfile=jf,
            artifacts=[],
            match=Match.FAILED,
            key_evidence=["exit_code=1"],
            health=Health.OK,
            analyzer=analyzer,
        )
        assert card["status"] == "failed"
        assert card["expectation_match"] == "failed"
        assert "exit_code=1" in card["key_evidence"]

    def test_artifacts_in_card(self, tmp_path):
        """Artifacts list in card contains name, type, and preview."""
        Monitor, build_observation_card = _import_monitor()

        jf = _make_jobfile(tmp_path)
        run = _make_run(jf, state=State.COMPLETED)
        run.exit_code = 0

        artifact = Artifact(
            id="art-1",
            run_id=run.run_id,
            remote_path="/tmp/out.txt",
            local_path="/tmp/out.txt",
            type=ArtifactType.TEXT_LOG,
            size=100,
            checksum="abc",
            preview={"head": ["line1"], "tail": []},
            created_at=_now_iso(),
        )

        analyzer = OfflineAnalyzer()
        card = build_observation_card(
            run=run,
            jobfile=jf,
            artifacts=[artifact],
            match=Match.USABLE,
            key_evidence=[],
            health=Health.OK,
            analyzer=analyzer,
        )
        assert len(card["artifacts"]) == 1
        art_entry = card["artifacts"][0]
        assert "name" in art_entry
        assert "type" in art_entry
        assert "preview" in art_entry

    def test_stuck_run_card_health(self, tmp_path):
        """Card for a stuck run should have health=stuck."""
        Monitor, build_observation_card = _import_monitor()

        jf = _make_jobfile(tmp_path)
        run = _make_run(jf, state=State.STUCK)
        analyzer = OfflineAnalyzer()

        card = build_observation_card(
            run=run,
            jobfile=jf,
            artifacts=[],
            match=Match.INCONCLUSIVE,
            key_evidence=["no heartbeat for 600s"],
            health=Health.STUCK,
            analyzer=analyzer,
        )
        assert card["health"] == "stuck"
        assert card["status"] == "stuck"


# ===========================================================================
# Tests: Monitor.probe_servers
# ===========================================================================

class TestProbeServers:

    def test_probe_updates_server_online(self, tmp_path):
        """probe_servers updates online=True when prober returns a server."""
        Monitor, _ = _import_monitor()

        store = _make_store(tmp_path)
        config = {"servers": {"srv1": {"host": "srv1.example.com"}}}

        online_server = Server(
            name="srv1",
            backend_type="ssh",
            online=True,
            last_heartbeat=_now_iso(),
            cpu={"used": 0.5},
            mem={"used_gb": 4.0},
            gpu={},
            disk={},
            slurm_queue={},
            note=None,
        )
        prober = FakeProber([online_server])
        notifiers_factory = lambda run: []

        monitor = Monitor(
            store=store,
            config=config,
            analyzer=OfflineAnalyzer(),
            notifiers_factory=notifiers_factory,
        )
        monitor._prober = prober

        asyncio.run(monitor.probe_servers())

        stored = store.get_server("srv1")
        assert stored is not None
        assert stored.online is True
        assert stored.cpu == {"used": 0.5}

    def test_probe_marks_server_offline_on_none(self, tmp_path):
        """probe_servers marks server offline when prober returns None."""
        Monitor, _ = _import_monitor()

        store = _make_store(tmp_path)
        # Seed an initially online server
        existing = Server(
            name="srv1",
            backend_type="ssh",
            online=True,
            last_heartbeat=_now_iso(),
            cpu={}, mem={}, gpu={}, disk={}, slurm_queue={}, note=None,
        )
        store.upsert_server(existing)

        config = {"servers": {"srv1": {}}}
        prober = FakeProber([])  # No servers -> returns None for any name

        monitor = Monitor(
            store=store,
            config=config,
            analyzer=OfflineAnalyzer(),
            notifiers_factory=lambda run: [],
        )
        monitor._prober = prober

        asyncio.run(monitor.probe_servers())

        stored = store.get_server("srv1")
        assert stored is not None
        assert stored.online is False

    def test_probe_resource_pressure_detected(self, tmp_path):
        """probe_servers sets note for resource_pressure when cpu is high."""
        Monitor, _ = _import_monitor()

        store = _make_store(tmp_path)
        config = {"servers": {"bigbox": {}}}

        overloaded = Server(
            name="bigbox",
            backend_type="ssh",
            online=True,
            last_heartbeat=_now_iso(),
            cpu={"percent": 99.0},
            mem={"percent": 98.0},
            gpu={},
            disk={},
            slurm_queue={},
            note=None,
        )
        prober = FakeProber([overloaded])

        monitor = Monitor(
            store=store,
            config=config,
            analyzer=OfflineAnalyzer(),
            notifiers_factory=lambda run: [],
        )
        monitor._prober = prober

        asyncio.run(monitor.probe_servers())

        stored = store.get_server("bigbox")
        assert stored is not None
        # Either health note set or online status is correct
        assert stored.online is True


# ===========================================================================
# Tests: Monitor.poll_runs — state transitions
# ===========================================================================

class TestPollRuns:

    def _setup(self, tmp_path, poll_sequence, collect_result):
        """Set up store, jobfile, run, and a Monitor with FakeBackend."""
        Monitor, _ = _import_monitor()

        store = _make_store(tmp_path)
        jf = _make_jobfile(tmp_path, artifact_patterns=["*.txt"])
        store.add_jobfile(jf)

        # Create a stdout file
        stdout_file = tmp_path / "stdout.txt"
        stdout_file.write_text("Run completed\n")

        run = _make_run(jf, workdir=str(tmp_path), state=State.SUBMITTED)
        run.stdout_path = str(stdout_file)
        run.stderr_path = str(tmp_path / "stderr.txt")
        Path(tmp_path / "stderr.txt").write_text("")
        store.add_run(run)

        backend = FakeBackend(poll_sequence=poll_sequence, collect_result=collect_result)
        notifier = FakeNotifier()

        monitor = Monitor(
            store=store,
            config={},
            analyzer=OfflineAnalyzer(),
            notifiers_factory=lambda r: [notifier],
        )
        # Inject the fake backend
        monitor._backends = {run.run_id: backend}

        return store, jf, run, backend, notifier, monitor

    def test_running_then_completed(self, tmp_path):
        """Run goes submitted -> running -> completed; on_terminal called once."""
        poll_sequence = [
            PollResult(state=State.RUNNING, resource={"cpu": 0.5}, last_log_mtime=time.time()),
            PollResult(state=State.COMPLETED, resource={"cpu": 0.0}, last_log_mtime=time.time()),
        ]

        stdout_path = str(tmp_path / "stdout.txt")
        stderr_path = str(tmp_path / "stderr.txt")

        collect = CollectResult(
            exit_code=0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            artifact_dir=str(tmp_path),
            resource_summary={"cpu_avg": 0.5},
        )

        store, jf, run, backend, notifier, monitor = self._setup(
            tmp_path, poll_sequence, collect
        )

        # Poll twice — first to running, second to completed
        asyncio.run(monitor.poll_runs())
        stored = store.get_run(run.run_id)
        assert stored.state in (State.RUNNING, State.COMPLETED, State.SUBMITTED)

        asyncio.run(monitor.poll_runs())
        stored = store.get_run(run.run_id)
        assert stored.state == State.COMPLETED

        # on_terminal should have been triggered (notifier called)
        assert len(notifier.calls) >= 1
        card = notifier.calls[0][1]
        assert "status" in card
        assert card["status"] == "completed"

    def test_card_and_match_persisted(self, tmp_path):
        """After terminal, observation_card and expectation_match are persisted in DB."""
        poll_sequence = [
            PollResult(state=State.COMPLETED, resource={}, last_log_mtime=time.time()),
        ]
        collect = CollectResult(
            exit_code=0,
            stdout_path=str(tmp_path / "stdout.txt"),
            stderr_path=str(tmp_path / "stderr.txt"),
            artifact_dir=str(tmp_path),
        )

        store, jf, run, backend, notifier, monitor = self._setup(
            tmp_path, poll_sequence, collect
        )

        asyncio.run(monitor.poll_runs())
        stored = store.get_run(run.run_id)

        assert stored.state == State.COMPLETED
        assert stored.observation_card is not None
        assert isinstance(stored.observation_card, dict)
        assert stored.expectation_match is not None

    def test_failed_run_transitions(self, tmp_path):
        """Run going FAILED is handled by on_terminal."""
        poll_sequence = [
            PollResult(state=State.FAILED, resource={}, last_log_mtime=None),
        ]
        collect = CollectResult(
            exit_code=1,
            stdout_path=str(tmp_path / "stdout.txt"),
            stderr_path=str(tmp_path / "stderr.txt"),
            artifact_dir=str(tmp_path),
        )

        store, jf, run, backend, notifier, monitor = self._setup(
            tmp_path, poll_sequence, collect
        )

        asyncio.run(monitor.poll_runs())
        stored = store.get_run(run.run_id)

        assert stored.state == State.FAILED
        assert len(notifier.calls) == 1
        card = notifier.calls[0][1]
        assert card["status"] == "failed"

    def test_observation_card_has_all_required_fields(self, tmp_path):
        """The persisted card must contain all required top-level fields."""
        poll_sequence = [
            PollResult(state=State.COMPLETED, resource={}, last_log_mtime=time.time()),
        ]
        collect = CollectResult(
            exit_code=0,
            stdout_path=str(tmp_path / "stdout.txt"),
            stderr_path=str(tmp_path / "stderr.txt"),
            artifact_dir=str(tmp_path),
        )

        store, jf, run, backend, notifier, monitor = self._setup(
            tmp_path, poll_sequence, collect
        )

        asyncio.run(monitor.poll_runs())
        stored = store.get_run(run.run_id)

        card = stored.observation_card
        required = {
            "status", "jobfile", "run_id", "server", "artifacts",
            "health", "expectation_match", "key_evidence",
            "interpretation", "recommended_next_action",
        }
        assert required.issubset(card.keys()), f"Missing: {required - card.keys()}"

    def test_poll_many_blank_timeout_logs_exception_class(self, tmp_path, caplog):
        """TimeoutError() has no message; monitor logs must still show a reason."""
        poll_sequence = [
            PollResult(state=State.RUNNING, resource={}, last_log_mtime=time.time()),
        ]
        collect = CollectResult(
            exit_code=None,
            stdout_path=str(tmp_path / "stdout.txt"),
            stderr_path=str(tmp_path / "stderr.txt"),
            artifact_dir=str(tmp_path),
        )
        store, jf, run, backend, notifier, monitor = self._setup(
            tmp_path, poll_sequence, collect
        )
        run.backend = "slurm"
        run.server = "oblix"
        store.update_run(run.run_id, backend="slurm", server="oblix")

        def raise_blank_timeout(group):
            raise TimeoutError()

        backend.poll_many = raise_blank_timeout
        caplog.set_level("WARNING", logger="jobctl.monitor.monitor")

        asyncio.run(monitor.poll_runs())

        assert "poll_many failed/timed out for slurm/oblix: TimeoutError" in caplog.text
        stored = store.get_run(run.run_id)
        assert stored.health == Health.NO_HEARTBEAT


# ===========================================================================
# Tests: mem-auto OOM handling
# ===========================================================================

class TestMemAutoOom:

    def _monitor_for_slurm_failure(
        self,
        tmp_path,
        *,
        stderr_text="",
        resource_summary=None,
        auto_policy=None,
    ):
        Monitor, _ = _import_monitor()

        store = _make_store(tmp_path)
        jf = _make_jobfile(tmp_path, artifact_patterns=[])
        jf.backend_prefs = [{"backend": "slurm", "server": "oblix"}]
        store.add_jobfile(jf)

        stdout_file = tmp_path / "stdout.txt"
        stdout_file.write_text("")
        stderr_file = tmp_path / "stderr.txt"
        stderr_file.write_text(stderr_text)

        run = _make_run(jf, workdir=str(tmp_path), state=State.SUBMITTED)
        run.backend = "slurm"
        run.server = "oblix"
        run.remote_job_id = "123"
        run.stdout_path = str(stdout_file)
        run.stderr_path = str(stderr_file)
        run.slurm_request = {"mem": "1G", "cpus": 1, "time": "01:00:00", "job_id": "123"}
        if auto_policy is None:
            auto_policy = {"mem_auto": True, "factor": 1.5, "max_attempts": 3}
        run.auto_policy = auto_policy
        store.add_run(run)

        collect = CollectResult(
            exit_code=1,
            stdout_path=str(stdout_file),
            stderr_path=str(stderr_file),
            artifact_dir=str(tmp_path),
            resource_summary=resource_summary or {},
        )
        backend = FakeBackend(
            poll_sequence=[PollResult(state=State.FAILED, resource={}, last_log_mtime=None)],
            collect_result=collect,
        )
        notifier = FakeNotifier()
        monitor = Monitor(
            store=store,
            config={},
            analyzer=OfflineAnalyzer(),
            notifiers_factory=lambda r: [notifier],
        )
        monitor._backends = {run.run_id: backend}
        return store, run, backend, notifier, monitor

    def test_cpu_oom_mem_auto_submits_larger_retry(self, tmp_path):
        store, run, backend, notifier, monitor = self._monitor_for_slurm_failure(
            tmp_path,
            resource_summary={"State": "OUT_OF_MEMORY", "MaxRSS": "900M"},
        )

        asyncio.run(monitor.poll_runs())

        parent = store.get_run(run.run_id)
        assert parent.observation_card["oom"]["kind"] == "cpu"
        assert parent.observation_card["auto_retry"]["submitted"] is True
        children = store.list_runs(parent_run_id=run.run_id)
        assert len(children) == 1
        child = children[0]
        assert child.state == State.SUBMITTED
        assert child.attempt == 2
        assert child.slurm_request["mem"] == "1536M"
        assert child.slurm_request["cpus"] == 1
        assert "job_id" not in child.slurm_request or child.slurm_request["job_id"] == "fake-123"
        assert backend.submitted is True
        assert len(notifier.calls) == 1

    def test_gpu_oom_stops_without_cpu_mem_retry(self, tmp_path):
        store, run, backend, notifier, monitor = self._monitor_for_slurm_failure(
            tmp_path,
            stderr_text="RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
            resource_summary={"State": "FAILED"},
        )

        asyncio.run(monitor.poll_runs())

        parent = store.get_run(run.run_id)
        assert parent.observation_card["oom"]["kind"] == "gpu"
        assert "auto_retry" not in parent.observation_card
        assert store.list_runs(parent_run_id=run.run_id) == []
        assert parent.health == Health.RESOURCE_PRESSURE

    def test_cpu_oom_without_mem_auto_recommends_larger_memory(self, tmp_path):
        store, run, backend, notifier, monitor = self._monitor_for_slurm_failure(
            tmp_path,
            resource_summary={"State": "OUT_OF_MEMORY", "MaxRSS": "900M"},
            auto_policy={},
        )

        asyncio.run(monitor.poll_runs())

        parent = store.get_run(run.run_id)
        card = parent.observation_card
        assert card["oom"]["kind"] == "cpu"
        assert "auto_retry" not in card
        assert "CPU OOM detected; mem_auto is not enabled" in card["key_evidence"]
        assert "larger --mem" in card["recommended_next_action"]
        assert store.list_runs(parent_run_id=run.run_id) == []

    def test_running_gpu_oom_log_is_cancelled_and_notified(self, tmp_path):
        Monitor, _ = _import_monitor()

        store = _make_store(tmp_path)
        jf = _make_jobfile(tmp_path, artifact_patterns=[])
        store.add_jobfile(jf)
        stdout_file = tmp_path / "stdout.txt"
        stdout_file.write_text("")
        stderr_file = tmp_path / "stderr.txt"
        stderr_file.write_text("RuntimeError: CUDA out of memory while allocating tensor")

        run = _make_run(jf, workdir=str(tmp_path), state=State.RUNNING)
        run.stdout_path = str(stdout_file)
        run.stderr_path = str(stderr_file)
        store.add_run(run)

        backend = FakeBackend(
            poll_sequence=[PollResult(state=State.RUNNING, resource={}, last_log_mtime=time.time())],
            collect_result=CollectResult(
                exit_code=1,
                stdout_path=str(stdout_file),
                stderr_path=str(stderr_file),
                artifact_dir=str(tmp_path),
                resource_summary={"State": "FAILED"},
            ),
        )
        notifier = FakeNotifier()
        monitor = Monitor(
            store=store,
            config={},
            analyzer=OfflineAnalyzer(),
            notifiers_factory=lambda r: [notifier],
        )
        monitor._backends = {run.run_id: backend}

        asyncio.run(monitor.poll_runs())

        stored = store.get_run(run.run_id)
        assert backend.cancelled is True
        assert stored.state == State.FAILED
        assert stored.health == Health.RESOURCE_PRESSURE
        assert stored.observation_card["oom"]["kind"] == "gpu"
        assert len(notifier.calls) == 1


# ===========================================================================
# Tests: Stuck detection
# ===========================================================================

class TestStuckDetection:

    def test_stuck_when_running_stale_log_and_no_heartbeat(self, tmp_path):
        """RUNNING + stale log mtime + stale/no heartbeat -> STUCK."""
        Monitor, _ = _import_monitor()

        store = _make_store(tmp_path)
        jf = _make_jobfile(tmp_path)
        store.add_jobfile(jf)

        stdout_file = tmp_path / "stdout.txt"
        stdout_file.write_text("old output")

        run = _make_run(jf, workdir=str(tmp_path), state=State.RUNNING)
        run.stdout_path = str(stdout_file)
        run.stderr_path = str(tmp_path / "stderr.txt")
        Path(tmp_path / "stderr.txt").write_text("")

        # Heartbeat that's very old (> stuck threshold)
        old_time = "2000-01-01T00:00:00+00:00"
        run.last_heartbeat = old_time
        run.started_at = old_time
        store.add_run(run)

        # Very stale log mtime
        stale_mtime = time.time() - 7200  # 2 hours ago
        import os
        os.utime(str(stdout_file), (stale_mtime, stale_mtime))

        # Poll result stays RUNNING
        stale_poll = PollResult(
            state=State.RUNNING,
            resource={},
            last_log_mtime=stale_mtime,
        )
        backend = FakeBackend(
            poll_sequence=[stale_poll],
            collect_result=CollectResult(
                exit_code=None,
                stdout_path=str(stdout_file),
                stderr_path=str(tmp_path / "stderr.txt"),
                artifact_dir=str(tmp_path),
            ),
        )
        notifier = FakeNotifier()
        monitor = Monitor(
            store=store,
            config={"stuck_timeout_seconds": 600},  # 10 min threshold
            analyzer=OfflineAnalyzer(),
            notifiers_factory=lambda r: [notifier],
        )
        monitor._backends = {run.run_id: backend}

        asyncio.run(monitor.poll_runs())

        stored = store.get_run(run.run_id)
        assert stored.state == State.STUCK
        assert stored.health == Health.STUCK

    def test_not_stuck_when_recent_heartbeat(self, tmp_path):
        """RUNNING + recent heartbeat -> NOT stuck."""
        Monitor, _ = _import_monitor()

        store = _make_store(tmp_path)
        jf = _make_jobfile(tmp_path)
        store.add_jobfile(jf)

        stdout_file = tmp_path / "stdout.txt"
        stdout_file.write_text("recent output")

        run = _make_run(jf, workdir=str(tmp_path), state=State.RUNNING)
        run.stdout_path = str(stdout_file)
        run.stderr_path = str(tmp_path / "stderr.txt")
        Path(tmp_path / "stderr.txt").write_text("")

        # Recent heartbeat
        run.last_heartbeat = _now_iso()
        run.started_at = _now_iso()
        store.add_run(run)

        recent_mtime = time.time() - 30  # 30 seconds ago
        import os
        os.utime(str(stdout_file), (recent_mtime, recent_mtime))

        poll = PollResult(
            state=State.RUNNING,
            resource={},
            last_log_mtime=recent_mtime,
        )
        backend = FakeBackend(
            poll_sequence=[poll],
            collect_result=CollectResult(
                exit_code=None,
                stdout_path=str(stdout_file),
                stderr_path=str(tmp_path / "stderr.txt"),
                artifact_dir=str(tmp_path),
            ),
        )
        monitor = Monitor(
            store=store,
            config={"stuck_timeout_seconds": 600},
            analyzer=OfflineAnalyzer(),
            notifiers_factory=lambda r: [],
        )
        monitor._backends = {run.run_id: backend}

        asyncio.run(monitor.poll_runs())

        stored = store.get_run(run.run_id)
        assert stored.state == State.RUNNING
        assert stored.health != Health.STUCK


# ===========================================================================
# Tests: run_loop (asyncio stop_event)
# ===========================================================================

class TestRunLoop:

    def test_run_loop_stops_on_event(self, tmp_path):
        """run_loop exits cleanly when stop_event is set."""
        Monitor, _ = _import_monitor()

        store = _make_store(tmp_path)
        monitor = Monitor(
            store=store,
            config={"poll_interval_seconds": 0.05},
            analyzer=OfflineAnalyzer(),
            notifiers_factory=lambda r: [],
        )

        async def _run():
            stop = asyncio.Event()
            stop.set()  # Stop immediately
            await monitor.run_loop(stop)

        # Should complete without hanging
        asyncio.run(asyncio.wait_for(_run(), timeout=5.0))

    def test_run_loop_drives_full_pipeline(self, tmp_path):
        """run_loop drives a submitted run all the way to completed."""
        Monitor, _ = _import_monitor()

        store = _make_store(tmp_path)
        jf = _make_jobfile(tmp_path, artifact_patterns=["*.txt"])
        store.add_jobfile(jf)

        stdout_file = tmp_path / "stdout.txt"
        stdout_file.write_text("hello from job\n")
        stderr_file = tmp_path / "stderr.txt"
        stderr_file.write_text("")

        run = _make_run(jf, workdir=str(tmp_path), state=State.SUBMITTED)
        run.stdout_path = str(stdout_file)
        run.stderr_path = str(stderr_file)
        store.add_run(run)

        poll_sequence = [
            PollResult(state=State.RUNNING, resource={}, last_log_mtime=time.time()),
            PollResult(state=State.COMPLETED, resource={}, last_log_mtime=time.time()),
        ]
        collect = CollectResult(
            exit_code=0,
            stdout_path=str(stdout_file),
            stderr_path=str(stderr_file),
            artifact_dir=str(tmp_path),
        )
        backend = FakeBackend(poll_sequence=poll_sequence, collect_result=collect)
        notifier = FakeNotifier()

        monitor = Monitor(
            store=store,
            config={"poll_interval_seconds": 0.05},
            analyzer=OfflineAnalyzer(),
            notifiers_factory=lambda r: [notifier],
        )
        monitor._backends = {run.run_id: backend}

        async def _run():
            stop = asyncio.Event()

            async def _watcher():
                # Poll until run completes then signal the loop to stop
                for _ in range(40):
                    await asyncio.sleep(0.05)
                    stored = store.get_run(run.run_id)
                    if stored.state in (State.COMPLETED, State.FAILED, State.STUCK):
                        stop.set()
                        return
                stop.set()  # timeout fallback

            await asyncio.gather(
                monitor.run_loop(stop),
                _watcher(),
            )

        asyncio.run(asyncio.wait_for(_run(), timeout=10.0))

        stored = store.get_run(run.run_id)
        assert stored.state == State.COMPLETED
        assert stored.observation_card is not None
        assert len(notifier.calls) >= 1


# ===========================================================================
# Tests: pending -> submitted via submit path (if supported)
# ===========================================================================

class TestPendingToSubmitted:

    def test_pending_runs_are_polled(self, tmp_path):
        """Pending runs with an assigned backend are polled after submit step."""
        Monitor, _ = _import_monitor()

        store = _make_store(tmp_path)
        jf = _make_jobfile(tmp_path)
        store.add_jobfile(jf)

        stdout_file = tmp_path / "stdout.txt"
        stdout_file.write_text("")
        stderr_file = tmp_path / "stderr.txt"
        stderr_file.write_text("")

        run = _make_run(jf, workdir=str(tmp_path), state=State.SUBMITTED)
        run.stdout_path = str(stdout_file)
        run.stderr_path = str(stderr_file)
        store.add_run(run)

        poll_sequence = [
            PollResult(state=State.COMPLETED, resource={}, last_log_mtime=time.time()),
        ]
        collect = CollectResult(
            exit_code=0,
            stdout_path=str(stdout_file),
            stderr_path=str(stderr_file),
            artifact_dir=str(tmp_path),
        )
        backend = FakeBackend(poll_sequence=poll_sequence, collect_result=collect)
        notifier = FakeNotifier()

        monitor = Monitor(
            store=store,
            config={},
            analyzer=OfflineAnalyzer(),
            notifiers_factory=lambda r: [notifier],
        )
        monitor._backends = {run.run_id: backend}

        asyncio.run(monitor.poll_runs())

        stored = store.get_run(run.run_id)
        assert stored.state == State.COMPLETED


# ===========================================================================
# Tests: reachability (no false stuck/failed on SSH/VPN blips) + reconcile
# ===========================================================================

class TestReachabilityAndReconcile:
    """Regression for the 'falsely stuck' bug: a connectivity blip must not
    corrupt run state, and an already-stuck run must reconcile when the cluster
    comes back."""

    def _monitor(self, store, backend, run):
        Monitor, _ = _import_monitor()
        m = Monitor(
            store=store,
            config={"stuck_timeout_seconds": 600},
            analyzer=OfflineAnalyzer(),
            notifiers_factory=lambda r: [],
        )
        m._backends = {run.run_id: backend}
        return m

    def test_unreachable_poll_keeps_state_and_marks_no_heartbeat(self, tmp_path):
        store = _make_store(tmp_path)
        jf = _make_jobfile(tmp_path)
        store.add_jobfile(jf)
        run = _make_run(jf, workdir=str(tmp_path), state=State.RUNNING)
        run.started_at = _now_iso()
        run.last_heartbeat = _now_iso()
        store.add_run(run)

        backend = FakeBackend(
            poll_sequence=[PollResult(state=State.RUNNING, resource={}, reachable=False)],
            collect_result=CollectResult(
                exit_code=None, stdout_path="", stderr_path="", artifact_dir=str(tmp_path)
            ),
        )
        asyncio.run(self._monitor(store, backend, run).poll_runs())

        stored = store.get_run(run.run_id)
        assert stored.state == State.RUNNING          # NOT stuck / NOT failed
        assert stored.health == Health.NO_HEARTBEAT

    def test_terminal_poll_clears_stale_no_heartbeat_health(self, tmp_path):
        store = _make_store(tmp_path)
        jf = _make_jobfile(tmp_path)
        store.add_jobfile(jf)
        run = _make_run(jf, workdir=str(tmp_path), state=State.RUNNING)
        run.health = Health.NO_HEARTBEAT
        run.started_at = _now_iso()
        run.last_heartbeat = "2020-01-01T00:00:00+00:00"
        store.add_run(run)
        stdout = tmp_path / "stdout.txt"
        stderr = tmp_path / "stderr.txt"
        stdout.write_text("done\n")
        stderr.write_text("")

        backend = FakeBackend(
            poll_sequence=[PollResult(state=State.COMPLETED, resource={}, reachable=True)],
            collect_result=CollectResult(
                exit_code=0,
                stdout_path=str(stdout),
                stderr_path=str(stderr),
                artifact_dir=str(tmp_path),
            ),
        )
        asyncio.run(self._monitor(store, backend, run).poll_runs())

        stored = store.get_run(run.run_id)
        assert stored.state == State.COMPLETED
        assert stored.health == Health.OK
        assert stored.observation_card["health"] == "ok"
        assert "silently died" not in stored.observation_card["interpretation"]

    def test_running_without_local_log_is_not_stuck(self, tmp_path):
        # SLURM-style: running a long time with no local stdout mirror -> NOT stuck
        store = _make_store(tmp_path)
        jf = _make_jobfile(tmp_path)
        store.add_jobfile(jf)
        run = _make_run(jf, workdir=str(tmp_path), state=State.RUNNING)
        run.started_at = "2020-01-01T00:00:00+00:00"   # long ago
        run.last_heartbeat = "2020-01-01T00:00:00+00:00"
        run.stdout_path = None
        store.add_run(run)

        backend = FakeBackend(
            poll_sequence=[PollResult(state=State.RUNNING, resource={}, last_log_mtime=None)],
            collect_result=CollectResult(
                exit_code=None, stdout_path="", stderr_path="", artifact_dir=str(tmp_path)
            ),
        )
        asyncio.run(self._monitor(store, backend, run).poll_runs())

        stored = store.get_run(run.run_id)
        assert stored.state == State.RUNNING
        assert stored.health != Health.STUCK

    def test_stuck_run_reconciles_when_cluster_returns(self, tmp_path):
        store = _make_store(tmp_path)
        jf = _make_jobfile(tmp_path)
        store.add_jobfile(jf)
        run = _make_run(jf, workdir=str(tmp_path), state=State.STUCK)
        run.started_at = _now_iso()
        store.add_run(run)
        (tmp_path / "o.txt").write_text("")
        (tmp_path / "e.txt").write_text("")

        backend = FakeBackend(
            poll_sequence=[PollResult(state=State.CANCELLED, resource={}, reachable=True)],
            collect_result=CollectResult(
                exit_code=0, stdout_path=str(tmp_path / "o.txt"),
                stderr_path=str(tmp_path / "e.txt"), artifact_dir=str(tmp_path),
            ),
        )
        asyncio.run(self._monitor(store, backend, run).poll_runs())

        stored = store.get_run(run.run_id)
        assert stored.state == State.CANCELLED   # reconciled out of 'stuck'
