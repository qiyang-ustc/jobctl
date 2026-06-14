"""Tests for Task 1: config, db/models, db/store, jobfile."""
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import pytest
import yaml


# ---------------------------------------------------------------------------
# config.py tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_load_cluster_yaml(self, tmp_path):
        """load_config reads servers/tasks/remote_path from cluster.yaml."""
        from jobctl.config import load_config

        cluster_yaml = tmp_path / "cluster.yaml"
        cluster_yaml.write_text(yaml.dump({
            "servers": {
                "hipster": {
                    "backend": "slurm",
                    "host": "hipster.example.com",
                    "note": "VPN required",
                },
                "oblix": {
                    "backend": "ssh",
                    "host": "oblix.example.com",
                }
            },
            "tasks": {
                "gpu1_l4": {"partition": "gpu", "account": "linuxusers"},
                "cpu": {"ntasks": 4},
            },
            "remote_path": "/scratch/user/jobctl",
        }))

        cfg = load_config(cluster_yaml_path=str(cluster_yaml))
        assert "hipster" in cfg.servers
        assert cfg.servers["hipster"]["backend"] == "slurm"
        assert cfg.servers["oblix"]["backend"] == "ssh"
        assert "gpu1_l4" in cfg.tasks
        assert cfg.remote_path == "/scratch/user/jobctl"

    def test_defaults_when_cluster_yaml_absent(self, tmp_path):
        """Returns sensible defaults when cluster.yaml doesn't exist."""
        from jobctl.config import load_config

        cfg = load_config(cluster_yaml_path=str(tmp_path / "nonexistent.yaml"))
        assert cfg.servers == {}
        assert cfg.tasks == {}
        assert cfg.remote_path is not None  # has a default

    def test_jobctl_settings_defaults(self, tmp_path):
        """jobctl settings have sane defaults (db_path, run_dir, port, etc.)."""
        from jobctl.config import load_config

        cfg = load_config(cluster_yaml_path=str(tmp_path / "nonexistent.yaml"))
        assert cfg.db_path is not None
        assert cfg.run_dir is not None
        assert isinstance(cfg.daemon_port, int)
        assert cfg.daemon_port > 0

    def test_jobctl_home_sets_state_paths(self, tmp_path, monkeypatch):
        """JOBCTL_HOME is the single state root for DB/config/run defaults."""
        from jobctl.config import load_config

        root = tmp_path / "jobctl-home"
        monkeypatch.setenv("JOBCTL_HOME", str(root))
        cfg = load_config(
            cluster_yaml_path=str(tmp_path / "nonexistent.yaml"),
            jobctl_config_path=str(root / "missing.toml"),
        )
        assert cfg.state_root == str(root)
        assert cfg.db_path == str(root / "jobctl.db")
        assert cfg.run_dir == str(root / "runs")
        assert cfg.jobctl_config_path == str(root / "missing.toml")
        assert cfg.cluster_yaml_path == str(tmp_path / "nonexistent.yaml")

    def test_jobctl_settings_from_toml(self, tmp_path):
        """load_config reads optional jobctl config toml."""
        import tomllib as tomllib_module
        try:
            import tomllib
        except ImportError:
            pytest.skip("tomllib not available")

        from jobctl.config import load_config

        config_dir = tmp_path / ".jobctl"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text(
            '[jobctl]\ndb_path = "/custom/db.sqlite"\ndaemon_port = 9999\n'
            '[jobctl.default_policies.oblix]\n'
            'mode = "cpu_fill_idle"\n'
            'target_idle_pct = 5\n'
            'kernel_cpus = 2\n'
        )

        cfg = load_config(
            cluster_yaml_path=str(tmp_path / "nonexistent.yaml"),
            jobctl_config_path=str(config_file),
        )
        assert cfg.db_path == "/custom/db.sqlite"
        assert cfg.daemon_port == 9999
        assert cfg.default_policies["oblix"]["target_idle_pct"] == 5


# ---------------------------------------------------------------------------
# db/models.py tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_state_enum_values(self):
        from jobctl.db.models import State
        assert State.PENDING == "pending"
        assert State.SUBMITTED == "submitted"
        assert State.RUNNING == "running"
        assert State.COMPLETED == "completed"
        assert State.FAILED == "failed"
        assert State.CANCELLED == "cancelled"
        assert State.STUCK == "stuck"
        assert State.TIMEOUT == "timeout"

    def test_health_enum_values(self):
        from jobctl.db.models import Health
        assert Health.OK == "ok"
        assert Health.WEAK == "weak"
        assert Health.NO_HEARTBEAT == "no_heartbeat"
        assert Health.RESOURCE_PRESSURE == "resource_pressure"
        assert Health.STUCK == "stuck"

    def test_match_enum_values(self):
        from jobctl.db.models import Match
        assert Match.USABLE == "usable"
        assert Match.WEAK_SIGNAL == "weak_signal"
        assert Match.BAD_SIGNAL == "bad_signal"
        assert Match.INCONCLUSIVE == "inconclusive"
        assert Match.FAILED == "failed"

    def test_artifact_type_enum_values(self):
        from jobctl.db.models import ArtifactType
        assert ArtifactType.IMAGE == "image"
        assert ArtifactType.PLOT == "plot"
        assert ArtifactType.CSV == "csv"
        assert ArtifactType.JSON == "json"
        assert ArtifactType.TEXT_LOG == "text_log"
        assert ArtifactType.BINARY == "binary"
        assert ArtifactType.OTHER == "other"

    def test_jobfile_dataclass_fields(self):
        from jobctl.db.models import JobFile
        import dataclasses
        fields = {f.name for f in dataclasses.fields(JobFile)}
        required = {
            "id", "name", "version", "source_path", "command_template",
            "params_schema", "backend_prefs", "artifact_patterns",
            "expectation_contract_id", "content_hash", "created_at",
        }
        assert required.issubset(fields)

    def test_run_dataclass_fields(self):
        from jobctl.db.models import Run
        import dataclasses
        fields = {f.name for f in dataclasses.fields(Run)}
        required = {
            "run_id", "jobfile_id", "jobfile_version", "params", "input_hashes",
            "backend", "server", "task", "remote_job_id", "state", "health",
            "exit_code", "submitted_at", "started_at", "finished_at",
            "last_heartbeat", "workdir", "stdout_path", "stderr_path",
            "resource_summary", "expectation_match", "observation_card",
            "slurm_request", "parent_run_id", "attempt", "auto_policy",
        }
        assert required.issubset(fields)

    def test_artifact_dataclass_fields(self):
        from jobctl.db.models import Artifact
        import dataclasses
        fields = {f.name for f in dataclasses.fields(Artifact)}
        required = {"id", "run_id", "remote_path", "local_path", "type", "size", "checksum", "preview", "created_at"}
        assert required.issubset(fields)

    def test_criterion_dataclass_fields(self):
        from jobctl.db.models import Criterion
        import dataclasses
        fields = {f.name for f in dataclasses.fields(Criterion)}
        required = {"id", "text", "kind", "check", "status", "strength", "evidence_run_ids"}
        assert required.issubset(fields)

    def test_expectation_contract_dataclass_fields(self):
        from jobctl.db.models import ExpectationContract
        import dataclasses
        fields = {f.name for f in dataclasses.fields(ExpectationContract)}
        required = {"id", "jobfile_id", "version", "criteria", "source", "created_at", "updated_at"}
        assert required.issubset(fields)

    def test_feedback_dataclass_fields(self):
        from jobctl.db.models import Feedback
        import dataclasses
        fields = {f.name for f in dataclasses.fields(Feedback)}
        required = {"id", "run_id", "kind", "text", "created_at"}
        assert required.issubset(fields)

    def test_server_dataclass_fields(self):
        from jobctl.db.models import Server
        import dataclasses
        fields = {f.name for f in dataclasses.fields(Server)}
        required = {"name", "backend_type", "online", "last_heartbeat", "cpu", "mem", "gpu", "disk", "slurm_queue", "note"}
        assert required.issubset(fields)


# ---------------------------------------------------------------------------
# db/store.py tests
# ---------------------------------------------------------------------------

def make_jobfile(suffix=""):
    from jobctl.db.models import JobFile
    from datetime import datetime, timezone
    return JobFile(
        id=f"jf-001{suffix}",
        name=f"test-job{suffix}",
        version=1,
        source_path="/tmp/test.jobfile.yaml",
        command_template="python {script} --lr {lr}",
        params_schema={"script": {"type": "path", "required": True}, "lr": {"type": "float", "default": 0.01}},
        backend_prefs=[{"backend": "local"}],
        artifact_patterns=["*.csv", "*.png"],
        expectation_contract_id=None,
        content_hash="abc123",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def make_run(jobfile_id="jf-001", run_id="run-001"):
    from jobctl.db.models import Run, State, Health
    from datetime import datetime, timezone
    return Run(
        run_id=run_id,
        jobfile_id=jobfile_id,
        jobfile_version=1,
        params={"script": "/tmp/train.py", "lr": 0.01},
        input_hashes={"/tmp/train.py": "sha256:deadbeef"},
        backend="local",
        server=None,
        task=None,
        remote_job_id=None,
        state=State.PENDING,
        health=Health.OK,
        exit_code=None,
        submitted_at=datetime.now(timezone.utc).isoformat(),
        started_at=None,
        finished_at=None,
        last_heartbeat=None,
        workdir="/tmp/runs/run-001",
        stdout_path=None,
        stderr_path=None,
        resource_summary={},
        expectation_match=None,
        observation_card=None,
    )


class TestStore:
    def test_init_schema(self, tmp_path):
        """init_schema creates all tables."""
        from jobctl.db.store import Store
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        assert {"jobfiles", "runs", "artifacts", "expectation_contracts", "feedback", "servers"}.issubset(tables)

    def test_add_and_get_jobfile(self, tmp_path):
        """add_jobfile + get_jobfile round-trip."""
        from jobctl.db.store import Store
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()

        jf = make_jobfile()
        store.add_jobfile(jf)
        fetched = store.get_jobfile(jf.id)
        assert fetched.id == jf.id
        assert fetched.name == jf.name
        assert fetched.params_schema == jf.params_schema
        assert fetched.backend_prefs == jf.backend_prefs
        assert fetched.artifact_patterns == jf.artifact_patterns

    def test_get_jobfile_by_name(self, tmp_path):
        from jobctl.db.store import Store
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()

        jf = make_jobfile()
        store.add_jobfile(jf)
        fetched = store.get_jobfile_by_name(jf.name)
        assert fetched.id == jf.id

    def test_list_jobfiles(self, tmp_path):
        from jobctl.db.store import Store
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()

        jf1 = make_jobfile("")
        jf2 = make_jobfile("-2")
        jf2.id = "jf-002"
        jf2.name = "test-job-2"
        store.add_jobfile(jf1)
        store.add_jobfile(jf2)
        all_jf = store.list_jobfiles()
        ids = {jf.id for jf in all_jf}
        assert "jf-001" in ids
        assert "jf-002" in ids

    def test_bump_version(self, tmp_path):
        from jobctl.db.store import Store
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()

        jf = make_jobfile()
        store.add_jobfile(jf)
        new_hash = "newcontent789"
        store.bump_version(jf.id, new_hash)
        updated = store.get_jobfile(jf.id)
        assert updated.version == 2
        assert updated.content_hash == new_hash

    def test_add_and_get_run(self, tmp_path):
        from jobctl.db.store import Store
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()

        jf = make_jobfile()
        store.add_jobfile(jf)
        run = make_run()
        store.add_run(run)
        fetched = store.get_run(run.run_id)
        assert fetched.run_id == run.run_id
        assert fetched.params == run.params
        assert fetched.input_hashes == run.input_hashes
        assert fetched.resource_summary == run.resource_summary

    def test_slurm_request_round_trips(self, tmp_path):
        """slurm_request persists via add_run and via update_run."""
        from jobctl.db.store import Store
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()
        store.add_jobfile(make_jobfile())
        run = make_run()
        store.add_run(run)
        assert store.get_run(run.run_id).slurm_request is None  # default

        store.update_run(run.run_id, slurm_request={"partition": "lln", "mem": "100M", "cpus": 1})
        got = store.get_run(run.run_id).slurm_request
        assert got == {"partition": "lln", "mem": "100M", "cpus": 1}

    def test_auto_policy_and_lineage_round_trips(self, tmp_path):
        """mem-auto lineage/policy fields persist via add_run and update_run."""
        from jobctl.db.store import Store
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()
        store.add_jobfile(make_jobfile())
        parent = make_run(run_id="run-parent")
        store.add_run(parent)

        child = make_run(run_id="run-child")
        child.parent_run_id = parent.run_id
        child.attempt = 2
        child.auto_policy = {"mem_auto": True, "factor": 1.5, "max_attempts": 3}
        store.add_run(child)

        got = store.get_run(child.run_id)
        assert got.parent_run_id == parent.run_id
        assert got.attempt == 2
        assert got.auto_policy == {"mem_auto": True, "factor": 1.5, "max_attempts": 3}
        assert [r.run_id for r in store.list_runs(parent_run_id=parent.run_id)] == ["run-child"]

        store.update_run(child.run_id, auto_policy={"mem_auto": True, "factor": 2.0})
        assert store.get_run(child.run_id).auto_policy == {"mem_auto": True, "factor": 2.0}

    def test_retry_lineage_allows_only_one_child_per_parent(self, tmp_path):
        """DB constraint prevents duplicate auto-retry children for one parent."""
        from jobctl.db.store import Store
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()
        store.add_jobfile(make_jobfile())
        parent = make_run(run_id="run-parent")
        store.add_run(parent)

        child1 = make_run(run_id="run-child-1")
        child1.parent_run_id = parent.run_id
        store.add_run(child1)

        child2 = make_run(run_id="run-child-2")
        child2.parent_run_id = parent.run_id
        with pytest.raises(sqlite3.IntegrityError):
            store.add_run(child2)

    def test_migration_adds_slurm_request_column(self, tmp_path):
        """A runs table created without slurm_request gets the column on init_schema."""
        db = str(tmp_path / "old.db")
        conn = sqlite3.connect(db)
        # Minimal pre-migration runs table (no slurm_request column)
        conn.execute(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, jobfile_id TEXT, "
            "jobfile_version INTEGER, params TEXT, input_hashes TEXT, backend TEXT, "
            "server TEXT, task TEXT, remote_job_id TEXT, state TEXT, health TEXT, "
            "exit_code INTEGER, submitted_at TEXT, started_at TEXT, finished_at TEXT, "
            "last_heartbeat TEXT, workdir TEXT, stdout_path TEXT, stderr_path TEXT, "
            "resource_summary TEXT, expectation_match TEXT, observation_card TEXT)"
        )
        conn.commit()
        conn.close()

        from jobctl.db.store import Store
        store = Store(db)
        store.init_schema()  # must ALTER TABLE to add the new column
        cols = {r[1] for r in store._get_conn().execute("PRAGMA table_info(runs)")}
        assert "slurm_request" in cols
        assert {"parent_run_id", "attempt", "auto_policy"} <= cols

    def test_update_run_state_health_card(self, tmp_path):
        """update_run mutates state/health/observation_card."""
        from jobctl.db.store import Store
        from jobctl.db.models import State, Health, Match
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()

        jf = make_jobfile()
        store.add_jobfile(jf)
        run = make_run()
        store.add_run(run)

        card = {"status": "completed", "interpretation": "all good"}
        store.update_run(run.run_id, state=State.COMPLETED, health=Health.OK,
                         exit_code=0, expectation_match=Match.USABLE, observation_card=card)

        updated = store.get_run(run.run_id)
        assert updated.state == State.COMPLETED
        assert updated.health == Health.OK
        assert updated.exit_code == 0
        assert updated.expectation_match == Match.USABLE
        assert updated.observation_card == card

    def test_list_runs_no_filter(self, tmp_path):
        from jobctl.db.store import Store
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()

        jf = make_jobfile()
        store.add_jobfile(jf)
        r1 = make_run(run_id="run-001")
        r2 = make_run(run_id="run-002")
        store.add_run(r1)
        store.add_run(r2)
        runs = store.list_runs()
        assert {r.run_id for r in runs} == {"run-001", "run-002"}

    def test_list_runs_filter_by_state(self, tmp_path):
        from jobctl.db.store import Store
        from jobctl.db.models import State
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()

        jf = make_jobfile()
        store.add_jobfile(jf)
        r1 = make_run(run_id="run-001")
        r2 = make_run(run_id="run-002")
        store.add_run(r1)
        store.add_run(r2)
        store.update_run("run-002", state=State.COMPLETED)

        pending = store.list_runs(state=State.PENDING)
        assert len(pending) == 1
        assert pending[0].run_id == "run-001"

        completed = store.list_runs(state=State.COMPLETED)
        assert len(completed) == 1
        assert completed[0].run_id == "run-002"

    def test_list_runs_filter_by_jobfile_id(self, tmp_path):
        from jobctl.db.store import Store
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()

        jf1 = make_jobfile("")
        jf2 = make_jobfile("-2")
        jf2.id = "jf-002"
        jf2.name = "other-job"
        store.add_jobfile(jf1)
        store.add_jobfile(jf2)

        r1 = make_run(jobfile_id="jf-001", run_id="run-001")
        r2 = make_run(jobfile_id="jf-002", run_id="run-002")
        store.add_run(r1)
        store.add_run(r2)

        results = store.list_runs(jobfile_id="jf-001")
        assert len(results) == 1
        assert results[0].run_id == "run-001"

    def test_add_and_list_artifacts(self, tmp_path):
        from jobctl.db.store import Store
        from jobctl.db.models import Artifact, ArtifactType
        from datetime import datetime, timezone
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()

        jf = make_jobfile()
        store.add_jobfile(jf)
        run = make_run()
        store.add_run(run)

        art = Artifact(
            id="art-001",
            run_id="run-001",
            remote_path="/remote/output.csv",
            local_path="/local/output.csv",
            type=ArtifactType.CSV,
            size=1024,
            checksum="sha256:abc",
            preview={"head": "col1,col2\n1,2", "shape": [10, 2]},
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        store.add_artifact(art)
        arts = store.list_artifacts("run-001")
        assert len(arts) == 1
        assert arts[0].id == "art-001"
        assert arts[0].preview == art.preview

    def test_upsert_and_list_servers(self, tmp_path):
        from jobctl.db.store import Store
        from jobctl.db.models import Server
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()

        srv = Server(
            name="hipster",
            backend_type="slurm",
            online=True,
            last_heartbeat="2026-06-04T10:00:00+00:00",
            cpu={"load": 1.2},
            mem={"used_gb": 8.0},
            gpu={"free": 1},
            disk={"used_pct": 60},
            slurm_queue={"running": 5, "pending": 2},
            note="GPU cluster",
        )
        store.upsert_server(srv)
        servers = store.list_servers()
        assert len(servers) == 1
        assert servers[0].name == "hipster"
        assert servers[0].slurm_queue == {"running": 5, "pending": 2}

        # upsert again (update)
        srv2 = Server(
            name="hipster",
            backend_type="slurm",
            online=False,
            last_heartbeat="2026-06-04T11:00:00+00:00",
            cpu={"load": 0.1},
            mem={"used_gb": 4.0},
            gpu={},
            disk={},
            slurm_queue={},
            note="offline",
        )
        store.upsert_server(srv2)
        servers2 = store.list_servers()
        assert len(servers2) == 1
        assert servers2[0].online is False

    def test_get_server(self, tmp_path):
        from jobctl.db.store import Store
        from jobctl.db.models import Server
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()

        srv = Server(
            name="oblix",
            backend_type="ssh",
            online=True,
            last_heartbeat=None,
            cpu={},
            mem={},
            gpu={},
            disk={},
            slurm_queue={},
            note=None,
        )
        store.upsert_server(srv)
        fetched = store.get_server("oblix")
        assert fetched.name == "oblix"
        assert fetched.backend_type == "ssh"
        assert store.get_server("nonexistent") is None

    def test_save_and_get_contract(self, tmp_path):
        from jobctl.db.store import Store
        from jobctl.db.models import ExpectationContract, Criterion
        from datetime import datetime, timezone
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()

        jf = make_jobfile()
        store.add_jobfile(jf)

        criterion = Criterion(
            id="crit-001",
            text="No NaN in logs",
            kind="absence",
            check={"pattern": "NaN"},
            status="confirmed",
            strength=3,
            evidence_run_ids=["run-001", "run-002"],
        )
        contract = ExpectationContract(
            id="ec-001",
            jobfile_id="jf-001",
            version=1,
            criteria=[criterion],
            source="expectation: no NaN",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        store.save_contract(contract)

        fetched = store.get_contract("jf-001")
        assert fetched.id == "ec-001"
        assert len(fetched.criteria) == 1
        assert fetched.criteria[0].id == "crit-001"
        assert fetched.criteria[0].evidence_run_ids == ["run-001", "run-002"]

        fetched_by_id = store.get_contract_by_id("ec-001")
        assert fetched_by_id.id == "ec-001"

    def test_add_and_list_feedback(self, tmp_path):
        from jobctl.db.store import Store
        from jobctl.db.models import Feedback
        from datetime import datetime, timezone
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()

        jf = make_jobfile()
        store.add_jobfile(jf)
        run = make_run()
        store.add_run(run)

        fb = Feedback(
            id="fb-001",
            run_id="run-001",
            kind="accept",
            text="Looks good, energy converged.",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        store.add_feedback(fb)
        feedbacks = store.list_feedback("run-001")
        assert len(feedbacks) == 1
        assert feedbacks[0].kind == "accept"

    def test_json_columns_deserialized(self, tmp_path):
        """JSON columns come back as Python dicts/lists, not strings."""
        from jobctl.db.store import Store
        store = Store(str(tmp_path / "test.db"))
        store.init_schema()

        jf = make_jobfile()
        store.add_jobfile(jf)
        run = make_run()
        store.add_run(run)

        fetched = store.get_run(run.run_id)
        assert isinstance(fetched.params, dict)
        assert isinstance(fetched.input_hashes, dict)
        assert isinstance(fetched.resource_summary, dict)


# ---------------------------------------------------------------------------
# jobfile.py tests
# ---------------------------------------------------------------------------

class TestJobfile:
    def _write_manifest(self, path, content):
        path.write_text(yaml.dump(content))

    def test_load_jobfile_from_manifest(self, tmp_path):
        """load_jobfile parses a .jobfile.yaml."""
        from jobctl.jobfile import load_jobfile

        manifest_path = tmp_path / "ipeps.jobfile.yaml"
        self._write_manifest(manifest_path, {
            "name": "ipeps-opt",
            "command": "julia +1.11 {script} --chi {chi} --D {D}",
            "params": {
                "script": {"type": "path", "required": True},
                "chi": {"type": "int", "default": 40},
                "D": {"type": "int", "default": 4},
            },
            "backends": [
                {"backend": "slurm", "server": "hipster", "task": "gpu1_l4"},
                {"backend": "local"},
            ],
            "artifacts": ["*.png", "energy*.csv"],
            "expectation": "energy converges below -0.66/site; no NaNs in logs",
        })

        jf = load_jobfile(str(manifest_path))
        assert jf.name == "ipeps-opt"
        assert jf.command_template == "julia +1.11 {script} --chi {chi} --D {D}"
        assert "chi" in jf.params_schema
        assert jf.params_schema["chi"]["default"] == 40
        assert len(jf.backend_prefs) == 2
        assert jf.artifact_patterns == ["*.png", "energy*.csv"]
        assert jf.version == 1
        assert jf.content_hash  # non-empty

    def test_autowrap_py(self, tmp_path):
        """Bare .py file -> python command."""
        from jobctl.jobfile import load_jobfile

        script = tmp_path / "train.py"
        script.write_text("print('hello')\n")
        jf = load_jobfile(str(script))
        assert "python" in jf.command_template
        assert str(script) in jf.command_template or "{script}" in jf.command_template

    def test_autowrap_sbatch(self, tmp_path):
        """Bare .sbatch file -> sbatch command."""
        from jobctl.jobfile import load_jobfile

        script = tmp_path / "job.sbatch"
        script.write_text("#!/bin/bash\necho hi\n")
        jf = load_jobfile(str(script))
        assert "sbatch" in jf.command_template

    def test_autowrap_sh(self, tmp_path):
        """Bare .sh file -> bash command."""
        from jobctl.jobfile import load_jobfile

        script = tmp_path / "run.sh"
        script.write_text("#!/bin/bash\necho hello\n")
        jf = load_jobfile(str(script))
        assert "bash" in jf.command_template

    def test_autowrap_jl(self, tmp_path):
        """Bare .jl file -> julia command."""
        from jobctl.jobfile import load_jobfile

        script = tmp_path / "sim.jl"
        script.write_text('println("hi")\n')
        jf = load_jobfile(str(script))
        assert "julia" in jf.command_template

    def test_autowrap_m(self, tmp_path):
        """Bare .m file -> matlab command."""
        from jobctl.jobfile import load_jobfile

        script = tmp_path / "compute.m"
        script.write_text("disp('hi')\n")
        jf = load_jobfile(str(script))
        assert "matlab" in jf.command_template

    def test_autowrap_r(self, tmp_path):
        """Bare .R file -> Rscript command."""
        from jobctl.jobfile import load_jobfile

        script = tmp_path / "analysis.R"
        script.write_text("print('hi')\n")
        jf = load_jobfile(str(script))
        assert "Rscript" in jf.command_template

    def test_resolve_params_applies_defaults(self, tmp_path):
        """resolve_params fills defaults for missing keys."""
        from jobctl.jobfile import load_jobfile, resolve_params

        manifest = tmp_path / "job.jobfile.yaml"
        self._write_manifest(manifest, {
            "name": "my-job",
            "command": "python {script} --lr {lr} --epochs {epochs}",
            "params": {
                "script": {"type": "path", "required": True},
                "lr": {"type": "float", "default": 0.001},
                "epochs": {"type": "int", "default": 10},
            },
            "backends": [{"backend": "local"}],
        })
        jf = load_jobfile(str(manifest))
        resolved = resolve_params(jf, {"script": "/tmp/train.py"})
        assert resolved["lr"] == 0.001
        assert resolved["epochs"] == 10
        assert resolved["script"] == "/tmp/train.py"

    def test_resolve_params_override(self, tmp_path):
        """resolve_params: override beats default."""
        from jobctl.jobfile import load_jobfile, resolve_params

        manifest = tmp_path / "job.jobfile.yaml"
        self._write_manifest(manifest, {
            "name": "my-job",
            "command": "python {script} --lr {lr}",
            "params": {
                "script": {"type": "path", "required": True},
                "lr": {"type": "float", "default": 0.001},
            },
            "backends": [{"backend": "local"}],
        })
        jf = load_jobfile(str(manifest))
        resolved = resolve_params(jf, {"script": "/tmp/train.py", "lr": "0.1"})
        assert abs(resolved["lr"] - 0.1) < 1e-9

    def test_resolve_params_required_missing(self, tmp_path):
        """resolve_params raises ValueError if required param is missing."""
        from jobctl.jobfile import load_jobfile, resolve_params

        manifest = tmp_path / "job.jobfile.yaml"
        self._write_manifest(manifest, {
            "name": "my-job",
            "command": "python {script}",
            "params": {
                "script": {"type": "path", "required": True},
            },
            "backends": [{"backend": "local"}],
        })
        jf = load_jobfile(str(manifest))
        with pytest.raises((ValueError, KeyError)):
            resolve_params(jf, {})

    def test_resolve_params_int_cast(self, tmp_path):
        """resolve_params casts string '5' to int for int-typed params."""
        from jobctl.jobfile import load_jobfile, resolve_params

        manifest = tmp_path / "job.jobfile.yaml"
        self._write_manifest(manifest, {
            "name": "my-job",
            "command": "run --n {n}",
            "params": {
                "n": {"type": "int", "default": 1},
            },
            "backends": [{"backend": "local"}],
        })
        jf = load_jobfile(str(manifest))
        resolved = resolve_params(jf, {"n": "5"})
        assert resolved["n"] == 5
        assert isinstance(resolved["n"], int)

    def test_render_command(self, tmp_path):
        """render_command substitutes params into command template."""
        from jobctl.jobfile import load_jobfile, render_command

        manifest = tmp_path / "job.jobfile.yaml"
        self._write_manifest(manifest, {
            "name": "my-job",
            "command": "julia {script} --chi {chi}",
            "params": {
                "script": {"type": "path", "required": True},
                "chi": {"type": "int", "default": 40},
            },
            "backends": [{"backend": "local"}],
        })
        jf = load_jobfile(str(manifest))
        cmd = render_command(jf, {"script": "/tmp/sim.jl", "chi": 60})
        assert cmd == "julia /tmp/sim.jl --chi 60"

    def test_render_command_preserves_slurm_env_placeholders(self, tmp_path):
        """Shell env placeholders are not treated as job params."""
        from jobctl.jobfile import load_jobfile, render_command

        manifest = tmp_path / "job.jobfile.yaml"
        self._write_manifest(manifest, {
            "name": "my-job",
            "command": "julia run.jl --threads ${SLURM_CPUS_PER_TASK} --chi {chi}",
            "params": {
                "chi": {"type": "int", "default": 40},
            },
            "backends": [{"backend": "slurm"}],
        })
        jf = load_jobfile(str(manifest))
        cmd = render_command(jf, {"chi": 60})
        assert cmd == "julia run.jl --threads ${SLURM_CPUS_PER_TASK} --chi 60"

    def test_render_command_still_rejects_missing_job_params(self, tmp_path):
        """Lowercase missing placeholders remain configuration errors."""
        from jobctl.jobfile import load_jobfile, render_command

        manifest = tmp_path / "job.jobfile.yaml"
        self._write_manifest(manifest, {
            "name": "my-job",
            "command": "julia {script} --chi {chi}",
            "params": {
                "chi": {"type": "int", "default": 40},
            },
            "backends": [{"backend": "local"}],
        })
        jf = load_jobfile(str(manifest))
        with pytest.raises(KeyError):
            render_command(jf, {"chi": 60})

    def test_render_command_rejects_missing_uppercase_job_params(self, tmp_path):
        """Uppercase physics-style params still need explicit values."""
        from jobctl.jobfile import load_jobfile, render_command

        manifest = tmp_path / "job.jobfile.yaml"
        self._write_manifest(manifest, {
            "name": "my-job",
            "command": "julia run.jl --D {D}",
            "params": {
                "D": {"type": "int", "required": True},
            },
            "backends": [{"backend": "local"}],
        })
        jf = load_jobfile(str(manifest))
        with pytest.raises(KeyError):
            render_command(jf, {})

    def test_content_hash_deterministic(self, tmp_path):
        """content_hash is stable across calls."""
        from jobctl.jobfile import load_jobfile, content_hash

        manifest = tmp_path / "job.jobfile.yaml"
        self._write_manifest(manifest, {
            "name": "my-job",
            "command": "python {script}",
            "params": {"script": {"type": "path", "required": True}},
            "backends": [{"backend": "local"}],
        })
        jf = load_jobfile(str(manifest))
        h1 = content_hash(jf)
        h2 = content_hash(jf)
        assert h1 == h2
        assert len(h1) > 0

    def test_content_hash_changes_with_script(self, tmp_path):
        """content_hash changes when the referenced script file changes."""
        from jobctl.jobfile import load_jobfile, content_hash

        script = tmp_path / "train.py"
        script.write_text("x = 1\n")
        jf1 = load_jobfile(str(script))
        h1 = content_hash(jf1)

        script.write_text("x = 2\n")
        jf2 = load_jobfile(str(script))
        h2 = content_hash(jf2)

        assert h1 != h2

    def test_input_hashes_deterministic(self, tmp_path):
        """input_hashes gives stable dict of {path: sha256} for path-typed params."""
        from jobctl.jobfile import load_jobfile, input_hashes

        script = tmp_path / "train.py"
        script.write_text("print('hello')\n")
        manifest = tmp_path / "job.jobfile.yaml"
        self._write_manifest(manifest, {
            "name": "my-job",
            "command": "python {script}",
            "params": {"script": {"type": "path", "required": True}},
            "backends": [{"backend": "local"}],
        })
        jf = load_jobfile(str(manifest))
        hashes1 = input_hashes(jf, {"script": str(script)})
        hashes2 = input_hashes(jf, {"script": str(script)})
        assert hashes1 == hashes2
        assert str(script) in hashes1
        assert hashes1[str(script)].startswith("sha256:")

    def test_input_hashes_changes_with_content(self, tmp_path):
        """input_hashes changes when script content changes."""
        from jobctl.jobfile import load_jobfile, input_hashes

        script = tmp_path / "train.py"
        script.write_text("x = 1\n")
        manifest = tmp_path / "job.jobfile.yaml"
        self._write_manifest(manifest, {
            "name": "my-job",
            "command": "python {script}",
            "params": {"script": {"type": "path", "required": True}},
            "backends": [{"backend": "local"}],
        })
        jf = load_jobfile(str(manifest))
        h1 = input_hashes(jf, {"script": str(script)})

        script.write_text("x = 999\n")
        h2 = input_hashes(jf, {"script": str(script)})
        assert h1[str(script)] != h2[str(script)]

    def test_content_hash_stored_on_jobfile(self, tmp_path):
        """JobFile.content_hash is set on creation."""
        from jobctl.jobfile import load_jobfile

        manifest = tmp_path / "job.jobfile.yaml"
        self._write_manifest(manifest, {
            "name": "my-job",
            "command": "python {script}",
            "params": {"script": {"type": "path", "required": True}},
            "backends": [{"backend": "local"}],
        })
        jf = load_jobfile(str(manifest))
        assert jf.content_hash
        assert isinstance(jf.content_hash, str)
        assert len(jf.content_hash) > 8
