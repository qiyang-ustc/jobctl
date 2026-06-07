"""Human-readable run identity: title / note / tags.

Covers the DB round-trip, additive migration, the display-title fallback, and
the API submit + PATCH surface.
"""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from jobctl.db.models import Run, State, Health
from jobctl.db.store import Store


def _make_jobfile():
    from jobctl.db.models import JobFile
    from datetime import datetime, timezone
    return JobFile(
        id="jf-id-1",
        name="my-experiment",
        version=1,
        source_path="/tmp/x.yaml",
        command_template="echo {lr}",
        params_schema={"lr": {"type": "float", "default": 0.01}},
        backend_prefs=[{"backend": "local"}],
        artifact_patterns=[],
        expectation_contract_id=None,
        content_hash="h",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_run(run_id="run-id-1", **over):
    from datetime import datetime, timezone
    base = dict(
        run_id=run_id,
        jobfile_id="jf-id-1",
        jobfile_version=1,
        params={"lr": 0.01, "chi": 60, "method": "exact", "extra": 9},
        input_hashes={},
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
        workdir=None,
        stdout_path=None,
        stderr_path=None,
        resource_summary={},
        expectation_match=None,
        observation_card=None,
    )
    base.update(over)
    return Run(**base)


# ---------------------------------------------------------------------------
# Store round-trip + migration
# ---------------------------------------------------------------------------

class TestIdentityStore:
    def test_title_note_tags_round_trip_via_add_run(self, tmp_path):
        store = Store(str(tmp_path / "t.db"))
        store.init_schema()
        store.add_jobfile(_make_jobfile())
        store.add_run(_make_run(title="chi scan", note="checking convergence", tags=["sweep", "chi"]))
        got = store.get_run("run-id-1")
        assert got.title == "chi scan"
        assert got.note == "checking convergence"
        assert got.tags == ["sweep", "chi"]

    def test_defaults_are_none(self, tmp_path):
        store = Store(str(tmp_path / "t.db"))
        store.init_schema()
        store.add_jobfile(_make_jobfile())
        store.add_run(_make_run())
        got = store.get_run("run-id-1")
        assert got.title is None and got.note is None and got.tags is None

    def test_update_run_patches_identity(self, tmp_path):
        store = Store(str(tmp_path / "t.db"))
        store.init_schema()
        store.add_jobfile(_make_jobfile())
        store.add_run(_make_run())
        store.update_run("run-id-1", title="renamed", tags=["a", "b"])
        got = store.get_run("run-id-1")
        assert got.title == "renamed"
        assert got.tags == ["a", "b"]
        # clearing
        store.update_run("run-id-1", title=None, tags=None)
        got = store.get_run("run-id-1")
        assert got.title is None and got.tags is None

    def test_migration_adds_identity_columns(self, tmp_path):
        """A runs table predating title/note/tags gets them on init_schema."""
        db = str(tmp_path / "old.db")
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, jobfile_id TEXT, "
            "jobfile_version INTEGER, params TEXT, input_hashes TEXT, backend TEXT, "
            "server TEXT, task TEXT, remote_job_id TEXT, state TEXT, health TEXT, "
            "exit_code INTEGER, submitted_at TEXT, started_at TEXT, finished_at TEXT, "
            "last_heartbeat TEXT, workdir TEXT, stdout_path TEXT, stderr_path TEXT, "
            "resource_summary TEXT, expectation_match TEXT, observation_card TEXT, "
            "slurm_request TEXT)"
        )
        conn.commit()
        conn.close()

        store = Store(db)
        store.init_schema()
        cols = {r[1] for r in store._get_conn().execute("PRAGMA table_info(runs)")}
        assert {"title", "note", "tags"} <= cols


# ---------------------------------------------------------------------------
# Display-title fallback
# ---------------------------------------------------------------------------

class TestDisplayTitle:
    def test_explicit_title_wins(self):
        from jobctl.api.server import _display_title
        run = _make_run(title="my label")
        assert _display_title(run, "my-experiment") == "my label"

    def test_fallback_uses_jobfile_and_three_params(self):
        from jobctl.api.server import _display_title
        run = _make_run()  # 4 params, no title
        out = _display_title(run, "my-experiment")
        assert out.startswith("my-experiment · ")
        # only the first three params are shown
        assert out.count("=") == 3
        assert "extra=" not in out

    def test_fallback_without_jobfile_name_uses_id(self):
        from jobctl.api.server import _display_title
        run = _make_run(params={})
        assert _display_title(run) == run.jobfile_id


# ---------------------------------------------------------------------------
# API submit + PATCH
# ---------------------------------------------------------------------------

@pytest.fixture
def api(tmp_path):
    from jobctl.api.server import create_app
    config = {
        "db_path": str(tmp_path / "app.db"),
        "run_dir": str(tmp_path / "runs"),
        "poll_interval_seconds": 0.1,
        "probe_interval_seconds": 9999,
        "stuck_timeout_seconds": 9999,
    }
    app = create_app(config=config, start_monitor=False)
    with TestClient(app) as client:
        yield client


def _register_jobfile(api, tmp_path):
    script = tmp_path / "hi.sh"
    script.write_text('#!/bin/bash\necho hi\n')
    jf = tmp_path / "hi.jobfile.yaml"
    jf.write_text(f'name: hi\ncommand: "bash {script}"\nparams: {{}}\nbackends:\n  - backend: local\n')
    r = api.post("/jobfiles", json={"path": str(jf)})
    assert r.status_code == 200
    return r.json()["id"]


def test_submit_carries_identity(api, tmp_path):
    jf_id = _register_jobfile(api, tmp_path)
    r = api.post("/runs", json={
        "jobfile_id": jf_id,
        "title": "smoke run",
        "note": "verifying identity",
        "tags": ["t1", "t2"],
    })
    assert r.status_code == 200
    data = r.json()
    assert data["title"] == "smoke run"
    assert data["note"] == "verifying identity"
    assert data["tags"] == ["t1", "t2"]
    assert data["display_title"] == "smoke run"


def test_submit_without_title_gets_derived_display_title(api, tmp_path):
    jf_id = _register_jobfile(api, tmp_path)
    r = api.post("/runs", json={"jobfile_id": jf_id})
    data = r.json()
    assert data["title"] is None
    # never just the hash — derived from jobfile name
    assert data["display_title"] and data["display_title"] != data["run_id"]
    assert data["display_title"].startswith("hi")


def test_patch_edits_identity(api, tmp_path):
    jf_id = _register_jobfile(api, tmp_path)
    run_id = api.post("/runs", json={"jobfile_id": jf_id}).json()["run_id"]
    r = api.patch(f"/runs/{run_id}", json={"title": "edited", "tags": ["x"]})
    assert r.status_code == 200
    data = r.json()
    assert data["title"] == "edited"
    assert data["tags"] == ["x"]
    assert data["display_title"] == "edited"


def test_patch_unknown_run_404(api):
    r = api.patch("/runs/run-does-not-exist", json={"title": "x"})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /ui/poll regressions (review findings)
# ---------------------------------------------------------------------------

def test_poll_servers_returns_server_fragment_not_buckets(api, tmp_path):
    """section=servers must return the server fragment, never the run buckets.
    Regression: it used to fall through to buckets, so HTMX overwrote the live
    server cards with the run list after the first 15s poll."""
    jf_id = _register_jobfile(api, tmp_path)
    api.post("/runs", json={"jobfile_id": jf_id})  # ensure buckets WOULD be non-empty
    body = api.get("/ui/poll", params={"section": "servers"},
                   headers={"Accept": "text/html"}).text
    assert "run-row" not in body          # not the buckets fragment
    assert "empty.servers" in body or "server-card" in body


def test_poll_run_id_swaps_only_pills_not_whole_hero(api, tmp_path):
    """The run-hero live poll must return just the status pills, not a bare
    fragment that wipes the display_title / id-chip / tags."""
    jf_id = _register_jobfile(api, tmp_path)
    run_id = api.post("/runs", json={"jobfile_id": jf_id}).json()["run_id"]
    resp = api.get("/ui/poll", params={"run_id": run_id}, headers={"Accept": "text/html"})
    body = resp.text
    assert "<h1>Run" not in body          # old hero-wiping markup is gone
    assert "badge-" in body               # returns the status pill(s)

