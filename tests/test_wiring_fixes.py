"""Regression tests for CLI/daemon wiring fixes found during the audit:

- feedback --accept/--reject map to kind good/bad (spec surface).
- expect list subcommand exists and returns contracts.
- run --callback persists a per-run callback URL into the monitor so the
  terminal pipeline can POST the card.
- run --reuse short-circuits to a prior usable run.
"""
from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner


def _seed_app(tmp_path, jobfile_yaml):
    from jobctl.api.server import create_app
    config = {
        "db_path": str(tmp_path / "w.db"),
        "run_dir": str(tmp_path / "wruns"),
        "poll_interval_seconds": 0.05,
        "probe_interval_seconds": 9999,
        "stuck_timeout_seconds": 9999,
    }
    return create_app(config=config, start_monitor=True)


@pytest.fixture
def csv_jobfile(tmp_path):
    script = tmp_path / "c.sh"
    script.write_text('#!/bin/bash\necho "v=1"\necho "a,b" > out.csv\necho "1,2" >> out.csv\n')
    script.chmod(0o755)
    jf = tmp_path / "c.jobfile.yaml"
    jf.write_text(
        "name: wire-job\n"
        f'command: "bash {script}"\n'
        "params: {}\n"
        "backends:\n  - backend: local\n"
        'artifacts:\n  - "*.csv"\n'
    )
    return str(jf)


def _wait_terminal(http, run_id, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = http.get(f"/runs/{run_id}").json()
        if d["state"] in {"completed", "failed", "cancelled", "stuck", "timeout"}:
            return d
        time.sleep(0.05)
    return http.get(f"/runs/{run_id}").json()


def test_feedback_accept_reject_map_to_good_bad(tmp_path, csv_jobfile):
    from jobctl.cli import main as cli_module
    from jobctl.cli.main import app
    from jobctl.api.client import ApiClient

    with TestClient(_seed_app(tmp_path, csv_jobfile)) as http:
        ac = ApiClient(base_url="http://testserver", transport=http)
        cli_module._OVERRIDE_CLIENT = ac
        try:
            jf_id = http.post("/jobfiles", json={"path": csv_jobfile}).json()["id"]
            rid = http.post("/runs", json={"jobfile_id": jf_id, "params": {}}).json()["run_id"]
            runner = CliRunner()

            r1 = runner.invoke(app, ["feedback", rid, "--accept", "--text", "great"])
            assert r1.exit_code == 0, r1.output
            r2 = runner.invoke(app, ["feedback", rid, "--reject"])
            assert r2.exit_code == 0, r2.output

            kinds = [f["kind"] for f in http.get(f"/runs/{rid}/feedback").json()]
            assert "good" in kinds and "bad" in kinds, kinds

            # mutually exclusive
            r3 = runner.invoke(app, ["feedback", rid, "--accept", "--reject"])
            assert r3.exit_code != 0
        finally:
            cli_module._OVERRIDE_CLIENT = None


def test_expect_list_subcommand(tmp_path, csv_jobfile):
    from jobctl.cli import main as cli_module
    from jobctl.cli.main import app
    from jobctl.api.client import ApiClient

    with TestClient(_seed_app(tmp_path, csv_jobfile)) as http:
        cli_module._OVERRIDE_CLIENT = ApiClient(base_url="http://testserver", transport=http)
        try:
            http.post("/jobfiles", json={"path": csv_jobfile})
            runner = CliRunner()
            r = runner.invoke(app, ["expect", "list", "wire-job", "--json"])
            assert r.exit_code == 0, r.output
            # default_contract is created at registration; list must return JSON (possibly [])
            json.loads(r.output)  # must be valid JSON, not an argparse error
        finally:
            cli_module._OVERRIDE_CLIENT = None


def test_callback_url_persisted_to_monitor(tmp_path, csv_jobfile):
    app = _seed_app(tmp_path, csv_jobfile)
    with TestClient(app) as http:
        jf_id = http.post("/jobfiles", json={"path": csv_jobfile}).json()["id"]
        resp = http.post("/runs", json={
            "jobfile_id": jf_id, "params": {},
            "callback_url": "http://127.0.0.1:9/cb",
        }).json()
        rid = resp["run_id"]
        monitor = app.state.monitor
        assert monitor._callback_urls.get(rid) == "http://127.0.0.1:9/cb"


def test_registration_persists_default_contract(tmp_path, csv_jobfile):
    app = _seed_app(tmp_path, csv_jobfile)
    with TestClient(app) as http:
        reg = http.post("/jobfiles", json={"path": csv_jobfile}).json()
        jf_id = reg["id"]
        # Contract is persisted and linked at registration.
        assert reg.get("expectation_contract_id"), reg
        contracts = http.get(f"/expect?jobfile_id={jf_id}").json()
        assert len(contracts) == 1, contracts
        assert len(contracts[0]["criteria"]) >= 3  # NaN/Traceback/CUDA absence


def test_observation_card_carries_per_criterion(tmp_path, csv_jobfile):
    app = _seed_app(tmp_path, csv_jobfile)
    with TestClient(app) as http:
        jf_id = http.post("/jobfiles", json={"path": csv_jobfile}).json()["id"]
        rid = http.post("/runs", json={"jobfile_id": jf_id, "params": {}}).json()["run_id"]
        d = _wait_terminal(http, rid)
        assert d["state"] == "completed", d
        card = d["observation_card"]
        assert "per_criterion" in card
        assert isinstance(card["per_criterion"], list) and len(card["per_criterion"]) >= 1


def test_run_reuse_short_circuits(tmp_path, csv_jobfile):
    app = _seed_app(tmp_path, csv_jobfile)
    with TestClient(app) as http:
        jf_id = http.post("/jobfiles", json={"path": csv_jobfile}).json()["id"]
        # First run to completion
        rid1 = http.post("/runs", json={"jobfile_id": jf_id, "params": {}}).json()["run_id"]
        d1 = _wait_terminal(http, rid1)
        assert d1["state"] == "completed", d1
        # Force it usable so reuse is eligible (classification may be inconclusive
        # for a contract with unconfirmed criteria; reuse keys on USABLE).
        http_match = http.post(f"/runs/{rid1}/feedback", json={"kind": "good", "text": "ok"})
        assert http_match.status_code in (200, 201)
        from jobctl.db.store import Store
        store = app.state.store
        store.update_run(rid1, expectation_match="usable")
        # Second submit with reuse=true and identical params -> reused
        resp2 = http.post("/runs", json={"jobfile_id": jf_id, "params": {}, "reuse": True}).json()
        assert resp2.get("reused") is True, resp2
        assert resp2["run_id"] == rid1, resp2
