"""Regression test: `jobctl run <path> --param script=<relative>` must work.

The local backend executes the job in its OWN workdir (cwd=run workdir), so a
relative script path resolved against the user's shell cwd is not visible there.
The CLI therefore must absolutize the jobfile path and any path-typed params
before submitting.  Without that, the job exits non-zero (file not found) and
the run is classified `failed`.

This reproduces the failure in-process via the TestClient-backed ApiClient.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner


def test_run_by_path_with_relative_param_completes(tmp_path, monkeypatch):
    from jobctl.cli import main as cli_module
    from jobctl.cli.main import app
    from jobctl.api.server import create_app
    from jobctl.api.client import ApiClient

    script = tmp_path / "emit.py"
    script.write_text(
        "import os\n"
        "wd = os.environ.get('JOBCTL_WORKDIR', '.')\n"
        "open(os.path.join(wd, 'out.csv'), 'w').write('a,b\\n1,2\\n')\n"
        "print('done')\n"
    )
    jf = tmp_path / "j.jobfile.yaml"
    jf.write_text(
        "name: pathjob\n"
        'command: "python {script}"\n'
        "params:\n"
        "  script: {type: path, required: true}\n"
        "backends:\n"
        "  - backend: local\n"
        "artifacts:\n"
        '  - "*.csv"\n'
    )

    config = {
        "db_path": str(tmp_path / "db.sqlite"),
        "run_dir": str(tmp_path / "runs"),
        "poll_interval_seconds": 0.05,
        "probe_interval_seconds": 9999,
        "stuck_timeout_seconds": 9999,
    }
    fast_app = create_app(config=config, start_monitor=True)

    with TestClient(fast_app) as http:
        cli_module._OVERRIDE_CLIENT = ApiClient(base_url="http://testserver", transport=http)
        try:
            # Run from tmp_path so the relative param "emit.py" resolves vs cwd,
            # while the backend runs elsewhere (its own workdir).
            monkeypatch.chdir(tmp_path)
            runner = CliRunner()
            res = runner.invoke(
                app,
                ["run", str(jf), "--param", "script=emit.py", "--wait", "--json"],
            )
            assert res.exit_code == 0, res.output
            data = json.loads(res.output)
            state = data.get("status") or data.get("state")
            assert state == "completed", f"expected completed, got {state}: {res.output}"
        finally:
            cli_module._OVERRIDE_CLIENT = None
