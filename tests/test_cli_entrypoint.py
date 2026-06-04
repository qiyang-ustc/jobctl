"""Regression test for the `python -m jobctl.cli.main` entry point.

The daemon auto-start path (`ensure_daemon`) spawns the CLI via
`python -m jobctl.cli.main serve`.  Running a module with `-m` does NOT call a
Typer app unless the module has an `if __name__ == "__main__": app()` guard.
Without it the spawned process imported the module and exited immediately, so
the daemon never started and `jobctl run` failed with "Connection refused".

The in-process CliRunner/TestClient unit tests cannot catch this because they
never go through a real subprocess.  This test does.
"""
import subprocess
import sys


def test_python_m_entrypoint_invokes_app():
    r = subprocess.run(
        [sys.executable, "-m", "jobctl.cli.main", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr
    # If the __main__ guard is missing, stdout is empty (module just defines app).
    assert "Usage" in r.stdout
    assert "serve" in r.stdout
