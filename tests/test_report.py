"""Bug-report assembly + issue submission (the 'submit issue to you' channel)."""
from __future__ import annotations

from jobctl import report


def test_build_report_includes_description_version_and_run(tmp_path):
    (tmp_path / "daemon.log").write_text("boot\nready\n")
    (tmp_path / "cli.log").write_text("invoke: run x\n")
    run = {"run_id": "run-abc", "state": "stuck", "backend": "slurm", "server": "oblix"}
    body = report.build_report(
        "monitor marked my running job stuck",
        version="0.1.0",
        run=run,
        recent=[{"run_id": "run-abc", "state": "stuck", "backend": "slurm", "server": "oblix", "exit_code": None}],
        log_dir=tmp_path,
    )
    assert "monitor marked my running job stuck" in body
    assert "0.1.0" in body
    assert "run-abc" in body
    assert "ready" in body          # daemon.log tail
    assert "invoke: run x" in body  # cli.log tail
    assert "Filed via `jobctl report-bug`" in body


def test_submit_issue_uses_injected_runner_and_returns_url():
    seen = {}
    def runner(repo, title, body):
        seen["repo"] = repo
        return "https://github.com/qiyang-ustc/jobctl/issues/7"
    url = report.submit_issue("[bug] x", "body", runner=runner)
    assert url == "https://github.com/qiyang-ustc/jobctl/issues/7"
    assert seen["repo"] == "qiyang-ustc/jobctl"


def test_submit_issue_returns_none_on_failure():
    def boom(repo, title, body):
        raise RuntimeError("gh not authed")
    assert report.submit_issue("t", "b", runner=boom) is None
