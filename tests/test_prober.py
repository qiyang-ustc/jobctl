"""Tests for SshProber parsing (no real SSH — runner is injected)."""
from __future__ import annotations

import subprocess
import logging

from jobctl.monitor.prober import SshProber


def _cp(stdout: str, rc: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["ssh"], returncode=rc, stdout=stdout, stderr="")


REACHABLE_SLURM = "\n".join([
    "JOBCTL_OK",
    "NPROC:8",
    "LOAD:2.0 1.5 1.0 1/300 999",
    "MEMTOTAL:16000000",
    "MEMAVAIL:4000000",
    "DISK:/dev/sda1 100 42 58 42% /",
    "GPU:37",
    "HASSLURM:1",
    "SQR:3",
    "SQP:5",
    "SQRA:120",
    "SQPA:340",
    "SINFOC:760/40/0/800",
    "SQJ:12345^R^alpha_scan^01:02^03:58^1^8^node-a",
    "SQJ:12346^PD^beta_scan^00:00^04:00^1^4^(Priority)",
])


def test_probe_parses_reachable_slurm_server():
    prober = SshProber({"hipster": {}}, runner=lambda t, r, to: _cp(REACHABLE_SLURM))
    srv = prober.probe("hipster")
    assert srv is not None and srv.online is True
    assert srv.cpu == {"pct": 25, "nproc": 8}          # load 2.0 / 8 cores = 25%
    assert srv.mem == {"pct": 75}                        # 1 - 4M/16M = 75%
    assert srv.disk == {"pct": 42}
    assert srv.gpu == {"pct": 37.0}
    assert srv.backend_type == "slurm"
    assert srv.slurm_queue == {
        "running": 3, "pending": 5, "running_all": 120, "pending_all": 340,
        "allocated_cpus": 760, "idle_cpus": 40, "other_cpus": 0,
        "total_cpus": 800, "idle_pct": 5.0,
        "jobs": [
            {
                "job_id": "12345",
                "state": "R",
                "name": "alpha_scan",
                "elapsed": "01:02",
                "time_left": "03:58",
                "nodes": 1,
                "cpus": 8,
                "where": "node-a",
            },
            {
                "job_id": "12346",
                "state": "PD",
                "name": "beta_scan",
                "elapsed": "00:00",
                "time_left": "04:00",
                "nodes": 1,
                "cpus": 4,
                "where": "(Priority)",
            },
        ],
    }


def test_probe_non_slurm_host_has_no_queue():
    out = "\n".join(["JOBCTL_OK", "NPROC:4", "LOAD:0.4 0 0", "MEMTOTAL:8000000",
                     "MEMAVAIL:6000000", "DISK:/x 1 1 1 10% /", "GPU:", "HASSLURM:0", "SQR:0", "SQP:0"])
    prober = SshProber({"oblix": {"backend": "ssh"}}, runner=lambda t, r, to: _cp(out))
    srv = prober.probe("oblix")
    assert srv.online is True
    assert srv.backend_type == "ssh"
    assert srv.slurm_queue == {}
    assert srv.gpu == {}
    assert srv.cpu["pct"] == 10                          # 0.4 / 4 = 10%


def test_probe_unreachable_returns_none():
    # non-zero rc
    p1 = SshProber({"x": {}}, runner=lambda t, r, to: _cp("", rc=255))
    assert p1.probe("x") is None
    # missing marker
    p2 = SshProber({"x": {}}, runner=lambda t, r, to: _cp("garbage"))
    assert p2.probe("x") is None
    # runner raises (timeout / ssh missing)
    def boom(t, r, to):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=to)
    p3 = SshProber({"x": {}}, runner=boom)
    assert p3.probe("x") is None


def test_probe_timeout_log_is_concise(caplog):
    def boom(t, r, to):
        raise subprocess.TimeoutExpired(cmd=["ssh", "x", "very long remote command"], timeout=to)

    prober = SshProber({"x": {}}, runner=boom)
    with caplog.at_level(logging.INFO, logger="jobctl.monitor.prober"):
        assert prober.probe("x") is None

    messages = " ".join(record.message for record in caplog.records)
    assert "timeout after 8s" in messages
    assert "very long remote command" not in messages


def test_probe_uses_ssh_alias_with_optional_user():
    captured = {}
    def runner(target, remote, to):
        captured["target"] = target
        return _cp(REACHABLE_SLURM)
    SshProber({"hipster": {"user": "qyang1"}}, runner=runner).probe("hipster")
    assert captured["target"] == "qyang1@hipster"
    SshProber({"oblix": {}}, runner=runner).probe("oblix")
    assert captured["target"] == "oblix"
