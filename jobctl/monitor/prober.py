"""SshProber: gather real server health over SSH for the monitor.

One SSH round-trip per server collects a marker + cpu/mem/disk/gpu/slurm metrics
using only POSIX-ish shell builtins (no awk / embedded quotes), parsed locally.
Relies on ~/.ssh/config host aliases (same convention as the ssh/slurm backends).
Returns a Server snapshot, or None when the host is unreachable.
"""
from __future__ import annotations

import logging
import subprocess
from typing import Callable

from jobctl.db.models import Server

logger = logging.getLogger(__name__)

# Single remote command — every value is emitted as KEY:<value> on its own line.
# Uses only nproc/cat/grep/tr/df/tail/head/wc/command/nvidia-smi/squeue so there
# are no embedded quotes to escape through ssh.
_REMOTE = "; ".join([
    "echo JOBCTL_OK",
    "echo NPROC:$(nproc 2>/dev/null)",
    "echo LOAD:$(cat /proc/loadavg 2>/dev/null)",
    "echo MEMTOTAL:$(grep -m1 MemTotal /proc/meminfo 2>/dev/null | tr -dc 0-9)",
    "echo MEMAVAIL:$(grep -m1 MemAvailable /proc/meminfo 2>/dev/null | tr -dc 0-9)",
    "echo DISK:$(df -P / 2>/dev/null | tail -1)",
    "echo GPU:$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)",
    "echo HASSLURM:$(command -v squeue >/dev/null 2>&1 && echo 1 || echo 0)",
    "echo SQR:$(squeue -h -u $USER -t R 2>/dev/null | wc -l)",
    "echo SQP:$(squeue -h -u $USER -t PD 2>/dev/null | wc -l)",
    "echo SQRA:$(squeue -h -t R 2>/dev/null | wc -l)",
    "echo SQPA:$(squeue -h -t PD 2>/dev/null | wc -l)",
    "echo SINFOC:$(sinfo -h -o %C 2>/dev/null | head -1)",
    "squeue -h -u $USER -t R,PD -o %i^%t^%j^%M^%L^%D^%C^%R 2>/dev/null | head -80 | sed s/^/SQJ:/",
])


def _ssh_runner(target: str, remote: str, timeout: float) -> subprocess.CompletedProcess:
    cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(timeout)}",
        "-o", "StrictHostKeyChecking=accept-new", target, remote,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)


def _num(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _parse_slurm_jobs(lines: list[str]) -> list[dict]:
    jobs: list[dict] = []
    for line in lines:
        parts = line.split("^", 7)
        if len(parts) < 3:
            continue
        job = {
            "job_id": parts[0],
            "state": parts[1],
            "name": parts[2],
        }
        if len(parts) > 3:
            job["elapsed"] = parts[3]
        if len(parts) > 4:
            job["time_left"] = parts[4]
        if len(parts) > 5:
            job["nodes"] = int(_num(parts[5])) if parts[5] else 0
        if len(parts) > 6:
            job["cpus"] = int(_num(parts[6])) if parts[6] else 0
        if len(parts) > 7:
            job["where"] = parts[7]
        jobs.append(job)
    return jobs


def _parse_sinfo_cpus(raw: str) -> dict:
    """Parse ``sinfo -o %C``: allocated/idle/other/total."""
    parts = (raw or "").strip().split("/")
    if len(parts) != 4:
        return {}
    alloc, idle, other, total = [int(_num(p)) for p in parts]
    idle_pct = round(idle / total * 100.0, 1) if total else 0.0
    return {
        "allocated_cpus": alloc,
        "idle_cpus": idle,
        "other_cpus": other,
        "total_cpus": total,
        "idle_pct": idle_pct,
    }


class SshProber:
    """Probe servers over SSH. ``runner(target, remote, timeout)`` is injectable."""

    def __init__(self, servers_config: dict, timeout: float = 8.0, runner: Callable | None = None) -> None:
        self._servers = servers_config or {}
        self._timeout = timeout
        self._runner = runner or _ssh_runner

    def probe(self, name: str) -> Server | None:
        cfg = self._servers.get(name, {}) or {}
        host = cfg.get("host", name)            # ssh alias by default (~/.ssh/config)
        user = cfg.get("user")
        target = f"{user}@{host}" if user else host
        try:
            r = self._runner(target, _REMOTE, self._timeout)
        except Exception as exc:                 # timeout, ssh missing, etc.
            logger.info("probe %s: unreachable (%s)", name, exc)
            return None
        if r.returncode != 0 or "JOBCTL_OK" not in (r.stdout or ""):
            return None
        return self._parse(name, cfg, r.stdout)

    def _parse(self, name: str, cfg: dict, out: str) -> Server:
        kv: dict[str, str] = {}
        slurm_job_lines: list[str] = []
        for line in out.splitlines():
            if line.startswith("SQJ:"):
                slurm_job_lines.append(line[4:].strip())
                continue
            if ":" in line:
                k, _, v = line.partition(":")
                kv[k.strip()] = v.strip()

        nproc = _num(kv.get("NPROC"), 1.0) or 1.0
        load1 = _num((kv.get("LOAD", "").split() or ["0"])[0])
        cpu_pct = max(0.0, min(100.0, load1 / nproc * 100.0))

        memtotal = _num(kv.get("MEMTOTAL"))
        memavail = _num(kv.get("MEMAVAIL"))
        mem_pct = max(0.0, min(100.0, (1 - memavail / memtotal) * 100.0)) if memtotal else 0.0

        disk_pct = 0.0
        for tok in kv.get("DISK", "").split():
            if tok.endswith("%"):
                disk_pct = _num(tok[:-1])
                break

        gpu: dict = {}
        gpu_raw = kv.get("GPU", "").strip()
        if gpu_raw and gpu_raw[0].isdigit():
            gpu = {"pct": _num(gpu_raw)}

        slurm_queue: dict = {}
        backend_type = cfg.get("backend", "ssh")
        if kv.get("HASSLURM") == "1":
            backend_type = "slurm"
            # "running"/"pending" are YOUR jobs; "*_all" are cluster-wide depth.
            # For a SLURM host the queue is the meaningful signal — the login-node
            # cpu/mem above are not where work runs.
            slurm_queue = {
                "running": int(_num(kv.get("SQR"))),
                "pending": int(_num(kv.get("SQP"))),
                "running_all": int(_num(kv.get("SQRA"))),
                "pending_all": int(_num(kv.get("SQPA"))),
            }
            slurm_queue.update(_parse_sinfo_cpus(kv.get("SINFOC", "")))
            jobs = _parse_slurm_jobs(slurm_job_lines)
            if jobs:
                slurm_queue["jobs"] = jobs

        return Server(
            name=name,
            backend_type=backend_type,
            online=True,
            last_heartbeat=None,            # probe_servers stamps this
            cpu={"pct": round(cpu_pct), "nproc": int(nproc)},
            mem={"pct": round(mem_pct)},
            gpu=gpu,
            disk={"pct": round(disk_pct)},
            slurm_queue=slurm_queue,
            note=cfg.get("note"),
        )
