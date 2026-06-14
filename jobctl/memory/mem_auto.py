"""Helpers for --mem-auto retry decisions."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jobctl.db.models import State

if TYPE_CHECKING:
    from jobctl.db.models import Run
    from jobctl.db.store import Store


_MEM_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[kmgtpeKMGTPE]?)i?[bB]?\s*$")
_UNITS_TO_MB = {
    "": 1.0,
    "K": 1.0 / 1024.0,
    "M": 1.0,
    "G": 1024.0,
    "T": 1024.0 * 1024.0,
    "P": 1024.0 * 1024.0 * 1024.0,
    "E": 1024.0 * 1024.0 * 1024.0 * 1024.0,
}

_GPU_OOM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"cuda\s+out\s+of\s+memory",
        r"outofmemoryerror:.*cuda",
        r"cublas_status_alloc_failed",
        r"cudaerrormemoryallocation",
        r"hiperroroutofmemory",
        r"hip\s+out\s+of\s+memory",
        r"mps\s+backend\s+out\s+of\s+memory",
        r"gpu\s+out\s+of\s+memory",
        r"resource_exhausted:.*gpu",
    )
]

_CPU_OOM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\boom[- ]kill",
        r"oom\s+killed\s+process",
        r"detected\s+\d+\s+oom-kill",
        r"memory\s+cgroup\s+out\s+of\s+memory",
        r"exceeded\s+(?:job\s+)?memory\s+limit",
        r"killed\s+by\s+signal\s+9",
        r"\bmemoryerror\b",
        r"\bout\s+of\s+memory\b",
    )
]


@dataclass(frozen=True)
class OomDiagnosis:
    kind: str | None
    evidence: list[str]

    @property
    def is_oom(self) -> bool:
        return self.kind is not None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "evidence": list(self.evidence)}


def parse_mem_to_mb(value: Any) -> int | None:
    """Parse SLURM-style memory strings into integer MiB.

    Bare numbers are treated as MiB, matching SLURM's default unit for --mem.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value <= 0:
            return None
        return max(1, int(math.ceil(float(value))))
    text = str(value).strip()
    if not text or text in {"0", "0K", "0M", "0G"}:
        return None
    match = _MEM_RE.match(text)
    if not match:
        return None
    num = float(match.group("num"))
    unit = (match.group("unit") or "").upper()
    factor = _UNITS_TO_MB.get(unit)
    if factor is None or num <= 0:
        return None
    return max(1, int(math.ceil(num * factor)))


def format_mem_mb(mb: int | float) -> str:
    """Format memory as an explicit MiB request accepted by SLURM."""
    return f"{max(1, int(math.ceil(float(mb))))}M"


def classify_oom(resource_summary: dict | None, stdout: str = "", stderr: str = "") -> OomDiagnosis:
    """Classify terminal evidence into CPU OOM, GPU OOM, or not OOM.

    GPU evidence wins over generic log text, because many GPU libraries include
    the words "out of memory" and would otherwise look like CPU memory pressure.
    """
    text = "\n".join([stdout or "", stderr or ""])
    evidence: list[str] = []

    for pattern in _GPU_OOM_PATTERNS:
        match = pattern.search(text)
        if match:
            evidence.append(f"GPU OOM evidence: {match.group(0)[:160]}")
            return OomDiagnosis(kind="gpu", evidence=evidence)

    resource = resource_summary or {}
    state_text = str(resource.get("State") or resource.get("state") or "")
    if re.search(r"(OUT_OF_MEMORY|\bOOM\b|OUT OF MEMORY)", state_text, re.IGNORECASE):
        evidence.append(f"SLURM state reports OOM: {state_text}")
        return OomDiagnosis(kind="cpu", evidence=evidence)

    for pattern in _CPU_OOM_PATTERNS:
        match = pattern.search(text)
        if match:
            evidence.append(f"CPU OOM evidence: {match.group(0)[:160]}")
            return OomDiagnosis(kind="cpu", evidence=evidence)

    return OomDiagnosis(kind=None, evidence=[])


def next_mem_request(
    current_mem: Any,
    resource_summary: dict | None = None,
    *,
    factor: float = 1.5,
    cap: Any = None,
) -> str | None:
    """Return the next conservative memory request, or None if it cannot grow."""
    current_mb = parse_mem_to_mb(current_mem)
    if current_mb is None:
        return None

    factor = max(float(factor or 1.5), 1.01)
    target_mb = current_mb * factor

    maxrss_mb = None
    if resource_summary:
        maxrss_mb = parse_mem_to_mb(
            resource_summary.get("MaxRSS")
            or resource_summary.get("max_rss")
            or resource_summary.get("maxrss")
        )
    if maxrss_mb is not None:
        target_mb = max(target_mb, maxrss_mb * factor)

    cap_mb = parse_mem_to_mb(cap)
    if cap_mb is not None:
        target_mb = min(target_mb, cap_mb)

    next_mb = int(math.ceil(target_mb))
    if next_mb <= current_mb:
        return None
    return format_mem_mb(next_mb)


def _apply_cap(candidate_mb: int | None, cap: Any) -> int | None:
    if candidate_mb is None:
        return None
    cap_mb = parse_mem_to_mb(cap)
    if cap_mb is None:
        return candidate_mb
    return min(candidate_mb, cap_mb)


def estimate_mem_from_history(
    store: "Store",
    *,
    jobfile_id: str,
    params: dict,
    input_hashes: dict,
    current_mem: Any = None,
    factor: float = 1.5,
    cap: Any = None,
) -> dict | None:
    """Estimate an initial --mem from prior runs with the same params/inputs."""
    current_mb = parse_mem_to_mb(current_mem) or 0
    best: tuple[int, dict] | None = None

    for run in store.list_runs(jobfile_id=jobfile_id):
        if run.params != params or run.input_hashes != input_hashes:
            continue

        req = run.slurm_request or {}
        req_mb = parse_mem_to_mb(req.get("mem"))
        resource = run.resource_summary or {}
        maxrss_mb = parse_mem_to_mb(
            resource.get("MaxRSS") or resource.get("max_rss") or resource.get("maxrss")
        )

        diagnosis = classify_oom(resource)
        candidate_mb: int | None = None
        reason = None
        if diagnosis.kind == "cpu" and req_mb is not None:
            candidate_mb = parse_mem_to_mb(next_mem_request(req.get("mem"), resource, factor=factor, cap=cap))
            reason = "prior_cpu_oom"
        elif run.state == State.COMPLETED and maxrss_mb is not None:
            candidate_mb = parse_mem_to_mb(next_mem_request(maxrss_mb, None, factor=factor, cap=cap))
            reason = "prior_maxrss"
        elif run.state == State.COMPLETED and req_mb is not None:
            candidate_mb = _apply_cap(req_mb, cap)
            reason = "prior_success_request"

        if candidate_mb is None or candidate_mb <= current_mb:
            continue
        info = {
            "mem": format_mem_mb(candidate_mb),
            "source_run_id": run.run_id,
            "reason": reason,
        }
        if best is None or candidate_mb > best[0]:
            best = (candidate_mb, info)

    return best[1] if best else None
