"""Backend ABC + result dataclasses + select_backend() + get_backend()."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jobctl.db.models import JobFile, Run, Server, State


def resolved_command(run: "Run", jobfile: "JobFile") -> str:
    """Render the jobfile command template with the run's resolved params.

    Backends must execute THIS, not ``jobfile.command_template`` directly —
    otherwise ``{param}`` placeholders are never substituted.
    """
    from jobctl.jobfile import render_command

    return render_command(jobfile, run.params or {})


_GPU_MARKERS = (
    "cuda",
    "nvidia-smi",
    "nvcc",
    "cupy",
    "torch.cuda",
    "--device cuda",
    "device=cuda",
    "gpu",
    "rocm",
    "hipcc",
)
_ACCELERATOR_PARTITIONS = {
    "h100": "gpu_h100",
    "a100": "gpu_a100",
    "l4": "gpu_l4",
    "rtx": "gpu_rtx",
    "mi300a": "gpu_mi300a",
    "mi300": "gpu_mi300",
    "mi250": "gpu_mi250",
}
_SLURM_RESOURCE_KEYS = {"partition", "account", "time", "mem", "cpus", "cpus_per_task", "gres"}


def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            parts.extend(_flatten_text(key))
            parts.extend(_flatten_text(item))
        return parts
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            parts.extend(_flatten_text(item))
        return parts
    return [str(value)]


def _gpu_terms_from_text(text: str) -> set[str]:
    lowered = text.lower()
    terms = {marker for marker in _GPU_MARKERS if marker in lowered}
    if re.search(r"(^|[^a-z0-9])hip([^a-z0-9]|$)", lowered):
        terms.add("hip")
    if re.search(r"(^|[^a-z0-9])rtx([^a-z0-9]|$)", lowered):
        terms.add("rtx")
    for accel in _ACCELERATOR_PARTITIONS:
        if re.search(rf"(^|[^a-z0-9]){re.escape(accel)}([^a-z0-9]|$)", lowered):
            terms.add(accel)
    return terms


def _gpu_terms_from_values(*values: Any) -> set[str]:
    terms: set[str] = set()
    for value in values:
        for text in _flatten_text(value):
            terms.update(_gpu_terms_from_text(text))
    return terms


def slurm_resources_request_gpu(resources: dict | None) -> bool:
    """Return True when a SLURM resource dict explicitly asks for a GPU path."""
    res = resources or {}
    return bool(_gpu_terms_from_values(res.get("gres"), res.get("partition"), res.get("constraint")))


def jobfile_gpu_terms(
    jobfile: "JobFile",
    *,
    params: dict | None = None,
    resources: dict | None = None,
    metadata: dict | None = None,
) -> set[str]:
    """Detect conservative GPU intent from a JobFile and submission metadata."""
    return _gpu_terms_from_values(
        getattr(jobfile, "name", ""),
        getattr(jobfile, "command_template", ""),
        getattr(jobfile, "params_schema", {}),
        getattr(jobfile, "artifact_patterns", []),
        params or {},
        resources or {},
        metadata or {},
    )


def jobfile_requires_gpu(
    jobfile: "JobFile",
    *,
    params: dict | None = None,
    resources: dict | None = None,
    metadata: dict | None = None,
) -> bool:
    """Return True when the submission appears to need an accelerator."""
    return bool(jobfile_gpu_terms(jobfile, params=params, resources=resources, metadata=metadata))


def backend_pref_slurm_resources(pref: dict | None) -> dict:
    """Extract SLURM resource defaults embedded in a backend preference."""
    if not isinstance(pref, dict):
        return {}
    resources: dict = {}
    nested = pref.get("resources")
    if isinstance(nested, dict):
        resources.update({k: v for k, v in nested.items() if v is not None})
    for key in _SLURM_RESOURCE_KEYS:
        if key in pref and pref[key] is not None:
            resources[key] = pref[key]
    return resources


def _server_config(config: dict | None, server: str | None) -> dict:
    if not server:
        return {}
    servers_cfg = (config or {}).get("servers") or {}
    cfg = servers_cfg.get(server) or {}
    return cfg if isinstance(cfg, dict) else {}


def _server_supports_gpu(server: "Server | None", server_cfg: dict) -> bool:
    gpu = getattr(server, "gpu", None) if server is not None else None
    if isinstance(gpu, dict) and gpu:
        return True
    return bool(
        _gpu_terms_from_values(
            getattr(server, "name", ""),
            getattr(server, "note", ""),
            server_cfg.get("gres"),
            server_cfg.get("partition"),
            server_cfg.get("partitions"),
            server_cfg.get("tasks"),
            server_cfg.get("gpu"),
            server_cfg.get("gpu_defaults"),
            server_cfg.get("gpu_resources"),
            server_cfg.get("default_gpu_resources"),
            server_cfg.get("accelerators"),
        )
    )


def _pref_supports_gpu(pref: dict, server_map: dict[str, "Server"], config: dict | None) -> bool:
    pref_resources = backend_pref_slurm_resources(pref)
    if slurm_resources_request_gpu(pref_resources):
        return True
    if _gpu_terms_from_values(pref.get("task"), pref.get("server"), pref.get("name")):
        return True
    server = pref.get("server")
    return _server_supports_gpu(server_map.get(server), _server_config(config, server))


def _server_online(server_name: str | None, server_map: dict[str, "Server"]) -> bool:
    if server_name is None:
        return True
    server = server_map.get(server_name)
    return bool(server and server.online)


def _pref_available(pref: dict, server_map: dict[str, "Server"]) -> bool:
    server = pref.get("server")
    return _server_online(server, server_map)


def select_backend_pref(
    jobfile: "JobFile",
    servers: "list[Server]",
    override: dict | None,
    *,
    resources: dict | None = None,
    params: dict | None = None,
    config: dict | None = None,
    metadata: dict | None = None,
) -> "tuple[str, str | None, str | None, dict | None]":
    """Choose a backend and return the selected backend preference as well.

    GPU-looking jobs are kept away from local/CPU-only fallbacks when possible.
    Explicit overrides still win; otherwise the selector first looks for an
    online GPU-capable preference, then falls back to the first online non-local
    preference so the submit path can fail fast with a resource message instead
    of silently running CUDA/HIP work on a CPU partition.
    """
    if override:
        pref = dict(override)
        return (
            pref.get("backend", "local"),
            pref.get("server"),
            pref.get("task"),
            pref,
        )

    server_map: dict[str, "Server"] = {s.name: s for s in servers}
    prefs = list(jobfile.backend_prefs or [])
    needs_gpu = jobfile_requires_gpu(
        jobfile,
        params=params,
        resources=resources,
        metadata=metadata,
    )

    if needs_gpu:
        for pref in prefs:
            if _pref_available(pref, server_map) and _pref_supports_gpu(pref, server_map, config):
                return (pref.get("backend", "local"), pref.get("server"), pref.get("task"), pref)
        for pref in prefs:
            backend = pref.get("backend", "local")
            if backend != "local" and _pref_available(pref, server_map):
                return (backend, pref.get("server"), pref.get("task"), pref)
        online_gpu = [
            server
            for server in servers
            if server.online and _server_supports_gpu(server, _server_config(config, server.name))
        ]
        if online_gpu:
            online_gpu.sort(key=lambda s: 0 if s.backend_type == "slurm" else 1)
            server = online_gpu[0]
            return (server.backend_type or "slurm", server.name, None, {"backend": server.backend_type, "server": server.name})

    for pref in prefs:
        if _pref_available(pref, server_map):
            return (pref.get("backend", "local"), pref.get("server"), pref.get("task"), pref)

    return ("local", None, None, None)


def _dict_resource_defaults(raw: Any) -> dict:
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if k in _SLURM_RESOURCE_KEYS and v is not None}


def _partition_gres_from_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("gpu:"):
        return text.split()[0]
    return None


def _choose_gpu_partition(partitions: Any, terms: set[str]) -> tuple[str | None, str | None]:
    if not isinstance(partitions, dict):
        return (None, None)
    accelerator_terms = [term for term in _ACCELERATOR_PARTITIONS if term in terms]
    entries = [(str(k), v) for k, v in partitions.items()]
    for wanted in accelerator_terms + ["gpu", "cuda"]:
        for name, value in entries:
            haystack = " ".join(_flatten_text([name, value])).lower()
            if wanted in haystack:
                return (name, _partition_gres_from_value(value))
    for name, value in entries:
        haystack = " ".join(_flatten_text([name, value])).lower()
        if _gpu_terms_from_text(haystack):
            return (name, _partition_gres_from_value(value))
    return (None, None)


def infer_gpu_slurm_resources(
    jobfile: "JobFile",
    *,
    params: dict | None = None,
    resources: dict | None = None,
    backend: str | None = None,
    server: str | None = None,
    task: str | None = None,
    selected_pref: dict | None = None,
    config: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    """Merge backend-pref resources and infer missing GPU SLURM defaults.

    This keeps CUDA/HIP/H100-style jobs from falling through to a cluster's CPU
    default partition (for example SLURM defaulting to ``rome`` when no
    partition/GRES was specified).
    """
    merged = dict(backend_pref_slurm_resources(selected_pref))
    merged.update(resources or {})
    if backend != "slurm":
        return merged

    terms = jobfile_gpu_terms(jobfile, params=params, resources=merged, metadata=metadata)
    if not terms:
        return merged
    if slurm_resources_request_gpu(merged):
        if "gres" not in merged and _gpu_terms_from_values(merged.get("partition")):
            merged["gres"] = "gpu:1"
        return merged

    server_cfg = _server_config(config, server)

    task_cfg = None
    if task:
        tasks = server_cfg.get("tasks") or {}
        task_cfg = tasks.get(task)
        if not isinstance(task_cfg, dict):
            task_cfg = ((config or {}).get("tasks") or {}).get(task)
    merged.update(_dict_resource_defaults(task_cfg))

    for key in ("accelerators", "gpu_defaults", "gpu_resources", "default_gpu_resources", "gpu"):
        raw = server_cfg.get(key)
        if key == "accelerators" and isinstance(raw, dict):
            for term in terms:
                if isinstance(raw.get(term), dict):
                    merged.update(_dict_resource_defaults(raw[term]))
                    break
        else:
            merged.update(_dict_resource_defaults(raw))

    if not slurm_resources_request_gpu(merged):
        partition, gres = _choose_gpu_partition(server_cfg.get("partitions"), terms)
        if partition:
            merged.setdefault("partition", partition)
        if gres:
            merged.setdefault("gres", gres)

    if not slurm_resources_request_gpu(merged):
        for term, partition in _ACCELERATOR_PARTITIONS.items():
            if term in terms:
                merged.setdefault("partition", partition)
                merged.setdefault("gres", "gpu:1")
                break

    if not slurm_resources_request_gpu(merged):
        raise ValueError(
            f"JobFile {jobfile.name!r} appears to require a GPU, but selected SLURM "
            f"server {server!r} has no GPU partition/GRES defaults. Pass --partition "
            "--gres or configure GPU defaults for that server."
        )

    if "gres" not in merged:
        merged["gres"] = "gpu:1"
    return merged


@dataclass
class SubmitResult:
    """Result returned by Backend.submit()."""
    remote_job_id: str | None
    workdir: str
    # Resolved SLURM submission request (None for non-SLURM backends).
    slurm_request: dict | None = None


@dataclass
class PollResult:
    """Result returned by Backend.poll().

    reachable=False means the backend could NOT determine the job's state
    (SSH/cluster unreachable, command timed out). The monitor must treat this
    as "unknown / no heartbeat" and NOT transition the run to failed/stuck —
    conflating "I can't see the job" with "the job died" was the core bug
    behind false 'stuck' classifications during VPN/SSH blips.
    """
    state: "State"
    resource: dict
    last_log_mtime: float | None = None
    reachable: bool = True


@dataclass
class CollectResult:
    """Result returned by Backend.collect()."""
    exit_code: int | None
    stdout_path: str
    stderr_path: str
    artifact_dir: str
    resource_summary: dict = field(default_factory=dict)


class Backend(ABC):
    """Abstract base class for all execution backends."""

    name: str

    @abstractmethod
    def submit(self, run: "Run", jobfile: "JobFile") -> SubmitResult:
        """Submit the job; returns remote_job_id and workdir."""

    @abstractmethod
    def poll(self, run: "Run") -> PollResult:
        """Poll job status; returns current State + resource snapshot."""

    @abstractmethod
    def collect(self, run: "Run") -> CollectResult:
        """Collect terminal results: stdout/stderr paths, artifact dir, exit code."""

    @abstractmethod
    def cancel(self, run: "Run") -> None:
        """Cancel/kill the job."""

    def poll_many(self, runs: "list[Run]") -> "dict[str, PollResult]":
        """Poll several runs at once, returning {run_id: PollResult}.

        Default = poll each individually. Backends that talk to a scheduler
        (SLURM) override this to make ONE query per cycle instead of one SSH
        per run (the connection storm that throttled the login node).
        """
        return {run.run_id: self.poll(run) for run in runs}


# ---------------------------------------------------------------------------
# Backend selector
# ---------------------------------------------------------------------------

def select_backend(
    jobfile: "JobFile",
    servers: "list[Server]",
    override: dict | None,
) -> "tuple[str, str | None, str | None]":
    """Choose (backend, server, task) based on preferences + server health.

    Priority:
    1. If *override* dict is provided, use it directly.
    2. Walk ``jobfile.backend_prefs`` in order:
       - If the pref has no ``server`` key (local), pick it immediately.
       - Otherwise pick it only if the named server is online.
    3. Fall back to ``("local", None, None)``.

    Returns:
        Tuple of (backend_name, server_name_or_None, task_or_None).
    """
    backend, server, task, _pref = select_backend_pref(jobfile, servers, override)
    return (backend, server, task)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def get_backend(backend: str, server: str | None, config: dict) -> Backend:
    """Instantiate and return the correct Backend implementation.

    Args:
        backend:  One of "local", "ssh", "slurm".
        server:   Server name (required for ssh/slurm).
        config:   Config dict (from jobctl.config) — used to look up server
                  settings such as host, user, remote_path.

    Raises:
        ValueError: if *backend* is not recognised.
    """
    if backend == "local":
        from jobctl.backends.local import LocalBackend
        return LocalBackend(workdir_root=config.get("run_dir"))

    if backend in ("ssh", "slurm"):
        servers_cfg: dict = config.get("servers", {})
        server_config: dict = dict(servers_cfg.get(server, {}) if server else {})
        if config.get("run_dir"):
            server_config.setdefault("run_dir", config["run_dir"])

    if backend == "ssh":
        from jobctl.backends.ssh import SshBackend
        return SshBackend(server=server, server_config=server_config)

    if backend == "slurm":
        from jobctl.backends.slurm import SlurmBackend
        return SlurmBackend(server=server, server_config=server_config)

    raise ValueError(f"Unknown backend: {backend!r}")
