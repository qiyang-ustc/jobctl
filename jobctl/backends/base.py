"""Backend ABC + result dataclasses + select_backend() + get_backend()."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    if override:
        return (
            override.get("backend", "local"),
            override.get("server"),
            override.get("task"),
        )

    server_map: dict[str, bool] = {s.name: s.online for s in servers}

    for pref in jobfile.backend_prefs:
        backend = pref.get("backend", "local")
        server = pref.get("server")
        task = pref.get("task")

        if server is None:
            # Local or no-server backend — always available
            return (backend, None, task)

        if server_map.get(server, False):
            return (backend, server, task)

    return ("local", None, None)


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
