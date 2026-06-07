"""jobctl configuration loader.

Loads:
- ~/.cluster.yaml (or a given path): servers, tasks, remote_path
- $JOBCTL_HOME/config.toml or ~/.jobctl/config.toml: jobctl daemon settings
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Unified configuration object."""

    # From cluster.yaml
    servers: dict[str, dict] = field(default_factory=dict)
    tasks: dict[str, dict] = field(default_factory=dict)
    remote_path: str = ""

    # jobctl daemon settings
    cluster_yaml_path: str = ""
    jobctl_config_path: str = ""
    state_root: str = ""
    db_path: str = ""
    run_dir: str = ""
    daemon_port: int = 7421
    daemon_host: str = "127.0.0.1"
    default_policies: dict[str, dict] = field(default_factory=dict)

    # Optional analysis
    deepseek_api_key: str = ""
    gemini_api_key: str = ""

    # Desktop notifications (macOS). Enabled by default; a silent no-op off macOS.
    notify_macos_enabled: bool = True
    notify_sound: str = ""  # e.g. "Glass"; empty => silent banner
    notify_window_seconds: float = 15.0  # coalesce a burst into one "series" banner


_DEFAULT_CLUSTER_YAML = os.path.expanduser("~/.cluster.yaml")


def _expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def default_state_root() -> str:
    """Return the configured jobctl state root.

    ``JOBCTL_HOME`` is the single environment override used by logs, DB, run
    mirrors, and generated config paths. Keeping this centralized prevents
    subagents/sandboxed sessions from accidentally writing to ``~/.jobctl``.
    """
    return _expand_path(os.environ.get("JOBCTL_HOME", "~/.jobctl"))


def default_jobctl_config_path(state_root: str | None = None) -> str:
    return str(Path(state_root or default_state_root()) / "config.toml")


def default_run_dir(state_root: str | None = None) -> str:
    return str(Path(state_root or default_state_root()) / "runs")


def default_db_path(state_root: str | None = None) -> str:
    return str(Path(state_root or default_state_root()) / "jobctl.db")


def load_config(
    cluster_yaml_path: str | None = None,
    jobctl_config_path: str | None = None,
) -> Config:
    """Load and merge configuration from cluster.yaml and optional config.toml.

    Args:
        cluster_yaml_path: Path to cluster.yaml. Defaults to ~/.cluster.yaml.
        jobctl_config_path: Path to config.toml. Defaults to ~/.jobctl/config.toml.

    Returns:
        Populated Config dataclass with sane defaults when files are absent.
    """
    cluster_yaml_path = cluster_yaml_path or _DEFAULT_CLUSTER_YAML
    state_root = default_state_root()
    jobctl_config_path = jobctl_config_path or os.environ.get("JOBCTL_CONFIG") or default_jobctl_config_path(state_root)

    # Start with defaults
    cfg = Config(
        cluster_yaml_path=_expand_path(cluster_yaml_path),
        jobctl_config_path=_expand_path(jobctl_config_path),
        state_root=state_root,
        db_path=default_db_path(state_root),
        run_dir=default_run_dir(state_root),
        remote_path=os.path.expanduser("~/jobctl-remote"),
    )

    # Load cluster.yaml
    cluster_path = Path(cluster_yaml_path)
    if cluster_path.exists():
        with open(cluster_path) as f:
            data = yaml.safe_load(f) or {}
        cfg.servers = data.get("servers", {}) or {}
        cfg.tasks = data.get("tasks", {}) or {}
        if "remote_path" in data:
            cfg.remote_path = data["remote_path"]

    # Load jobctl config.toml
    config_path = Path(jobctl_config_path)
    if config_path.exists():
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore
            except ImportError:
                tomllib = None  # type: ignore

        if tomllib is not None:
            with open(config_path, "rb") as f:
                toml_data = tomllib.load(f)
            jc = toml_data.get("jobctl", {})
            if "state_root" in jc:
                cfg.state_root = _expand_path(str(jc["state_root"]))
                cfg.db_path = default_db_path(cfg.state_root)
                cfg.run_dir = default_run_dir(cfg.state_root)
            if "db_path" in jc:
                cfg.db_path = _expand_path(str(jc["db_path"]))
            if "run_dir" in jc:
                cfg.run_dir = _expand_path(str(jc["run_dir"]))
            if "daemon_port" in jc:
                cfg.daemon_port = int(jc["daemon_port"])
            if "daemon_host" in jc:
                cfg.daemon_host = jc["daemon_host"]
            if "default_policies" in jc and isinstance(jc["default_policies"], dict):
                cfg.default_policies = jc["default_policies"]
            if "notify_macos_enabled" in jc:
                cfg.notify_macos_enabled = bool(jc["notify_macos_enabled"])
            if "notify_sound" in jc:
                cfg.notify_sound = str(jc["notify_sound"])
            if "notify_window_seconds" in jc:
                cfg.notify_window_seconds = float(jc["notify_window_seconds"])

    # Override from environment
    if os.environ.get("DEEPSEEK_API_KEY"):
        cfg.deepseek_api_key = os.environ["DEEPSEEK_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        cfg.gemini_api_key = os.environ["GEMINI_API_KEY"]

    return cfg
