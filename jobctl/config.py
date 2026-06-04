"""jobctl configuration loader.

Loads:
- ~/.cluster.yaml (or a given path): servers, tasks, remote_path
- ~/.jobctl/config.toml (or a given path): jobctl daemon settings
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
    db_path: str = ""
    run_dir: str = ""
    daemon_port: int = 7421
    daemon_host: str = "127.0.0.1"

    # Optional analysis
    deepseek_api_key: str = ""


_DEFAULT_CLUSTER_YAML = os.path.expanduser("~/.cluster.yaml")
_DEFAULT_JOBCTL_CONFIG = os.path.expanduser("~/.jobctl/config.toml")
_DEFAULT_RUN_DIR = os.path.expanduser("~/.jobctl/runs")
_DEFAULT_DB_PATH = os.path.expanduser("~/.jobctl/jobctl.db")


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
    jobctl_config_path = jobctl_config_path or _DEFAULT_JOBCTL_CONFIG

    # Start with defaults
    cfg = Config(
        db_path=_DEFAULT_DB_PATH,
        run_dir=_DEFAULT_RUN_DIR,
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
            if "db_path" in jc:
                cfg.db_path = jc["db_path"]
            if "run_dir" in jc:
                cfg.run_dir = jc["run_dir"]
            if "daemon_port" in jc:
                cfg.daemon_port = int(jc["daemon_port"])
            if "daemon_host" in jc:
                cfg.daemon_host = jc["daemon_host"]

    # Override from environment
    if os.environ.get("DEEPSEEK_API_KEY"):
        cfg.deepseek_api_key = os.environ["DEEPSEEK_API_KEY"]

    return cfg
