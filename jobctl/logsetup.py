"""Central file logging so every jobctl invocation leaves a trail.

Both the CLI and the daemon call ``configure_logging`` at startup, writing to
``~/.jobctl/<component>.log`` (rotating). This is what makes "check the logs"
actually work — previously the daemon was spawned with stdout/stderr → DEVNULL
and nothing was ever recorded.
"""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED: set[str] = set()


def log_dir() -> Path:
    """Return (creating if needed) the directory holding jobctl logs."""
    d = Path(os.environ.get("JOBCTL_HOME", os.path.expanduser("~/.jobctl")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_path(component: str) -> Path:
    """Path of the log file for a component ('cli' or 'daemon')."""
    return log_dir() / f"{component}.log"


def configure_logging(component: str, level: int = logging.INFO) -> Path:
    """Attach a rotating file handler for *component* to the root logger.

    Idempotent per component within a process. Returns the log file path.
    """
    path = log_path(component)
    if component in _CONFIGURED:
        return path

    handler = RotatingFileHandler(
        path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter(
            f"%(asctime)s %(levelname)s [{component}:%(process)d] %(name)s: %(message)s"
        )
    )
    root = logging.getLogger()
    if root.level == logging.WARNING or root.level == 0:  # default/unset
        root.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED.add(component)
    return path
