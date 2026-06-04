"""Database models: enums, dataclasses, and DDL strings."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class State(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STUCK = "stuck"
    TIMEOUT = "timeout"


class Health(str, Enum):
    OK = "ok"
    WEAK = "weak"
    NO_HEARTBEAT = "no_heartbeat"
    RESOURCE_PRESSURE = "resource_pressure"
    STUCK = "stuck"


class Match(str, Enum):
    USABLE = "usable"
    WEAK_SIGNAL = "weak_signal"
    BAD_SIGNAL = "bad_signal"
    INCONCLUSIVE = "inconclusive"
    FAILED = "failed"


class ArtifactType(str, Enum):
    IMAGE = "image"
    PLOT = "plot"
    CSV = "csv"
    JSON = "json"
    TEXT_LOG = "text_log"
    BINARY = "binary"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class JobFile:
    id: str
    name: str
    version: int
    source_path: str
    command_template: str
    params_schema: dict
    backend_prefs: list[dict]
    artifact_patterns: list[str]
    expectation_contract_id: str | None
    content_hash: str
    created_at: str


@dataclass
class Run:
    run_id: str
    jobfile_id: str
    jobfile_version: int
    params: dict
    input_hashes: dict
    backend: str
    server: str | None
    task: str | None
    remote_job_id: str | None
    state: State
    health: Health
    exit_code: int | None
    submitted_at: str | None
    started_at: str | None
    finished_at: str | None
    last_heartbeat: str | None
    workdir: str | None
    stdout_path: str | None
    stderr_path: str | None
    resource_summary: dict
    expectation_match: Match | None
    observation_card: dict | None
    # Resolved SLURM submission request (partition/account/time/mem/cpus + job_id).
    # None for non-SLURM backends. Surfaced in the run-detail panel.
    slurm_request: dict | None = None


@dataclass
class Artifact:
    id: str
    run_id: str
    remote_path: str
    local_path: str
    type: ArtifactType
    size: int
    checksum: str
    preview: dict
    created_at: str


@dataclass
class Criterion:
    id: str
    text: str
    kind: str  # "numeric|presence|absence|pattern"
    check: dict
    status: str  # "proposed|confirmed"
    strength: int
    evidence_run_ids: list[str]


@dataclass
class ExpectationContract:
    id: str
    jobfile_id: str
    version: int
    criteria: list[Criterion]
    source: str
    created_at: str
    updated_at: str


@dataclass
class Feedback:
    id: str
    run_id: str
    kind: str  # "accept|reject|note"
    text: str
    created_at: str


@dataclass
class Server:
    name: str
    backend_type: str
    online: bool
    last_heartbeat: str | None
    cpu: dict
    mem: dict
    gpu: dict
    disk: dict
    slurm_queue: dict
    note: str | None


# ---------------------------------------------------------------------------
# DDL strings
# ---------------------------------------------------------------------------

DDL_JOBFILES = """
CREATE TABLE IF NOT EXISTS jobfiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    source_path TEXT,
    command_template TEXT,
    params_schema TEXT NOT NULL DEFAULT '{}',
    backend_prefs TEXT NOT NULL DEFAULT '[]',
    artifact_patterns TEXT NOT NULL DEFAULT '[]',
    expectation_contract_id TEXT,
    content_hash TEXT,
    created_at TEXT
)
"""

DDL_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    jobfile_id TEXT NOT NULL,
    jobfile_version INTEGER NOT NULL DEFAULT 1,
    params TEXT NOT NULL DEFAULT '{}',
    input_hashes TEXT NOT NULL DEFAULT '{}',
    backend TEXT,
    server TEXT,
    task TEXT,
    remote_job_id TEXT,
    state TEXT NOT NULL DEFAULT 'pending',
    health TEXT NOT NULL DEFAULT 'ok',
    exit_code INTEGER,
    submitted_at TEXT,
    started_at TEXT,
    finished_at TEXT,
    last_heartbeat TEXT,
    workdir TEXT,
    stdout_path TEXT,
    stderr_path TEXT,
    resource_summary TEXT NOT NULL DEFAULT '{}',
    expectation_match TEXT,
    observation_card TEXT,
    slurm_request TEXT,
    FOREIGN KEY (jobfile_id) REFERENCES jobfiles(id)
)
"""

DDL_ARTIFACTS = """
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    remote_path TEXT,
    local_path TEXT,
    type TEXT,
    size INTEGER,
    checksum TEXT,
    preview TEXT NOT NULL DEFAULT '{}',
    created_at TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
)
"""

DDL_EXPECTATION_CONTRACTS = """
CREATE TABLE IF NOT EXISTS expectation_contracts (
    id TEXT PRIMARY KEY,
    jobfile_id TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    criteria TEXT NOT NULL DEFAULT '[]',
    source TEXT,
    created_at TEXT,
    updated_at TEXT,
    FOREIGN KEY (jobfile_id) REFERENCES jobfiles(id)
)
"""

DDL_FEEDBACK = """
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    kind TEXT,
    text TEXT,
    created_at TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
)
"""

DDL_SERVERS = """
CREATE TABLE IF NOT EXISTS servers (
    name TEXT PRIMARY KEY,
    backend_type TEXT,
    online INTEGER NOT NULL DEFAULT 0,
    last_heartbeat TEXT,
    cpu TEXT NOT NULL DEFAULT '{}',
    mem TEXT NOT NULL DEFAULT '{}',
    gpu TEXT NOT NULL DEFAULT '{}',
    disk TEXT NOT NULL DEFAULT '{}',
    slurm_queue TEXT NOT NULL DEFAULT '{}',
    note TEXT
)
"""

ALL_DDL = [
    DDL_JOBFILES,
    DDL_RUNS,
    DDL_ARTIFACTS,
    DDL_EXPECTATION_CONTRACTS,
    DDL_FEEDBACK,
    DDL_SERVERS,
]
