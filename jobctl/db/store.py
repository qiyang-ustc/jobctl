"""Store: SQLite repository. The ONLY writer for all tables.

JSON columns (params, input_hashes, etc.) are serialized/deserialized here.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from jobctl.db.models import (
    ALL_DDL,
    Artifact,
    ArtifactType,
    Criterion,
    DDL_RUN_PARENT_RETRY_INDEX,
    ExpectationContract,
    Feedback,
    Health,
    JobFile,
    Match,
    Run,
    Server,
    State,
)


def _j(obj: Any) -> str:
    """Serialize to JSON string."""
    return json.dumps(obj)


def _dj(s: str | None, default: Any = None) -> Any:
    """Deserialize JSON string, returning default if None or empty."""
    if s is None:
        return default
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return default


class Store:
    """SQLite repository. Single-writer, handles all tables."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def init_schema(self) -> None:
        """Create all tables if they don't exist, then run lightweight migrations."""
        conn = self._get_conn()
        for ddl in ALL_DDL:
            conn.execute(ddl)
        conn.commit()
        self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Additive column migrations for DBs created before a field existed."""
        run_cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        if "slurm_request" not in run_cols:
            conn.execute("ALTER TABLE runs ADD COLUMN slurm_request TEXT")
        for col in ("title", "note", "tags", "parent_run_id", "auto_policy"):
            if col not in run_cols:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col} TEXT")
        if "attempt" not in run_cols:
            conn.execute("ALTER TABLE runs ADD COLUMN attempt INTEGER NOT NULL DEFAULT 1")
        conn.execute(DDL_RUN_PARENT_RETRY_INDEX)
        conn.commit()

    # ------------------------------------------------------------------
    # JobFile
    # ------------------------------------------------------------------

    def add_jobfile(self, jf: JobFile) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO jobfiles
                (id, name, version, source_path, command_template,
                 params_schema, backend_prefs, artifact_patterns,
                 expectation_contract_id, content_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                jf.id, jf.name, jf.version, jf.source_path, jf.command_template,
                _j(jf.params_schema), _j(jf.backend_prefs), _j(jf.artifact_patterns),
                jf.expectation_contract_id, jf.content_hash, jf.created_at,
            ),
        )
        conn.commit()

    def get_jobfile(self, jobfile_id: str) -> JobFile | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM jobfiles WHERE id = ?", (jobfile_id,)
        ).fetchone()
        return self._row_to_jobfile(row) if row else None

    def get_jobfile_by_name(self, name: str) -> JobFile | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM jobfiles WHERE name = ? ORDER BY version DESC LIMIT 1",
            (name,),
        ).fetchone()
        return self._row_to_jobfile(row) if row else None

    def list_jobfiles(self) -> list[JobFile]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM jobfiles ORDER BY created_at").fetchall()
        return [self._row_to_jobfile(r) for r in rows]

    def bump_version(self, jobfile_id: str, new_content_hash: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE jobfiles SET version = version + 1, content_hash = ? WHERE id = ?",
            (new_content_hash, jobfile_id),
        )
        conn.commit()

    def update_jobfile_revision(self, jobfile_id: str, jf: JobFile) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            UPDATE jobfiles SET
                version = version + 1,
                source_path = ?,
                command_template = ?,
                params_schema = ?,
                backend_prefs = ?,
                artifact_patterns = ?,
                content_hash = ?
            WHERE id = ?
            """,
            (
                jf.source_path,
                jf.command_template,
                _j(jf.params_schema),
                _j(jf.backend_prefs),
                _j(jf.artifact_patterns),
                jf.content_hash,
                jobfile_id,
            ),
        )
        conn.commit()

    def _row_to_jobfile(self, row: sqlite3.Row) -> JobFile:
        return JobFile(
            id=row["id"],
            name=row["name"],
            version=row["version"],
            source_path=row["source_path"],
            command_template=row["command_template"],
            params_schema=_dj(row["params_schema"], {}),
            backend_prefs=_dj(row["backend_prefs"], []),
            artifact_patterns=_dj(row["artifact_patterns"], []),
            expectation_contract_id=row["expectation_contract_id"],
            content_hash=row["content_hash"],
            created_at=row["created_at"],
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def add_run(self, run: Run) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO runs
                (run_id, jobfile_id, jobfile_version, params, input_hashes,
                 backend, server, task, remote_job_id, state, health,
                 exit_code, submitted_at, started_at, finished_at, last_heartbeat,
                 workdir, stdout_path, stderr_path, resource_summary,
                 expectation_match, observation_card, slurm_request,
                 title, note, tags, parent_run_id, attempt, auto_policy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id, run.jobfile_id, run.jobfile_version,
                _j(run.params), _j(run.input_hashes),
                run.backend, run.server, run.task, run.remote_job_id,
                run.state.value if isinstance(run.state, State) else run.state,
                run.health.value if isinstance(run.health, Health) else run.health,
                run.exit_code, run.submitted_at, run.started_at, run.finished_at,
                run.last_heartbeat, run.workdir, run.stdout_path, run.stderr_path,
                _j(run.resource_summary),
                run.expectation_match.value if isinstance(run.expectation_match, Match) else run.expectation_match,
                _j(run.observation_card) if run.observation_card is not None else None,
                _j(run.slurm_request) if run.slurm_request is not None else None,
                run.title, run.note,
                _j(run.tags) if run.tags is not None else None,
                run.parent_run_id,
                run.attempt,
                _j(run.auto_policy) if run.auto_policy is not None else None,
            ),
        )
        conn.commit()

    def get_run(self, run_id: str) -> Run | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return self._row_to_run(row) if row else None

    def update_run(self, run_id: str, **kwargs) -> None:
        """Update specific fields on a run.

        Supported kwargs: state, health, exit_code, started_at, finished_at,
        last_heartbeat, workdir, stdout_path, stderr_path, remote_job_id,
        resource_summary, expectation_match, observation_card, slurm_request,
        title, note, tags, parent_run_id, attempt, auto_policy.
        """
        if not kwargs:
            return

        conn = self._get_conn()
        set_parts = []
        values = []

        for key, val in kwargs.items():
            if key in ("state",):
                val = val.value if isinstance(val, State) else val
            elif key in ("health",):
                val = val.value if isinstance(val, Health) else val
            elif key in ("expectation_match",):
                val = val.value if isinstance(val, Match) else val
            elif key in ("resource_summary", "observation_card", "slurm_request", "tags", "auto_policy"):
                val = _j(val) if val is not None else None

            set_parts.append(f"{key} = ?")
            values.append(val)

        values.append(run_id)
        sql = f"UPDATE runs SET {', '.join(set_parts)} WHERE run_id = ?"
        conn.execute(sql, values)
        conn.commit()

    def list_runs(
        self,
        state: State | None = None,
        jobfile_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> list[Run]:
        conn = self._get_conn()
        conditions = []
        params: list[Any] = []

        if state is not None:
            conditions.append("state = ?")
            params.append(state.value if isinstance(state, State) else state)

        if jobfile_id is not None:
            conditions.append("jobfile_id = ?")
            params.append(jobfile_id)

        if parent_run_id is not None:
            conditions.append("parent_run_id = ?")
            params.append(parent_run_id)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = conn.execute(
            f"SELECT * FROM runs {where} ORDER BY submitted_at", params
        ).fetchall()
        return [self._row_to_run(r) for r in rows]

    def _row_to_run(self, row: sqlite3.Row) -> Run:
        return Run(
            run_id=row["run_id"],
            jobfile_id=row["jobfile_id"],
            jobfile_version=row["jobfile_version"],
            params=_dj(row["params"], {}),
            input_hashes=_dj(row["input_hashes"], {}),
            backend=row["backend"],
            server=row["server"],
            task=row["task"],
            remote_job_id=row["remote_job_id"],
            state=State(row["state"]),
            health=Health(row["health"]),
            exit_code=row["exit_code"],
            submitted_at=row["submitted_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            last_heartbeat=row["last_heartbeat"],
            workdir=row["workdir"],
            stdout_path=row["stdout_path"],
            stderr_path=row["stderr_path"],
            resource_summary=_dj(row["resource_summary"], {}),
            expectation_match=Match(row["expectation_match"]) if row["expectation_match"] else None,
            observation_card=_dj(row["observation_card"]),
            slurm_request=_dj(row["slurm_request"], None) if "slurm_request" in row.keys() else None,
            title=row["title"] if "title" in row.keys() else None,
            note=row["note"] if "note" in row.keys() else None,
            tags=_dj(row["tags"], None) if "tags" in row.keys() else None,
            parent_run_id=row["parent_run_id"] if "parent_run_id" in row.keys() else None,
            attempt=row["attempt"] if "attempt" in row.keys() and row["attempt"] is not None else 1,
            auto_policy=_dj(row["auto_policy"], None) if "auto_policy" in row.keys() else None,
        )

    # ------------------------------------------------------------------
    # Artifact
    # ------------------------------------------------------------------

    def add_artifact(self, artifact: Artifact) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO artifacts
                (id, run_id, remote_path, local_path, type, size, checksum, preview, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.id, artifact.run_id, artifact.remote_path, artifact.local_path,
                artifact.type.value if isinstance(artifact.type, ArtifactType) else artifact.type,
                artifact.size, artifact.checksum,
                _j(artifact.preview), artifact.created_at,
            ),
        )
        conn.commit()

    def list_artifacts(self, run_id: str) -> list[Artifact]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        ).fetchall()
        return [self._row_to_artifact(r) for r in rows]

    def _row_to_artifact(self, row: sqlite3.Row) -> Artifact:
        return Artifact(
            id=row["id"],
            run_id=row["run_id"],
            remote_path=row["remote_path"],
            local_path=row["local_path"],
            type=ArtifactType(row["type"]) if row["type"] else ArtifactType.OTHER,
            size=row["size"],
            checksum=row["checksum"],
            preview=_dj(row["preview"], {}),
            created_at=row["created_at"],
        )

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------

    def upsert_server(self, server: Server) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO servers
                (name, backend_type, online, last_heartbeat, cpu, mem, gpu, disk, slurm_queue, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                backend_type = excluded.backend_type,
                online = excluded.online,
                last_heartbeat = excluded.last_heartbeat,
                cpu = excluded.cpu,
                mem = excluded.mem,
                gpu = excluded.gpu,
                disk = excluded.disk,
                slurm_queue = excluded.slurm_queue,
                note = excluded.note
            """,
            (
                server.name,
                server.backend_type,
                1 if server.online else 0,
                server.last_heartbeat,
                _j(server.cpu), _j(server.mem), _j(server.gpu), _j(server.disk),
                _j(server.slurm_queue),
                server.note,
            ),
        )
        conn.commit()

    def list_servers(self) -> list[Server]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM servers ORDER BY name").fetchall()
        return [self._row_to_server(r) for r in rows]

    def get_server(self, name: str) -> Server | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM servers WHERE name = ?", (name,)
        ).fetchone()
        return self._row_to_server(row) if row else None

    def _row_to_server(self, row: sqlite3.Row) -> Server:
        return Server(
            name=row["name"],
            backend_type=row["backend_type"],
            online=bool(row["online"]),
            last_heartbeat=row["last_heartbeat"],
            cpu=_dj(row["cpu"], {}),
            mem=_dj(row["mem"], {}),
            gpu=_dj(row["gpu"], {}),
            disk=_dj(row["disk"], {}),
            slurm_queue=_dj(row["slurm_queue"], {}),
            note=row["note"],
        )

    # ------------------------------------------------------------------
    # ExpectationContract
    # ------------------------------------------------------------------

    def save_contract(self, contract: ExpectationContract) -> None:
        """Insert or replace a contract (upsert by id)."""
        conn = self._get_conn()
        criteria_data = [
            {
                "id": c.id,
                "text": c.text,
                "kind": c.kind,
                "check": c.check,
                "status": c.status,
                "strength": c.strength,
                "evidence_run_ids": c.evidence_run_ids,
            }
            for c in contract.criteria
        ]
        conn.execute(
            """
            INSERT INTO expectation_contracts
                (id, jobfile_id, version, criteria, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                version = excluded.version,
                criteria = excluded.criteria,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                contract.id, contract.jobfile_id, contract.version,
                _j(criteria_data), contract.source,
                contract.created_at, contract.updated_at,
            ),
        )
        conn.commit()

    def get_contract(self, jobfile_id: str) -> ExpectationContract | None:
        """Get the latest contract for a jobfile."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM expectation_contracts WHERE jobfile_id = ? ORDER BY version DESC LIMIT 1",
            (jobfile_id,),
        ).fetchone()
        return self._row_to_contract(row) if row else None

    def get_contract_by_id(self, contract_id: str) -> ExpectationContract | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM expectation_contracts WHERE id = ?", (contract_id,)
        ).fetchone()
        return self._row_to_contract(row) if row else None

    def _row_to_contract(self, row: sqlite3.Row) -> ExpectationContract:
        criteria_data = _dj(row["criteria"], [])
        criteria = [
            Criterion(
                id=c["id"],
                text=c["text"],
                kind=c["kind"],
                check=c.get("check", {}),
                status=c["status"],
                strength=c["strength"],
                evidence_run_ids=c.get("evidence_run_ids", []),
            )
            for c in criteria_data
        ]
        return ExpectationContract(
            id=row["id"],
            jobfile_id=row["jobfile_id"],
            version=row["version"],
            criteria=criteria,
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def add_feedback(self, feedback: Feedback) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO feedback (id, run_id, kind, text, created_at) VALUES (?, ?, ?, ?, ?)",
            (feedback.id, feedback.run_id, feedback.kind, feedback.text, feedback.created_at),
        )
        conn.commit()

    def list_feedback(self, run_id: str) -> list[Feedback]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM feedback WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        ).fetchall()
        return [
            Feedback(
                id=r["id"],
                run_id=r["run_id"],
                kind=r["kind"],
                text=r["text"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
