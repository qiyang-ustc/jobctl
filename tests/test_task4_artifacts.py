"""Tests for Task 4 — artifacts/indexer.py.

Exercises:
- index_run: discover files by artifact_patterns, checksum, detect type, build preview,
  persist Artifact rows linked to the run.
- detect_type: returns correct ArtifactType for .png, .csv, .json, .log, .bin, etc.
- build_preview: csv head+shape, json keys, log head+tail, image thumbnail.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import csv as csv_module
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path):
    """Return an initialised in-memory store."""
    from jobctl.db.store import Store
    db = str(tmp_path / "test.db")
    store = Store(db)
    store.init_schema()
    return store


def _make_run(run_id: str, workdir: str, jobfile_id: str):
    from jobctl.db.models import Run, State, Health
    return Run(
        run_id=run_id,
        jobfile_id=jobfile_id,
        jobfile_version=1,
        params={},
        input_hashes={},
        backend="local",
        server=None,
        task=None,
        remote_job_id=None,
        state=State.COMPLETED,
        health=Health.OK,
        exit_code=0,
        submitted_at=datetime.now(timezone.utc).isoformat(),
        started_at=None,
        finished_at=None,
        last_heartbeat=None,
        workdir=workdir,
        stdout_path=None,
        stderr_path=None,
        resource_summary={},
        expectation_match=None,
        observation_card=None,
    )


def _make_jobfile(jobfile_id: str, artifact_patterns: list[str]):
    from jobctl.db.models import JobFile
    return JobFile(
        id=jobfile_id,
        name="test-job",
        version=1,
        source_path="",
        command_template="echo hi",
        params_schema={},
        backend_prefs=[],
        artifact_patterns=artifact_patterns,
        expectation_contract_id=None,
        content_hash="abc",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Fixtures: build a temp workdir with various file types
# ---------------------------------------------------------------------------

@pytest.fixture()
def workdir(tmp_path: Path):
    """Create a temp workdir with one file of each important type."""
    wd = tmp_path / "workdir"
    wd.mkdir()

    # --- PNG via Pillow ---
    from PIL import Image
    img = Image.new("RGB", (80, 60), color=(255, 128, 0))
    img.save(str(wd / "result.png"))

    # --- CSV ---
    csv_path = wd / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["step", "loss", "acc"])
        for i in range(20):
            w.writerow([i, 1.0 / (i + 1), i / 20.0])

    # --- JSON ---
    json_path = wd / "metrics.json"
    json_path.write_text(json.dumps({"loss": 0.05, "accuracy": 0.98, "step": 100}))

    # --- log ---
    log_path = wd / "run.log"
    lines = [f"line {i}\n" for i in range(30)]
    log_path.write_text("".join(lines))

    # --- binary ---
    bin_path = wd / "model.bin"
    bin_path.write_bytes(b"\x00\x01\x02\x03" * 100)

    return wd


# ---------------------------------------------------------------------------
# detect_type
# ---------------------------------------------------------------------------

class TestDetectType:
    def test_png_is_image(self, workdir):
        from jobctl.artifacts.indexer import detect_type
        from jobctl.db.models import ArtifactType
        assert detect_type(str(workdir / "result.png")) == ArtifactType.IMAGE

    def test_jpg_is_image(self, tmp_path):
        from jobctl.artifacts.indexer import detect_type
        from jobctl.db.models import ArtifactType
        from PIL import Image
        p = tmp_path / "img.jpg"
        Image.new("RGB", (10, 10)).save(str(p))
        assert detect_type(str(p)) == ArtifactType.IMAGE

    def test_csv_is_csv(self, workdir):
        from jobctl.artifacts.indexer import detect_type
        from jobctl.db.models import ArtifactType
        assert detect_type(str(workdir / "results.csv")) == ArtifactType.CSV

    def test_json_is_json(self, workdir):
        from jobctl.artifacts.indexer import detect_type
        from jobctl.db.models import ArtifactType
        assert detect_type(str(workdir / "metrics.json")) == ArtifactType.JSON

    def test_log_is_text_log(self, workdir):
        from jobctl.artifacts.indexer import detect_type
        from jobctl.db.models import ArtifactType
        assert detect_type(str(workdir / "run.log")) == ArtifactType.TEXT_LOG

    def test_txt_is_text_log(self, tmp_path):
        from jobctl.artifacts.indexer import detect_type
        from jobctl.db.models import ArtifactType
        p = tmp_path / "output.txt"
        p.write_text("hello\n")
        assert detect_type(str(p)) == ArtifactType.TEXT_LOG

    def test_bin_is_binary(self, workdir):
        from jobctl.artifacts.indexer import detect_type
        from jobctl.db.models import ArtifactType
        assert detect_type(str(workdir / "model.bin")) == ArtifactType.BINARY

    def test_pdf_is_other(self, tmp_path):
        from jobctl.artifacts.indexer import detect_type
        from jobctl.db.models import ArtifactType
        p = tmp_path / "paper.pdf"
        p.write_bytes(b"%PDF-1.4 fake content")
        # .pdf has no specific enum value → OTHER or BINARY (extension-based)
        result = detect_type(str(p))
        assert result in (ArtifactType.BINARY, ArtifactType.OTHER)


# ---------------------------------------------------------------------------
# build_preview
# ---------------------------------------------------------------------------

class TestBuildPreview:
    def test_csv_preview_has_shape_and_head(self, workdir):
        from jobctl.artifacts.indexer import build_preview
        from jobctl.db.models import ArtifactType
        prev = build_preview(str(workdir / "results.csv"), ArtifactType.CSV)
        assert "shape" in prev
        rows, cols = prev["shape"]
        assert cols == 3
        assert rows > 0  # data rows (may be full or capped)
        assert "head" in prev
        assert isinstance(prev["head"], list)
        # head should have header + at most 5 rows
        assert len(prev["head"]) <= 6  # header + 5 data rows

    def test_csv_preview_head_includes_header(self, workdir):
        from jobctl.artifacts.indexer import build_preview
        from jobctl.db.models import ArtifactType
        prev = build_preview(str(workdir / "results.csv"), ArtifactType.CSV)
        assert prev["head"][0] == ["step", "loss", "acc"]

    def test_json_preview_has_keys(self, workdir):
        from jobctl.artifacts.indexer import build_preview
        from jobctl.db.models import ArtifactType
        prev = build_preview(str(workdir / "metrics.json"), ArtifactType.JSON)
        assert "keys" in prev
        assert set(prev["keys"]) == {"loss", "accuracy", "step"}

    def test_log_preview_has_head_and_tail(self, workdir):
        from jobctl.artifacts.indexer import build_preview
        from jobctl.db.models import ArtifactType
        prev = build_preview(str(workdir / "run.log"), ArtifactType.TEXT_LOG)
        assert "head" in prev
        assert "tail" in prev
        # 30 lines total; head should have 10, tail should have 10
        assert len(prev["head"]) <= 10
        assert len(prev["tail"]) <= 10
        assert prev["head"][0] == "line 0"
        assert prev["tail"][-1] == "line 29"

    def test_image_preview_creates_thumbnail(self, workdir, tmp_path):
        from jobctl.artifacts.indexer import build_preview
        from jobctl.db.models import ArtifactType
        prev = build_preview(str(workdir / "result.png"), ArtifactType.IMAGE)
        assert "thumbnail_path" in prev
        assert prev["thumbnail_path"] is not None
        # Thumbnail file should exist
        assert Path(prev["thumbnail_path"]).exists()
        # Original dimensions
        assert "width" in prev
        assert "height" in prev

    def test_binary_preview_is_minimal(self, workdir):
        from jobctl.artifacts.indexer import build_preview
        from jobctl.db.models import ArtifactType
        prev = build_preview(str(workdir / "model.bin"), ArtifactType.BINARY)
        # Should not crash; returns a dict (possibly empty or with size)
        assert isinstance(prev, dict)


# ---------------------------------------------------------------------------
# index_run
# ---------------------------------------------------------------------------

class TestIndexRun:
    def test_discovers_all_matched_files(self, workdir, tmp_path):
        from jobctl.artifacts.indexer import index_run
        store = _make_store(tmp_path)
        jf = _make_jobfile("jf-1", ["*.csv", "*.json", "*.png", "*.log", "*.bin"])
        run = _make_run("run-1", str(workdir), "jf-1")
        store.add_jobfile(jf)
        store.add_run(run)

        artifacts = index_run(store, run, jf)
        names = {Path(a.local_path).name for a in artifacts}
        assert "results.csv" in names
        assert "metrics.json" in names
        assert "result.png" in names
        assert "run.log" in names
        assert "model.bin" in names

    def test_returns_artifact_objects(self, workdir, tmp_path):
        from jobctl.artifacts.indexer import index_run
        from jobctl.db.models import Artifact
        store = _make_store(tmp_path)
        jf = _make_jobfile("jf-2", ["*.csv"])
        run = _make_run("run-2", str(workdir), "jf-2")
        store.add_jobfile(jf)
        store.add_run(run)

        artifacts = index_run(store, run, jf)
        assert len(artifacts) == 1
        a = artifacts[0]
        assert isinstance(a, Artifact)
        assert a.run_id == "run-2"

    def test_checksums_are_stable_sha256(self, workdir, tmp_path):
        from jobctl.artifacts.indexer import index_run
        store = _make_store(tmp_path)
        jf = _make_jobfile("jf-3", ["*.csv"])
        run = _make_run("run-3", str(workdir), "jf-3")
        store.add_jobfile(jf)
        store.add_run(run)

        artifacts = index_run(store, run, jf)
        assert len(artifacts) == 1
        a = artifacts[0]
        expected = _sha256(str(workdir / "results.csv"))
        assert a.checksum == expected

    def test_types_are_correct(self, workdir, tmp_path):
        from jobctl.artifacts.indexer import index_run
        from jobctl.db.models import ArtifactType
        store = _make_store(tmp_path)
        jf = _make_jobfile("jf-4", ["*.csv", "*.json", "*.png", "*.log", "*.bin"])
        run = _make_run("run-4", str(workdir), "jf-4")
        store.add_jobfile(jf)
        store.add_run(run)

        artifacts = index_run(store, run, jf)
        by_name = {Path(a.local_path).name: a for a in artifacts}
        assert by_name["results.csv"].type == ArtifactType.CSV
        assert by_name["metrics.json"].type == ArtifactType.JSON
        assert by_name["result.png"].type == ArtifactType.IMAGE
        assert by_name["run.log"].type == ArtifactType.TEXT_LOG
        assert by_name["model.bin"].type == ArtifactType.BINARY

    def test_artifacts_persisted_in_store(self, workdir, tmp_path):
        from jobctl.artifacts.indexer import index_run
        store = _make_store(tmp_path)
        jf = _make_jobfile("jf-5", ["*.csv"])
        run = _make_run("run-5", str(workdir), "jf-5")
        store.add_jobfile(jf)
        store.add_run(run)

        artifacts = index_run(store, run, jf)
        stored = store.list_artifacts("run-5")
        assert len(stored) == 1
        assert stored[0].run_id == "run-5"
        assert stored[0].checksum == artifacts[0].checksum

    def test_no_duplicate_on_second_index(self, workdir, tmp_path):
        """Calling index_run twice should not duplicate stored artifacts."""
        from jobctl.artifacts.indexer import index_run
        store = _make_store(tmp_path)
        jf = _make_jobfile("jf-6", ["*.csv"])
        run = _make_run("run-6", str(workdir), "jf-6")
        store.add_jobfile(jf)
        store.add_run(run)

        index_run(store, run, jf)
        index_run(store, run, jf)
        stored = store.list_artifacts("run-6")
        # should still be 1 (idempotent by checksum or skipped)
        assert len(stored) == 1

    def test_size_matches_file_size(self, workdir, tmp_path):
        from jobctl.artifacts.indexer import index_run
        store = _make_store(tmp_path)
        jf = _make_jobfile("jf-7", ["*.json"])
        run = _make_run("run-7", str(workdir), "jf-7")
        store.add_jobfile(jf)
        store.add_run(run)

        artifacts = index_run(store, run, jf)
        a = artifacts[0]
        expected_size = (workdir / "metrics.json").stat().st_size
        assert a.size == expected_size

    def test_preview_populated(self, workdir, tmp_path):
        from jobctl.artifacts.indexer import index_run
        store = _make_store(tmp_path)
        jf = _make_jobfile("jf-8", ["*.csv"])
        run = _make_run("run-8", str(workdir), "jf-8")
        store.add_jobfile(jf)
        store.add_run(run)

        artifacts = index_run(store, run, jf)
        assert artifacts[0].preview  # non-empty dict
        assert "head" in artifacts[0].preview

    def test_glob_patterns_match_subdirs(self, tmp_path):
        """Patterns with ** or subdir prefix should work via recursive glob."""
        from jobctl.artifacts.indexer import index_run
        wd = tmp_path / "wd"
        wd.mkdir()
        subdir = wd / "outputs"
        subdir.mkdir()
        (subdir / "result.csv").write_text("a,b\n1,2\n")

        store = _make_store(tmp_path)
        jf = _make_jobfile("jf-9", ["**/*.csv"])
        run = _make_run("run-9", str(wd), "jf-9")
        store.add_jobfile(jf)
        store.add_run(run)

        artifacts = index_run(store, run, jf)
        assert len(artifacts) == 1
        assert Path(artifacts[0].local_path).name == "result.csv"

    def test_image_thumbnail_in_preview(self, workdir, tmp_path):
        from jobctl.artifacts.indexer import index_run
        store = _make_store(tmp_path)
        jf = _make_jobfile("jf-10", ["*.png"])
        run = _make_run("run-10", str(workdir), "jf-10")
        store.add_jobfile(jf)
        store.add_run(run)

        artifacts = index_run(store, run, jf)
        assert len(artifacts) == 1
        prev = artifacts[0].preview
        assert "thumbnail_path" in prev
        assert Path(prev["thumbnail_path"]).exists()

    def test_empty_workdir_returns_empty(self, tmp_path):
        """No files matching patterns → empty list."""
        from jobctl.artifacts.indexer import index_run
        wd = tmp_path / "empty_wd"
        wd.mkdir()

        store = _make_store(tmp_path)
        jf = _make_jobfile("jf-11", ["*.csv"])
        run = _make_run("run-11", str(wd), "jf-11")
        store.add_jobfile(jf)
        store.add_run(run)

        artifacts = index_run(store, run, jf)
        assert artifacts == []

    def test_none_workdir_returns_empty(self, tmp_path):
        """run.workdir is None → return empty list gracefully."""
        from jobctl.artifacts.indexer import index_run
        store = _make_store(tmp_path)
        jf = _make_jobfile("jf-12", ["*.csv"])
        run = _make_run("run-12", None, "jf-12")
        store.add_jobfile(jf)
        store.add_run(run)

        artifacts = index_run(store, run, jf)
        assert artifacts == []
