"""Artifact indexer: discover, checksum, type, preview, and persist artifacts.

Public API:
    index_run(store, run, jobfile) -> list[Artifact]
    detect_type(path) -> ArtifactType
    build_preview(path, atype) -> dict
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from jobctl.db.models import Artifact, ArtifactType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from jobctl.db.models import JobFile, Run
    from jobctl.db.store import Store


# ---------------------------------------------------------------------------
# Extension → ArtifactType mappings
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".svg"}
_PLOT_EXTS = {".eps", ".ps"}  # vector plot formats not covered by Pillow
_CSV_EXTS = {".csv", ".tsv"}
_JSON_EXTS = {".json", ".jsonl"}
_LOG_EXTS = {".log", ".txt", ".out", ".err"}
_BINARY_EXTS = {".bin", ".pkl", ".pickle", ".pt", ".pth", ".h5", ".hdf5", ".npz", ".npy"}


def detect_type(path: str) -> ArtifactType:
    """Detect the ArtifactType for a file using extension first, then magic bytes."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext in _IMAGE_EXTS:
        return ArtifactType.IMAGE
    if ext in _PLOT_EXTS:
        return ArtifactType.PLOT
    if ext in _CSV_EXTS:
        return ArtifactType.CSV
    if ext in _JSON_EXTS:
        return ArtifactType.JSON
    if ext in _LOG_EXTS:
        return ArtifactType.TEXT_LOG
    if ext in _BINARY_EXTS:
        return ArtifactType.BINARY

    # Fall back to magic bytes inspection
    try:
        with open(path, "rb") as f:
            header = f.read(16)
    except OSError:
        return ArtifactType.OTHER

    # PNG: \x89PNG
    if header[:4] == b"\x89PNG":
        return ArtifactType.IMAGE
    # JPEG: FF D8 FF
    if header[:3] == b"\xff\xd8\xff":
        return ArtifactType.IMAGE
    # GIF: GIF8
    if header[:4] in (b"GIF8", b"GIF9"):
        return ArtifactType.IMAGE
    # PDF: %PDF
    if header[:4] == b"%PDF":
        return ArtifactType.BINARY

    # Try to detect text vs binary by sampling the first 512 bytes
    try:
        with open(path, "rb") as f:
            sample = f.read(512)
        if b"\x00" in sample:
            return ArtifactType.BINARY
        # Printable-ish
        sample.decode("utf-8", errors="strict")
        return ArtifactType.TEXT_LOG
    except (UnicodeDecodeError, OSError):
        return ArtifactType.BINARY


# ---------------------------------------------------------------------------
# Preview builders
# ---------------------------------------------------------------------------

_CSV_HEAD_ROWS = 5   # data rows (after header)
_LOG_HEAD_LINES = 10
_LOG_TAIL_LINES = 10
_THUMBNAIL_SIZE = (128, 128)


def build_preview(path: str, atype: ArtifactType) -> dict:
    """Build a preview dict for the artifact at *path* with the given type.

    Returns:
        ArtifactType.CSV   → {"shape": [rows, cols], "head": [[...], ...]}
        ArtifactType.JSON  → {"keys": [...]}
        ArtifactType.TEXT_LOG → {"head": [...lines], "tail": [...lines]}
        ArtifactType.IMAGE → {"thumbnail_path": str, "width": int, "height": int}
        others             → {}
    """
    if atype == ArtifactType.CSV:
        return _preview_csv(path)
    if atype == ArtifactType.JSON:
        return _preview_json(path)
    if atype == ArtifactType.TEXT_LOG:
        return _preview_log(path)
    if atype in (ArtifactType.IMAGE, ArtifactType.PLOT):
        return _preview_image(path)
    return {}


def _preview_csv(path: str) -> dict:
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return {"shape": [0, 0], "head": []}
        num_cols = max(len(r) for r in rows)
        # rows[0] is header, remaining are data
        data_rows = rows[1:]
        total_data = len(data_rows)
        head = [rows[0]] + data_rows[:_CSV_HEAD_ROWS]
        return {"shape": [total_data, num_cols], "head": head}
    except Exception:
        return {"error": "csv parse failed"}


def _preview_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return {"keys": list(obj.keys())}
        if isinstance(obj, list):
            return {"length": len(obj), "keys": list(obj[0].keys()) if obj and isinstance(obj[0], dict) else []}
        return {"type": type(obj).__name__}
    except Exception:
        return {"error": "json parse failed"}


def _preview_log(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = [line.rstrip("\n") for line in f]
        head = all_lines[:_LOG_HEAD_LINES]
        tail = all_lines[-_LOG_TAIL_LINES:] if len(all_lines) > _LOG_HEAD_LINES else []
        return {"head": head, "tail": tail}
    except Exception:
        return {"error": "log read failed"}


def _preview_image(path: str) -> dict:
    try:
        from PIL import Image as PILImage
        img = PILImage.open(path)
        width, height = img.size
        # Save thumbnail next to original
        thumb_path = Path(path).with_suffix(".thumb.png")
        thumb = img.copy()
        thumb.thumbnail(_THUMBNAIL_SIZE)
        thumb.save(str(thumb_path))
        return {"thumbnail_path": str(thumb_path), "width": width, "height": height}
    except Exception:
        return {"error": "image preview failed"}


# ---------------------------------------------------------------------------
# SHA-256 checksum
# ---------------------------------------------------------------------------

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _glob_artifact_pattern(workdir: Path, pattern: str) -> list[Path]:
    """Return paths matching a relative-to-workdir or absolute artifact glob."""
    text = os.path.expanduser(str(pattern))
    try:
        if Path(text).is_absolute():
            return [Path(p) for p in glob.glob(text, recursive=True)]
        return list(workdir.glob(text))
    except (NotImplementedError, OSError, ValueError) as exc:
        logger.warning(
            "artifact pattern ignored for workdir=%s pattern=%r: %s",
            workdir,
            pattern,
            exc,
        )
        return []


# ---------------------------------------------------------------------------
# index_run
# ---------------------------------------------------------------------------

def index_run(store: "Store", run: "Run", jobfile: "JobFile") -> list[Artifact]:
    """Discover files matching jobfile.artifact_patterns in run.workdir.

    For each discovered file:
    - Compute SHA-256 checksum
    - Detect ArtifactType
    - Build preview dict
    - Persist as Artifact row linked to run (idempotent: skip if checksum already stored)

    Returns the full list of Artifact objects (including any previously stored).
    """
    if run.workdir is None:
        return []

    workdir = Path(run.workdir)
    if not workdir.exists():
        return []

    patterns = jobfile.artifact_patterns or []
    if not patterns:
        return []

    # Gather matching paths
    matched: list[Path] = []
    for pattern in patterns:
        found = _glob_artifact_pattern(workdir, pattern)
        for p in found:
            if p.is_file() and p not in matched:
                matched.append(p)

    # Build set of already-stored checksums to avoid duplicates
    existing = store.list_artifacts(run.run_id)
    existing_checksums = {a.checksum for a in existing}

    now = datetime.now(timezone.utc).isoformat()
    new_artifacts: list[Artifact] = []

    for fpath in matched:
        checksum = _sha256(str(fpath))
        if checksum in existing_checksums:
            continue  # already stored (idempotent)

        atype = detect_type(str(fpath))
        preview = build_preview(str(fpath), atype)
        size = fpath.stat().st_size

        artifact = Artifact(
            id=str(uuid.uuid4()),
            run_id=run.run_id,
            remote_path=str(fpath),
            local_path=str(fpath),
            type=atype,
            size=size,
            checksum=checksum,
            preview=preview,
            created_at=now,
        )
        store.add_artifact(artifact)
        new_artifacts.append(artifact)
        existing_checksums.add(checksum)

    # Return all artifacts for this run (previously stored + new ones)
    return store.list_artifacts(run.run_id)
