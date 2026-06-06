"""Garbage-collect orphan run directories.

Each ssh/slurm run mirrors its workdir to ``~/.jobctl/runs/<run_id>/``. Over
many sweeps (and DB resets) these accumulate far beyond the runs the DB still
knows about. ``find_orphans`` lists dirs with no DB record; ``gc_runs`` removes
them. A dir is an orphan ONLY if its name isn't a known run_id, so a run that
still exists in the DB is never touched.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def find_orphans(runs_dir: str | Path, known_ids: set[str]) -> list[Path]:
    """Return run dirs under *runs_dir* whose name is not a known run_id."""
    d = Path(runs_dir)
    if not d.is_dir():
        return []
    return sorted(
        p for p in d.iterdir() if p.is_dir() and p.name not in known_ids
    )


def gc_runs(
    runs_dir: str | Path, known_ids: set[str], dry_run: bool = False
) -> tuple[list[Path], list[Path]]:
    """Remove orphan run dirs. Returns (orphans_found, removed)."""
    orphans = find_orphans(runs_dir, known_ids)
    removed: list[Path] = []
    if not dry_run:
        for p in orphans:
            try:
                shutil.rmtree(p)
                removed.append(p)
            except OSError:
                pass
    return orphans, removed
