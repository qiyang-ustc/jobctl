"""Orphan run-dir garbage collection."""
from __future__ import annotations

from jobctl import gc


def _make_run_dirs(base, names):
    for n in names:
        (base / n).mkdir()
        (base / n / "stdout.txt").write_text("x")


def test_find_orphans_excludes_known_runs(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _make_run_dirs(runs, ["run-keep", "run-orphan1", "run-orphan2"])
    orphans = gc.find_orphans(runs, {"run-keep"})
    names = {p.name for p in orphans}
    assert names == {"run-orphan1", "run-orphan2"}


def test_gc_runs_removes_only_orphans(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _make_run_dirs(runs, ["run-keep", "run-gone"])
    orphans, removed = gc.gc_runs(runs, {"run-keep"})
    assert {p.name for p in removed} == {"run-gone"}
    assert (runs / "run-keep").exists()
    assert not (runs / "run-gone").exists()


def test_gc_dry_run_removes_nothing(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _make_run_dirs(runs, ["run-gone"])
    orphans, removed = gc.gc_runs(runs, set(), dry_run=True)
    assert {p.name for p in orphans} == {"run-gone"}
    assert removed == []
    assert (runs / "run-gone").exists()  # untouched


def test_find_orphans_missing_dir_is_empty(tmp_path):
    assert gc.find_orphans(tmp_path / "nope", set()) == []
