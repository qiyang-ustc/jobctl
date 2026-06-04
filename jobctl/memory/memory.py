"""Run memory: query past runs and find reuse candidates.

query()          — summarise what the store knows about a jobfile + optionally
                   check for an exact-match prior run.
reuse_candidate() — return a Run only when all three conditions hold:
                   1. exact input_hash + params match
                   2. expectation_match == USABLE
                   3. at least one artifact whose local_path exists on disk
"""
from __future__ import annotations

import os
from typing import Any

from jobctl.db.models import JobFile, Match, Run
from jobctl.db.store import Store


def query(
    store: Store,
    jobfile_id: str | None = None,
    name: str | None = None,
    params: dict | None = None,
    input_hashes: dict | None = None,
) -> dict:
    """Return a summary dict describing memory for a jobfile.

    Keys:
        has_jobfile        bool  – jobfile is registered
        runs               int   – total run count for the jobfile
        exact_match_run_id str|None – run_id of first run with identical
                                     params AND input_hashes (or None)
        artifacts_dir      str|None – workdir of the exact-match run
        server             str|None – server of the exact-match run
        outcome            str|None – state.value of the exact-match run
        reuse_eligible     bool  – True only when exact match, USABLE, artifacts on disk
    """
    # Resolve jobfile by name if id not provided
    if jobfile_id is None and name is not None:
        jf = store.get_jobfile_by_name(name)
        if jf is not None:
            jobfile_id = jf.id

    base: dict[str, Any] = {
        "has_jobfile": False,
        "runs": 0,
        "exact_match_run_id": None,
        "artifacts_dir": None,
        "server": None,
        "outcome": None,
        "reuse_eligible": False,
    }

    if jobfile_id is None:
        return base

    jf = store.get_jobfile(jobfile_id)
    if jf is None:
        return base

    base["has_jobfile"] = True

    all_runs = store.list_runs(jobfile_id=jobfile_id)
    base["runs"] = len(all_runs)

    # Look for exact match when caller supplied params + input_hashes
    if params is not None and input_hashes is not None:
        exact = _find_exact_match(all_runs, params, input_hashes)
        if exact is not None:
            base["exact_match_run_id"] = exact.run_id
            base["artifacts_dir"] = exact.workdir
            base["server"] = exact.server
            base["outcome"] = exact.state.value if exact.state is not None else None
            base["reuse_eligible"] = _is_reuse_eligible(store, exact)

    return base


def reuse_candidate(
    store: Store,
    jobfile: JobFile,
    params: dict,
    input_hashes: dict,
) -> Run | None:
    """Return a prior Run that can be safely reused, or None.

    All three conditions must hold:
    1. Exact match on input_hashes AND params (no missing/extra keys, values equal).
    2. expectation_match == USABLE.
    3. At least one artifact with a local_path that exists on disk.
    """
    all_runs = store.list_runs(jobfile_id=jobfile.id)
    exact = _find_exact_match(all_runs, params, input_hashes)
    if exact is None:
        return None

    if not _is_reuse_eligible(store, exact):
        return None

    return exact


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _params_equal(a: dict, b: dict) -> bool:
    """Deep equality between two params dicts (handles float ≈ float)."""
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        av, bv = a[k], b[k]
        if type(av) != type(bv):
            # Allow numeric cross-type comparison (int vs float)
            try:
                if float(av) != float(bv):
                    return False
            except (TypeError, ValueError):
                return False
        else:
            if av != bv:
                return False
    return True


def _find_exact_match(runs: list[Run], params: dict, input_hashes: dict) -> Run | None:
    """Return the first run whose params AND input_hashes exactly match."""
    for run in runs:
        if _params_equal(run.params, params) and run.input_hashes == input_hashes:
            return run
    return None


def _is_reuse_eligible(store: Store, run: Run) -> bool:
    """True iff run.expectation_match == USABLE AND at least one artifact exists on disk."""
    if run.expectation_match != Match.USABLE:
        return False

    artifacts = store.list_artifacts(run.run_id)
    if not artifacts:
        return False

    # At least one artifact must be present on disk
    for art in artifacts:
        if art.local_path and os.path.exists(art.local_path):
            return True

    return False
