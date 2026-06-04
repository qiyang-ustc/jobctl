"""Expectation Distiller: propose() and confirm().

propose() uses an Analyzer to generate new criteria from user feedback and
persists them to the contract in the Store. Proposals start with
status='proposed', strength=1.

confirm() promotes a criterion to status='confirmed' and increments strength.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from jobctl.db.models import (
    Criterion,
    ExpectationContract,
    JobFile,
    Run,
)
from jobctl.db.store import Store
from jobctl.expectations.contracts import default_contract


# ---------------------------------------------------------------------------
# Analyzer protocol (duck-typed so we don't depend on the ABC in tests)
# ---------------------------------------------------------------------------

class _AnalyzerProtocol(Protocol):
    def propose_criteria(
        self,
        feedback: dict,
        history: list,
        jobfile: dict,
    ) -> list[dict]:
        ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict_to_criterion(d: dict) -> Criterion:
    """Convert an analyzer-returned dict to a Criterion dataclass."""
    return Criterion(
        id=d["id"],
        text=d["text"],
        kind=d["kind"],
        check=d.get("check", {}),
        status=d.get("status", "proposed"),
        strength=int(d.get("strength", 1)),
        evidence_run_ids=list(d.get("evidence_run_ids", [])),
    )


def _get_or_create_contract(store: Store, run: Run) -> ExpectationContract:
    """Load existing contract for the run's jobfile, or create a default one."""
    contract = store.get_contract(run.jobfile_id)
    if contract is not None:
        return contract

    # Build default contract; we need the jobfile
    jf = store.get_jobfile(run.jobfile_id)
    if jf is not None:
        contract = default_contract(jf)
    else:
        # Fallback minimal contract
        contract = ExpectationContract(
            id=f"ec-distiller-{run.jobfile_id[:8]}",
            jobfile_id=run.jobfile_id,
            version=1,
            criteria=[],
            source="distiller-created",
            created_at=_now(),
            updated_at=_now(),
        )
    store.save_contract(contract)
    return contract


def propose(
    store: Store,
    run: Run,
    feedback: dict,
    analyzer: Any,
) -> list[Criterion]:
    """Propose new expectation criteria from user feedback via the analyzer.

    Uses the analyzer's propose_criteria() to generate new criteria, merges
    them into the contract for the run's jobfile, persists to store.

    Args:
        store:    The Store repository.
        run:      The Run being evaluated.
        feedback: Dict with keys like 'kind', 'text', 'run_id'.
        analyzer: Any object with propose_criteria(feedback, history, jobfile) -> list[dict].

    Returns:
        List of newly proposed Criterion objects (not including pre-existing).
    """
    # Collect run history for context
    history_runs = store.list_runs(jobfile_id=run.jobfile_id)
    history = [
        {
            "run_id": r.run_id,
            "state": r.state.value if hasattr(r.state, "value") else str(r.state),
            "exit_code": r.exit_code,
            "expectation_match": (
                r.expectation_match.value
                if hasattr(r.expectation_match, "value") and r.expectation_match
                else r.expectation_match
            ),
        }
        for r in history_runs
    ]

    # Build jobfile context dict
    jf = store.get_jobfile(run.jobfile_id)
    jobfile_dict: dict = {}
    if jf is not None:
        jobfile_dict = {
            "id": jf.id,
            "name": jf.name,
            "command_template": jf.command_template,
            "artifact_patterns": jf.artifact_patterns,
        }
        # Embed expectation text if present in the source
        if jf.source_path:
            try:
                from pathlib import Path
                src = Path(jf.source_path)
                if src.suffix.lower() in (".yaml", ".yml") and src.exists():
                    import yaml
                    raw = yaml.safe_load(src.read_text()) or {}
                    jobfile_dict["expectation"] = raw.get("expectation", "")
            except Exception:
                pass

    # Ask the analyzer for proposed criteria
    raw_criteria: list[dict] = analyzer.propose_criteria(feedback, history, jobfile_dict)

    # Convert to Criterion dataclass instances
    proposed: list[Criterion] = []
    for d in raw_criteria:
        # Ensure status=proposed and strength=1 for freshly proposed criteria
        d_copy = dict(d)
        d_copy["status"] = "proposed"
        d_copy["strength"] = 1
        proposed.append(_dict_to_criterion(d_copy))

    # Merge into existing contract
    contract = _get_or_create_contract(store, run)
    existing_ids = {c.id for c in contract.criteria}
    new_criteria = [c for c in proposed if c.id not in existing_ids]

    if new_criteria:
        contract.criteria.extend(new_criteria)
        contract.updated_at = _now()
        store.save_contract(contract)

    return proposed


def confirm(store: Store, criterion_id: str) -> Criterion:
    """Promote a criterion to confirmed and increment its strength.

    Searches all contracts in the store for the criterion_id.

    Args:
        store:        The Store repository.
        criterion_id: ID of the criterion to confirm.

    Returns:
        The updated Criterion with status='confirmed' and strength incremented.

    Raises:
        ValueError: if no criterion with that ID is found in any contract.
    """
    # Search all contracts for this criterion
    # We need to iterate through all contracts; we'll use list_jobfiles + get_contract
    all_jobfiles = store.list_jobfiles()

    found_contract: ExpectationContract | None = None
    found_criterion: Criterion | None = None

    for jf in all_jobfiles:
        contract = store.get_contract(jf.id)
        if contract is None:
            continue
        for crit in contract.criteria:
            if crit.id == criterion_id:
                found_contract = contract
                found_criterion = crit
                break
        if found_criterion is not None:
            break

    if found_criterion is None or found_contract is None:
        raise ValueError(f"Criterion '{criterion_id}' not found in any contract.")

    # Update the criterion in place
    found_criterion.status = "confirmed"
    found_criterion.strength += 1

    # Persist the updated contract
    found_contract.updated_at = _now()
    store.save_contract(found_contract)

    return found_criterion
