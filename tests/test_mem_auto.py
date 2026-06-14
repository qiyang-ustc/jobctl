from __future__ import annotations

from datetime import datetime, timezone

from jobctl.db.models import Health, JobFile, Run, State
from jobctl.db.store import Store
from jobctl.memory.mem_auto import (
    classify_oom,
    estimate_mem_from_history,
    next_mem_request,
    parse_mem_to_mb,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jobfile() -> JobFile:
    return JobFile(
        id="jf-auto",
        name="auto-job",
        version=1,
        source_path="/tmp/auto.jobfile.yaml",
        command_template="echo run",
        params_schema={},
        backend_prefs=[{"backend": "slurm", "server": "oblix"}],
        artifact_patterns=[],
        expectation_contract_id=None,
        content_hash="abc",
        created_at=_now(),
    )


def _run(run_id: str, *, state: State = State.FAILED, slurm_request=None, resource_summary=None) -> Run:
    return Run(
        run_id=run_id,
        jobfile_id="jf-auto",
        jobfile_version=1,
        params={"x": 1},
        input_hashes={"input": "sha"},
        backend="slurm",
        server="oblix",
        task=None,
        remote_job_id=None,
        state=state,
        health=Health.OK,
        exit_code=1 if state == State.FAILED else 0,
        submitted_at=_now(),
        started_at=None,
        finished_at=_now(),
        last_heartbeat=None,
        workdir=None,
        stdout_path=None,
        stderr_path=None,
        resource_summary=resource_summary or {},
        expectation_match=None,
        observation_card=None,
        slurm_request=slurm_request,
    )


def test_parse_and_next_mem_request():
    assert parse_mem_to_mb("1G") == 1024
    assert parse_mem_to_mb("64GB") == 65536
    assert parse_mem_to_mb("64GiB") == 65536
    assert parse_mem_to_mb("956K") == 1
    assert next_mem_request("1G", {"MaxRSS": "900M"}, factor=1.5) == "1536M"
    assert next_mem_request("60G", {}, factor=1.5, cap="64G") == "65536M"
    assert next_mem_request("60G", {}, factor=1.5, cap="64GB") == "65536M"
    assert next_mem_request("64G", {}, factor=1.5, cap="64G") is None


def test_classify_gpu_oom_wins_over_generic_out_of_memory():
    diagnosis = classify_oom({}, stderr="RuntimeError: CUDA out of memory. Tried to allocate")
    assert diagnosis.kind == "gpu"


def test_classify_cpu_oom_from_slurm_state():
    diagnosis = classify_oom({"State": "OUT_OF_MEMORY"}, stdout="", stderr="")
    assert diagnosis.kind == "cpu"


def test_estimate_mem_from_history_uses_prior_cpu_oom(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    store.init_schema()
    store.add_jobfile(_jobfile())
    store.add_run(
        _run(
            "run-old",
            slurm_request={"mem": "2G", "cpus": 1},
            resource_summary={"State": "OUT_OF_MEMORY"},
        )
    )

    estimate = estimate_mem_from_history(
        store,
        jobfile_id="jf-auto",
        params={"x": 1},
        input_hashes={"input": "sha"},
        factor=1.5,
    )
    assert estimate == {
        "mem": "3072M",
        "source_run_id": "run-old",
        "reason": "prior_cpu_oom",
    }


def test_estimate_mem_from_history_caps_prior_success_request(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    store.init_schema()
    store.add_jobfile(_jobfile())
    store.add_run(
        _run(
            "run-success",
            state=State.COMPLETED,
            slurm_request={"mem": "128G", "cpus": 1},
            resource_summary={},
        )
    )

    estimate = estimate_mem_from_history(
        store,
        jobfile_id="jf-auto",
        params={"x": 1},
        input_hashes={"input": "sha"},
        current_mem="4G",
        cap="64GB",
    )
    assert estimate == {
        "mem": "65536M",
        "source_run_id": "run-success",
        "reason": "prior_success_request",
    }
