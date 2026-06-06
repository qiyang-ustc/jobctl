"""Per-server SLURM directive resolution.

Regression for the oblix gap: the backend used to hardcode hipster's
`--account=linuxusers` / `--partition=capacity` defaults for any server,
so a submit to oblix (which uses partition `lln` and *no* account) would
be wrong. Directives must come from per-server config + per-run overrides,
and account/partition must be omitted when not configured.
"""
from __future__ import annotations

from jobctl.backends.slurm import SlurmBackend
from jobctl.db.models import Health, Run, State


def _run(**kw) -> Run:
    d = dict(
        run_id="run-x", jobfile_id="jf", jobfile_version=1, params={}, input_hashes={},
        backend="slurm", server="oblix", task=None, remote_job_id=None,
        state=State.PENDING, health=Health.OK, exit_code=None,
        submitted_at=None, started_at=None, finished_at=None, last_heartbeat=None,
        workdir=None, stdout_path=None, stderr_path=None, resource_summary={},
        expectation_match=None, observation_card=None,
    )
    d.update(kw)
    return Run(**d)


def _backend(cfg) -> SlurmBackend:
    return SlurmBackend(server="oblix", server_config=cfg, run_cmd=lambda c, **k: None)


def test_omits_account_and_partition_when_not_configured():
    # oblix-like: no account, no partition in config -> let SLURM use its defaults
    d = _backend({"remote_path": "/tmp"})._build_directives(_run(), None, "/wd")
    joined = " ".join(d)
    assert "--account" not in joined
    assert "--partition" not in joined
    # still emits sensible time/mem/cpus
    assert "--time=" in joined and "--mem=" in joined and "--cpus-per-task=" in joined


def test_partition_from_server_config_without_account():
    d = _backend({"partition": "lln"})._build_directives(_run(), None, "/wd")
    joined = " ".join(d)
    assert "--partition=lln" in joined
    assert "--account" not in joined


def test_account_emitted_only_when_configured():
    d = _backend({"partition": "capacity", "account": "linuxusers"})._build_directives(
        _run(), None, "/wd"
    )
    joined = " ".join(d)
    assert "--account=linuxusers" in joined
    assert "--partition=capacity" in joined


def test_per_run_override_wins_over_server_defaults():
    b = _backend({"partition": "capacity", "mem": "1G", "cpus_per_task": 8})
    run = _run(slurm_request={"partition": "lln", "mem": "100M", "cpus": 1, "time": "00:11:00"})
    joined = " ".join(b._build_directives(run, None, "/wd"))
    assert "--partition=lln" in joined
    assert "--mem=100M" in joined
    assert "--cpus-per-task=1" in joined
    assert "--time=00:11:00" in joined


def test_resource_request_dict_shape():
    b = _backend({"partition": "lln"})
    req = b._resource_request(_run(slurm_request={"mem": "100M", "cpus": 1}))
    assert req["partition"] == "lln"
    assert req["mem"] == "100M"
    assert req["cpus"] == 1
    assert "account" not in req  # oblix: omitted


# --- poll reachability: distinguish "job aged out of squeue" from "SSH down" ---

def test_poll_aged_out_job_falls_through_to_sacct():
    """squeue rc!=0 'Invalid job id' (aged out) must consult sacct, not fail."""
    import subprocess
    def runner(cmd, **kw):
        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Invalid job id specified")
        if cmd[0] == "sacct":
            return subprocess.CompletedProcess(cmd, 0, stdout="315650|CANCELLED|0:0|08:44:00|956K|08:44:12\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    b = SlurmBackend(server="oblix", server_config={}, run_cmd=runner)
    poll = b.poll(_run(remote_job_id="315650", state=State.STUCK))
    assert poll.reachable is True
    assert poll.state == State.CANCELLED


def test_poll_ssh_failure_is_unreachable_not_failed():
    """ssh exit 255 = connection failure -> reachable=False, state kept."""
    import subprocess
    def runner(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 255, stdout="", stderr="ssh: connect to host timed out")
    b = SlurmBackend(server="oblix", server_config={}, run_cmd=runner)
    poll = b.poll(_run(remote_job_id="315650", state=State.RUNNING))
    assert poll.reachable is False
    assert poll.state == State.RUNNING


def test_parse_sacct_prefers_main_row_state_over_substeps():
    """Main job CANCELLED but .extern COMPLETED -> State is CANCELLED."""
    from jobctl.backends.slurm import SlurmBackend
    out = "\n".join([
        "315650|CANCELLED|0:0|08:44:00|0|08:44:12",
        "315650.batch|CANCELLED|0:15|08:44:00|956K|08:44:12",
        "315650.extern|COMPLETED|0:0|08:44:00|0|08:44:12",
    ])
    exit_code, resource = SlurmBackend._parse_sacct(out, "315650")
    assert resource["State"] == "CANCELLED"
