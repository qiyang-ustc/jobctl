# jobctl Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:test-driven-development for every task. Steps use checkbox (`- [ ]`) syntax. This plan is executed by a Workflow of Sonnet subagents, one coherent layer per task, in dependency order. Each task: write tests first, implement minimally, run tests green, commit.

**Goal:** Build jobctl — a CLI-first, JobFile-native gateway that runs remote research jobs like local commands and manages their full lifecycle (monitoring, artifacts, health, expectations, cheap-model analysis, result return).

**Architecture:** A FastAPI daemon owns a SQLite DB and an asyncio monitor loop, drives local/ssh/slurm backend adapters, indexes artifacts, classifies runs against versioned Expectation Contracts (deterministic source of truth), and narrates results via a pluggable cheap-model analyzer (DeepSeek with an offline fallback). A Typer CLI is a thin HTTP client; a Jinja/HTMX UI renders the dashboard.

**Tech Stack:** Python 3.11+, Typer, FastAPI + uvicorn, httpx, SQLite (stdlib `sqlite3`), Jinja2, HTMX, Pillow, PyYAML, `openai` SDK (DeepSeek-compatible), pytest.

Spec: `docs/superpowers/specs/2026-06-04-jobctl-research-run-gateway-design.md` (subagents read it for detail).

---

## File Structure & Ownership

```
pyproject.toml                 packaging + deps + pytest config + console_script jobctl
jobctl/__init__.py             version; exports wired in the final integration task
jobctl/config.py               load ~/.cluster.yaml + ~/.jobctl/config.toml; paths; settings
jobctl/db/models.py            dataclasses + enums + DDL strings
jobctl/db/store.py             Store: sqlite repository (the ONLY writer)
jobctl/jobfile.py              manifest parse + bare-script autowrap + versioning/hashing
jobctl/analysis/base.py        Analyzer ABC + get_analyzer()
jobctl/analysis/offline.py     deterministic template analyzer
jobctl/analysis/deepseek.py    OpenAI-compatible DeepSeek analyzer
jobctl/artifacts/indexer.py    discover/checksum/type/preview/thumbnail
jobctl/memory/memory.py        query() + reuse_candidate()
jobctl/expectations/contracts.py  criteria engine: evaluate()/classify()
jobctl/expectations/distiller.py  propose()/confirm() (uses analyzer)
jobctl/backends/base.py        Backend ABC + result dataclasses + select_backend()/get_backend()
jobctl/backends/local.py       LocalBackend
jobctl/backends/ssh.py         SshBackend
jobctl/backends/slurm.py       SlurmBackend
jobctl/notify/notify.py        Notifier ABC + Log/Webhook/Slack/Callback + get_notifiers()
jobctl/monitor/monitor.py      Monitor loop + build_observation_card()
jobctl/api/server.py           FastAPI app: REST + UI routes + monitor startup
jobctl/api/client.py           ApiClient + ensure_daemon()
jobctl/cli/main.py             Typer app (run/await/status/logs/artifacts/inspect/cancel/rerun/servers/memory + serve/register/jobfiles/feedback/expect)
jobctl/ui/templates/*.html     dashboard / run / jobfile (Jinja + HTMX)
jobctl/ui/static/*             minimal css
tests/...                      mirror of the above; tests/fakebin/ holds fake sbatch/squeue/sacct/scancel
```

---

## Interface Contract (authoritative — all tasks must match these names)

```python
# db/models.py
class State(str, Enum):  PENDING="pending"; SUBMITTED="submitted"; RUNNING="running"; \
    COMPLETED="completed"; FAILED="failed"; CANCELLED="cancelled"; STUCK="stuck"; TIMEOUT="timeout"
class Health(str, Enum): OK="ok"; WEAK="weak"; NO_HEARTBEAT="no_heartbeat"; \
    RESOURCE_PRESSURE="resource_pressure"; STUCK="stuck"
class Match(str, Enum):  USABLE="usable"; WEAK_SIGNAL="weak_signal"; BAD_SIGNAL="bad_signal"; \
    INCONCLUSIVE="inconclusive"; FAILED="failed"
class ArtifactType(str, Enum): IMAGE="image"; PLOT="plot"; CSV="csv"; JSON="json"; \
    TEXT_LOG="text_log"; BINARY="binary"; OTHER="other"

@dataclass JobFile: id,name; version:int; source_path,command_template; params_schema:dict; \
    backend_prefs:list[dict]; artifact_patterns:list[str]; expectation_contract_id:str|None; \
    content_hash:str; created_at:str
@dataclass Run: run_id,jobfile_id; jobfile_version:int; params:dict; input_hashes:dict; \
    backend,server,task,remote_job_id; state:State; health:Health; exit_code:int|None; \
    submitted_at,started_at,finished_at,last_heartbeat,workdir,stdout_path,stderr_path; \
    resource_summary:dict; expectation_match:Match|None; observation_card:dict|None
@dataclass Artifact: id,run_id,remote_path,local_path; type:ArtifactType; size:int; checksum:str; \
    preview:dict; created_at:str
@dataclass Criterion: id,text; kind:str("numeric|presence|absence|pattern"); check:dict; \
    status:str("proposed|confirmed"); strength:int; evidence_run_ids:list[str]
@dataclass ExpectationContract: id,jobfile_id; version:int; criteria:list[Criterion]; source:str; \
    created_at,updated_at
@dataclass Feedback: id,run_id,kind,text,created_at
@dataclass Server: name,backend_type; online:bool; last_heartbeat; cpu;mem;gpu;disk; \
    slurm_queue:dict; note

# db/store.py  — Store(db_path). JSON columns (de)serialized inside.
Store.init_schema(); add_jobfile/get_jobfile/get_jobfile_by_name/list_jobfiles/bump_version;
add_run/get_run/update_run/list_runs(state=None,jobfile_id=None);
add_artifact/list_artifacts(run_id); upsert_server/list_servers/get_server;
save_contract/get_contract(jobfile_id)/get_contract_by_id; add_feedback/list_feedback(run_id).

# jobfile.py
load_jobfile(path) -> JobFile            # manifest OR bare script (autowrap by extension)
resolve_params(jobfile, overrides:dict) -> dict   # apply defaults, types, required check
render_command(jobfile, params) -> str
content_hash(jobfile) -> str             # hash(command_template + referenced script bytes)
input_hashes(jobfile, params) -> dict    # {path: sha256} for script + path-typed params

# analysis/base.py
class Analyzer(ABC):
    analyze_run(facts:dict)->dict        # {interpretation, key_evidence?, recommended_next_action}
    summarize_log(text:str)->str
    explain_bad_signal(facts:dict)->str
    suggest_next_action(facts:dict, history:list)->str
    propose_criteria(feedback:dict, history:list, jobfile:dict)->list[dict]
    summarize_failures(history:list)->str
get_analyzer(config)->Analyzer           # DeepSeek if DEEPSEEK_API_KEY else Offline

# artifacts/indexer.py
index_run(store, run, jobfile)->list[Artifact]
detect_type(path)->ArtifactType
build_preview(path, atype)->dict         # csv head+shape / json keys / log head+tail / image thumb path

# memory/memory.py
query(store, jobfile_id=None, name=None, params=None, input_hashes=None)->dict
    # {has_jobfile, runs:int, exact_match_run_id, artifacts_dir, server, outcome, reuse_eligible}
reuse_candidate(store, jobfile, params, input_hashes)->Run|None

# expectations/contracts.py
evaluate(contract, run, artifacts, stdout, stderr)->tuple[Match, list[str], list[dict]]
    # returns (match, key_evidence, per_criterion[{id,text,passed,detail}])
default_contract(jobfile)->ExpectationContract   # seeds absence(NaN/Traceback/CUDA error)+expectation text
# expectations/distiller.py
propose(store, run, feedback, analyzer)->list[Criterion]
confirm(store, criterion_id)->Criterion          # status->confirmed, strength+=1

# backends/base.py
@dataclass SubmitResult: remote_job_id:str|None; workdir:str
@dataclass PollResult: state:State; resource:dict; last_log_mtime:float|None
@dataclass CollectResult: exit_code:int|None; stdout_path:str; stderr_path:str; \
    artifact_dir:str; resource_summary:dict
class Backend(ABC): name:str; submit(run,jobfile)->SubmitResult; poll(run)->PollResult; \
    collect(run)->CollectResult; cancel(run)->None
get_backend(backend:str, server:str|None, config)->Backend
select_backend(jobfile, servers:list[Server], override:dict|None)->tuple[str,str|None,str|None]

# notify/notify.py
class Notifier(ABC): notify(run, card:dict)->None
get_notifiers(config, run)->list[Notifier]       # LogNotifier always; +Webhook/Slack/Callback if configured

# monitor/monitor.py
class Monitor:
    __init__(store, config, analyzer, notifiers_factory)
    async run_loop(stop_event)                    # probe servers + poll runs
    probe_servers(); poll_runs(); on_terminal(run)
build_observation_card(run, jobfile, artifacts, match, key_evidence, health, analyzer)->dict

# api/client.py
class ApiClient(base_url): submit/get_run/list_runs/cancel/rerun/logs/artifacts/jobfiles/
    register/servers/feedback/expect/memory_query/await_run
ensure_daemon(config)->base_url                   # auto-start `jobctl serve` if down
```

Observation card shape (built by `build_observation_card`):
`{status, jobfile, run_id, server, artifacts:[{name,type,preview}], health, expectation_match,
key_evidence:[...], interpretation, recommended_next_action}`

---

## Tasks (dependency-ordered; each is TDD + commit)

### Task 0 — Scaffold
**Files:** `pyproject.toml`, `jobctl/__init__.py`, all empty subpackage `__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`.
- [ ] Create `pyproject.toml`: package `jobctl`, deps listed in Tech Stack, `[project.scripts] jobctl="jobctl.cli.main:app"`, pytest config (`testpaths=tests`).
- [ ] Create empty modules per File Structure so imports resolve.
- [ ] `tests/test_smoke.py`: `import jobctl` works and `jobctl.__version__` is a str.
- [ ] `pip install -e .` (or `uv pip install -e .`), run `pytest -q` → green. Commit.

### Task 1 — Foundation: config + models + store + jobfile
**Files:** `jobctl/config.py`, `jobctl/db/models.py`, `jobctl/db/store.py`, `jobctl/jobfile.py`, tests for each.
- [ ] Tests: config loads a temp `~/.cluster.yaml`-style file (servers/tasks/remote_path) + jobctl settings; defaults when absent.
- [ ] Tests: `Store` round-trips JobFile/Run/Artifact/Contract/Feedback/Server incl. JSON columns; `list_runs(state=...)` filters; `update_run` mutates state/health/card.
- [ ] Tests: `load_jobfile` parses a manifest; autowraps `foo.py`→`python foo.py`, `foo.sbatch`→sbatch, `foo.jl`→`julia`, etc.; `resolve_params` applies defaults/required/int-cast; `render_command` substitutes; `content_hash`/`input_hashes` deterministic and change when script bytes change.
- [ ] Implement; all green. Commit.

### Task 2 — Analysis layer (offline + deepseek + selector)
**Files:** `jobctl/analysis/{base,offline,deepseek}.py`, tests.
- [ ] Tests: `get_analyzer` returns Offline when no `DEEPSEEK_API_KEY`, DeepSeek when set (patched). `OfflineAnalyzer.analyze_run` builds deterministic interpretation + next-action from facts (no network). `propose_criteria` returns structured dicts. DeepSeek adapter: mock the `openai` client, assert it sends compact facts and parses JSON; never raises on facts-only input.
- [ ] Implement; green. Commit.

### Task 3 — Expectations engine + distiller
**Files:** `jobctl/expectations/{contracts,distiller}.py`, tests. (Depends on models + analysis.)
- [ ] Tests for `evaluate`: numeric extract-and-compare from csv/json/log (pass/near-threshold→weak/fail→bad); presence (glob exists); absence (NaN present→bad_signal); pattern; missing artifact→inconclusive; nonzero exit→failed. Returns key_evidence + per_criterion list.
- [ ] Tests for `default_contract` (seeds absence of NaN/Traceback/CUDA error + the manifest expectation text).
- [ ] Tests for distiller `propose` (uses a fake analyzer returning canned criteria → persisted as status=proposed,strength=1) and `confirm` (→confirmed, strength+1).
- [ ] Implement; green. Commit.

### Task 4 — Artifact indexer
**Files:** `jobctl/artifacts/indexer.py`, tests.
- [ ] Tests: write a temp workdir with png (tiny via Pillow), csv, json, .log, .bin; `index_run` discovers per patterns, checksums (sha256 stable), types correctly, previews (csv head+shape, json keys, log head+tail, image thumbnail file created), persists Artifacts linked to run.
- [ ] Implement; green. Commit.

### Task 5 — Run memory
**Files:** `jobctl/memory/memory.py`, tests.
- [ ] Tests: with seeded runs, `query` reports has_jobfile, run count, exact input_hash+params match → prior run_id + artifacts_dir + server + outcome; `reuse_candidate` returns a run only when exact match AND `expectation_match==USABLE` AND artifacts present, else None.
- [ ] Implement; green. Commit.

### Task 6 — Notify layer
**Files:** `jobctl/notify/notify.py`, tests.
- [ ] Tests: `get_notifiers` always includes LogNotifier; adds Webhook/Slack when config has urls; adds Callback when run has callback_url. Webhook/Slack/Callback `notify` POST the card (mock httpx, assert payload). No-config → only Log.
- [ ] Implement; green. Commit.

### Task 7 — Backends: base + local + ssh + slurm (+ fake-slurm harness)
**Files:** `jobctl/backends/{base,local,ssh,slurm}.py`, `tests/fakebin/{sbatch,squeue,sacct,scancel}`, tests.
- [ ] Tests `select_backend`: picks first pref whose server is online; honors override; falls back to local.
- [ ] Tests LocalBackend end-to-end: submit a `bash -c "echo hi; <emit csv/png>"` job in a temp workdir → poll→running/completed→collect exit_code 0, stdout captured, artifact_dir set.
- [ ] Tests SlurmBackend against `tests/fakebin` on PATH: `sbatch` prints a jobid, `squeue` cycles PD→R→(empty), `sacct` reports CD + resource fields; assert state mapping (PD→submitted,R→running,CD→completed,F→failed,TO→timeout,CA→cancelled), jobid capture, resource_summary parse, stdout/stderr collection, `cancel`→scancel called.
- [ ] SshBackend: unit-test command construction (rsync push/pull, nohup+pidfile, poll-by-pid) with a mocked runner; mark a real-cluster smoke test `@pytest.mark.cluster` (skipped by default).
- [ ] Implement; green. Commit.

### Task 8 — Monitor + observation cards
**Files:** `jobctl/monitor/monitor.py`, tests. (Depends on backends, artifacts, expectations, analysis, notify.)
- [ ] Tests: a fake backend drives a run pending→running→completed; `Monitor.poll_runs` updates state, on terminal runs the pipeline (index→evaluate→build_observation_card→notify), persists card + expectation_match. Stuck detection: running + stale log mtime + stale heartbeat → state STUCK/health STUCK. `probe_servers` updates Server rows from a fake prober (online/offline/resource_pressure).
- [ ] `build_observation_card` always returns all required fields; never just "finished".
- [ ] Implement; green. Commit.

### Task 9 — API daemon + client
**Files:** `jobctl/api/server.py`, `jobctl/api/client.py`, tests (FastAPI TestClient).
- [ ] Tests: POST /runs registers/loads jobfile, resolves params, attaches memory hint, selects backend, creates Run (pending), returns run dict; GET /runs & /runs/{id}; list filter by state; cancel/rerun; logs endpoint streams stdout/stderr file tail; artifacts endpoint; /jobfiles register+list; /servers; /feedback; /expect (list/confirm/propose); /memory/query. Monitor started on startup (use a fast tick + local backend) so a submitted local job reaches completed and a card appears.
- [ ] `ApiClient` methods hit each endpoint (TestClient transport); `await_run` long-polls to terminal; `ensure_daemon` starts uvicorn if `/health` unreachable.
- [ ] Implement; green. Commit.

### Task 10 — CLI
**Files:** `jobctl/cli/main.py`, tests (Typer CliRunner with a TestClient-backed ApiClient).
- [ ] Tests for each command's `--json` output shape: `run --wait` blocks to terminal and prints the observation card; `run --background` prints `{run_id}`; `await`, `status`, `inspect`, `artifacts`, `memory query` print JSON; `logs` prints text; `servers` prints a table/JSON; `cancel`/`rerun`/`register`/`jobfiles`/`feedback`/`expect` work; `serve` boots uvicorn (smoke).
- [ ] Implement; green. Commit.

### Task 11 — Web UI
**Files:** `jobctl/ui/templates/{dashboard,run,jobfile}.html`, `jobctl/ui/static/app.css`, UI routes in `api/server.py`, tests.
- [ ] Tests (TestClient): `/` dashboard renders server-health + run buckets (running/queued/stuck/weak-signal/completed/failed) from seeded data; `/runs/{id}` shows record, stdout/stderr tail, artifact previews (img tag for image, table for csv), observation card, per-criterion contract table; `/jobfiles/{id}` shows params schema, historical runs, contract versions; HTMX poll partial endpoint returns updated fragments.
- [ ] Implement; green. Commit.

### Task 12 — Integration & verification
**Files:** `jobctl/__init__.py` exports, `tests/test_e2e_local.py`, `README.md`.
- [ ] Wire package exports; ensure `jobctl --help` lists all commands.
- [ ] E2E test: register a tiny local jobfile that emits a csv+png and prints a value; `run --wait --json` → run reaches `completed`, artifacts indexed, contract classifies, observation card populated; `memory query` then reports the run and reuse eligibility; `rerun` works.
- [ ] Run full suite `pytest -q` → all green; fix any cross-layer mismatch. Write a short README (install, `jobctl serve`, `jobctl run`, UI URL). Commit.

---

## Self-Review — spec coverage
- CLI commands (req 1): Tasks 9–10. Blocking/background (req 2): Task 9 `await_run` + Task 10. JobFile Registry (req 3): Task 1. Run Registry (req 4): Task 1+9. local/ssh/slurm (req 5): Task 7. UI (req 6): Task 11. Health monitoring (req 7): Task 8. Artifact indexing (req 8): Task 4. Run Memory (req 9): Task 5. Cheap-model analysis (req 10): Task 2 + cards in Task 8. Expectation Contracts (req 11): Task 3. Observation Cards (req 12): Task 8. Notifications/callbacks (req 13): Task 6 + Task 9/10 JSON/await. MCP optional (req 14): satisfied by CLI/daemon/DB being the source of truth — no MCP in scope. Clean architecture (req 15): the module layout. No gaps.
