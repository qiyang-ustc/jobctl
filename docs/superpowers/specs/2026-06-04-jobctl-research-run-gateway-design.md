# jobctl — JobFile-native Research Run Gateway

**Date:** 2026-06-04
**Status:** Approved design — ready for implementation planning

## Purpose

A lightweight, CLI-first gateway that lets LLM agents (and humans) run **remote
research jobs as if they were local commands**, while the platform manages
lifecycle, monitoring, artifacts, health, expectations, cheap-model analysis,
and structured result return.

The system is **not model-centric**. The primary compute unit is a **JobFile** —
any executable research job: sbatch, shell, Python, R, Julia, MATLAB, binary,
simulation input, container command, or custom instrument script.

MCP is explicitly **optional**: the source of truth is the CLI + daemon +
database. MCP can be bolted on later as a model-facing adapter without being
required for core lifecycle management.

## Decisions (locked during brainstorming)

- **Stack:** Python — Typer CLI + FastAPI daemon + SQLite + Jinja/HTMX UI.
- **Scope:** a **complete thin vertical slice** across all 15 requirement areas
  — every layer present and wired end-to-end, none gold-plated — then extend.
- **Cheap-model analysis:** **DeepSeek** (OpenAI-compatible API) as default,
  with a deterministic **offline analyzer** fallback so the system works with no
  API key and tests cost zero tokens. Analyzer is pluggable.
- **Backend validation:** real **cheap, short jobs on oblix (ssh/CPU) and
  hipster (slurm)**; deterministic local tests use a fake `sbatch/squeue/...`.

## Non-Goals (YAGNI for this slice)

- Model training/serving lifecycle (this is job-centric, not model-centric).
- Multi-user auth / RBAC (single-user, localhost daemon).
- Distributed/HA daemon; the daemon is a single local process.
- A SPA front-end with a build pipeline (server-rendered HTMX only).
- Full email/GitHub notification implementations (interface + stubs only).

---

## 1. Architecture

```
jobctl/
  cli/         Typer commands -> thin HTTP client to the daemon
  api/         FastAPI daemon: REST + UI + monitor loop; owns the DB
  db/          SQLite schema + repository (only the daemon writes -> no lock fights)
  jobfile.py   manifest parse + bare-script auto-wrap + versioning/hashing
  backends/    base ABC + local / ssh / slurm + selector
  monitor/     asyncio loop: server health + run-state transitions + stuck/stale
  artifacts/   discover -> checksum -> type -> preview
  expectations/ versioned contracts + criteria engine + distiller
  analysis/    Analyzer ABC + deepseek adapter + offline fallback
  memory/      has-run-before / input-hash match / reuse decision
  notify/      UI always + optional slack/webhook (email/github stubbed)
  ui/          Jinja + HTMX, polling (no SPA build step)
  config.py    loads ~/.cluster.yaml + ~/.jobctl/config
```

**Daemon-centric flow.** The FastAPI daemon holds SQLite, runs the monitor as a
background asyncio task, and serves REST + UI. The CLI is a thin localhost HTTP
client that auto-starts the daemon if it is down. Only the daemon writes to
SQLite, avoiding lock contention.

- `run --wait` submits, then long-polls `/runs/{id}` until a terminal state, so
  it **behaves like a local command** from the agent's perspective.
- `run --background` returns a `run_id` handle immediately; `await <run_id>`
  blocks on it later.

```
Agent/User -> jobctl CLI -> HTTP -> FastAPI daemon
                                      |- DB (SQLite)
                                      |- Monitor loop (asyncio)
                                      |- Backends (local/ssh/slurm)
                                      |- Artifact indexer
                                      |- Expectation engine + DeepSeek/offline analyzer
                                      |- Notify (UI + optional webhook/callback)
                                      \- UI (Jinja/HTMX)
```

---

## 2. The JobFile

A JobFile is a `.jobfile.yaml` manifest wrapping any executable:

```yaml
name: ipeps-opt
command: "julia +1.11 {script} --chi {chi} --D {D}"
params:
  script: { type: path, required: true }
  chi:    { type: int, default: 40 }
  D:      { type: int, default: 4 }
backends:
  - { backend: slurm, server: hipster, task: gpu1_l4 }
  - { backend: ssh,   server: oblix,   task: cpu }
  - { backend: local }
artifacts:
  - "*.png"
  - "energy*.csv"
  - "out/**/*.json"
expectation: "energy converges below -0.66/site; no NaNs in logs"
```

- **Bare-script mode:** `jobctl run train.sbatch` is accepted directly. The
  command is auto-inferred from the extension (`.sbatch`->`sbatch`,
  `.py`->`python`, `.sh`->`bash`, `.jl`->`julia`, `.m`->`matlab`,
  `.R`->`Rscript`) and an implicit JobFile is registered.
- **Versioning:** the JobFile `version` is bumped when the content hash of
  `command_template` + referenced script changes. Old runs keep their version.
- The free-text `expectation` seeds the JobFile's Expectation Contract v1.

---

## 3. Data model (SQLite)

**jobfiles**: `id, name, version, source_path, command_template, params_schema(JSON),
backend_prefs(JSON), artifact_patterns(JSON), expectation_contract_id, content_hash,
created_at`

**runs**: `run_id, jobfile_id, jobfile_version, params(JSON), input_hashes(JSON),
backend, server, task, remote_job_id, state, health, exit_code, submitted_at,
started_at, finished_at, last_heartbeat, workdir, stdout_path, stderr_path,
resource_summary(JSON), expectation_match, observation_card(JSON)`

**artifacts**: `id, run_id, remote_path, local_path, type, size, checksum, preview(JSON),
created_at`

**expectation_contracts**: `id, jobfile_id, version, criteria(JSON), source, created_at,
updated_at`

**feedback**: `id, run_id, kind(accept|reject|note), text, created_at`

**servers**: `name, backend_type, online, last_heartbeat, cpu, mem, gpu, disk,
slurm_queue(JSON), note`

### Enums

- `state` ∈ `pending · submitted · running · completed · failed · cancelled · stuck · timeout`
- `health` ∈ `ok · weak · no_heartbeat · resource_pressure · stuck`
- `expectation_match` ∈ `usable · weak_signal · bad_signal · inconclusive · failed`
- `artifact.type` ∈ `image · plot · csv · json · text_log · binary · other`

---

## 4. Backend adapters

Interface (ABC):

```python
class Backend(ABC):
    def submit(self, run, jobfile) -> SubmitResult:   # remote_job_id, workdir
    def poll(self, run) -> PollResult:                # state + resource snapshot
    def collect(self, run) -> CollectResult:          # stdout/stderr, artifact paths, exit code
    def cancel(self, run) -> None:
```

- **local** — subprocess in `~/.jobctl/runs/<run_id>/`; PID tracking; poll via
  PID liveness; collect from the workdir.
- **ssh** (e.g. oblix) — rsync inputs to `remote_path`, launch via `nohup` and
  record the remote PID, poll over ssh, rsync artifacts back. Honors per-server
  notes from `~/.cluster.yaml` (oblix: VPN, conservative resources).
- **slurm** (e.g. hipster) — ssh + render the resolved command into the chosen
  `~/.slurm` task template, `sbatch` it, capture the jobid, poll
  `squeue`/`sacct` mapping SLURM states (`PD/R/CG/CD/F/TO/CA`) to run states,
  collect stdout/stderr + a `sacct` resource summary. **Stuck** detection:
  state `running` but stdout mtime stale beyond a threshold and no heartbeat.

**Selector:** chooses the first backend in the JobFile's `backend_prefs` whose
server is online (per the monitor's server table), unless `--backend/--server`
override.

---

## 5. Monitor layer

A single asyncio loop inside the daemon:

- **Server health** (every `T_server` s): probe each known server —
  `uptime/nproc/free/df` and `nvidia-smi` if present; slurm servers also
  `squeue -u`. Update the `servers` row: online/offline, heartbeat freshness,
  resource pressure.
- **Run state** (every `T_run` s): for each non-terminal run call
  `backend.poll()`, update `state`/`health`; detect **stuck** (running but no
  log growth + stale heartbeat) and **stale** (submitted but never started
  within a window). On a terminal transition, trigger the completion pipeline
  (collect -> index artifacts -> analyze -> observation card -> classify ->
  notify).

---

## 6. Artifact layer

`indexer.discover(run)` globs the JobFile's `artifact_patterns` in the run's
local-mirror workdir. For each match: compute `sha256`, determine `type` from
extension + magic bytes, record `size`, and build a `preview`:

- **image/plot** (png/jpg/svg/pdf): store path; generate a raster thumbnail
  (Pillow); UI renders it.
- **csv**: head N rows + shape (rows x cols).
- **json**: top-level keys (and inline small objects).
- **text_log**: head + tail.
- **binary/other**: size + detected magic type only.

Artifacts are linked to the run and surfaced in the observation card + UI.

---

## 7. Run Memory

`memory.query` answers, before launching:

- Has this **JobFile** run before? (by id/name)
- Has this **input_hash + params** combination run before? (exact -> prior
  `run_id` + outcome)
- Where are the artifacts? Which server ran it? What was the outcome
  (usable/weak/bad/failed/inconclusive)?
- **Can a previous result be reused?** — if an exact input+params match exists
  with `expectation_match = usable` and artifacts present, reuse is suggested;
  `run --reuse` short-circuits to the prior run.

Exposed as `jobctl memory query ... --json` and consulted automatically before
each launch (returned as a "memory hint" in the run response).

---

## 8. Expectation Contracts (source of truth) + cheap-model analysis (narration only)

**The criteria engine is the source of truth.** Machine-checkable criteria:

- `numeric`: extract a value (regex / jsonpath / csv-column) from a named
  artifact or log and compare to a threshold (e.g. "final energy < -0.66").
- `presence`: an artifact matching a glob exists.
- `absence`: a pattern ("NaN", "Traceback", "CUDA error") does **not** appear in
  logs.
- `pattern`: a regex must match.

**Classification** (deterministic) -> `expectation_match`:

- `failed` — non-zero exit / backend failure.
- `usable` — all confirmed criteria pass.
- `bad_signal` — an `absence` criterion violated, or a hard `numeric` fails.
- `weak_signal` — structurally fine but a numeric is near threshold / some
  non-hard criteria unmet.
- `inconclusive` — criteria could not be evaluated (missing/unparsable artifact).

**Contracts are versioned and linked to JobFiles.** Criteria carry
`{id, text, kind, check, status: proposed|confirmed, strength, evidence_run_ids}`.

**Distiller.** When the user gives feedback (`jobctl feedback <run_id>
--accept|--reject "text"`), the distiller proposes new/updated criteria from:
user NL feedback, accepted/rejected artifacts, historical runs, similar
failures, JobFile context, and agent goals. Proposals start
`status=proposed, strength=1`. `jobctl expect confirm <criterion_id>` promotes to
`confirmed` and increments `strength`. **User confirmation strengthens the
criterion.**

**Cheap-model analyzer — never the source of truth.** It consumes only compact
**structured facts** and returns JSON. Responsibilities:

- observation-card interpretation, concise log summary,
- possible bad-signal explanation, next-action suggestion,
- **proposed** expectation criteria (human-confirmed), historical-failure
  summary.

Adapters:

- **DeepSeekAnalyzer** — OpenAI-compatible client (`base_url=https://api.deepseek.com`,
  `model=deepseek-chat`), key from `DEEPSEEK_API_KEY`.
- **OfflineAnalyzer** — deterministic templates from structured facts; used when
  no key is set and in all tests (zero token cost).

---

## 9. Observation Cards

Built for **every** terminal run — never just "finished". The engine fills the
facts; the analyzer fills the narrative fields:

```json
{
  "status": "completed",
  "jobfile": "ipeps-opt v3",
  "run_id": "...",
  "server": "hipster",
  "artifacts": [{ "name": "...", "type": "csv", "preview": "..." }],
  "health": "ok",
  "expectation_match": "usable",
  "key_evidence": ["final energy=-0.662 < -0.66", "no NaN in log"],
  "interpretation": "Converged cleanly; energy meets target.",
  "recommended_next_action": "Accept; try chi=60 for a tighter bound."
}
```

Returned by `inspect`, `status --json`, and `await --json`.

---

## 10. CLI (Typer)

Spec commands (all support `--json` where specified):

```
run <jobfile> [--param k=v ...] [--wait|--background] [--backend B] [--server S]
              [--reuse] [--callback URL] [--json]
await <run_id> [--timeout S] [--json]
status <run_id> [--json]
logs <run_id> [--follow] [--stderr]
artifacts <run_id> [--json]
inspect <run_id> [--json]
cancel <run_id>
rerun <run_id>
servers [--json]
memory query [--jobfile X] [--params ...] [--json]
```

Minimal enablers (needed to exercise the 15 layers, kept thin):

```
serve                                  # start the daemon
register <jobfile>                     # register/version a JobFile
jobfiles [--json]                      # list JobFiles + history
feedback <run_id> --accept|--reject [text]
expect [list|confirm <id>|propose] <jobfile>
```

---

## 11. Notifications & callbacks

- **UI** — always reflects current state.
- **Agents** — structured JSON (`--json`), `await` long-poll handles, stored
  retrieval (`inspect`), and an optional `run --callback URL` that POSTs the
  observation card on terminal state.
- **Humans** — optional Slack/generic webhook fired on terminal/`bad_signal`;
  email + GitHub sit behind the same `Notifier` interface as stubs.

---

## 12. Web UI (Jinja + HTMX, polling)

- **Dashboard** — server-health cards (online, cpu/mem/gpu/disk, slurm queue)
  and run buckets: running / queued / **stuck** / **weak-signal** / completed /
  failed.
- **Run detail** — full record, stdout/stderr tail (auto-refresh), artifact
  previews (image thumbnails, csv head), observation card, expectation contract
  with per-criterion pass/fail.
- **JobFile detail** — metadata, params schema, historical runs, contract
  versions.

HTMX polling for live updates; no SPA build step.

---

## 13. Testing strategy (TDD)

- **Unit:** jobfile parse/version/hash, criteria engine + classification,
  artifact typing/preview, memory matching, offline analyzer, backend selector.
- **Integration (local):** real short local jobs end-to-end
  (submit -> monitor -> collect -> index -> card -> classify).
- **Slurm adapter (deterministic):** a fake `sbatch/squeue/sacct/scancel` on
  `PATH` to exercise state mapping and collection without a cluster.
- **Real validation (user-driven):** cheap `sleep + emit tiny csv/png` jobs on
  **oblix** (ssh/CPU, conservative) and **hipster** (slurm, capacity/l4
  partition, `account=linuxusers`) to confirm ssh/slurm adapters, artifact
  rsync-back, and observation cards against real infrastructure.

---

## 14. Build approach

After spec approval -> author an implementation plan -> execute it with a
**Workflow** of **Sonnet** subagents, layer by layer, tests-first, with a final
verification pass. This keeps per-token cost down per the user's instruction.
```

