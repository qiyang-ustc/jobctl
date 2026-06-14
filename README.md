# jobctl

Run any research job — a script, an `sbatch` file, a container, an instrument
command — on your laptop or a remote cluster (`ssh` / `slurm`) **as if it were a
local command**, and get back a *result you can act on*, not just "done".

You give jobctl a **JobFile** (any executable + how to run it + what counts as
success). It submits, watches health, collects artifacts, checks the output
against an expectation, and returns an **observation card**: what happened,
whether the signal is usable / weak / bad, and the recommended next step.

It's CLI-first and every command speaks `--json`, so an LLM agent can drive it
straight from the shell. The daemon and dashboard use the same SQLite state, so
CLI, API, monitor, and UI views do not need separate bookkeeping.

## Use it

```bash
jobctl run job.jobfile.yaml --wait --json   # submit, block, get the observation card
jobctl run job.jobfile.yaml                 # background → prints a run_id
jobctl run job.jobfile.yaml --title "chi sweep χ=64" --tag sweep   # name what it's for
jobctl run job.jobfile.yaml --mem-auto      # CPU OOM → retry with larger --mem; GPU OOM → stop + notify
jobctl await  <run_id> --json               # block on a backgrounded run
jobctl status <run_id> --json               # state + health (+ title/tags)
jobctl inspect <run_id> --json              # persisted observation card + record
jobctl logs <run_id> --json                 # stdout/stderr tail
jobctl artifacts <run_id> --json            # indexed artifacts
jobctl memory query --name <job> --json     # has this run before? can I reuse it?
jobctl servers --json                       # cluster health, queue, capacity, policy
```

`jobctl --help` lists the rest. Give a run a human identity with
`--title` / `--note` / `--tag` (or `PATCH /runs/{id}`); it shows up across the
CLI, the web panel, and the JSON API as `display_title` (with a sensible
`<jobfile> · key params` fallback) so a run is never just a hash.

## Dashboard

The daemon serves a dashboard at `http://127.0.0.1:7421`.

It is built for repeated monitoring rather than presentation:
- top summary: managed active runs, scheduler-visible jobs, attention count,
  and completed history
- server cards: online state, SLURM queue, idle CPU capacity, configured policy,
  and scheduler-visible jobs
- run lanes: jobctl-managed runs plus explicit scheduler-only rows when server
  status can see jobs that are not in the run registry
- run detail: logs, artifacts, observation card, SLURM request, resources, and
  expectation criteria
- configuration panel: resolved state paths, daemon host/port, analyzer,
  notification settings, default policies, and configured servers

Long server notes and config snippets are collapsed by default so operational
text does not take over the page.

## Configuration

Set `JOBCTL_HOME=/path/to/state` to move the whole state root: config lookup,
DB, logs, and run mirrors all use that root. The same can be pinned in TOML:

```toml
[jobctl]
state_root = "/scratch/you/jobctl-state"
daemon_port = 7421
```

Remote backends still read server definitions from `~/.cluster.yaml` by
default. jobctl shows the resolved `config.toml` and `cluster.yaml` paths in the
dashboard configuration panel so agents do not need to guess where state lives.

Default scheduling policy is user-configured and exposed in `jobctl servers
--json` plus the web UI. For an `oblix` CPU fill policy that keeps about 5% of
cluster CPU capacity idle for CPU kernels:

```toml
[jobctl.default_policies.oblix]
mode = "cpu_fill_idle"
target_idle_pct = 5
kernel_cpus = 1
```

When SLURM `sinfo` capacity is available, server status reports idle/total CPU
and the policy row computes how many kernel slots can be submitted while
preserving the configured idle reserve. If capacity is missing or stale, the UI
reports that explicitly instead of substituting login-node CPU load.

On macOS, jobctl posts a desktop notification when a run (or a burst of runs)
finishes; disable with `notify_macos_enabled = false` under `[jobctl]`.

## Observation Cards

A terminal run should be inspectable from every surface:
- `jobctl run --wait --json`
- `jobctl await <run_id> --json`
- `jobctl status <run_id> --json`
- `jobctl inspect <run_id> --json`
- `GET /runs/{run_id}`
- the run detail page

The persisted observation card carries the real state, health, artifacts,
expectation match, key evidence, interpretation, and recommended next action.
Even backend submit failures now persist a minimal card so a failed run is not
just an opaque state transition.

## Logs & reporting bugs

Every invocation leaves a trail in `$JOBCTL_HOME` (default `~/.jobctl/`):
- `cli.log` — every `jobctl` command
- `daemon.log` — the daemon + monitor loop (state polls, unreachable warnings,
  terminal pipeline)

Hit a jobctl bug? File it straight from the CLI — it bundles the version, log
tails, the run record, and recent failures, then opens a GitHub issue:

```bash
jobctl report-bug "monitor marked my running job stuck" --run <run_id>
```

Run dirs are mirrored under the configured `run_dir`; `jobctl gc` clears ones
with no DB record (`--dry-run` to preview).

## A JobFile

```yaml
name: my-experiment
command: "python train.py --lr {lr}"
params:  { lr: { type: float, default: 0.001 } }
artifacts: ["*.csv", "checkpoints/**"]
expectation: "final loss below 0.1"
```

Or skip the manifest and point it at a bare script — `jobctl run train.py` —
and it wraps `.py` / `.sh` / `.jl` / `.R` / `.sbatch` automatically.

## Install

```bash
pip install -e .       # Python 3.11+
```

Optional: set `GEMINI_API_KEY` (or `DEEPSEEK_API_KEY`) for richer cheap-model
run narration — without a key, jobctl falls back to a built-in offline analyzer.
Gemini uses `gemini-2.5-flash-lite` by default (override with `GEMINI_MODEL`).
