# jobctl

Run any research job — a script, an `sbatch` file, a container, an instrument
command — on your laptop or a remote cluster (`ssh` / `slurm`) **as if it were a
local command**, and get back a *result you can act on*, not just "done".

You give jobctl a **JobFile** (any executable + how to run it + what counts as
success). It submits, watches health, collects artifacts, checks the output
against an expectation, and returns an **observation card**: what happened,
whether the signal is usable / weak / bad, and the recommended next step.

It's CLI-first and every command speaks `--json`, so an LLM agent can drive it
straight from the shell.

## Use it

```bash
jobctl run job.jobfile.yaml --wait --json   # submit, block, get the observation card
jobctl run job.jobfile.yaml                 # background → prints a run_id
jobctl run job.jobfile.yaml --title "chi sweep χ=64" --tag sweep   # name what it's for
jobctl await  <run_id> --json               # block on a backgrounded run
jobctl status <run_id>                      # state + health (+ title/tags)
jobctl logs / artifacts / inspect <run_id>  # look at what came back
jobctl memory query --name <job> --json     # has this run before? can I reuse it?
jobctl servers --json                       # cluster health (online, load, queue)
```

`jobctl --help` lists the rest. Give a run a human identity with
`--title` / `--note` / `--tag` (or `PATCH /runs/{id}`); it shows up across the
CLI, the web panel, and the JSON API as `display_title` (with a sensible
`<jobfile> · key params` fallback) so a run is never just a hash.

A web UI runs at `http://127.0.0.1:7421` — server health, run list, per-run
logs / artifacts / observation cards. On macOS, jobctl posts a desktop
notification when a run (or a burst of runs) finishes; disable with
`notify_macos_enabled = false` under `[jobctl]` in `~/.jobctl/config.toml`.

## Logs & reporting bugs

Every invocation leaves a trail in `~/.jobctl/`:
- `cli.log` — every `jobctl` command
- `daemon.log` — the daemon + monitor loop (state polls, unreachable warnings, terminal pipeline)

Hit a jobctl bug? File it straight from the CLI — it bundles the version, log
tails, the run record, and recent failures, then opens a GitHub issue:

```bash
jobctl report-bug "monitor marked my running job stuck" --run <run_id>
```

Run dirs are mirrored under `~/.jobctl/runs/`; `jobctl gc` clears ones with no
DB record (`--dry-run` to preview).

## A JobFile

```yaml
name: my-experiment
command: "python train.py --lr {lr}"
params:  { lr: { type: float, default: 0.001 } }
artifacts: ["*.csv", "checkpoints/**"]
expectation: "final loss below 0.1"
```

Or skip the manifest and point it at a bare script — `jobctl run train.py` —
and it wraps `.py` / `.sh` / `.jl` / `.R` / `.sbatch` automatically. Remote
backends (`ssh` / `slurm`) read server config from `~/.cluster.yaml`.

## Install

```bash
pip install -e .       # Python 3.11+
```

Optional: set `GEMINI_API_KEY` (or `DEEPSEEK_API_KEY`) for richer cheap-model
run narration — without a key, jobctl falls back to a built-in offline analyzer.
Gemini uses `gemini-2.5-flash-lite` by default (override with `GEMINI_MODEL`).
