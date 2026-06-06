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
jobctl await  <run_id> --json               # block on a backgrounded run
jobctl status <run_id>                      # state + health
jobctl logs / artifacts / inspect <run_id>  # look at what came back
jobctl memory query --name <job> --json     # has this run before? can I reuse it?
jobctl servers --json                       # cluster health (online, load, queue)
```

`jobctl --help` lists the rest. A web UI runs at `http://127.0.0.1:7421`
(server health, run buckets, per-run logs / artifacts / observation cards).

## Logs & reporting bugs

Every invocation leaves a trail in `~/.jobctl/`:
- `cli.log` — every `jobctl` command
- `daemon.log` — the daemon + monitor loop (state polls, unreachable warnings, terminal pipeline)

Hit a jobctl bug? File it straight from the CLI — it bundles the version, log
tails, the run record, and recent failures, then opens a GitHub issue:

```bash
jobctl report-bug "monitor marked my running job stuck" --run <run_id>
```

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

Optional: set `DEEPSEEK_API_KEY` for richer (cheap-model) run narration —
without it, jobctl falls back to a built-in offline analyzer.
