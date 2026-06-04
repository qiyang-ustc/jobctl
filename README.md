# jobctl

A CLI-first, JobFile-native gateway that runs research jobs locally or on
remote clusters and manages their full lifecycle: monitoring, artifact
indexing, expectation contracts, cheap-model analysis, and result delivery.

## Install

```bash
pip install -e .
```

Requires Python 3.11+. Optional: `DEEPSEEK_API_KEY` for AI-powered run analysis.

## Quick Start

### 1. Start the daemon

```bash
jobctl serve
```

Starts the FastAPI daemon on `http://127.0.0.1:7421` with a SQLite store and
asyncio monitor loop.

### 2. Register a JobFile

Create a manifest `my_job.jobfile.yaml`:

```yaml
name: my-experiment
command: "python train.py --lr {lr} --epochs {epochs}"
params:
  lr:
    type: float
    default: 0.001
  epochs:
    type: int
    default: 10
backends:
  - backend: local
artifacts:
  - "*.csv"
  - "*.png"
  - "checkpoints/**"
expectation: "loss below 0.1 at epoch 10"
```

Register it:

```bash
jobctl register my_job.jobfile.yaml
```

### 3. Submit a run

```bash
# Background (returns run_id immediately)
jobctl run my-experiment --param lr=0.01 --param epochs=20

# Wait for completion and print the observation card
jobctl run my-experiment --param lr=0.01 --wait --json
```

### 4. Inspect results

```bash
jobctl status <run_id>
jobctl logs <run_id>
jobctl artifacts <run_id> --json
jobctl inspect <run_id> --json
```

### 5. Query run memory

```bash
jobctl memory query --name my-experiment --json
```

Returns: `{has_jobfile, runs, exact_match_run_id, reuse_eligible, outcome, ...}`

### 6. Rerun

```bash
jobctl rerun <run_id>
```

### 7. Web UI

Open `http://127.0.0.1:7421` in a browser for:
- Server health cards
- Run buckets (running / queued / stuck / weak-signal / completed / failed)
- Per-run detail: logs, artifact previews, observation card, expectation criteria
- JobFile detail: params schema, run history, contract versions

## Commands

| Command | Description |
|---------|-------------|
| `jobctl serve` | Start the daemon |
| `jobctl run <jobfile>` | Submit a run (`--wait` to block) |
| `jobctl await <run_id>` | Block until terminal state |
| `jobctl status <run_id>` | Current state |
| `jobctl logs <run_id>` | Tail stdout/stderr |
| `jobctl artifacts <run_id>` | List artifacts |
| `jobctl inspect <run_id>` | Full run record + observation card |
| `jobctl cancel <run_id>` | Cancel a run |
| `jobctl rerun <run_id>` | Copy params and resubmit |
| `jobctl servers` | Server health rows |
| `jobctl register <path>` | Register a JobFile |
| `jobctl jobfiles` | List registered JobFiles |
| `jobctl feedback <run_id> --text "..."` | Post feedback |
| `jobctl expect` | List expectation contracts |
| `jobctl expect propose <run_id>` | Propose criteria from feedback |
| `jobctl expect confirm <criterion_id>` | Confirm a criterion |
| `jobctl memory query` | Query run memory |

All structured commands support `--json` for machine-readable output.

## Backends

- **local** — subprocess in a temp workdir (default)
- **ssh** — rsync + nohup on a remote host
- **slurm** — `sbatch` / `squeue` / `sacct` on an HPC cluster

Configure servers in `~/.cluster.yaml` (standard cluster config) or
`~/.jobctl/config.toml`.

## Architecture

```
CLI (Typer)  →  ApiClient (httpx)  →  FastAPI daemon
                                           │
                                    ┌──────┴──────┐
                                 SQLite DB    Monitor loop
                                 (Store)      (asyncio)
                                    │              │
                              Backends ←──── poll_runs()
                              (local/ssh/slurm)    │
                                            on_terminal()
                                                   │
                              Artifact Indexer ────┤
                              Expectation Engine ──┤
                              Analyzer (offline) ──┘
```
