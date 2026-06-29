---
name: jobctl-followups
description: Use when working on jobctl product follow-ups: UI redesign/readability, Codex or subagent state-path writes under ~/.jobctl, default server scheduling policies such as oblix 5% idle CPU capacity, fixing monitor/inspect completion behavior, resolving job visibility mismatches where running views miss jobs that server status can see, or enforcing bug-report submission after unexpected jobctl behavior.
---

# jobctl Follow-Ups

Use this skill for jobctl development work that goes beyond simply running jobs.
The current open concerns come from the user's Codex migration request and the
Claude transcript for `/Users/yangqi/source/jobctl`.

## Source Context

- The original jobctl design was a complete thin slice across CLI, daemon,
  backends, monitor, artifacts, expectations, analysis, memory, notifications,
  and Jinja/HTMX UI.
- The source of truth is the CLI plus daemon plus SQLite database. MCP is
  optional and should not be required for core lifecycle management.
- Claude transcript evidence: session
  `7c9882fe-64ca-4278-9d41-8de9b2d8e6f3`, project
  `-Users-yangqi-source-jobctl`, especially the initial design/plan and later
  real E2E fixes around `run --wait`.
- The exact four follow-ups below are from the current user request, not exact
  quoted Claude-log lines.

## UI Redesign

Use the installed `frontend-design:frontend-design` skill for substantial UI
work when available. Apply it with jobctl-specific constraints:

- The UI is an operational monitoring tool, not a landing page.
- Prioritize readability, density, quick scanning, stable status colors, clear
  run identity, and obvious next actions.
- Avoid decorative redesigns that make job state, server capacity, logs,
  artifacts, and observation cards harder to compare.
- Verify desktop and mobile layouts with the in-app browser when a dev server
  or local UI target is available.

Relevant files usually include:

- `jobctl/ui/templates/base.html`
- `jobctl/ui/templates/dashboard.html`
- `jobctl/ui/templates/runs.html`
- `jobctl/ui/templates/run.html`
- `jobctl/ui/templates/partials/*.html`
- `jobctl/ui/static/app.css`
- `tests/test_task11_ui.py`

## State Root And ~/.jobctl Writes

If a Codex session, subagent, or test fails because it tries to write directly
to `~/.jobctl`, treat that as a product/interface problem to fix. Do not just
ask for broader filesystem permissions.

Preferred direction:

- Add or verify a configurable jobctl state root, for example via
  `~/.jobctl/config.toml`, an environment override, or an explicit CLI/API
  option.
- Keep direct persistent writes owned by the daemon/CLI layer when possible.
- Make logs, run directories, database, and generated files discoverable from
  the API/CLI so agents do not need to infer private paths.
- Add tests that can run with a temporary state root outside the user's home
  directory.

Relevant files usually include:

- `jobctl/config.py`
- `jobctl/api/client.py`
- `jobctl/api/server.py`
- `jobctl/backends/local.py`
- `jobctl/db/store.py`
- `tests/test_task1_foundation.py`
- `tests/test_e2e_local.py`

## Bug Reporting Contract

When jobctl itself behaves unexpectedly, create a local diagnostic bug report
after checking `~/.jobctl/cli.log` and `~/.jobctl/daemon.log`. This is required
for state mismatches, false stuck/running/completed states, missing observation
cards, crashes, broken JSON output, UI/API visibility mismatches, or sandbox
state-root failures. Reports include local log tails, so the default is a local
Markdown file for the current user to review; do not upload it unless the user
explicitly asks for `--submit`.

Use the top-level shortcut:

```bash
jobctl --report-bug "<what went wrong>" --report-run <run_id>
```

The subcommand form is equivalent:

```bash
jobctl report-bug "<what went wrong>" --run <run_id>
```

Omit the run flag when no run id exists. Add `--submit` only when the user
explicitly wants a GitHub issue created from those diagnostics. If you directly
fix and merge the jobctl bug in the same turn, report the PR or commit instead
of filing a separate issue.

## Default Scheduling Policy

Add a user-config-level default policy rather than hardcoding one-off server
rules. The motivating example is `oblix`: the user expects CPU kernel
submission to keep capacity saturated until roughly 5% remains idle.

Implementation should make these policy fields explicit and testable:

- server or task selector, such as `oblix` and `cpu`
- target idle capacity, such as `0.05`
- maximum concurrent submissions or queued jobs when needed
- interaction with explicit CLI overrides
- CPU-only jobs must never be placed on GPU partitions or GPU nodes; reserve
  H100/L4/RTX/MI300A and other accelerator nodes for jobs that explicitly
  use CUDA, ROCm/HIP, `nvidia-smi`, `--device cuda`, or another accelerator path
  in the executable command or resolved parameters. `--gres` or a GPU partition
  by itself is a resource request, not proof of a GPU workload; CPU-looking jobs
  that request GPU resources should be rejected before submission.
- behavior when `jobctl servers --json` reports stale or missing capacity data
- conservative fallback when server health is weak or unknown

Treat the 5% idle target as a configurable user policy, not a universal default.

Relevant files usually include:

- `jobctl/config.py`
- `jobctl/backends/base.py`
- `jobctl/backends/ssh.py`
- `jobctl/backends/slurm.py`
- `jobctl/monitor/prober.py`
- `jobctl/monitor/monitor.py`
- `jobctl/cli/main.py`
- `tests/test_prober.py`
- `tests/test_task7_backends.py`
- `tests/test_feature_wiring.py`

## Monitor, Inspect, And Observation Cards

If `jobctl inspect <run_id> --json` appears ineffective after completion, debug
the terminal pipeline end to end:

1. Confirm the backend moves the run into a terminal state.
2. Confirm `Monitor.poll_runs` observes that transition.
3. Confirm artifacts are indexed and expectation criteria are evaluated.
4. Confirm the observation card and expectation match are persisted in SQLite.
5. Confirm `inspect`, `status --json`, `await --json`, API JSON, and UI all read
   the same persisted state.
6. Inspect `~/.jobctl/cli.log` and `~/.jobctl/daemon.log` before concluding.

Do not claim the original design requirements are complete unless a terminal
run visibly produces a persisted observation card with state, health, artifacts,
expectation match, evidence, interpretation, and recommended next action.

Relevant files usually include:

- `jobctl/monitor/monitor.py`
- `jobctl/artifacts/indexer.py`
- `jobctl/expectations/contracts.py`
- `jobctl/analysis/offline.py`
- `jobctl/api/server.py`
- `jobctl/api/client.py`
- `jobctl/cli/main.py`
- `jobctl/db/store.py`
- `tests/test_task8_monitor.py`
- `tests/test_task10_cli.py`
- `tests/test_e2e_local.py`

## Running View And Server Status Mismatch

If `jobctl running`, dashboard running views, or API run listings show no jobs
while `jobctl servers --json` or server status can see active capacity, queued
work, or running jobs, treat it as a source-of-truth mismatch.

Debug this as a product issue, not as user confusion:

1. Identify whether the visible jobs are jobctl-managed runs, backend-native
   jobs discovered by probes, or stale server-status observations.
2. Check whether the run registry, daemon API, CLI `running` view, and UI use
   the same filters for active, queued, terminal, and externally discovered jobs.
3. Confirm server probes persist enough job identity for drill-down or explain
   why a server-visible job is not listed as a jobctl run.
4. Make empty states explicit when a view only shows jobctl-managed runs.
5. Add a regression test for the case where server status sees active work but
   the running view would otherwise render empty.

Relevant files usually include:

- `jobctl/cli/main.py`
- `jobctl/api/server.py`
- `jobctl/api/client.py`
- `jobctl/monitor/prober.py`
- `jobctl/monitor/monitor.py`
- `jobctl/backends/ssh.py`
- `jobctl/backends/slurm.py`
- `jobctl/ui/templates/dashboard.html`
- `jobctl/ui/templates/runs.html`
- `tests/test_task10_cli.py`
- `tests/test_prober.py`
- `tests/test_task11_ui.py`
