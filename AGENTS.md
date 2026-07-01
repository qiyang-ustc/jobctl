# jobctl Codex Guidance

Use the global `$jobctl` skill when running, submitting, monitoring, inspecting,
or recovering research jobs from this repository. Read `README.md` before using
the CLI model in a new thread.

For product-development work on jobctl itself, especially the open follow-ups
below, use the repo skill `$jobctl-followups` if it is available.

Open follow-up requirements:

- UI readability needs a redesign pass. Use the installed
  `$frontend-design` / `frontend-design:frontend-design` skill for UI work, but
  keep jobctl's dashboard operational: dense, scan-friendly, restrained, and
  suitable for repeated monitoring rather than a marketing-style page.
- Subagents or Codex sessions that fail because jobctl writes directly under
  `~/.jobctl` indicate an interface/configuration problem to solve. Prefer a
  configurable state root, daemon/CLI mediated writes, and clear docs over
  ad hoc direct writes to the home directory.
- Add a user-config-level default scheduling policy. For example, the `oblix`
  CPU policy should be configurable to keep submitting CPU kernels until only
  about 5% capacity remains idle, while still respecting explicit user caps and
  server health.
- Treat ineffective completion monitoring or `jobctl inspect <run_id> --json`
  as a core design gap. A completed run should have state, health, artifacts,
  expectation match, observation card, and recommended next action persisted
  and visible through `inspect`, `status --json`, `await --json`, the API, and
  the UI.
- If `jobctl running` or the UI cannot see jobs while `jobctl servers` /
  server status can see active capacity or jobs, treat it as a run visibility
  and source-of-truth bug. Debug the run registry, backend probes, daemon API,
  and UI query path before assuming no jobs exist.

When debugging jobctl itself, inspect `~/.jobctl/cli.log` and
`~/.jobctl/daemon.log` before guessing. If jobctl misreports state, marks a run
stuck incorrectly, crashes, fails to persist the terminal observation card, or
otherwise behaves differently from the documented contract, agents must create
a local bug report with diagnostics instead of only mentioning it in chat:
`jobctl --report-bug "<what went wrong>" --report-run <run_id>`. The subcommand
form `jobctl report-bug "<what went wrong>" --run <run_id>` is equivalent. If no
run id exists, omit the run flag. Reports are saved locally for the current user
to review; use `--submit` only when the user explicitly asks to upload the
diagnostics. If the agent fixes the bug in the same turn, the PR/commit is the
tracking record and should be reported explicitly.

The user authorizes agents to cancel jobctl-managed validation, smoke, or pilot
runs that the same agent submitted in the current task when those runs are no
longer useful, are blocking cleanup, or were superseded by a code fix. Use
`jobctl cancel <run_id> --agent-owned-validation --reason "<why>" --json` so
the reason is recorded in the run. Do not cancel production runs, user-submitted
runs, scheduler-only jobs, or runs with unclear ownership without asking the
user first.
