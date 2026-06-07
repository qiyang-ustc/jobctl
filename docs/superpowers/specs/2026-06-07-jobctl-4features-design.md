# jobctl — 4-feature enhancement design (2026-06-07)

Four user-requested enhancements, built on the maps + research in workflow `wf_ff262db9-4a6`.

## 1. Human-readable run identity (foundational — do first)

**Problem:** runs are opaque `run-<12hex>` hashes; you can't tell *what* a run was doing.

**Design:**
- Add three optional fields to `Run` (`db/models.py`) + `DDL_RUNS` + `_migrate()` ALTER:
  `title: str|None`, `note: str|None`, `tags: list[str]|None` (tags stored as JSON TEXT).
  Stored *in the run row* (not derived) so they survive jobfile rename/delete.
- `store.py`: add/update/_row_to_run round-trip + migration for existing DBs (NULL-safe).
- API submit (`server.py`) accepts `title`/`note`/`tags`; `_run_to_dict` serializes them.
- `client.py submit()` forwards them.
- CLI `run`: `--title`, `--note`, repeatable `--tag`; background output prints `run_id — title`.
- **Display fallback** (`_display_title`): `title or "<jobfile_name> · <≤3 key params>"` computed at
  serialization so *old* runs and untitled new runs still read well — never just a hash.
- No batch schema; "series" handled in notifications (feature 3) by time-window coalescing.

## 2. macOS local notifications (single + series)

**Design (`jobctl/notify/macos.py`, new):**
- `notify_macos(message, *, title, subtitle, sound, timeout)` — zero-dep `osascript` via
  `subprocess.run`, **injection-safe** (all user text passed as AppleScript `on run argv` items,
  never interpolated). Gated on `sys.platform=="darwin"` + `shutil.which("osascript")`; silent
  no-op elsewhere; `geteuid()==0` → `launchctl asuser <console_uid>`; 7s timeout, DEVNULL streams.
- `summarize_terminal_events(events)` — pure: 1 event → "✅ completed: <title>"; N → "N jobs
  finished (✅k ❌m)" + subtitle of a few titles. (Testable without a GUI.)
- `MacNotifyCoalescer(window=15s)` — async debounce: `add(event)` accumulates terminal events for
  `window`s then fires ONE notification via `to_thread(notify_macos,…)`. This is the "series" signal.
- Config: `notify_macos_enabled` (default **True** — user asked for it; harmless no-op off-mac),
  `notify_sound` (default `None`). Wired `cli serve` → `create_app` → monitor.
- `monitor.on_terminal`: if enabled + available, `coalescer.add({title,state,match})`. Fires on the
  NEW-transition path only (avoid double-fire on STUCK→reconcile).

## 3. Gemini summarize/report API hook

**Design (`jobctl/analysis/gemini.py`, new):**
- `GeminiAnalyzer(Analyzer)` implementing all 6 ABC methods, mirroring `DeepSeekAnalyzer` but using
  **httpx only** (no SDK): `POST v1beta/models/{model}:generateContent`, header `x-goog-api-key`,
  model `gemini-2.5-flash-lite` (env `GEMINI_MODEL` override), `maxOutputTokens` cap, temp 0.2.
- Robust: any `httpx.HTTPError|KeyError|IndexError|ValueError` → offline-style fallback (reuse the
  deepseek `_parse_json_or_fallback` pattern). Timeout 12s so it can't stall the monitor.
- `analysis/base.py get_analyzer()`: add `elif GEMINI_API_KEY → GeminiAnalyzer` (after DeepSeek).
- `config.py`: `gemini_api_key` field + env load. **Key value is never read by me** — only the name
  (`GEMINI_API_KEY`) was confirmed.
- Monitor: wrap `analyze_run` in `to_thread` if it's a blocking sync call (protects the loop for both
  providers).

## 4. UI redesign (last — consumes the new identity fields)

**Direction (from research):** Linear/Vercel/Anthropic-console refinement, *not* a rewrite. Keep the
bundled Archivo + JetBrains Mono fonts. Three moves:
1. **De-theme the chrome:** drop phosphor grid + scanlines + neon glow → one soft ambient wash on
   flat layered neutral-slate surfaces with hairline white-alpha borders.
2. **Palette:** single blue accent; green reserved for success only; distinct status hues
   (running=sky, ok=emerald, queued=indigo, weak=amber, stuck=orange, bad=red, idle=slate).
3. **Type + density + layout:** Archivo display / system body / mono ids; left **sidebar** nav
   (Dashboard, Runs) + thin context bar; **grouped 40px rows** in one bordered container;
   **run-id chips** (`run-e483…a10e`, mono, hover copy button) in lists, full id + copy on detail.
- New `/runs` list page (makes the sidebar real). `base.html` passes `page` for active nav; new nav
  links carry `data-i18n`. Bucket classification stays DRY (one helper used by `/` and `/ui/poll`).
- Show `display_title` + tag chips on run rows and the run hero. Logs rendered as inset wells (not
  green). Motion <150ms, `prefers-reduced-motion` respected.

## Build order & verification
identity → (parallel: gemini + macos leaf modules) → wiring → UI → full test suite →
adversarial review workflow on the diff → live smoke (daemon restart, submit, screenshot, notify,
gemini-fallback) → commit per feature → push.
