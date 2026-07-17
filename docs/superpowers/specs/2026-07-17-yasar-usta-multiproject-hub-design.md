# Yaşar Usta → Multi-Project Hub — Sub-Project 1: Hub Core

**Date:** 2026-07-17
**Status:** Design (approved for spec review)
**Scope:** Sub-project 1 of 3. Local process supervision of N targets. Remote health-checks (sub-project 2) and deploy/job triggers (sub-project 3) are OUT of scope.

## Goal

Evolve Yaşar Usta from a single-process manager (one `ProcessGuard` supervising the KutAI orchestrator) into a hub that supervises **N local process targets across multiple projects**, controlled from **one Telegram surface**. KutAI becomes the first registry entry — no longer special-cased in the entry point.

## Locked decisions (from brainstorming)

- **Single hub process, N in-process asyncio supervisors, one shared Telegram poller.** (Not per-project wrapper subprocesses.)
- **Declarative YAML registry** (`registry.yaml`) + **optional per-project Python hook module** for custom lifecycle logic.
- **Inline-button Telegram dashboard**; target-scoped callbacks encode project id.
- **Big-bang rewrite** of `kutai_wrapper.py` into the hub, KutAI as the first registry block. No parallel old code path. Bound by rails: **behavior-preserving + TDD (characterization tests first) + restart-gated push after live-verify.** Git history is the fallback, not a second code path.

## Decomposition (this spec = sub-project 1 only)

1. **Hub core** (this doc) — `process` kind, local supervision, N targets, YAML registry, hook modules, inline dashboard, KutAI migrated.
2. Remote health-check kind (`healthcheck`) — later spec.
3. Deploy/job kind (`job`) — later spec.

Each layers onto the same registry; 2 and 3 are additive target kinds.

---

## Architecture

Yaşar Usta stays the package (`packages/yasar_usta/src/yasar_usta/`). Today's `ProcessGuard` owns **three** responsibilities that must be split:

| Responsibility | Today | After |
|---|---|---|
| Supervise one managed subprocess (start/stop/restart/backoff/heartbeat/sidecars/claude-signal watch) | `ProcessGuard` | **`TargetSupervisor`** (one per target) |
| Telegram poll loop + command/callback dispatch | `ProcessGuard` | **`Hub`** (one shared poller) |
| Single-instance lock, coordinated shutdown, hub self-restart, process-global signal handlers | `ProcessGuard` + `kutai_wrapper.py` | **`Hub`** + hub entry point |

### `Hub` (new — `hub.py`)
Owns:
- The **full `TelegramAPI` client** — the poll loop, the callback dispatch table, and **all callback-side ops** (`answer_callback`, `delete`, `edit` with Markdown→plain retry). The single Telegram **token + chat_id** live on `HubConfig`, not on targets (there is one shared bot; see finding R3).
- The single `hub` lock (`name="hub"`, in a hub-level log dir — **not** any target's log dir).
- The registry of `TargetSupervisor` instances keyed by `project_id`.
- **Status rendering.** `_send_status`/dashboard is Hub-owned (reads `supervisor.status()` from each) — not supervisor-owned, because it needs `edit`/`answer_callback` which only the Hub holds (finding R1).
- Coordinated shutdown (fan out to all supervisors) and **hub self-restart** (see finding #1 below).
- Process-global signal handlers + Windows console ctrl handler (installed once).

### `TargetSupervisor` (`ProcessGuard` generalized)
Owns supervision of exactly **one** target: the `run()` state machine (crash/backoff/exit-42/exit-0/hung), heartbeat staleness detection, sidecars, the claude-signal watcher, per-target intent flags. **Does not** own the poller, the lock, signal handlers, callback ops, or status rendering. It is injected **only** a one-way `notify(text, reply_markup=...)` sender for its own `run()` transition messages (crash/hung/restart notices) — it never edits messages or answers callbacks (finding R1).

Exposes an explicit intent API the Hub's poller calls (never reaching into internals):
- `request_start()`, `request_restart()`, `request_stop()`
- `is_running` / `status()` snapshot
- `stop(timeout)` for coordinated shutdown

---

## Components

1. **`config.py`** — add:
   - `HubConfig(telegram_token, telegram_chat_id, log_dir)` — hub-level; the shared bot's token/chat move here from per-target `GuardConfig` (finding R3). Per-target Telegram fields are **removed**.
   - `ProjectConfig(id, name, targets: list[GuardConfig], hook_module: str | None)`.
   - `load_registry(path) -> (HubConfig, list[ProjectConfig])` — parse YAML → validate → construct. **Fail fast** on parse/validation error (no partial start).
   - Keep `GuardConfig` per target (already parameterized: `command`, `cwd`, `heartbeat_file`, `restart_exit_code`, `log_dir`, `log_file`, `sidecars`, `backoff_steps`, `on_exit`, `extra_processes`, `extra_commands` — retained as-is, unused by KutAI). Add optional `env: dict[str,str]` for per-target subprocess env (see finding #4). Drop `telegram_token`/`telegram_chat_id` (now Hub-level).
2. **`hub.py`** (new) — `Hub` class per Architecture above.
3. **`guard.py`** — refactor `ProcessGuard` → `TargetSupervisor`: strip poller + lock + signal handlers; convert intent flags to the `request_*()` API; accept injected `send`.
4. **`telegram.py`** — reuse the API client; add per-target callback routing helpers. Dispatch table lives in `Hub`.
5. **`status.py`** — render a **multi-project dashboard**: one health line per project/target + inline buttons (▶️ start / ⏹ stop / ♻️ restart / 📋 logs), callback data carries project id.
6. **`projects/kutai/hooks.py`** (new) — KutAI's `_kill_orphan_processes` (→ `on_exit`), `_kill_stale_orchestrators` + `_reconcile_stray_llama` (→ `pre_boot`). See finding #4.
7. **`remote.py`** — claude sessions already per-`log_dir` namespaced; fix the temp-log filename collision (finding #7).

### YAML registry shape (illustrative)

```yaml
projects:
  kutai:
    name: Kutay
    hook_module: yasar_usta.projects.kutai.hooks
    targets:
      - id: orchestrator
        command: ["${venv_python}", "src/app/run.py"]
        cwd: "${project_root}"
        env:
          NERD_HERD_PROJECT_ROOT: "${project_root}"
        heartbeat_file: "logs/orchestrator.heartbeat"
        restart_exit_code: 42
        log_dir: "logs"
        log_file: "logs/orchestrator.jsonl"
        sidecars: [yazbunu, nerd_herd]
  # future project blocks add here — no code change
```

`${...}` tokens are resolved by the loader (venv python discovery, project root). KutAI's paths stay `logs/`-relative because its app-side IPC contract is hardcoded there (finding #3).

---

## Data flow

**Startup:** hub entry point → load `.env` → hub venv guard → `load_registry(registry.yaml)` → construct `Hub` → acquire `hub` lock → **for each project: run `pre_boot(project)` hook once** (per-**project**, after the lock, before any of that project's supervisors spawn — KutAI's stale-orchestrator + stray-llama cleanup) → construct N `TargetSupervisor` → spawn N supervisor tasks + 1 telegram poll task → announce.

**Runtime — command routing (finding R4):** inline-dashboard buttons are the **only** per-target control surface; their callbacks carry `verb:project_id[:arg]` → Hub routes to that supervisor's `request_*()`. Bare **text** commands are **Hub-global**: `/status` = aggregate dashboard of all projects, `/restart_hub` = restart the hub, `/logs` = hub log. A per-target text verb with no id (e.g. bare `/restart`) is **rejected** with a hint to use the dashboard button — never silently applied to a guessed target. Each supervisor's `run()` loops independently; Hub-global text and per-target callbacks are dispatched by the single poller.

**Status:** `/status` (or dashboard refresh) → `Hub` collects `supervisor.status()` from all → renders dashboard.

**Shutdown:** OS signal / console ctrl → **Hub** `request_shutdown()` → fan out `stop(timeout)` to every supervisor (graceful, per-target `stop_timeout`) → release `hub` lock.

**Hub self-restart** (`/restart_hub`): Hub stops all supervisors, respawns the hub entry point, releases lock, exits. (Replaces per-guard `_restart_self`; see finding #1.)

---

## Review findings folded into the design

An adversarial code review (verdict: SOUND-WITH-FIXES) surfaced coupling the first-pass design missed. Each is now a design constraint:

- **#1 (blocker) — `_restart_self` uses `os._exit(0)` (`guard.py:540-575`).** In one shared process this kills every target. Hub self-restart is **Hub-owned**; it respawns the hub entry point and re-reads the registry. `TargetSupervisor` has no process-exit path. Per-target "restart" = kill+restart the managed subprocess only (existing `confirm_restart` semantics), never a process exit.
- **#2 (major) — intent flags are shared state.** `_restart_requested`/`_stop_requested` are set by the poller (`guard.py:402,410`) and read by `run()` (`guard.py:700,711`); the poller also calls `subprocess.stop()` directly (`guard.py:404`). Resolution: flags + `_write_shutdown_signal` stay on `TargetSupervisor`; the poller only calls `request_restart()/request_stop()/request_start()`. The poller must **never** touch `supervisor.subprocess`.
- **#3 (major) — app-side IPC contract hardcoded to `logs/`.** The managed app reads `logs/shutdown.signal` and writes `logs/orchestrator.heartbeat` + `logs/orchestrator.state.json` (`orchestrator.py:359,487-492`), CWD-relative — while the guard uses config-driven paths (`guard.py:116`, `subprocess_mgr.py:213-219`). They coincide today only because KutAI's `cwd`/`log_dir` align. **Constraint:** the shutdown-signal/heartbeat/state paths are a **per-app contract**, not freely relocatable by the registry. KutAI-the-target keeps `logs/`. A future target that wants a different `cwd` must read these paths from env/args on the app side; the registry cannot silently relocate them. Documented as a known KutAI-specific coupling.
- **#4 (major) — startup ordering + global env.** `_kill_stale_orchestrators()` and `_reconcile_stray_llama()` run at **module-import time** (`kutai_wrapper.py:130-131`), before lock/supervisor. The hook system therefore needs a **`pre_boot(cfg)` phase** that runs before that target's supervisor starts — distinct from `on_exit`. Also `os.environ["NERD_HERD_PROJECT_ROOT"]` (`kutai_wrapper.py:30-31`) is a process-global mutation that N targets would clobber → move to **per-target `env=`** passed into `create_subprocess_exec`, not `os.environ`. Signal handlers, the Windows console ctrl handler, and the venv guard are **irreducibly Hub-global** (installed once, on Hub).
- **#5 (major) — confirmation callbacks are stateless.** `confirm_restart`/`confirm_stop` carry no target identity (works today because there is one app). Encode identity **in the callback string**: `confirm_restart:{pid}`, `confirm_stop:{pid}`, `restart_sidecar:{pid}:{name}`, `refresh:{pid}`. Hub-global callbacks stay unqualified (`restart_hub`, `dashboard_refresh`). No server-side pending-confirm dict (a hub restart would lose it; `flush_updates` already drops in-flight callbacks).
- **#6 (minor) — lock hoist is clean.** `lock.py` uses module-global singletons + `atexit` (`lock.py:10-12,72`) — acceptable because there is exactly **one** hub lock. Add a guard so `acquire_lock` can't be called twice (it would silently clobber the handle). No target ever touches `lock.py`.
- **#7 (minor) — claude temp-log collision.** `remote.py:119` names the starting-log `_starting_{os.getpid()}.log` — now the shared **hub** PID, so concurrent per-target launches collide before the rename to `{child_pid}.log`. Use `_starting_{project_id}_{uuid}.log`. Session dirs are already per-`log_dir` namespaced (`guard.py:97`).
- **#8 (minor, highest mechanical risk) — `_start_signal_watcher` has 7 call sites** (`guard.py:650,678,694,708,720,730,741`) tied to app-lifecycle transitions, **plus 2 `_stop_signal_watcher` sites (`guard.py:664,785`)**. The watcher (claude-signal + sidecar `ensure()`, `guard.py:293-313`) moves whole to `TargetSupervisor`. **Enumerate all 9 in the implementation plan** — missing a start site silently stops a target's sidecar health checks after certain restart paths; missing a stop site leaks watcher tasks per target.
- **#10 (minor) — preserve, don't clean.** `_send_start_prompt` (`guard.py:130`) takes already-formatted `reason` strings from callers (`guard.py:511,524`). It is a slightly confused contract but **behavior-preserving-sensitive** (exact user-facing down-state wording). Do not "tidy" it during the split.

### Second review pass (findings folded)

A second adversarial pass (verdict: SOUND-WITH-FIXES; architecture + locked decisions confirmed) caught four spec-text gaps the first folding left — treated as constraints:

- **R1 (major) — injected sender was too thin.** A single `send` callback can't express status refresh (`edit` + Markdown→plain retry, `guard.py:198-211`) or callback acknowledgement (`answer_callback`/`delete`, `guard.py:364-416`). Resolution (already reflected in Architecture): Hub owns the full `TelegramAPI` and all callback-side ops **and** status rendering; the supervisor is injected only a one-way `notify()` for its `run()` transition messages.
- **R2 (major) — per-target `env` (#4) was unwired + incomplete.** (a) `SubprocessManager.start()` never passes `env=` to `create_subprocess_exec` (`subprocess_mgr.py:122-129`) — add it with an explicit **`{**os.environ, **cfg.env}` merge** (bare `cfg.env` would drop PATH/venv). (b) The nerd_herd **sidecar** also reads `NERD_HERD_PROJECT_ROOT`/`LLAMA_SERVER_PORT` from `os.environ`, but `SidecarManager` (`sidecar.py:31-49`) has no `env` param — thread per-project `env` through sidecars too, or moving project-root off `os.environ` breaks them. (c) `load_dotenv()` is process-global today; policy for sub-project 1 = **KutAI keeps the single process `.env`**; per-project `.env` files are deferred (only KutAI exists now). Env in the registry is per-target and merged onto `os.environ` at spawn.
- **R3 (major) — shared-poller Telegram token/chat migration was unstated.** Per-target `telegram_token`/`telegram_chat_id` (filtered at `guard.py:359,425`; KutAI token `YASAR_USTA_BOT_TOKEN` at `kutai_wrapper.py:141`) move to `HubConfig`. Per-target Telegram fields are dropped and **excluded from the migration-equivalence assertion** (see Testing).
- **R4 (major) — text-command routing was undefined.** Resolved in Data flow: inline buttons are the only per-target surface; bare text commands are Hub-global; ambiguous per-target text verbs are rejected with a hint, never applied to a guessed target.
- **Tidies:** `_stop_signal_watcher` sites added to #8 (above); `extra_commands` retained (Components); `pre_boot` pinned as per-**project**, once, after lock (Data flow); migration-guard relabeled as a config-equivalence check (Testing).

## Error handling

- One target crash-loops → its own `BackoffTracker`, others unaffected (isolation is the point of per-target supervisors).
- Missing/broken hook module → log + continue (hooks optional). A KutAI `pre_boot` failure is surfaced to Telegram, not swallowed.
- Registry parse/validation error → fail fast with a clear message, no partial start.
- `hub` lock held by a live PID → exit 1 (unchanged single-instance semantics).
- Telegram poller exception → isolated + retried (existing pattern), never takes down supervisors.

## Testing (TDD — characterization-first)

**Finding #9 is the real gate:** the `run()` state machine (`guard.py:579-787`) and `_telegram_poll_loop` (~180 lines) currently have **zero tests** (`test_guard.py` covers only construction + two formatters). "Behavior-preserving" is unverifiable by inspection. Therefore:

1. **Characterization tests FIRST (before any split):**
   - `run()` branches — mock `subprocess.wait_for_exit()` to return each of -1 (hung), 0 (clean), 42 (restart), and crash codes; assert the resulting notifications + flag transitions + restart/backoff behavior.
   - Poller callback dispatch — assert each existing callback (`confirm_restart`, `confirm_stop`, `restart_sidecar:{name}`, `restart_guard`, refresh) drives the expected action.
2. **New behavior after the split:**
   - Registry loader: YAML → `ProjectConfig`; fail-fast on bad input.
   - Hub multi-supervisor: 2 fake targets — independent restart; one crash-loop does **not** affect the other; coordinated shutdown stops both.
   - Per-target routing: `restart:foo` hits foo's supervisor, not kutai's; `confirm_restart:{pid}` targets the right one.
   - Dashboard render for N projects.
   - Hook load: `pre_boot` fires before supervisor start; `on_exit` fires on crash; KutAI cleanup fns invoked.
3. **Migration guard test (config-equivalence only):** the KutAI registry block + loader produce a `GuardConfig` equal to today's hardcoded `kutai_wrapper.py` config on: command, cwd, env, heartbeat_file, restart_exit_code, log paths, sidecars. **Excludes** the dropped per-target Telegram fields (now on `HubConfig`). This proves 1:1 *config*, not runtime behavior — behavior preservation is gated by the characterization suite (test 1), since the split changes execution structure even when config is identical.
4. All 10 existing `yasar_usta` test modules stay green (or are updated in lockstep where the split moves a responsibility — e.g. lock/poller ownership).

## Scope (YAGNI)

**IN:** `process` kind, local supervision, N targets, YAML registry + `${}` resolution, `pre_boot`/`on_exit` hook phases, per-target `env`, inline multi-project dashboard, KutAI migrated 1:1, characterization + new tests.

**OUT (later specs):** `healthcheck` kind, `job`/deploy kind, remote SSH/webhook, multi-bot, web UI, relocating KutAI's app-side IPC paths, converging the two recipe systems.
