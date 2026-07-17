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
- The single Telegram poller and the callback dispatch table.
- The single `hub` lock (`name="hub"`, in a hub-level log dir — **not** any target's log dir).
- The registry of `TargetSupervisor` instances keyed by `project_id`.
- Aggregated multi-project status dashboard.
- Coordinated shutdown (fan out to all supervisors) and **hub self-restart** (see finding #1 below).
- Process-global signal handlers + Windows console ctrl handler (installed once).

### `TargetSupervisor` (`ProcessGuard` generalized)
Owns supervision of exactly **one** target: the `run()` state machine (crash/backoff/exit-42/exit-0/hung), heartbeat staleness detection, sidecars, the claude-signal watcher, per-target intent flags. **Does not** own the poller, the lock, or signal handlers. Sends Telegram messages via an **injected sender callback** (`send(text, reply_markup=...)`), not a poller it owns.

Exposes an explicit intent API the Hub's poller calls (never reaching into internals):
- `request_start()`, `request_restart()`, `request_stop()`
- `is_running` / `status()` snapshot
- `stop(timeout)` for coordinated shutdown

---

## Components

1. **`config.py`** — add:
   - `ProjectConfig(id, name, targets: list[GuardConfig], hook_module: str | None)`.
   - `load_registry(path) -> list[ProjectConfig]` — parse YAML → validate → construct. **Fail fast** on parse/validation error (no partial start).
   - Keep `GuardConfig` per target (already parameterized: `command`, `cwd`, `heartbeat_file`, `restart_exit_code`, `log_dir`, `log_file`, `sidecars`, `backoff_steps`, `on_exit`, `extra_processes`). Add optional `env: dict[str,str]` for per-target subprocess env (see finding #4).
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

**Startup:** hub entry point → load `.env` → hub venv guard → `load_registry(registry.yaml)` → construct `Hub` → acquire `hub` lock → **for each project: run `pre_boot` hook** (KutAI's stale-orchestrator + stray-llama cleanup) → construct N `TargetSupervisor` → spawn N supervisor tasks + 1 telegram poll task → announce.

**Runtime:** each supervisor's `run()` loops independently. Poller receives a command/callback → parses `verb` or `verb:project_id[:arg]` → calls the addressed supervisor's `request_*()` (or Hub-global handler) → replies via the injected sender.

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
- **#8 (minor, highest mechanical risk) — `_start_signal_watcher` has 7 call sites** (`guard.py:650,678,694,708,720,730,741`) tied to app-lifecycle transitions. The watcher (claude-signal + sidecar `ensure()`, `guard.py:293-313`) moves whole to `TargetSupervisor`. **Enumerate all 7 in the implementation plan** — missing one silently stops a target's sidecar health checks after certain restart paths.
- **#10 (minor) — preserve, don't clean.** `_send_start_prompt` (`guard.py:130`) takes already-formatted `reason` strings from callers (`guard.py:511,524`). It is a slightly confused contract but **behavior-preserving-sensitive** (exact user-facing down-state wording). Do not "tidy" it during the split.

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
3. **Migration guard test:** the KutAI registry block + loader produce a `GuardConfig` equivalent to today's hardcoded `kutai_wrapper.py` config (proves 1:1). Assert command, cwd, env, heartbeat_file, restart_exit_code, log paths, sidecars.
4. All 10 existing `yasar_usta` test modules stay green.

## Scope (YAGNI)

**IN:** `process` kind, local supervision, N targets, YAML registry + `${}` resolution, `pre_boot`/`on_exit` hook phases, per-target `env`, inline multi-project dashboard, KutAI migrated 1:1, characterization + new tests.

**OUT (later specs):** `healthcheck` kind, `job`/deploy kind, remote SSH/webhook, multi-bot, web UI, relocating KutAI's app-side IPC paths, converging the two recipe systems.
