# Yaşar Usta — Shared Hub Relocation ⨝ Always-Live (Unified Design)

**Date:** 2026-07-21
**Status:** Design approved (brainstorming), pending spec review → implementation plan
**Branch:** `feat/yasar-usta-multiproject-hub` (local-only, not pushed)
**Supersedes:** the (unwritten) relocation-only design discussed earlier this session.
**Extends / references (does NOT duplicate):** `docs/superpowers/specs/2026-07-17-yasar-usta-always-live-singleton-design.md` — the source of truth for singleton (M1/M2), watchdog (M4a/M4b), and auto-start (M3) design detail.
**Handoff merged:** `docs/handoff/2026-07-21-yasar-usta-always-live-handoff.md`.

---

## 1. Why

Two efforts are converging on the same files and must ship as one:

1. **Relocation / share-ability** — more projects will connect to Yaşar Usta. Today it lives *inside* the KutAI monorepo (`packages/yasar_usta/`) with KutAI-specific code baked into the shared package. Other projects cannot cleanly consume it, and the hub currently runs *inside kutay's venv*, importing a KutAI package (`dallama`) in-process.
2. **Always-live / never-duplicate** — the singleton mutex is done and live-verified; the watchdog + auto-start work is in flight. Its deferred **M5 (state out of Dropbox)** touches exactly the files the relocation touches (`registry.yaml`, `config.py`, `hub.py`, entry point, orchestrator), and its **split-brain heartbeat** risk is triggered by any state relocation.

Doing them separately means two high-blast-radius passes over the same code with a live-verify each, plus a real collision risk (the handoff explicitly flags M5 as "collides with the parallel session"). Merging them removes the parallel-session collision (one effort) and lets a single coherent change be verified once per phase.

## 2. Goals / Non-goals

**Goals**
- Yaşar Usta lives in its own standalone git repo, consumable by N projects.
- Exactly **one** shared hub daemon for all projects (enforced by the existing kernel mutex).
- The hub runs in its **own venv** and never imports any project's packages in-process.
- Runtime state lives **outside Dropbox**; code stays in Dropbox (fine — infrequent, versioned writes).
- Adding a new project = drop a registry block + a `yasar_hooks.py`, install the client into its venv. No shared-venv pollution.
- The always-live guarantees (singleton, watchdog, auto-start) survive the relocation intact.

**Non-goals (YAGNI / deferred)**
- Splitting a separate zero-dep `yasar_usta.client` distribution (consumers pull `aiohttp`+`pyyaml` for now — light).
- Per-project registry *fragment* files (single hub-side registry with per-project `root` is enough for 2 projects).
- Publishing to a private package index (editable installs from the sibling repo, like `yazbunu`).
- Running the M3 auto-start installer (stays user-gated: needs elevation + auto-logon security tradeoff).

## 3. Key facts established (verified this session)

- `packages/yasar_usta/` is already a clean installable package (`pyproject`: deps `aiohttp`, `pyyaml`). Registry-driven.
- KutAI coupling exists in exactly three places: `registry.yaml` (kutay root — project-side, correct), `kutai_wrapper.py` (kutay root entry — correct side), and `src/yasar_usta/projects/kutai/` (KutAI code living **inside** the shared package — wrong side).
- The one in-process cross-package import: `projects/kutai/hooks.py:149` → `from dallama.platform import PlatformHelper`. Works only because the hub runs in kutay's venv today.
- `yasar_usta` is **also a client library**: the managed orchestrator imports it from *inside kutay's venv* — `src/app/run.py:378` and `src/core/orchestrator.py:468` (`HeartbeatWriter`, `write_heartbeat`). So consuming projects DO install `yasar_usta` (heartbeat client); the hub venv installs it too (daemon).
- Precedent: `yazbunu` already lives at `Workspaces/yazbunu` (sibling, outside kutay), editable-installed into kutay's venv.
- **Split-brain confirmed:** orchestrator writes `logs/orchestrator.heartbeat` **CWD-relative** (`run.py:379` `_hb_paths`, `orchestrator.py:488`), hub reads the registry's **absolute** path. They agree only because CWD=kutay. Any state relocation breaks this unless the child is told the path.
- **Entry-point ripple:** `scripts/install_yasar_autostart.ps1:33` launches `kutai_wrapper.py`; `hub.py:248` self-restarts via `sys.argv[0]`. Both must move to the generic entry.

## 4. Architecture

### 4.1 Two location axes

Code and runtime state are separated. Dropbox is acceptable for code (infrequent, versioned) and toxic for runtime state (sync churn, conflicted-copy corrupts locks; `hub.alive` is written ~1440×/day).

| Axis | Location | Registry token |
|---|---|---|
| Code / venv / scripts | `C:\Users\sakir\Dropbox\Workspaces\yasar_usta\` (own git repo) | `${project_root}` (per project, code dir) |
| Runtime state — hub | `%LOCALAPPDATA%\YasarUsta\hub\` (`hub.alive`, hub lock, `.mutex_fault.json`, `hub.stopped`, `.watchdog_killed`, `guard.jsonl`) | `${state_dir}` (hub) |
| Runtime state — per project | `%LOCALAPPDATA%\YasarUsta\<project_id>\` (`orchestrator.heartbeat`, per-project runtime, pid files) | `${state_dir}` (project-scoped) |

`YASAR_USTA_STATE_DIR` env is passed from hub → each managed child so the child never re-derives a relative state path.

Note: application logs that are genuinely a project's own concern (e.g. KutAI's `orchestrator.jsonl`) may remain in the project's `logs/`. Only Yaşar-owned state + the heartbeat contract move. The heartbeat is the one file both hub and child touch, so it is the split-brain-critical one.

### 4.2 Repo layout (target)

```
Workspaces/
  kutay/                       consumer project 1
  yazbunu/                     (existing sibling — precedent)
  yasar_usta/                  NEW own git repo
    src/yasar_usta/            generic ONLY — no projects/kutai
      hub.py supervisor.py registry.py config.py telegram.py
      singleton.py watchdog.py heartbeat.py sidecar.py subprocess_mgr.py
      backoff.py lock.py remote.py status.py commands.py hooks.py
      __main__.py              NEW generic entry (was kutai_wrapper.main)
    tests/
    scripts/install_yasar_autostart.ps1   (rewritten for generic entry + LOCALAPPDATA)
    pyproject.toml             deps aiohttp, pyyaml; console_script yasar-usta
    registry.yaml              multi-project registry (hub-side)
    start.bat                  .venv python -m yasar_usta --registry registry.yaml
    .venv/                     hub's OWN venv
```

The move preserves history via `git subtree split` of `packages/yasar_usta` into the new repo (fallback: plain copy if subtree is fiddly on Windows).

### 4.3 Two roles, one package

- **Hub venv** (`yasar_usta/.venv`): `pip install -e .` → runs the single daemon (`python -m yasar_usta --registry …`).
- **Consumer venv** (e.g. `kutay/.venv`): keeps an editable install of `yasar_usta`, only for `HeartbeatWriter`/`write_heartbeat`. The path in `requirements.txt` changes from `-e ./packages/yasar_usta` to `-e ../yasar_usta`. `run.py`/`orchestrator.py` imports are unchanged (except heartbeat path, see §4.5).

### 4.4 Decoupling: subprocess hooks

`src/yasar_usta/projects/kutai/` is deleted from the shared package. It is reborn as **`kutay/yasar_hooks.py`** (project-owned, at the kutay repo root), which imports `dallama`/`psutil` freely because it runs in **kutay's** venv.

The shared `hooks.py` changes from in-process `importlib` dispatch to **subprocess** dispatch: the hub spawns

```
<project_venv_python> yasar_hooks.py <phase> --context <json>
```

- `<phase>` ∈ `pre_boot` | `on_exit`.
- `--context` carries what the in-process hook used to read from `project`: the target run-script absolute path(s) (for M6 stale-orchestrator kill) and, for `on_exit`, the exit code. `LLAMA_SERVER_PORT` continues to flow via inherited env.
- Result via exit code (+ stdout for logging). `pre_boot` failure is surfaced (must stay visible, per the existing hook contract); `on_exit` is fail-soft.

Consequences: the hub venv never carries `dallama`/`psutil`-for-project; **M6's** precise absolute-path orphan-kill (psutil, no `wmic`) is preserved verbatim inside `yasar_hooks.py`.

### 4.5 Split-brain heartbeat fix (mandatory with §4.1)

Because the heartbeat file is the one artifact both the hub (reader) and the orchestrator (writer) touch, and its path is about to leave `logs/`:

- The hub passes `YASAR_USTA_STATE_DIR` (absolute) to the orchestrator child (and to sidecars that need state).
- `src/app/run.py` and `src/core/orchestrator.py` build the heartbeat path from that env var (`<state_dir>/orchestrator.heartbeat`) and **never** hardcode `logs/orchestrator.heartbeat`. A missing env var falls back to the current relative path (back-compat for a non-hub launch), but under the hub the env var always wins.
- The registry's `heartbeat_file` is resolved from `${state_dir}` so hub and child compute the identical absolute path.

This is the single highest-blast-radius change; it gets its own live-verify (confirm the hub does **not** false-kill a healthy orchestrator after the state move).

### 4.6 Fully declarative registry (dissolve `kutai_wrapper.py`)

`kutai_wrapper.py`'s `_apply_runtime_values` (Turkish `Messages`, `claude_cmd`, sidecar `--db-path`, `auto_restart`) dissolves into `registry.yaml` + a loader that supports `${env:VAR}` and per-project `${project_root}`/`${state_dir}`:

- `messages:` block per project → parsed into `Messages` (hub keyboard + Turkish crash/stop notifications).
- `claude_cmd: "${env:APPDATA}/npm/claude.cmd"`; sidecar `--db-path ${env:DB_PATH}`.
- `auto_restart` → hub CLI flag `--no-auto-restart`.
- Per-project `root:` field (fixes the current single global `project_root` token — required for project 2 living at a different path) and a per-project `state_dir` derived as `%LOCALAPPDATA%/YasarUsta/<id>`.

`kutai_wrapper.py` is then **deleted** — a shared multi-project hub has one launcher (`start.bat` → generic entry), not a per-project wrapper. Update `.claude/settings.local.json` launch commands and `CLAUDE.md`.

### 4.7 Entry point + always-live wiring

- Generic entry: `python -m yasar_usta --registry <path> [--no-auto-restart]` (the signal handling + `hub.run()` currently in `kutai_wrapper.main`). Optional `console_scripts: yasar-usta`.
- `hub.py::_do_restart_hub` self-fork changes from `[sys.executable, sys.argv[0]] + argv[1:]` to `[sys.executable, "-m", "yasar_usta"] + registry_args` — running `__main__.py` as a bare script would break package-relative imports.
- `install_yasar_autostart.ps1` main task action → `-Execute <hub_venv_python> -Argument "-m yasar_usta --registry <path>"`; the watchdog task's `--alive` points at `%LOCALAPPDATA%\YasarUsta\hub\hub.alive`. (The watchdog task already invokes `-m yasar_usta.watchdog`.)
- **"Never stack two relaunchers"** is preserved: hub self-restart is a clean mutex-release→spawn→exit(0); Task Scheduler restart-on-failure only fires on nonzero exit. (Optional later cleanup: `os._exit(42)` so Task Scheduler becomes the sole relauncher — deferred, the Popen bridge works.)

### 4.8 Always-live carryover (unchanged by relocation)

- **Singleton M1/M2** (`singleton.py`, `Global\YasarUstaHub` mutex, fail-closed, circuit-breaker) — done + live-verified; moves with the repo. One hub for all projects ⇒ one mutex name (unchanged). Directly reinforces the shared-hub model.
- **Watchdog M4b must-fixes** (still required *before* activating M3), from handoff §4:
  1. grace-after-kill: write `.watchdog_killed` (ts) in `run_once`; skip killing if `now - kill_ts < grace (~360s)`.
  2. `hub.stopped` gate: kill only if stale AND no `hub.stopped` AND pid alive; write `hub.stopped` on any deliberate hub-down.
  3. kill-death verification: confirm the mutex-holding real interpreter actually died; log/alert on failure (else silent zero-effective-hub).
  These marker/gate files live in `%LOCALAPPDATA%\YasarUsta\hub\`.
- **M3 auto-start** — installer rewritten now (§4.7); **running it stays user-gated** (elevation + `netplwiz` auto-logon + reboot test).

## 5. New-project onboarding (the payoff)

1. `pip install -e ../yasar_usta` into the project venv (heartbeat client); call `write_heartbeat()` in its loop.
2. Add a block to the hub's `registry.yaml`: `root`, `venv_python`, `hook` (path to its `yasar_hooks.py`), `targets`, `messages`.
3. Drop a `yasar_hooks.py` in the project repo (runs in its own venv; may import that project's packages). Done — zero shared-venv pollution.

## 6. Phased delivery (each phase independently live-verified)

1. **Relocate + decouple.** `git subtree` move; delete `projects/kutai/`; subprocess-hook dispatch + `kutay/yasar_hooks.py` (with M6); generic `__main__` entry; dissolve `_apply_runtime_values` into declarative registry + `${env:}` loader; delete `kutai_wrapper.py`; update `requirements.txt`, `.claude/settings.local.json`, `CLAUDE.md`. **State still `${project_root}/logs`** — no state change, so this phase's live-verify is a clean regression check (hub still manages KutAI from its own venv/new location).
2. **State-dir / M5 + split-brain fix.** Introduce `${state_dir}` (LOCALAPPDATA, hub + per-project) and `YASAR_USTA_STATE_DIR`; make `run.py`/`orchestrator.py` read the heartbeat path from env. Dedicated live-verify: hub does not false-kill a healthy orchestrator; state files appear under LOCALAPPDATA, none churn Dropbox.
3. **Watchdog M4b + installer rewrite.** The three must-fixes (TDD) + repoint `install_yasar_autostart.ps1` at the generic entry and LOCALAPPDATA alive-path.
4. **User activation (gated).** Run the installer elevated + auto-logon + reboot test; push branch; merge to `main`.

## 7. Testing

- Generic tests move to the new repo. Subprocess hook-dispatch gets new tests (fake project + fake venv python, assert argv/context/exit-code handling).
- `test_kutai_hooks.py` / `test_migration_kutai.py` move to **kutay** (they now test kutay code).
- Split-brain: a test asserting hub and child resolve the identical heartbeat path from `YASAR_USTA_STATE_DIR`.
- Watchdog M4b: TDD for grace marker, `hub.stopped` gate, kill-death verification (all three from the handoff's adversarial pass).
- Windows gotchas honored: targeted pytest, foreground, with timeout; kill only own hung processes; never taskkill llama-server.

## 8. Risks

- **Split-brain (§4.5)** — highest. Mitigation: env-passed path + a dedicated test + phase-2 live-verify before proceeding.
- **Deleting `kutai_wrapper.py`** touches the documented entry point + always-live launch/self-restart. Mitigation: all three consumers (installer, self-fork, docs) updated in phase 1; regression live-verify.
- **`git subtree` history preservation on Windows** — fallback to plain copy if it misbehaves; history is nice-to-have, not blocking.
- **Editable-install path change** in `kutay/requirements.txt` (`./packages/yasar_usta` → `../yasar_usta`) — re-run the editable install into kutay's venv; verify `HeartbeatWriter` still imports.

## 9. Open items intentionally deferred

- Zero-dep `yasar_usta.client` split; per-project registry fragments; private index publishing.
- `os._exit(42)` self-restart cleanup; `record_fault` per-type counters; other handoff §6 residuals.
- M5's `systemprofile` boot-assertion — include if cheap during phase 2, else track.
