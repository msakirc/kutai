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

Two adjacent facts (verified this session): (1) The LOCALAPPDATA move **improves** stale-lock robustness — `lock.py` PID-liveness recovery is dir-independent, and a per-user non-synced dir removes the Dropbox conflicted-copy risk that could corrupt the lock/`hub.alive`. The lock, `hub.alive`, and `.mutex_fault.json` must share the one `\hub\` dir so they relocate together (they do, per the table). (2) **CWD trap to close in phase 2:** `HubConfig.log_dir` **defaults to the relative `"logs"`** (`config.py:153`); under `-m yasar_usta` the CWD is the hub venv dir, not the project, so a registry-less/misconfigured launch would drop state in the wrong place and the installer's absolute `--alive` path would never match. Phase 2 must make the `${state_dir}` path absolute end-to-end and treat the relative default as a fail-loud misconfig, not a silent fallback.

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

`guard.py` (legacy `TargetSupervisor` precursor, still exported by `__init__.py` and covered by `test_guard_*_characterization.py`) is **not dead** — the migration must explicitly decide move-with-repo vs. delete-and-drop-its-tests. Default: move it (keeps history green), retire in a later cleanup.

### 4.3 Two roles, one package

- **Hub venv** (`yasar_usta/.venv`): `pip install -e .` → runs the single daemon (`python -m yasar_usta --registry …`).
- **Consumer venv** (e.g. `kutay/.venv`): keeps an editable install of `yasar_usta`, only for `HeartbeatWriter`/`write_heartbeat`. The path in `requirements.txt` changes from `-e ./packages/yasar_usta` to `-e ../yasar_usta`. `run.py`/`orchestrator.py` imports are unchanged (except heartbeat path, see §4.5).

**Boot ordering (req-2 completeness).** `from yasar_usta import HeartbeatWriter` fires at orchestrator start (`run.py:378`, `orchestrator.py:468`); if the sibling repo has not been editable-installed into the consumer venv, that import raises and the child never boots. Therefore: (a) the sibling repo must exist and be `pip install -e ../yasar_usta`'d into **both** the hub venv and each consumer venv **before** first managed launch; (b) at reboot this must hold before Task Scheduler starts the hub. Guard it: the hub asserts the **symbols actually resolve** for each project's `venv_python` — `-c "from yasar_usta import HeartbeatWriter, write_heartbeat"` (not a bare `import yasar_usta`, which passes on a stale/partial install where the symbol was renamed) — and fails loud with a clear "run pip install -e ../yasar_usta in <venv>" message rather than a late ImportError inside a managed child. **Placement:** run this **after** `_acquire_singleton()` (inside `run()`, not pre-mutex `__main__`) so a mutex-loser exits immediately without paying N subprocess spawns, and so the assert's cost is counted in — not added blind to — the boot-latency budget the watchdog grace window depends on (§4.8). Ordering is clean: registry load → mutex gate → import+creds asserts → file lock → pre_boot.

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

**Windows arg-passing:** the `--context` JSON carries backslash Windows paths (from `_orchestrator_script_paths`). The hub must build the subprocess as an **argv list** (`subprocess.Popen([venv_python, "yasar_hooks.py", phase, "--context", json_str])`), never a shell string — a shell string mangles backslashes/quotes. The in-process hook never had this concern; the subprocess boundary introduces it.

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

**[BLOCKER] Secrets loading is orphaned by the deletion.** `load_dotenv()` exists in exactly one place — `kutai_wrapper.py:13-15` — which this section deletes. The hub reads its own credentials via `os.getenv("YASAR_USTA_BOT_TOKEN")` / `os.getenv("TELEGRAM_ADMIN_CHAT_ID")` (`registry.py:90-91`). After relocation the hub runs `python -m yasar_usta` from its own repo with **no `load_dotenv` and no `.env`** → `telegram_token=""` → the poller disables, `_sync_alert` degrades to a silent log, and the singleton-fault / crash alerts go **dark** — the precise failure this whole effort prevents. Resolution:
  - The hub **owns its own `.env`** at `Workspaces/yasar_usta/.env` holding `YASAR_USTA_BOT_TOKEN` + `TELEGRAM_ADMIN_CHAT_ID` (hub-global creds, not any one project's). The generic `__main__.py` calls `load_dotenv()` (replacing the wrapper's).
  - **Fail loud:** after the mutex gate, boot-assert `telegram_token` is non-empty; a credential-less hub must never boot quietly (same spirit as the import assert, §4.3).
  - **Env-token scoping:** `${env:VAR}` in the registry resolves against the **hub** process env, so reserve it for genuinely-global OS vars (`APPDATA`). Project-specific values (`DB_PATH`) must **not** ride the hub's env — express them as path tokens (`${project_root}/data/kutai.db`) or per-project registry literals, so the hub never has to absorb each project's `.env`.

`kutai_wrapper.py` is then **deleted** — a shared multi-project hub has one launcher (`start.bat` → generic entry), not a per-project wrapper. Update `CLAUDE.md`. Note: `.claude/settings.local.json` entries referencing the wrapper are a cached **permission allowlist** (not live launchers) — stale ones are harmless (they simply never match again), but refresh them to avoid confusion.

### 4.7 Entry point + always-live wiring

- Generic entry: `python -m yasar_usta --registry <path> [--no-auto-restart]` (the signal handling + `hub.run()` currently in `kutai_wrapper.main`). Optional `console_scripts: yasar-usta`.
- `hub.py::_do_restart_hub` self-fork changes from `[sys.executable, sys.argv[0]] + argv[1:]` to `[sys.executable, "-m", "yasar_usta"] + registry_args` — running `__main__.py` as a bare script would break package-relative imports.
- `install_yasar_autostart.ps1` main task action → `-Execute <hub_venv_python> -Argument "-m yasar_usta --registry <path>"`; the watchdog task's `--alive` points at `%LOCALAPPDATA%\YasarUsta\hub\hub.alive`. (The watchdog task already invokes `-m yasar_usta.watchdog`.)
- **[BLOCKER] Watchdog process matcher.** `watchdog.py::find_hub_pids` (`:54`) hardcodes `"kutai_wrapper.py" in cmdline`. Deleting the wrapper + launching `-m yasar_usta` makes it return `[]` forever → the M4b watchdog **never fires** → always-lives silently defeated (same bug-class as M6's bare-substring match). Repoint the matcher to the new invocation. **Empirically confirmed on this box:** `-m yasar_usta` puts the literal tokens `-m yasar_usta` in the psutil cmdline of **both** the venv launcher-stub and its real Python interpreter child — but the `__main__.py` *file path* never appears (so drop any `__main__.py`-path matching). Match the **adjacent `-m` + `yasar_usta` token pair**, not a loose `"yasar_usta"` substring (which would over-match any `-e ../yasar_usta` pip line or this spec's filename). Fix the docstring too. `status.py`'s duplicate-detection takes `guard_script` as a param but is wired **only into legacy `guard.py`** (the hub dashboard keys off `app_script`, not `guard_script`) — effectively dead on the hub path; fold its fate into the `guard.py` move/retire decision (§4.2), do not repoint a corpse.
- **"Never stack two relaunchers"** is preserved: hub self-restart is a clean mutex-release→spawn→exit(0); Task Scheduler restart-on-failure only fires on nonzero exit. (Optional later cleanup: `os._exit(42)` so Task Scheduler becomes the sole relauncher — deferred, the Popen bridge works.)

### 4.8 Always-live carryover (unchanged by relocation)

- **Singleton M1/M2** (`singleton.py`, `Global\YasarUstaHub` mutex, fail-closed, circuit-breaker) — done + live-verified; moves with the repo. One hub for all projects ⇒ one mutex name (unchanged). Directly reinforces the shared-hub model.
- **Watchdog M4b must-fixes** (still required *before* activating M3), from handoff §4:
  1. grace-after-kill: write `.watchdog_killed` (ts) in `run_once`; skip killing if `now - kill_ts < grace (~360s)`.
  2. `hub.stopped` gate: kill only if stale AND no `hub.stopped` AND pid alive; write `hub.stopped` on any deliberate hub-down.
  3. kill-death verification: confirm the mutex-holding real interpreter actually died; log/alert on failure (else silent zero-effective-hub).
  These marker/gate files live in `%LOCALAPPDATA%\YasarUsta\hub\`.
  **Two relocation caveats on M4b:** (a) finding #3 was written against the `kutai_wrapper.py` process tree (launcher stub + real interpreter, both matched by `find_hub_pids`); under `-m yasar_usta` from the hub venv the tree may differ, so the phase-3 TDD for #3 must assert against the **actual `-m yasar_usta` process shape**, not the wrapper's. (b) The grace window (`~360s`) is exactly two watchdog ticks (`DEFAULT_INTERVAL=180s`); if hub boot (venv import + `_reconcile_stray_llama`) ever exceeds one tick, the margin is thin — size the grace with headroom (≥3 ticks) or make it boot-completion-signalled rather than time-only.
- **M3 auto-start** — installer rewritten now (§4.7); **running it stays user-gated** (elevation + `netplwiz` auto-logon + reboot test).

## 5. New-project onboarding (the payoff)

1. `pip install -e ../yasar_usta` into the project venv (heartbeat client); call `write_heartbeat()` in its loop.
2. Add a block to the hub's `registry.yaml`: `root`, `venv_python`, `hook` (path to its `yasar_hooks.py`), `targets`, `messages`.
3. Drop a `yasar_hooks.py` in the project repo (runs in its own venv; may import that project's packages). Done — zero shared-venv pollution.

**Shared Telegram surface (acknowledged, not a blocker).** One hub = one `YASAR_USTA_BOT_TOKEN` + one admin chat. The dashboard is already multi-project (`hub.py` iterates `self.supervisors`; bare-verb commands reject when N>1 → "use the buttons"), so the UI is ready. But every project's crash/hung alerts land in the **same** chat — there is no per-project chat routing. Fine for 2 projects; if per-project alert routing is ever needed, that's a follow-up (per-project `telegram_chat_id` in the registry block), explicitly out of scope now.

## 6. Phased delivery (each phase independently live-verified)

1. **Relocate + decouple + re-point the entry.** `git subtree` move; delete `projects/kutai/`; subprocess-hook dispatch + `kutay/yasar_hooks.py` (with M6); generic `__main__` entry; dissolve `_apply_runtime_values` into declarative registry + `${env:}` loader; delete `kutai_wrapper.py`. **State is neutral this phase (`${project_root}/logs`), but the entry point is NOT — deleting the wrapper is blast-radius-wide.** Enumerate and update *every* consumer that names `kutai_wrapper.py`, or one is silently orphaned (grep the whole repo — the full set found this session):
  - **Live launchers:** `start_kutai.bat:3` (real relauncher parallel to the installer — easy to miss), `install_yasar_autostart.ps1`, `hub.py::_do_restart_hub` self-fork.
  - **Watchdog matcher (blocker, §4.7):** `watchdog.py::find_hub_pids:54` + docstring.
  - **Secrets (blocker, §4.6):** `load_dotenv()` moves into `__main__.py`.
  - **Tests that assert on the wrapper:** `tests/integration/test_restart_shutdown.py:96-109` (checks the wrapper contains exit-code 42), `tests/test_wrapper_logs.py:8,14` (opens the wrapper) — rewrite against the new entry. `packages/yasar_usta/tests/test_kutai_hooks.py` + `test_migration_kutai.py` move to kutay (also §7).
  - **Docs / cached perms:** `CLAUDE.md`; `.claude/settings.local.json` (a permission allowlist, not launchers — stale entries are harmless, refresh for hygiene).

  Live-verify = clean regression: hub still manages KutAI from its own venv/new location, `start_kutai.bat`/`start.bat` launch it, **and** a manually hung hub is still found+killed by the watchdog (proves the matcher repoint).
2. **State-dir / M5 + split-brain fix.** Introduce `${state_dir}` (LOCALAPPDATA, hub + per-project) and `YASAR_USTA_STATE_DIR`; make `run.py`/`orchestrator.py` read the heartbeat path from env. Dedicated live-verify: hub does not false-kill a healthy orchestrator; state files appear under LOCALAPPDATA, none churn Dropbox.
3. **Watchdog M4b + installer rewrite.** The three must-fixes (TDD) + repoint `install_yasar_autostart.ps1` at the generic entry and LOCALAPPDATA alive-path.
4. **User activation (gated).** Run the installer elevated + auto-logon + reboot test; push branch; merge to `main`.

## 7. Testing

- Generic tests move to the new repo. Subprocess hook-dispatch gets new tests (fake project + fake venv python, assert argv/context/exit-code handling).
- `test_kutai_hooks.py` / `test_migration_kutai.py` move to **kutay** (they now test kutay code).
- Split-brain: a test asserting hub and child resolve the identical heartbeat path from `YASAR_USTA_STATE_DIR`.
- Watchdog matcher: a test asserting `find_hub_pids` matches a `-m yasar_usta` cmdline (adjacent `-m`+`yasar_usta` tokens) and does **not** depend on `"kutai_wrapper.py"` nor over-match a loose `"yasar_usta"` substring (regression guard for the blocker).
- Boot asserts: tests that the hub fails loud (non-zero, clear message) when (a) a project venv can't `from yasar_usta import HeartbeatWriter, write_heartbeat`, and (b) `telegram_token` is empty — and that both run **after** the singleton gate.
- Watchdog M4b: TDD for grace marker, `hub.stopped` gate, kill-death verification (all three from the handoff's adversarial pass); finding-#3 test asserts against the `-m yasar_usta` process shape.
- Windows gotchas honored: targeted pytest, foreground, with timeout; kill only own hung processes; never taskkill llama-server.

## 8. Risks

- **Orphaned secrets (§4.6)** — deleting `kutai_wrapper.py` (sole `load_dotenv` caller) boots the hub credential-less → alerts silently dark. Mitigation: `load_dotenv()` in `__main__.py` + hub-owned `.env` + fail-loud non-empty-`telegram_token` boot-assert. **Highest-consequence for the always-live goal** (the safety net goes dark without a crash).
- **Split-brain (§4.5)** — highest structural. Mitigation: env-passed path + a dedicated test + phase-2 live-verify before proceeding.
- **Deleting `kutai_wrapper.py`** touches the documented entry point + always-live launch/self-restart. **Highest-miss risk: `watchdog.py::find_hub_pids` hardcodes the wrapper name (`:54`)** — if missed, the watchdog silently never fires (blocker, §4.7). Mitigation: enumerate all launch-command consumers in phase 1 (§6) — installer, self-fork, watchdog matcher, `status.py` caller, settings.local.json ×6, docs; live-verify includes a deliberate hung-hub kill.
- **Consumer editable-install boot ordering** — `from yasar_usta import HeartbeatWriter` at `run.py:378` raises if `-e ../yasar_usta` isn't installed in the consumer venv before first managed launch (critical at reboot). Mitigation: hub `__main__` boot-asserts `import yasar_usta` resolves per project `venv_python` and fails loud (§4.3).
- **`git subtree` history preservation on Windows** — fallback to plain copy if it misbehaves; history is nice-to-have, not blocking.
- **Editable-install path change** in `kutay/requirements.txt` (`./packages/yasar_usta` → `../yasar_usta`) — re-run the editable install into kutay's venv; verify `HeartbeatWriter` still imports.

## 9. Open items intentionally deferred

- Zero-dep `yasar_usta.client` split; per-project registry fragments; private index publishing.
- `os._exit(42)` self-restart cleanup; `record_fault` per-type counters; other handoff §6 residuals.
- M5's `systemprofile` boot-assertion — include if cheap during phase 2, else track.
