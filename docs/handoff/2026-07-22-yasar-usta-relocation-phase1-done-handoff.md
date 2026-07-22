# Handoff — Yaşar Usta Relocation ⨝ Always-Live: Phase 1 DONE (live), Phase 2–4 remain

**Date:** 2026-07-22
**Effort:** Move Yaşar Usta out of the KutAI monorepo into a standalone, multi-project sibling repo + merge the always-live/singleton work. Delivered in 4 phases.
**Spec:** `docs/superpowers/specs/2026-07-21-yasar-usta-shared-hub-and-always-live-design.md`
**Plan:** `docs/superpowers/plans/2026-07-21-yasar-usta-shared-hub-and-always-live.md` (task-by-task, with real code)
**Prior handoff (superseded, merged in):** `docs/handoff/2026-07-21-yasar-usta-always-live-handoff.md`

---

## 0. TL;DR / status

| Phase | State |
|---|---|
| **Phase 1 — relocate + decouple + entry + secrets** | ✅ **DONE + LIVE-VERIFIED.** Hub runs standalone from `../yasar_usta`, managing the live KutAI stack. |
| **Phase 2 — state-dir (%LOCALAPPDATA%) + split-brain heartbeat fix** | ⬜ NOT STARTED. Riskiest phase (can false-kill the orchestrator). Safety pre-analyzed (see §5). |
| **Phase 3 — watchdog M4b must-fixes + installer rewrite** | ⬜ NOT STARTED. |
| **Phase 4 — activation (Task Scheduler + auto-logon, USER-gated)** | ⬜ NOT STARTED. |

**Right now the live hub is running detached and healthy.** Do NOT relaunch it unless you've stopped it first (singleton mutex blocks a 2nd instance).

---

## 1. What Phase 1 delivered (the big picture)

Yaşar Usta is now a **standalone sibling git repo** at `C:\Users\sakir\Dropbox\Workspaces\yasar_usta` with:
- Its **own venv** (`../yasar_usta/.venv`, py3.10) — installs only `aiohttp, pyyaml, python-dotenv, psutil` (+ pytest-asyncio/aiohttp for tests). Never imports KutAI packages.
- A **generic entry**: `python -m yasar_usta --registry <path>` (was `kutai_wrapper.py`, now deleted).
- A **declarative `registry.yaml`** (hub-side, at `../yasar_usta/registry.yaml`) with a `kutai` project block: per-project `root`/`venv_python`/`hook` path/`messages`/targets. Loader supports `${env:VAR}` + per-project `${project_root}`.
- The **singleton kernel mutex** (`Global\YasarUstaHub`), heartbeat/watchdog code (from the prior always-live work), all carried over.
- Its **own `.env`** (`../yasar_usta/.env`, gitignored) holding `YASAR_USTA_BOT_TOKEN` + `TELEGRAM_ADMIN_CHAT_ID` (copied from kutay's `.env`). `__main__.main()` calls `load_dotenv()`.

KutAI-side (in the kutay repo):
- **`yasar_hooks.py`** (repo root) — subprocess CLI the hub invokes in kutay's venv for `pre_boot`/`on_exit` (stale-orchestrator kill = M6, stray-llama reconcile, orphan-kill). Imports `dallama`/`psutil` freely.
- KutAI installs `yasar_usta` editable (`-e ../yasar_usta` in `requirements.txt`) ONLY for the `HeartbeatWriter` client (`run.py:378`, `orchestrator.py:468`).
- **Deleted:** `kutai_wrapper.py`, `packages/yasar_usta/`, `scripts/install_yasar_autostart.ps1` (moved into the hub repo).

Architecture win: consumer projects contribute a registry block + a `yasar_hooks.py` + the editable client install. Zero shared-venv pollution. Project 2 onboards the same way.

## 2. Commits

**Hub repo `../yasar_usta`** (own git, branch `master`, **NOT pushed anywhere** — it's a fresh local repo, no remote):
- 68 commits of preserved history (subtree-split) + 14 Phase-1 commits, head `b21c223`.
- Key: `0d6513d` pyproject v0.2.0, `8ce0112` registry ${env:}/root, `9ec8525` config fields, `0956cb1` messages, `aac6ae1` subprocess hooks, `2c511a4` hub on_exit rewire + drop projects/, `2351c24` __main__ + boot asserts, `84c9439` self-restart -m, `5037df8` watchdog matcher, `b21c223` registry.yaml + start.bat.
- Suite: **184 passed / 0 failed.**

**KutAI repo `kutay`** (branch `main`):
- `4496d129` — `yasar_hooks.py` CLI (+ later psutil-race hardening, in e3f1cb71).
- `e3f1cb71` — the switchover (requirements repoint, start_kutai.bat, deletions, test rewrites, CLAUDE.md, yasar_hooks hardening).
- Design/plan docs committed earlier: `193de78e`, `a4d9c0f1`, `bb0b41e6` (spec), `348163d7`, `3423abe5` (plan).
- **Not pushed** (branch = main per user; push deferred — decide later).

## 3. LIVE STATE (as of handoff)

The hub is **running detached** (launched via PowerShell `Start-Process -WindowStyle Hidden`). Verified healthy:
- Singleton mutex `Global\YasarUstaHub` HELD (reliable probe: `New-Object System.Threading.Mutex($false,'Global\YasarUstaHub',[ref]$createdNew)` → `createdNew=False`).
- Clean single process lineage: hub (`-m yasar_usta`, venv-stub + Python310 child) → orchestrator (venv-stub + run.py child) → all one tree. Plus nerd_herd + yazbunu sidecars. **Exactly one of each** (the doubled PIDs are the venv launcher pattern, NOT duplicates).
- `hub.alive` (in `%LOCALAPPDATA%\YasarUsta\hub\`) refreshing at 60s; `orchestrator.heartbeat` (in `kutay\logs\`, Phase-1 location) refreshing at ~15s.
- `/restart` verified working by the user (clean mutex handoff → fresh hub lineage).
- Singleton dedup verified: a 2nd `start_kutai.bat` exits instantly (rc=0), mutex refuses it.

To re-check liveness in a new session: reliable mutex probe above, or scan for `-m yasar_usta` + `src\app\run.py` processes, or check `%LOCALAPPDATA%\YasarUsta\hub\hub.alive` mtime < 90s.

## 4. ⚠️ THE INCIDENT + THE LAUNCH RULE (critical)

During Phase-1 live-verify, the user launched via `! start_kutai.bat` (Claude Code inline shell) → **both KutAI and the hub died**. Root cause: **`!` runs the hub ATTACHED to Claude's transient shell**; when that shell context closes, the hub gets a console-close event → clean shutdown → which also stops the orchestrator it spawned → both dead. The relocation CODE was never at fault (a standalone run stays up: `> file` repro hit exit 124 = alive; the misleading exit-0 repro was a `| head` pipe artifact).

**RULE:** never launch the hub via Claude's `!`. Launch **detached**: double-click `start_kutai.bat` from Explorer (own console), or PowerShell `Start-Process ... -WindowStyle Hidden`. Phase-4's Task Scheduler auto-start makes this moot. This is now in the memory file too.

Also learned: raw `CreateMutexW` + a *separate* `GetLastError()` P/Invoke is UNRELIABLE (CLR resets last-error between calls → false `err=0`). Use the managed `Mutex(...,[ref]$createdNew)` for a trustworthy singleton probe.

## 5. Phase 2 — pre-analyzed design + SAFETY (do this next)

**Goal:** move Yaşar-owned runtime state out of Dropbox → `%LOCALAPPDATA%\YasarUsta\` (hub state is already there since Phase 1; this phase moves the per-project **heartbeat** + kills the split-brain), via a `${state_dir}` registry token + a `YASAR_USTA_STATE_DIR` env passed to the orchestrator child.

**The split-brain (the whole point):** today the orchestrator writes `logs/orchestrator.heartbeat` CWD-relative (`run.py:379` `_hb_paths`, `orchestrator.py:487-488`), and the hub reads the registry's absolute `heartbeat_file`. They agree only because CWD=kutay. If you move the registry path to `${state_dir}` (LOCALAPPDATA) but the orchestrator keeps writing project `logs/`, **the hub sees no heartbeat → false-kills a healthy orchestrator.**

**Tasks (from the plan §Phase 2):**
- **T2.1** — `${state_dir}` token in `registry.py` (per-project `%LOCALAPPDATA%/YasarUsta/<id>`). TDD `test_registry_statedir.py`. (HUB)
- **T2.2** — hub passes `YASAR_USTA_STATE_DIR` to each managed child (`supervisor.py` `build_child_env`; thread `ProjectConfig.state_dir` from hub `__init__`). TDD `test_child_env_statedir.py`. (HUB)
- **T2.3** — orchestrator reads the heartbeat path from `YASAR_USTA_STATE_DIR` (new `src/app/hb_paths.py::heartbeat_paths()`, used in `run.py` + `orchestrator.py`; falls back to relative `logs/` when env unset). Switch registry `heartbeat_file` → `${state_dir}/orchestrator.heartbeat`. TDD `tests/yasar/test_heartbeat_path.py`. (KUTAY + HUB registry)
- **T2.4** — `hub.py` fail-loud on a relative `log_dir` (CWD trap). TDD `test_logdir_absolute.py`. (HUB)
- **T2.5** — LIVE-VERIFY: restart the hub **detached**, confirm the orchestrator is NOT false-killed over 5+ min, state under LOCALAPPDATA, nothing new churning in `kutay\logs` except `orchestrator.jsonl`.

**SAFETY (why landing T2.1–2.4 + the registry change is safe even before restart):** the coupling holds in BOTH states because the env-fallback matches the old registry path:
- Old hub still in memory → reads OLD registry (`project logs`). Orchestrator (new code) with no `YASAR_USTA_STATE_DIR` → falls back to `project logs`. **MATCH.**
- New (restarted) hub → reads NEW registry (`${state_dir}`) + sets the env. Orchestrator reads env → `${state_dir}`. **MATCH.**
- Any hub restart is a full `-m yasar_usta` re-exec → loads ALL new code + new registry consistently. The only cross-process contract is heartbeat path, and it's coupled in both states.
So: implement T2.1–2.4, commit, then restart the hub detached to activate, then verify (T2.5). Keep the `heartbeat_paths()` env-fallback intact — it's the safety net.

## 6. Phase 3 + 4 (after Phase 2 verified)

- **Phase 3** (from plan): watchdog M4b must-fixes in `../yasar_usta/src/yasar_usta/watchdog.py` — (1) grace-after-kill `.watchdog_killed` marker, (2) `hub.stopped` gate, (3) kill-death verification + alert. Then rewrite `../yasar_usta/scripts/install_yasar_autostart.ps1` for `-m yasar_usta` + LOCALAPPDATA `--alive`/`--marker`/`--stopped`. Marker/gate files live in `%LOCALAPPDATA%\YasarUsta\hub\`. TDD tests exist in the plan.
- **Phase 4** (USER-gated, host actions): run the installer elevated (registers at-logon + 3-min watchdog tasks), `netplwiz` auto-logon, reboot test, remove any legacy startup launcher. Optional: `os._exit(0)`→`os._exit(42)` self-restart cleanup; retire legacy `guard.py` + its 2 characterization tests.

## 7. How to resume (subagent-driven, same as Phase 1)

The plan is task-by-task with real code. Execution was subagent-driven (implementer + two-stage review per code task; direct for trivial config). To continue:
1. Read the plan's Phase 2 section (`docs/superpowers/plans/2026-07-21-...`).
2. Implement T2.1–2.4 (HUB tasks use `../yasar_usta/.venv/Scripts/python.exe -m pytest`; the shell resets cwd to kutay between Bash calls — `cd` each time).
3. Commit; **restart the hub DETACHED** (Start-Process, §4) to activate; verify no false-kill (T2.5) before declaring Phase 2 done.
4. Then Phase 3, then hand Phase 4 to the user.

## 8. Gotchas / notes

- **Never launch the hub via Claude `!`** (§4). Detached only.
- **Never taskkill llama-server.** To restart KutAI use Telegram `/restart` or the hub.
- Hub tests run in `../yasar_usta/.venv` (has pytest-asyncio/aiohttp). KutAI tests: targeted, foreground, `--timeout`; kill only your own hung pytest.
- The hub repo has **no git remote** — it's local-only. Decide whether to publish/remote it later.
- `settings.local.json` still has stale `kutai_wrapper.py` permission entries — harmless (never match), left untouched deliberately.
- `on_exit` runs blocking `subprocess.run` on the hub's event loop (≤60s) — a faithful match of the old blocking `_kill_orphan_processes`; future-harden via `asyncio.to_thread` if desired (out of scope).
- `yasar_hooks._iter_python_processes` was hardened to survive psutil `process_iter` races (batch `as_dict` raised on a vanishing proc → would have crashed pre_boot). Keep that pattern (attrs-less iter + per-proc guard).
- Memory file: `~/.claude/.../memory/project_yasar_relocation_alwayslive_merge_20260721.md` (full status).
