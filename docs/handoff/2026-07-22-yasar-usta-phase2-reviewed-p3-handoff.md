# Handoff — Yaşar Usta: Phase 2 DONE + LIVE + REVIEWED, Phase 3 next

**Date:** 2026-07-22
**Spec:** `docs/superpowers/specs/2026-07-21-yasar-usta-shared-hub-and-always-live-design.md`
**Plan:** `docs/superpowers/plans/2026-07-21-yasar-usta-shared-hub-and-always-live.md` (task-by-task, real code)
**Prior handoff:** `docs/handoff/2026-07-22-yasar-usta-relocation-phase1-done-handoff.md` (Phase 1)

---

## 0. Status

| Phase | State |
|---|---|
| **Phase 1** — relocate + decouple + entry + secrets | ✅ DONE + LIVE (hub runs standalone from `../yasar_usta`) |
| **Phase 2** — state-dir (`%LOCALAPPDATA%`) + split-brain heartbeat fix | ✅ **DONE + LIVE-VERIFIED + THOROUGHLY REVIEWED + fixes shipped** |
| **Phase 3** — watchdog M4b must-fixes + installer rewrite | ⬜ NOT STARTED (this handoff sets it up) |
| **Phase 4** — activation (Task Scheduler + auto-logon, USER-gated) | ⬜ NOT STARTED |

**The live hub is running detached and healthy on the new Phase-2 code.** State is under `%LOCALAPPDATA%\YasarUsta\`. Do NOT relaunch without stopping first (singleton mutex blocks a 2nd instance). To activate new code: Telegram → `🔁 Yaşar Usta` self-restart button (detached re-exec; never launch via Claude `!` — §5 of the Phase-1 handoff).

---

## 1. What Phase 2 delivered + how it was verified

**The fix:** the orchestrator now writes its heartbeat + state snapshot to a hub-supplied absolute dir (`%LOCALAPPDATA%\YasarUsta\kutai\`) read from env var `YASAR_USTA_STATE_DIR`, instead of CWD-relative `logs/`. The hub reads the same path from `registry.yaml`'s `${state_dir}` token. This kills the split-brain where the hub could false-kill a healthy orchestrator, and gets hot state (heartbeat, 15s cadence) out of Dropbox.

**Live-verify (T2.5) PASS** — user self-restarted the hub via the `🔁 Yaşar Usta` Telegram button; over 30+ min:
- `%LOCALAPPDATA%\YasarUsta\kutai\{orchestrator.heartbeat, heartbeat, orchestrator.state.json}` FRESH (~13s) — orchestrator writing there.
- OLD `kutay\logs\{heartbeat, orchestrator.state.json, claude_remote.signal}` STALE (~45 min) — writer moved off Dropbox.
- `hub.alive` fresh (60s); single `-m yasar_usta` lineage (2 PIDs = venv-stub + child, normal).
- **No false-kill** (a split-brain would kill within the 120s stale threshold).

**Design coupling (why it's safe in every intermediate state):** env-set → both sides use `${state_dir}`; env-unset fallback → both sides use `logs/…` (CWD=kutay). `load_registry` is read once at boot (no hot-reload), so editing the registry can't perturb a running hub. `write_heartbeat`/`write_state_snapshot` both `os.makedirs(exist_ok=True)`, so a non-existent state_dir is created (no false-kill from a missing dir).

## 2. Commits

**HUB `../yasar_usta`** (branch `master`, **LOCAL-ONLY — no git remote**), head `dd30876`, **199 tests green**:
- `23141ef` T2.1 `${state_dir}` per-project registry token (LOCALAPPDATA/YasarUsta/<pid>, else <root>/logs)
- `c892f65` T2.2 hub passes `YASAR_USTA_STATE_DIR` to child (`build_child_env` + `SubprocessManager.state_dir`, threaded via supervisor+hub; **fix: build env even when tgt.env empty** — old `if self.env:`→None would have split-brained)
- `2f19ac9` T2.3 registry `heartbeat_file`+`claude_signal_file`→`${state_dir}`; hub freeze-snapshot reader `state_snapshot_candidates` (state_dir sibling first)
- `e7b2636` T2.4 `assert_hub_log_dir_absolute` fail-loud on relative hub log_dir
- `749605a` harden: `start()` calls the tested `build_child_env` (shipped==tested) + coupling guard
- `dd30876` **review fixes**: 3 new tests (child receives env via real `start()`; multi-project state_dir isolation; log_dir guard fires in `run()`); renamed `assert_state_dir_absolute`→`assert_hub_log_dir_absolute`; `build_child_env` docstring fixed

**KUTAY `kutay`** (branch `main`, **PUSHED** `origin/main` @ `59395ff4`):
- `1ddf5513` T2.3 new `src/app/hb_paths.py` (`heartbeat_paths()`+`state_snapshot_path()` read env, fallback to OLD relative = safety net); wired into `run.py:379` + `orchestrator.py:487`
- `2b9a40df` harden: writer-side coupling guard (strict `== os.path.join(sd, "orchestrator.heartbeat")`)
- `59395ff4` **review fixes**: revert-to-literal regression guard (run.py/orchestrator.py must use `hb_paths`, not a `logs/…` literal); deleted dead `kutay/registry.yaml`; DRY `_state_dir()` in hb_paths

## 3. The thorough review pass (2026-07-22) — outcome

3 parallel opus reviewers (KUTAY integration · HUB edge-cases · test+code quality), on top of the earlier opus adversarial false-kill review. **Zero blockers. Code correct + live-proven.** Fixes applied this session:
- **Deleted `kutay/registry.yaml`** — git-tracked orphan from pre-relocation, pointed at OLD `logs/` paths + a dead `hook_module`; the hub loads `../yasar_usta/registry.yaml` (via `start_kutai.bat`), nothing loaded the kutay copy. Actively misleading → gone.
- **Closed 4 test-coverage gaps** (the reviewer's headline — coverage previously couldn't catch the phase's own regression): child actually *receiving* the env via real `start()`; `run.py`/`orchestrator.py` actually *using* `heartbeat_paths()` (revert-to-literal now fails a test); the log_dir guard *firing* on boot; multi-project state_dir isolation. FIX 4/5 revealed no hidden bugs (loop binds `pid` correctly; guard is first statement in `run()`).
- **Renamed** the misleading `assert_state_dir_absolute` (it checks `cfg.log_dir`) → `assert_hub_log_dir_absolute`; fixed the stale `build_child_env` docstring.

### Deferred / known-issues carried into later phases (NOT bugs — all verified harmless today)
1. **Multi-target state_dir aliasing (latent):** every target of a project shares `proj.state_dir`, and `hb_paths` hardcodes the filename `orchestrator.heartbeat` — two targets in one project would collide. Fine for single-target kutai; a trap for any future multi-target project. Fix when a 2nd target is added (per-target sub-path or per-target filename).
2. **`shutdown.signal` still in Dropbox `logs/`:** the hub writes `${project_root}/logs/shutdown.signal` (supervisor.py:152) and the orchestrator reads `logs/shutdown.signal` (CWD-relative) — still coupled via CWD, so it works, but it's the same "shared-path contract in Dropbox" class as the deferred pid files. Migrate together (see #3).
3. **Pid-file migration deferred (yazbunu/nerd_herd):** their `pid_file` stayed at `${project_root}/logs`. nerd_herd's `--pid-file` is registry-controlled (safe to move in lockstep), but **yazbunu writes its pid CWD-relative via `--log-dir ./logs`** (uncontrolled) — moving its registry `pid_file` blind would create a new split-brain. Needs a per-sidecar coupling analysis before moving. Low-churn (written once), not on any kill path.
4. **HUB repo has no git remote** — local-only. Decide whether to publish it.
5. **Accepted nits (left as-is):** the `logs/orchestrator.state.json` legacy candidate in `state_snapshot_candidates` (now CWD-relative to the hub, never matches — harmless last-resort, diagnostic-only path); the bare `${state_dir}/heartbeat` second file (written, no hub reader — legacy parity); `${env:VAR}`-unset raises `ValueError` not `SystemExit` (still fails loud, just inconsistent presentation).

---

## 4. PHASE 3 — what to do next (all HUB repo)

**Goal:** make the hung-hub watchdog safe to *activate* (no crash-loop, respects a deliberate stop, verifies its kill), and point the auto-start installer at the generic entry + LOCALAPPDATA. **Running the installer stays USER-gated (Phase 4).** Full task-by-task code is in the plan §"PHASE 3" (lines ~1668–1930).

- **T3.1 — M4b#1 grace-after-kill marker** (`watchdog.py::run_once` + `tests/test_watchdog_grace.py`): after a kill, write `.watchdog_killed`; skip further kills within a grace window (default 3 ticks) so a slow reboot isn't crash-looped.
- **T3.2 — M4b#2 `hub.stopped` gate** (`watchdog.py` + `hub.py` + `tests/test_watchdog_stopped.py`): if `hub.stopped` exists, a stale hub is a *deliberate* stop → no kill. Wire `run()` to `unlink(hub.stopped)` on boot now; the *create-on-deliberate-stop* half lands with a future `/shutdown-hub` command (flagged deferral, not silent).
- **T3.3 — M4b#3 verify-the-kill** (`watchdog.py` + `tests/test_watchdog_killverify.py`): after killing, re-check each PID with `is_pid_alive`; if any survive, `alert(...)` ("hub PID survived kill — possible zero-effective-hub"). Then wire `watchdog.main` to real defaults + `--marker`/`--stopped` args (siblings of `--alive` in `%LOCALAPPDATA%\YasarUsta\hub\`).
- **T3.4 — rewrite `scripts/install_yasar_autostart.ps1`**: target `-m yasar_usta --registry <hub>\registry.yaml`, `WorkingDirectory` = hub root, `--alive/--marker/--stopped` under `%LOCALAPPDATA%\YasarUsta\hub\`, keep the elevated at-logon trigger + 3-min watchdog repetition. Parse-check only (do NOT register — that's Phase 4).
- **T3.5 — verify**: full HUB suite green; USER watchdog dry-run (healthy hub → kills nothing; simulate a hang → kills + writes marker; immediate 2nd tick skips via grace). Then decide push (HUB is remote-less today).

**Before T3.1:** confirm the current `watchdog.py::run_once` signature/behavior (Phase 1's T1.11 added `cmdline_is_hub`/`find_hub_pids`; verify `run_once` exists and its current params) so the extension is accurate.

## 5. PHASE 4 (USER-gated, hand to Sakir — not code)
Run the installer elevated (registers at-logon + 3-min watchdog tasks); `netplwiz` auto-logon (stored-password tradeoff — user's call); remove any legacy startup-folder/`.vbs` launcher; reboot test (`Get-ScheduledTask YasarUsta*`, singleton holds). Optional: flip `_do_restart_hub` `os._exit(0)`→`os._exit(42)` so Task Scheduler is the sole relauncher; retire legacy `guard.py` + its 2 characterization tests once nothing imports `ProcessGuard`.

## 6. How to resume + rules
- **Method:** subagent-driven, TDD per task (as Phase 1/2 were). HUB tests: `../yasar_usta/.venv/Scripts/python.exe -m pytest <file> --timeout=60`. Git Bash resets cwd to kutay between calls — `cd` to the hub each command.
- **NEVER** launch the hub via Claude `!` (attaches to the transient shell → dies on close, taking the orchestrator with it). Detached only; the `🔁 Yaşar Usta` self-restart is detached by design.
- **NEVER** taskkill llama-server. Restart KutAI via Telegram `/restart` or the hub.
- **NEVER** run the full KUTAY pytest suite while the live app is up (zombie pytest holds SQLite locks → crash-loops KutAI). Targeted, foreground, `--timeout`; kill only your own hung pytest. (HUB full suite is safe — separate venv, no live dependency.)
- Memory file: `~/.claude/.../memory/project_yasar_relocation_alwayslive_merge_20260721.md`.
