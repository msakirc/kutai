# Handoff — Yaşar Usta "Always Lives, Never Duplicates"

**Date:** 2026-07-21
**Branch:** `feat/yasar-usta-multiproject-hub` (local-only — **NOT pushed to origin**)
**Design:** `docs/superpowers/specs/2026-07-17-yasar-usta-always-live-singleton-design.md` (§9 forensics, §10 pass-2, §11 impl status)
**Reviews:** design 2 passes + a full-code adversarial pass (2026-07-21, verdict SOUND-WITH-FIXES).

---

## 0. TL;DR / status at a glance

| Goal | State |
|---|---|
| **NEVER duplicates** | ✅ **DONE + live-verified in prod.** Named kernel mutex gates the hub; a 2nd launch exits(0) immediately. |
| **ALWAYS lives** | ⚠️ **Code mostly written, NOT active, and the watchdog has must-fix bugs.** No OS relauncher is registered yet; the only relaunch today is the hub's own Popen self-fork. |

**The reboot on 2026-07-21 killed the hub and nothing restarted it** — the exact always-lives gap. It was relaunched manually. That gap closes only when the auto-start task is installed (deferred by user) **and** the watchdog must-fixes (§4 below) land first.

---

## 1. What shipped (commits, in order)

All on `feat/yasar-usta-multiproject-hub`:

| Commit | What |
|---|---|
| `b121a685` | **M1** `packages/yasar_usta/src/yasar_usta/singleton.py` — named-mutex authority: `decide_singleton` (`Global\`→`Local\` on ACCESS_DENIED; NULL handle never "free"; fail-closed), `enforce_singleton` (OWNED proceed / ALREADY_RUNNING exit0 / ERROR → circuit-breaker), `record_fault` (JSON marker, dedup 1h, give-up-after-K=5), `release_singleton`, real `_win32_create_mutex`. 19 TDD tests incl. real Win32. |
| `ec938b63` | **M2** wired into `hub.py`: `Hub._acquire_singleton()` gates `run()` **before** `acquire_lock`/pre_boot; `_do_restart_hub` releases mutex+lock **before** its Popen self-fork; `_sync_alert` blocking urllib. |
| `d03101ca` | **FIX1** (from review) `tests/conftest.py` autouse fixture neutralizes the real Win32 mutex so tests never touch/hold the machine-global prod mutex; regression guard test. |
| `8f6a7278` | **M6** `projects/kutai/hooks.py` `_kill_stale_orchestrators(project)` matches the target's **absolute run-script path** (psutil), not a bare `"run.py"` substring (which force-killed torch/pexpect/openai/watchfiles run.py + sibling projects). Dropped deprecated `wmic`. 8 TDD. |
| `2adee180` | **M4a** `watchdog.py` (hung-hub watchdog decision + CLI) + `hub.py` writes `logs/hub.alive` via a decoupled `HeartbeatWriter` task + `scripts/install_yasar_autostart.ps1` (main at-logon task + 3-min watchdog task). |

Full `yasar_usta` suite at handoff: **174 passed** (2 pre-existing asyncio teardown warnings, unrelated).

---

## 2. Verification state (what's proven vs assumed)

- **M1/M2/FIX1/M6 — LIVE-VERIFIED (2026-07-20 restart):** hub holds `Global\YasarUstaHub` (probe `err=183`); 2nd manual `python kutai_wrapper.py` exits 0 immediately; orchestrator + sidecars up (M6 did NOT over-kill the live orchestrator).
- **Real-token probe:** this box **can** create `Global\` mutexes (`err=0`) — the pass-1 reviewer's "token lacks SeCreateGlobalPrivilege" was wrong. Prod runs on `Global\` (cross-session); `Local\` fallback is moot here but correct for other tokens.
- **M4a — NOT live-verified.** The hub currently running is **pre-M4a** (relaunched before M4a was written) → it does not yet write `hub.alive`. Picks up on next restart. No urgency (watchdog isn't active).
- **Layer 0 (auto-start) — NOT registered.** `Get-ScheduledTask YasarUsta*` → none. Always-lives is therefore **not in effect**; the sole relauncher is the un-retried Popen in `_do_restart_hub`.

---

## 3. How to operate it right now

- **Manual (re)launch** (survives the shell):
  ```
  Start-Process -FilePath "C:\Users\sakir\Dropbox\Workspaces\kutay\.venv\Scripts\python.exe" `
    -ArgumentList "kutai_wrapper.py" -WorkingDirectory "C:\Users\sakir\Dropbox\Workspaces\kutay" -WindowStyle Hidden
  ```
- **Verify it's the new code + singleton works:** probe `Global\YasarUstaHub` → `err==183` means the live hub holds it (healthy). A 2nd manual wrapper launch must exit 0 immediately.
- **Restart / stop:** via Telegram (Yaşar Usta bot). `/restart_hub` (now behind a confirm) uses the Popen self-fork; `/stop` stops the **orchestrator child only** — the hub stays alive polling.

---

## 4. Review findings — MUST-FIX before the auto-start/watchdog is activated

The full-code adversarial pass (2026-07-21) found the watchdog can *harm* liveness. **These block running `install_yasar_autostart.ps1`:**

1. **[crash-loop] Watchdog has no "grace after kill".** `watchdog.py` is stateless — every tick is a fresh process. After it kills a hung hub, `hub.alive` keeps the **old stale timestamp** (the killed hub never updated it). The next tick (180s later) reads it while the new hub is still booting (venv import + `pre_boot` `_reconcile_stray_llama` can be slow) → kills the **new, healthy** hub → loop. **Fix:** write a `.watchdog_killed` marker (ts) in `run_once`; on the next tick, if `now - kill_ts < grace` (~360s) return `[]` without killing. (§7 design already calls for this.)

2. **[resurrect-stopped] `hub.stopped` gate missing.** §7 requires kill only if stale **AND no `hub.stopped`** **AND** pid alive. `decide_kill` checks only stale + pid. Latent today (no deliberate hub-down command), but ship the gate + write `hub.stopped` on any future `/shutdown-hub`.

3. **[zero-hub] Silent kill failure leaves the mutex-holder alive.** `find_hub_pids` returns BOTH venv PIDs (stub + real child that holds the mutex). `kill_pid` swallows all exceptions (`watchdog.py`). If the mutex-holding child survives a failed kill, every relaunch hits `ERROR_ALREADY_EXISTS` → exit(0) → **permanent zero-effective-hub, silently.** **Fix:** after killing, confirm death (`is_pid_alive`) and log/alert on failure; prefer killing the mutex-holding real interpreter.

**Do items 1–3 (TDD) before installing the tasks.** Activation was deferred anyway, so there's time.

---

## 5. Remaining milestones (ordered)

- **M4b (watchdog must-fixes 1–3 above)** — TDD, `watchdog.py`. *Next code task.*
- **M3 activation (host, USER go-ahead — DEFERRED):** run `scripts/install_yasar_autostart.ps1` **elevated** (registers main at-logon + watchdog tasks), then `netplwiz` auto-logon (security tradeoff: stored password) for reboot-without-login. Remove any `start_kutai.vbs` from `shell:startup`. After this, `os._exit(0)` in `_do_restart_hub` could become `os._exit(42)` so Task Scheduler is the sole relauncher — but the current Popen bridge also works, so this is optional cleanup.
- **M5 state relocation (RISKY, deferred):** move `logs/` runtime state (lock, `hub.alive`, `orchestrator.heartbeat`, `.mutex_fault.json`, 42MB `guard.jsonl`) out of Dropbox → `%LOCALAPPDATA%\YasarUsta` + pass an absolute `YASAR_USTA_STATE_DIR` env to the orchestrator child + sidecars (they must NOT re-derive it → the heartbeat-path-split bug that would make the hub false-kill a healthy orchestrator). Add a `systemprofile` boot-assertion. Touches `config.py`, `registry.yaml`, `hub.py`, and the orchestrator (`src/app`). **High blast radius + collides with the parallel session** — do carefully, coordinated, with live-verify.

---

## 6. Residuals / deferred (not blocking, tracked)

- **DETACHED_PROCESS in `_do_restart_hub`** — benign for a python re-spawn (unlike the remote.py/Claude-CLI regression), but the new hub gets no stdout. Consider `CREATE_NO_WINDOW` alone for symmetry. Defer.
- **`record_fault` constant signature `"mutex_error"`** — all fault types share one give-up counter. Fail-closed-safe, coarser than §10c. Defer.
- **`_held_handle` double-acquire leak** — only if `enforce_singleton` runs twice/process (not reachable in prod). Defer.
- **`TestRealMutexSeam` leaks a real (PID-unique) handle** for the pytest process — harmless (can't collide with prod name). Add `release_singleton()` teardown. Defer.
- **`HubConfig.log_dir` defaults to `"logs"` (relative)** — prod overrides absolute via `registry.yaml`, but a non-registry caller reintroduces the §3.2 CWD trap. Fixed by M5.
- **`hub.alive` written 1440×/day into Dropbox** — sync churn + conflicted-copy risk (but `read_alive_ts` returns None on a partial read → watchdog fails **safe**). Fixed by M5.
- **Not pushed to origin** — branch is local; parallel session also committing here. Decide when to push/merge to `main`.

---

## 7. Intentions / rationale (why it's built this way)

- **Task Scheduler @ elevated logon (user session), NOT an nssm/Session-0 service.** A Session-0 service breaks the S13/S14 desktop-presence sensors (`s13_presence.py`/`s14_contention.py` would read an empty session → "always away" → wrong load-mode). This is the choice validated in the design after the user's prior service duplicated endlessly.
- **Mutex is the singleton authority, not the file lock.** The old CWD-relative file lock (`log_dir="logs"`) was the root of the service-era dupes (service CWD=System32 → lock in the wrong dir). The kernel mutex is CWD/session-independent, auto-released on death. `Global\` preferred (cross-session), `Local\` fallback.
- **Fail-CLOSED on ambiguous mutex errors** — the user cancelled the last attempt over *duplication*, not downtime; never proceed into a possible 2nd hub. Circuit-breaker prevents a permanent fault from an alert storm.
- **Never stack two relaunchers** — hub self-restart is a clean handoff (release mutex → spawn → exit); the scheduler's restart-on-failure only fires on nonzero exit, so they don't double.

---

## 8. Key file map

| File | Role |
|---|---|
| `packages/yasar_usta/src/yasar_usta/singleton.py` | Named-mutex authority (M1). |
| `packages/yasar_usta/src/yasar_usta/watchdog.py` | Hung-hub watchdog (M4a) — **has must-fixes §4**. |
| `packages/yasar_usta/src/yasar_usta/hub.py` | Gate wiring, self-restart, `hub.alive` writer (M2/M4a). Parallel session also edits this. |
| `packages/yasar_usta/src/yasar_usta/projects/kutai/hooks.py` | Precise orphan-orchestrator kill (M6). |
| `packages/yasar_usta/tests/conftest.py` | Autouse mutex isolation for tests (FIX1). |
| `scripts/install_yasar_autostart.ps1` | Task Scheduler installer (M3) — **do not run until §4 fixes land**. |
| `docs/superpowers/specs/2026-07-17-yasar-usta-always-live-singleton-design.md` | The design + all review history. |

---

## 9. Next steps (concrete, in order)

1. **Fix watchdog §4 items 1–3** (TDD in `watchdog.py`) — the `.watchdog_killed` grace marker, the `hub.stopped` gate, and kill-death verification. Blocks activation.
2. **Restart-verify M4a** — after next `/restart`, confirm `logs/hub.alive` is being written fresh.
3. **User activates M3** — run `install_yasar_autostart.ps1` elevated + auto-logon. Then reboot-test: PC restart → hub auto-starts.
4. **Push the branch** (decide origin/merge-to-main timing; coordinate with parallel session).
5. **M5** (state out of Dropbox) — carefully, with live-verify.

---

## 10. Operational gotchas

- **Zombie pytest** on this box holds SQLite locks / can collide — run targeted, foreground, with a timeout; kill only your own hung processes.
- **LF→CRLF** git warnings on commit are expected (Windows), harmless.
- **A parallel session is actively committing to this same branch** — files (esp. `hub.py`) move; re-read before editing.
- **Do NOT taskkill llama-server**; to restart KutAI use Telegram `/restart` or the hub, not process kills.
