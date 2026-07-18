# Yaşar Usta — "Always Lives, Never Duplicates" Hardening Design

**Date:** 2026-07-17
**Status:** Design — decision recorded, awaiting implementation
**Scope:** Singleton guarantee + auto-start/relaunch guarantee for the Yaşar Usta process manager (and its multi-project hub successor). Companion to `2026-07-17-yasar-usta-multiproject-hub-design.md` — this doc hardens the process-lifecycle layer that spec assumes.
**Decision owner:** Sakir. Owner directive: *"whichever GUARANTEES my requirements."* Scope this session: design doc only, no code.

---

## 1. Requirements (verbatim intent)

1. **ALWAYS lives** — survives crash, kill, hang, and reboot; comes back without manual action.
2. **NEVER duplicates** — at most one wrapper/hub process, at most one orchestrator per project, ever. No second process under any launch path.

Past pain: registering as a Windows service produced **multi-wrappers and multi-orchestrators**. This design finds the root cause and removes it structurally, not by luck.

---

## 2. Current state (audited + live-verified 2026-07-17)

### Singleton mechanism (works today, but fragile)
- Two-file lock in `logs/`: `guard.lock` (10-digit zero-padded PID, always readable) + `guard.lk` (msvcrt `LK_NBLCK` exclusive sentinel).
  - `packages/yasar_usta/src/yasar_usta/lock.py`: `acquire_lock()` 41–73, `_acquire_lock_msvcrt()` 75–132, `is_pid_alive()` 16–38 (Win32 `OpenProcess(0x1000)`), `release_lock()` 161–181, `atexit` hook L72.
  - Stale recovery: if `guard.lk` can't be locked, read PID from `guard.lock`; if alive → `sys.exit(1)`; if dead → unlink sentinel + retry. Sound logic.
- Acquired in `ProcessGuard.run()` (`guard.py:618`) via `acquire_lock(cfg.log_dir, name="guard")`.

### Lifecycle (sound)
- `SubprocessManager.start()` (`subprocess_mgr.py:102`) spawns orchestrator: `CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`.
- Hang detection: `wait_for_exit()` 187–241 polls in 30s slices; `is_heartbeat_stale()` 243–252 kills if `logs/orchestrator.heartbeat` > 120s old (`kutai_wrapper.py:148`).
- Backoff `[5,15,60,300]`, reset after 600s (`backoff.py`, `kutai_wrapper.py:144`).
- Graceful stop escalation `CTRL_BREAK → terminate → kill` (`subprocess_mgr.py:155`).

### Auto-start (as shipped)
- `start_kutai.vbs` → hidden `start_kutai.bat` (`cd /d ...kutay` then `.venv\Scripts\python.exe kutai_wrapper.py`).
- **No** Windows service / Task Scheduler / registry Run key / startup `.lnk` in-repo. "Always live" today = whatever launches the VBS + Yaşar Usta's own restart loop.

### Live verification (this machine, 2026-07-17)
- 1 wrapper (PID 247584), 1 orchestrator, sidecars up (yazbunu 9880, nerd_herd 9881). `guard.lock` = `0000247584`. **No live duplication.** Singleton healthy *right now*.
- `wmic` **present** (`C:\Windows\System32\Wbem\WMIC.exe`) — orphan-killer works today (but deprecated; see §4.3).
- ⚠️ **30+ zombie pytest processes** running from parallel sessions — the known SQLite-write-lock crash-loop hazard. Out of scope here; flagged for separate cleanup.
- ⚠️ Runtime state lives in **Dropbox**: `guard.jsonl` 42 MB + 3× ~50 MB rotated, plus lock/PID/heartbeat files — all syncing. Waste + correctness hazard (§4.2).

---

## 3. Root cause: why a Windows service duplicated

Two independent flaws, both real, both must be fixed regardless of which auto-start we choose.

### 3.1 Two supervisors fighting (double-relaunch)
A service (nssm/`sc` with restart-on-failure) is **itself** a supervisor. Yaşar Usta is **also** a supervisor with its own self-restart (`exit 42` / `os._exit`). When the wrapper self-exits to restart, the SCM sees the exit and **also** relaunches → two wrappers race the lock in the same instant. No single owner of "who brings the wrapper back."

### 3.2 CWD-relative lock silently disabled mutual exclusion
`acquire_lock(cfg.log_dir=...)` with `log_dir="logs"` resolves `Path("logs")` **against the current working directory**. The `.bat` does `cd /d ...kutay`, so today it resolves correctly. A **service defaults CWD to `C:\Windows\System32`** → the lock becomes `C:\Windows\System32\logs\guard.lk` — a *different file* → the msvcrt exclusion is against the wrong path → **two wrappers each "acquire" their own lock** → two orchestrators. This is the smoking gun.

### 3.3 Force-kill skipped cleanup
Service stop/restart force-kills the wrapper → its cleanup (`_kill_stale_orchestrators`, kill `llama-server`) never runs → orphan orchestrator + llama-server survive → next launch adds a second → dual + VRAM corruption.

**Conclusion:** the service didn't fail because "services are bad." It failed because two supervisors owned lifecycle and the lock key was CWD-relative. Fix those and any launcher is safe.

---

## 4. Design

Three rules make the guarantees structural. Then one Layer-0 decision.

### 4.1 Rule A — a named kernel mutex is the singleton authority (NEVER duplicates)

**Revised after live validation (§9).** The naive "acquire `Global\` mutex as the first action" is under-specified against this box's proven realities: the launch token lacks `SeCreateGlobalPrivilege`, the orphan-killer runs before any lock, and a mishandled `CreateMutex` failure would reintroduce unlimited duplicates. The corrected spec:

**Primitive + namespace.** Prefer `Global\YasarUstaHub`; **fall back to `Local\YasarUstaHub`** on ACCESS_DENIED. Both are correct here — the design keeps every instance in one interactive session (§4.4), so `Local\` (which needs **no** privilege) already dedups; `Global\` only adds cross-session protection *when* the token is elevated. Never make correctness depend on an elevation the historical launch path provably lacked (`whoami /priv` on the launching `sakir` token has no `SeCreateGlobalPrivilege`, §9).

**Exact decision tree (this is the contract — a wrong branch = the old bug returns):**
```
h = CreateMutexW(NULL, FALSE, "Global\\YasarUstaHub")
err = GetLastError()
if h and err == 0:                      # we own it
    hold h for entire process lifetime (never CloseHandle); proceed
elif err == ERROR_ALREADY_EXISTS (183): # another hub owns it
    log("hub already running — exiting"); sys.exit(0)
elif err == ERROR_ACCESS_DENIED (5) or h is NULL for a privilege reason:
    retry once with "Local\\YasarUstaHub"; apply the same 3 branches
else:                                    # any other/ambiguous failure
    FAIL CLOSED: loud one-time Telegram alert via Yaşar bot; sys.exit(nonzero)
```
- **A NULL handle is NEVER "lock is free."** The single most dangerous misread. Only `ERROR_ALREADY_EXISTS` means "someone else has it → exit 0". Everything else is an error, handled above.
- **Fail CLOSED, not open.** The user cancelled the last attempt over *duplication*, not downtime. A duplicate is silent and burned him; downtime is recoverable (Task Scheduler retries in 1 min) and made loud via the alert. So on any ambiguous result, exit rather than risk a second wrapper.

**Acquired by the Python-running child, in one unambiguous sequence.** On this box the venv `.venv\Scripts\python.exe` is a **code-less C launcher stub** (PID 245204: 7 modules, no `python310.dll`) that spawns the base interpreter child (PID 247584: 63 modules incl. `python310.dll`) which is the *only* process that runs Python and holds the lock (§9). So the mutex is acquired exactly once, in the child — **safe against the venv pair, proven by the fact that today exactly one healthy wrapper exists, not a self-collision.** Exact order (this replaces the vague "strict first statement"):

> **`venv-guard` (`kutai_wrapper.py:26` `sys.exit(1)` if not the real interp) → acquire mutex → `_kill_stale_orchestrators` (:130) / `_reconcile_stray_llama` (:131) / config / `pre_boot`.**

- This means the mutex **moves out of `ProcessGuard.run()` (`guard.py:618`) up into `kutai_wrapper.py`**, immediately after the venv-guard (after L26) and **before** L130. Leaving it in `run()` would satisfy neither "before the killers" nor the anti-race goal.
- On the `ERROR_ALREADY_EXISTS` → `exit(0)` branch, **no killer has run yet** — closing the §3-era race where a mutex-loser still `taskkill /F`s the winner's orchestrator.
- The venv-guard-first ordering is safe: the `sys.exit(1)` at L26 runs in the code-running child, which is the *same* process that then acquires the mutex — consistent, no gap.
- **Do NOT** "eliminate the pair" by pointing the task at `...Python310\python.exe kutai_wrapper.py` directly — that bypasses the venv → missing site-packages → broken imports. Keep the venv launch; pin the mutex to the child; treat the module-load evidence as a **canary** — re-verify if the venv tooling (currently virtualenv 20.24.5, launcher model) ever changes to a layout where the stub also runs Python.

Keep `guard.lock` (PID file) **for diagnostics only**, at an absolute path (§4.2). The **mutex is the sole singleton authority**; retire the CWD-relative `guard.lk` msvcrt sentinel as authority (a future refactor must not re-promote a file lock). No file to sync, stale, or Dropbox-conflict.

### 4.2 Rule B — runtime state out of Dropbox, absolute paths only
- Relocate logs / lock / PID / heartbeat / IPC signal files → `%LOCALAPPDATA%\YasarUsta\` (or `%PROGRAMDATA%\YasarUsta\` if ever run pre-logon).
- Resolve **all** such paths absolute at startup from `Path(__file__).resolve()` / the env var — **never** from CWD. This alone would have prevented §3.2; the mutex is the belt, this is the suspenders.
- Removes ~190 MB of churning Dropbox sync and the synced-lock hazard.
- App-side IPC contract (`logs/…heartbeat`, shutdown signal) is hardcoded today (hub spec finding #3). KutAI-the-target may keep `logs/` for now; new targets read these paths from env/args. Document the seam; don't block on it.

### 4.3 Rule C — one relauncher, hub never self-forks (ALWAYS lives, without §3.1)
- **Layer 0 (OS)** is the *sole* outer relauncher. **Layer 1 (hub)** is the *sole* supervisor of everything below it (orchestrators, sidecars) and owns its *internal* restarts.
- **DELETE the existing self-fork, don't just "not use it."** `guard.py:540-575` `_restart_self()` currently does `Popen([sys.executable, script], DETACHED_PROCESS)` then `release_lock()` then `os._exit(0)`. Left in place it **races Layer 0**: the self-forked child would hit `ERROR_ALREADY_EXISTS` and `exit(0)` before the dying parent releases the mutex → a **zero-wrapper window**. Remove the `Popen` path entirely. (It also carries the `sys.executable`-vs-venv-python bug and the `DETACHED_PROCESS` pattern already shown fragile in the Yaşar remote-button incident.)
- **TWO restart kinds — do NOT collapse them (P5c).** (i) **Orchestrator-child** crash/hang → **stays in-hub**: the existing backoff loop (`guard.py:744-768`, `[5,15,60,300]`) respawns the child; the hub process does **not** exit. Converting this to a hub exit would restart the whole hub + sidecars + Telegram poller on every orchestrator hiccup — a regression. (ii) **Hub-self** restart → **plain `os._exit(42)`** (delete the `_restart_self` Popen); Layer 0 relaunches. Only kind (ii) exits the process.
- Hub-self restart is clean because Layer 0 relaunches only **after** the prior process fully exited → mutex already free → new instance acquires cleanly, zero overlap. (Matches hub spec finding #1: self-restart is Hub-owned; `TargetSupervisor` has no process-exit path.)
- Exit-code contract to Layer 0:
  - **crash / hang-kill / `taskkill /F` / hub-self-restart (exit 42) → nonzero exit → Layer 0 relaunches.** (Exit 42 stays the restart signal; `_kill_orphan_processes` keeps special-casing 42 so it does **not** kill llama-server on a clean restart.)
  - **`/stop` does NOT exit the hub.** Correcting an earlier draft (P1): `/stop` stops only the **orchestrator child** (`guard.py:405-412` sets `_stop_requested`, signals the child); the **hub stays alive and keeps polling Telegram** so `▶️ Start` can revive KutAI. If the hub exited on `/stop`, nothing would be left to receive `/start`. A full hub shutdown is a distinct rare action (`/shutdown-hub` → `os._exit(0)` → Layer 0's restart-on-failure ignores success → stays down until the user re-triggers the task).
- **Orchestrator (child) singleton is guaranteed by the hub's own start/stop ordering, NOT by the wrapper mutex.** Two orchestrators can still arise if a child-restart spawns while the old one is mid-teardown (hung, kill not yet landed). Rule: the supervisor **awaits confirmed old-exit before spawning** — "confirmed" = the existing `wait_for_exit()` return (which force-kills a hung child after 120s heartbeat-stale); owned-PID `pre_boot` cleanup then reaps any survivor **by PID** *after* confirmed exit and *behind* the mutex.
- **Replace the `wmic`-substring orphan killer — it is a live foot-gun, not just multi-project hygiene.** `kutai_wrapper.py:84` matches bare `"run.py"` and `taskkill /F`s every hit; the tree already contains `torch/distributed/run.py`, `pexpect/run.py`, `openai/.../run.py`, `watchfiles/run.py` — any Python process whose cmdline contains one gets force-killed. It also runs **pre-mutex** (§4.1), so a race-loser kills the winner's orchestrator. Move cleanup into each project's `pre_boot()` hook, scoped by **full command + cwd** (owned-PID), gated behind the mutex. Drops the deprecated `wmic` dependency too.

### 4.4 Layer 0 decision — Task Scheduler @ logon (elevated), NOT a service

| Candidate | Guarantees liveness? | Collateral cost | Verdict |
|---|---|---|---|
| **Task Scheduler @ logon, elevated, user session** + Windows auto-logon | Yes (see below) | None — presence/GPU/Dropbox native | **CHOSEN** |
| NSSM service (Session 0) | Yes, incl. no-login | **Breaks `s13_presence.py`/`s14_contention.py`** — reads Session-0's empty desktop → "always away" → wrong load-mode; plus Session-0 GPU/env friction that caused past pain | Rejected (silent feature regression) |
| Keep Startup-folder VBS | Partial — no relaunch after kill until next logon | none | Rejected (weakest liveness) |

Why Task Scheduler @ logon *guarantees* the requirements for this box:
- **Never duplicates:** the named mutex (§4.1 — `Local\` primary, upgraded to `Global\` when the elevated token allows) makes a second instance impossible even if two triggers ever coexist. Also set the task's **"If the task is already running: Do not start a new instance."**
- **Survives reboot:** enable **Windows auto-logon** (netplwiz / Sysinternals Autologon). Boot → auto-logon → "At log on" trigger fires → hub starts. At-logon becomes effectively at-boot for a single-user always-on machine.
- **Survives crash/kill:** task setting **"If the task fails, restart every 1 minute, up to 999 times."** Nonzero exit (crash/kill) = task failure → relaunch. Exit 0 (clean stop) = success → no relaunch (honors §4.3 contract).
- **Preserves the interactive features** the agent depends on: runs in the user's session, so S13/S14 presence sensing, CUDA, and a mounted/synced Dropbox all work exactly as today.

Documented boundary: a full **log-off** (not lock) ends the user session and stops the hub. On a personal always-on agent box you *lock and walk away* (S13 is built for exactly that) — locking keeps the session and the hub alive. If you ever need it alive with **nobody logged on at all**, that requires a non-interactive session (service or "run whether logged on or not"), which forfeits presence sensing — accept that trade explicitly before switching; it is not the default.

---

## 5. Failure-mode guarantee matrix (post-design)

| Scenario | Outcome |
|---|---|
| Reboot, auto-logon completes | Task fires at logon → hub up. ✅ lives |
| Reboot, auto-logon does NOT complete (BitLocker/PIN pre-boot, Windows Update "finishing setup"/lock screen, Fast Startup hiberboot, policy-cleared `AutoAdminLogon`) | Hub down until a human reaches the desktop. ❌ **honest non-guarantee** — see §7. "Always lives" is bounded by "an interactive session exists." |
| Hub crashes (nonzero exit) | Task restart-on-failure → relaunch after prior fully exited; mutex free → clean single instance. ✅ lives, ✅ single |
| Hub `taskkill /F`'d | Nonzero task result → restart-on-failure. ✅ |
| Hub hangs (event loop stuck, still "running") | **NOT covered by Layer 0** (task isn't "failed"). The hub watchdogs its *children*'s heartbeat but nothing watchdogs the *hub itself*. Requirement #1 is absolute → an outer liveness-file watchdog task is **REQUIRED, not optional** (§7). |
| Two auto-start triggers somehow both active | Second `CreateMutexW` → `ERROR_ALREADY_EXISTS` → `exit(0)`. ✅ single |
| Mutex `CreateMutexW` fails (ACCESS_DENIED / other) | Retry `Local\`; if still ambiguous → fail-closed exit + dedup'd alert + circuit-breaker (§10). Never proceeds → ✅ single (never a silent second) |
| Service run with CWD=System32 (the old bug) | Irrelevant now: mutex is CWD-independent; paths absolute. ✅ single |
| Power failure mid-run leaves stale lock file | No authority in the file anymore; kernel mutex already gone with the dead process → next start acquires cleanly. Stale PID file is cosmetic. ✅ |
| Explicit `/stop` (orchestrator) | Hub stays **alive** polling Telegram; only the KutAI child stops. `▶️ Start` revives it. ✅ lives, ✅ single |
| Hung hub falsely seen stale by watchdog | Watchdog kills only if `hub.alive` stale **AND** no `hub.stopped` marker **AND** hub PID alive; threshold > max backoff (§10). No flap. ✅ |
| Dropbox offline / not synced | Runtime state no longer in Dropbox → unaffected. ✅ |
| Orphan orchestrator from a hard kill | Per-target `pre_boot()` cleanup (full cmd+cwd match) reaps it before respawn; never touches other projects. ✅ single orchestrator |

---

## 6. How it slots into the multi-project hub spec

The 2026-07-17 hub spec already chose **one hub process, N in-process `TargetSupervisor`s, one shared Telegram poller** — that architecture *inherently* prevents N-wrappers. This design adds the missing lifecycle guarantees on top:
- Mutex acquired in the **`Hub` entry point** (the spec's finding-#1 self-restart owner) — one mutex for the whole hub, not per-target.
- State relocation + absolute paths belong in `config.py` path resolution.
- `wmic`→per-target owned-PID orphan cleanup lives in `projects/<id>/hooks.py::pre_boot()`.
- The exit-code→Layer-0 contract replaces `os._exit`-style self-restart.

No conflict with the hub spec; this is its process-lifecycle foundation.

---

## 7. Required backstops + honest guarantee boundary

**Honest statement of the guarantee (do not oversell — the last attempt disappointed by overpromising):**
> *"NEVER duplicates"* is **unconditional** — the mutex makes a second wrapper structurally impossible in every launch/race/reboot path.
> *"ALWAYS lives"* is **bounded**: the hub stays alive whenever an interactive user session exists (**locked screen is fine**; only a full log-off ends it), and recovers from crash/kill/hang within ≤1 min. **Reboot recovery is best-effort** — it depends on auto-logon actually completing, which the states in §5 can block. It is not a headless-server guarantee (that path is the rejected Session-0 service, which breaks presence sensing).

**Required (promoted from "optional") — hardened after pass 2 (§10 has the exact numbers):**
- **Outer hub-liveness watchdog.** The hub writes `hub.alive` (timestamp) from a **dedicated async task on a fixed cadence, independent of the crash/backoff loop** — otherwise a legitimate 300s backoff sleep looks like a hang. A second, tiny Task Scheduler task (interval **3 min**) reads it and **kills only if ALL hold: `hub.alive` stale > 360s AND no `hub.stopped` marker exists AND the hub PID is actually alive.** The kill → nonzero exit → the **main task's** restart-on-failure brings it back.
  - **Watchdog kills, never relaunches** — relaunch is solely the main task (one owner of "bring the hub back", per §3.1). It must never spawn a hub itself.
  - **Distinguish hung from intentionally-stopped:** on `/shutdown-hub` the hub writes `hub.stopped`; the watchdog then does nothing. `/stop` (child only) leaves the hub alive writing `hub.alive`, so no false trigger.
  - **Anti-flap:** threshold 360s > the 300s max backoff step, so a hub sleeping between orchestrator respawns is never killed. After a kill, the watchdog skips its next tick (grace for the mutex-restart window).
  - **Watchdog is single-instance, same session:** task set "do not start a new instance" and **"run only when user is logged on"** (same interactive session as the main task — a Session-0 watchdog would have a *different* `Local\` namespace and could launch a second hub the interactive `Local\` mutex can't see).

**Host config steps (runbook, not code):**
- Enable Windows **auto-logon** (netplwiz / Sysinternals Autologon) — required for the reboot path.
- Consider **disabling Fast Startup** so "shutdown" is a true cold boot and the at-logon trigger fires predictably.
- Create the main task **elevated** ("run with highest privileges") so `Global\` succeeds; the `Local\` fallback keeps it correct even if that's ever missing (correctness never *depends* on elevation).
- **Remove every other auto-start trigger — hard precondition, not a nicety.** The `Local\`-primary choice assumes a single interactive session; a stray VBS/task in a *different* logon session (2nd user, RDP) would escape a `Local\` mutex. One trigger only.

**Out of scope (later specs):**
- Zombie pytest swarm cleanup — separate, immediate ops task (30+ live now).
- App-side IPC path relocation for non-KutAI targets — deferred to a later hub sub-spec (finding #3).

---

## 8. Rollout (when implementation is approved; restart-gated per repo convention)
1. TDD: characterization tests for current lock + restart state machine first, then new mutex/path behavior. Add a test asserting a NULL/ACCESS_DENIED CreateMutex result does **not** proceed (fail-closed), and one asserting the mutex gate precedes `_kill_stale_orchestrators`.
2. Add the mutex per §4.1 as the **strict-first statement** (`Global\` → `Local\` fallback → fail-closed + alert); relocate state to `%LOCALAPPDATA%`; absolute paths from `__file__`; retire CWD-relative sentinel as authority.
3. **Delete** `guard.py:540-575` `_restart_self` Popen self-fork; **hub-self** restart → `os._exit(42)` (Layer 0 relaunches). **Orchestrator-child** crash/hang restarts **stay in-hub** (keep the `guard.py:744-768` backoff loop) — do not collapse the two (§4.3).
4. Replace `wmic` `"run.py"` substring killer with owned-PID (full cmd+cwd) cleanup, run **behind** the mutex; enforce await-`wait_for_exit()`-return before respawn (§4.3).
5. Add the `hub.alive` writer (dedicated cadence task) + `hub.stopped` marker; create the outer watchdog task per §7.
6. Create the main scheduled task (elevated, at-logon, do-not-start-new-instance, restart-on-failure) + watchdog task + enable auto-logon; remove every other auto-start trigger.
7. Commit to `main` (conventional message). **Push after live-verify**: reboot test, `taskkill /F` test, hang-injection test (watchdog fires ≤~6.5 min), `/stop` test (**hub stays alive, orchestrator stops, `▶️ Start` revives**), `/shutdown-hub` test (stays down, watchdog does NOT resurrect), and a deliberate double-launch test (second instance exits 0) all pass.

---

## 9. Validation — against the not-working version + adversarial review

This design was validated two ways: (a) forensic reconstruction of the prior cancelled attempt, (b) a skeptical Windows-systems review tasked to *break* it by replaying that history. Both were run 2026-07-17.

### 9a. What the prior (cancelled) attempt actually was
- **Never committed.** No service/nssm/schtasks/mutex artifact exists in git — the cancelled mechanism was a **host-level `nssm`/`sc` service**, set up outside the repo, so there is no diff to revert. Git only shows the *lock code* reacting to real incidents.
- **Real dual-wrapper incident** — commit `7a29641f` (Mar 26): the **system Python** (`AppData\...\Python310\python.exe`) ran `kutai_wrapper.py` *alongside* the venv Python — two different interpreters, two real wrappers. "Fixed" by a venv-guard `sys.exit(1)`.
- **Chronic false-positive detection** — the DUAL-WRAPPER panel warning "**always** reported" duplicates because the venv launcher stub + child share a cmdline; took two fixes (`c2fc5afb`, `612c0ba0`) to collapse by parent-chain. **Some "duplication" the user saw was a false alarm on normal venv behavior** — worth knowing before trusting any process-count alarm.
- **Confirmed root of the service-era real dupes:** `config.py:115` `log_dir="logs"` (relative) → a service with CWD=`System32` put the lock at `System32\logs\guard.lk` → each wrapper "acquired" its own lock. Plus two competing supervisors and force-kill-skips-cleanup (§3).

### 9b. Adversarial review verdict: **SOUND-WITH-FIXES**
The diagnosis (§3) and direction (mutex + absolute paths + single relauncher) were upheld. The review broke the *original* mutex spec on this box's proven realities and required these changes — **all folded into §4.1/§4.3/§5/§7/§8 above**:

| # | Finding (evidence) | Where fixed |
|---|---|---|
| 1 | `Global\` needs `SeCreateGlobalPrivilege`; the launching `sakir` token **lacks it** (`whoami /priv`). Naive "no handle = free lock" → unlimited dupes. | §4.1 decision tree + `Local\` fallback + fail-closed |
| 2 | Orphan-killer runs at **import (line 130), before any lock** → race-loser `taskkill /F`s winner's orchestrator. | §4.1 strict-first ordering; §4.3 |
| 3 | `_restart_self` **Popen self-fork still exists** (`guard.py:540-575`) → races Layer 0 → zero-wrapper window. | §4.3 mandate delete |
| 4 | Hang-alive death mode uncaught by Task Scheduler. | §5 + §7 watchdog **required** |
| 5 | Reboot "always lives" oversold (BitLocker/PIN, Windows Update, Fast Startup, policy). | §5 + §7 honest boundary |
| 6 | `"run.py"` substring already matches ≥4 library files → live `taskkill /F` foot-gun. | §4.3 owned-PID |
| — | venv pair is **safe** (stub is code-less: 7 modules vs child's 63 incl. `python310.dll`; only child runs Python + holds lock). | §4.1 pinned + canary |

### 9c. One reviewer suggestion REJECTED (verified, not rubber-stamped)
The review proposed pointing the task at `...Python310\python.exe kutai_wrapper.py` **directly** to eliminate the venv pair. **Rejected:** that bypasses the venv → missing site-packages → broken imports. Correct fix is to keep the venv launch and **pin the mutex to the Python-running child** (§4.1), with the module-load evidence as a canary if venv tooling changes.

### 9d. Does the new design defeat every historical failure?
| Historical defeat | New design answer |
|---|---|
| CWD-relative lock (System32) | Mutex is CWD-independent + absolute paths. ✅ |
| Two supervisors race-relaunch | One relauncher (Task Scheduler); self-fork deleted. ✅ |
| Force-kill skipped cleanup → orphans | Owned-PID `pre_boot` cleanup, behind mutex. ✅ |
| Pre-lock orphan-killer oscillation | Mutex is strict-first → loser exits before any kill. ✅ |
| system-vs-venv dual interpreter (`7a29641f`) | venv-guard retained + mutex in child. ✅ |
| Dropbox conflicted-copy of lock file | Mutex has no file; state out of Dropbox. ✅ |
| venv false-positive alarm | Cosmetic detector already chain-collapsed; mutex unaffected. ✅ |

Net: with §8's steps, the guarantee is **structural, not lucky**. The failure mode that made the user cancel last time (silent duplication) is closed on every replayed path; the residual risk is bounded *downtime* (hang→watchdog ≤~6.5 min; reboot→auto-logon best-effort), made loud, never silent duplication.

---

## 10. Second-pass review — resolutions

A second, two-lens review (adversarial + implementation-readiness) ran 2026-07-17. Verdict both lenses: **SOUND-WITH-FIXES; NEVER-duplicates survived intact — no path to two wrappers/orchestrators found.** All defects were on the *ALWAYS-lives* side, introduced by the first-pass fixes. Fixed inline above; the net-new decisions are pinned here so §8 is code-ready.

### 10a. Defects the first-pass fixes introduced (now corrected)
| Sev | Defect | Fix (section) |
|---|---|---|
| dead-hub | My `/stop → hub exits 0` killed the Telegram poller → nothing left to receive `/start` (contradicts `guard.py:405-412` keep-alive). | §4.3 + §5: `/stop` stops **child only**, hub stays alive; `/shutdown-hub` is the distinct hub-down action. |
| regression | "every restart → `os._exit(nonzero)`" would restart the whole hub+sidecars on every orchestrator hiccup. | §4.3: child crashes **stay in-hub** (backoff loop kept); only hub-self → `os._exit(42)`. |
| spam/loop | "one-time alert" re-fires every 60s across relaunches; no circuit-breaker; `backoff.py` doesn't apply (mutex exit is *pre*-loop). | §10c below. |
| dead/flap | Watchdog couldn't tell hung from `/stop`; no anti-flap; no own singleton. | §7 hardened (marker + threshold + kill-never-relaunch + same-session). |
| ref bug | Cited `_self_restart` @ `555-575`; actual is `_restart_self` @ `540-575`. | Corrected everywhere. |
| wording | §4.4/§5 said "Global mutex" as if primary. | Reworded to "named mutex (`Local\` primary)". |

### 10b. Resolved decisions (were blocking TDD)
1. **State root:** `%LOCALAPPDATA%\YasarUsta\` (single-user interactive box). `%PROGRAMDATA%` only if a future headless target appears. Absolute, resolved once at hub startup.
2. **State dir passed to children as env `YASAR_USTA_STATE_DIR`.** The hub resolves it once and injects it into the orchestrator + sidecar `env=`; **children MUST read the env, never re-expand `%LOCALAPPDATA%`** (re-expansion under a different token = the dual-truth bug). Boot assertion: if the resolved path contains `systemprofile`, fail-closed + alert (catches a task mis-set to SYSTEM).
3. **Mutex code home:** new `packages/yasar_usta/src/yasar_usta/singleton.py` (`ctypes` `CreateMutexW` wrapper + the §4.1 decision tree). Not `lock.py` (that's the file lock being retired to diagnostics).
4. **Watchdog:** interval **3 min**; `hub.alive` stale threshold **360s**; `hub.stopped` marker gates resurrection; kill only if `stale AND no-marker AND pid-alive`. Worst-case hang recovery ≈ 360s + one 3-min tick ≈ **6.5 min**.
5. **Exit-42 preserved** as the hub-self-restart signal; `_kill_orphan_processes` keeps special-casing 42 (don't kill llama-server on clean restart).
6. **await-old-exit** = the existing `wait_for_exit()` return (force-kills a hung child after 120s heartbeat-stale); owned-PID cleanup reaps any survivor by PID before respawn.
7. **`hub.alive` writer** is a dedicated async task on a fixed cadence, decoupled from the crash/backoff loop (so a 300s backoff sleep isn't read as a hang).

### 10c. Fail-closed circuit-breaker (replaces "one-time alert")
On an ambiguous `CreateMutex` failure (not `ERROR_ALREADY_EXISTS`):
- **Cross-process dedup:** before alerting, check/write `%LOCALAPPDATA%\YasarUsta\.mutex_fault` (timestamp + fault signature); suppress the Telegram alert if an identical one fired < 1h ago. If `%LOCALAPPDATA%` itself is the fault, fall back to `%TEMP%\yasar_mutex_fault`.
- **Give-up-after-K:** the same marker counts consecutive fail-closed exits. For the first **K=5**, `exit(nonzero)` (Task Scheduler retries in 1 min). At K=5, send one terminal *"giving up — human needed"* alert and `exit(0)` so restart-on-failure **stops the hammer** (no 999× loop). Human clears the marker on fix.
- Rationale: a permanent perms/path fault must be loud **once**, not a per-minute alert storm that buries the signal — while still never proceeding into a possible duplicate.

### 10d. Multi-project note (P6)
One hub mutex = **one hub process, N in-process `TargetSupervisor`s** (matches hub spec). Per-target restart backoff MUST be non-blocking w.r.t. other targets **and** w.r.t. the `hub.alive` writer — else a crash-looping project A stalls project B's heartbeat and the child-watchdog kills a healthy B. State this in the hub-implementation spec.

### 10e. Code references — verified 14/15 accurate (only the `_restart_self` name/line was wrong, now fixed). Guarantee wording consistent across §1/§5/§7/§9d. Consistent with the companion hub spec (finding #1: hub-owned self-restart, no `TargetSupervisor` exit path).

---

## 11. Implementation status + re-baseline (2026-07-18)

**Re-baseline:** the multi-project **hub shipped** since this design was written (commits `1bef56ec`→`5c63c9c2`). The entry point is now `Hub` (`hub.py`), not `ProcessGuard`. So the hardening targets moved: singleton + self-restart now live in `hub.py` (`Hub.run()` acquire_lock at ~`hub.py:274`; `_do_restart_hub` self-fork at ~`hub.py:150`), and orphan cleanup **already moved** to `projects/kutai/hooks.py::pre_boot` (my §4.3 recommendation — done). `guard.py` `ProcessGuard`/`_restart_self` is now legacy (Hub uses `TargetSupervisor`). `/stop` already stops the child only and keeps the hub alive (confirms the §4.3/P1 correction). The design (§4/§7/§10) is unchanged; only the wiring locations shifted.

**DONE (TDD, committed, restart-gated — not yet live-verified):**
- **M1 — `yasar_usta/singleton.py`** (`b121a685`): `decide_singleton` (Global→Local→fail-closed), `enforce_singleton` (circuit-breaker), `record_fault`, `release_singleton`, real Win32 seam. 19 tests incl. real mutex.
- **M2 — wired into `Hub`**: `Hub._acquire_singleton()` gates `run()` **before** `acquire_lock`/pre_boot; `_do_restart_hub` calls `release_singleton()` before re-spawn (avoids the zero-hub window while there is still a Popen self-fork). Full `yasar_usta` suite: 128 passed.

**REMAINING (each its own milestone):**
- **M3 — Layer 0 (Task Scheduler @ elevated logon + auto-logon) + convert `_do_restart_hub` from Popen self-fork to `os._exit(42)`.** The Popen→exit conversion is **gated on Layer 0 existing** (else self-restart has no relauncher). Until then M2's mutex-safe Popen is the bridge.
- **M4 — outer hub-liveness watchdog** (`hub.alive` writer + `hub.stopped` marker + watchdog task, §7).
- **M5 — state relocation** to `%LOCALAPPDATA%\YasarUsta` + `YASAR_USTA_STATE_DIR` env to children + `systemprofile` boot assertion (§10b).
- **M6 — `_kill_stale_orchestrators` wmic-substring → owned-PID (full cmd+cwd)** (`projects/kutai/hooks.py`, live foot-gun §4.3).

**Next gate:** user `/restart` to live-verify M1+M2 (healthy hub still boots + holds the mutex; a second manual launch exits immediately) before proceeding to M3 (which needs host config).
