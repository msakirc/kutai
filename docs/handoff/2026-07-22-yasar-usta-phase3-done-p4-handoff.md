# Handoff — Yaşar Usta: Phase 3 DONE + REVIEWED + PUBLISHED, Phase 4 next

**Date:** 2026-07-22
**Spec:** `docs/superpowers/specs/2026-07-21-yasar-usta-shared-hub-and-always-live-design.md`
**Plan:** `docs/superpowers/plans/2026-07-21-yasar-usta-shared-hub-and-always-live.md` (task-by-task, real code)
**Prior handoff:** `docs/handoff/2026-07-22-yasar-usta-phase2-reviewed-p3-handoff.md` (Phase 2)

---

## 0. Status

| Phase | State |
|---|---|
| **Phase 1** — relocate + decouple + entry + secrets | ✅ DONE + LIVE |
| **Phase 2** — state-dir (`%LOCALAPPDATA%`) + split-brain heartbeat fix | ✅ DONE + LIVE-VERIFIED + REVIEWED |
| **Phase 3** — watchdog M4b must-fixes + installer rewrite | ✅ **CODE DONE + THOROUGHLY REVIEWED (opus APPROVE, 0 issues)** — restart/live-verify-gated |
| **Phase 4** — activation (Task Scheduler + auto-logon, USER-gated) | ⬜ NOT STARTED (this handoff sets it up) |

**Phase 3 is code-complete, reviewed clean, and the HUB repo is published to its private `playground` remote.** What's left for Phase 3 is one USER live-verify (§4). Phase 4 is USER-gated activation (§5). The live hub is still running the Phase-2 code — Phase 3 watchdog/installer changes do NOT affect the running hub until Phase 4 registers the scheduled tasks (and the watchdog only runs when scheduled). No `/restart` is required to "activate" Phase 3 the way earlier phases were; the new code is dormant until Phase 4 wires it into Task Scheduler.

---

## 1. What Phase 3 delivered (HUB repo `../yasar_usta`, branch `main`)

Method: subagent-driven, TDD per task (2 sonnet implementers), then a full **opus adversarial review** (APPROVE, zero Critical/Important/Minor). Suite **205 passed** (was 199; +6 tests). The watchdog got a dedicated opus review first (split-brain path invariant); the final review re-verified everything end-to-end incl. a full git-history secret scan.

**T3.1 `5f0ec87` — grace-after-kill marker.** `watchdog.run_once` gained `marker_path`/`grace` (default `3*DEFAULT_INTERVAL_SECONDS` = 540s). On a stale hub, if a prior kill's timestamp (`.watchdog_killed`) is within the grace window → skip (no crash-loop on a slow reboot); else kill and write `str(now)` to the marker. Test: `tests/test_watchdog_grace.py`.

**T3.2 `ec32125` — `hub.stopped` gate.** `run_once` gained `stopped_path`; checked **right after** the not-stale early-return (before grace): if the file exists → deliberate stop, no kill. `hub.py run()` boot-unlinks a stale `hub.stopped` (`(Path(self.cfg.log_dir)/"hub.stopped").unlink(missing_ok=True)`). Test: `tests/test_watchdog_stopped.py`. **NOTE:** the *create-on-deliberate-stop* half is deferred to a future `/shutdown-hub` command (flagged in-code at `hub.py`, not silent) — so the gate is currently **inert** (the sentinel is only ever unlinked, never written). See residual R1.

**T3.3 `e80f736` — verify-the-kill.** `run_once` gained `is_alive`/`alert`; module `is_pid_alive` (psutil). After the kill loop, re-check each killed pid; any survivor → `alert(...)` ("survived kill — possible zero-effective-hub"), default print. `main()` wired with `--marker`/`--stopped` argparse args defaulting to `--alive` siblings + `is_alive=is_pid_alive`. Test: `tests/test_watchdog_killverify.py`.

**T3.4 `6b39012` — installer rewrite.** `scripts/install_yasar_autostart.ps1` (was an untracked pre-relocation stub): `$root`=hub repo, main action `-m yasar_usta --registry "<hub>\registry.yaml"`, alive/marker/stopped under `%LOCALAPPDATA%\YasarUsta\hub\`, kept elevated at-logon trigger + 3-min watchdog repetition. Parse-checked, **NOT registered** (that's Phase 4).

**Split-brain invariant (opus-verified 3 ways):** hub WRITES `hub.alive` and CLEARS `hub.stopped` at `Path(cfg.log_dir)/…`; watchdog defaults resolve marker/stopped as `--alive` siblings; installer's `--alive` = `%LOCALAPPDATA%\YasarUsta\hub\hub.alive`. `registry.yaml` `log_dir: "${env:LOCALAPPDATA}/YasarUsta/hub"` resolves to that same dir → all four agree. `assert_hub_log_dir_absolute` forces it absolute; `hub.stopped` has exactly one touchpoint (the unlink), no divergent create site.

## 2. Repo unification + publish (this session)

- **Dirs unified:** the stale sibling `../yasar-usta` (hyphen — an old 2026-06-05 per-package extraction, single commit `d34b4c3` v0.1.0 standalone `ProcessGuard`, superseded by the hub) was **deleted**. The live hub is `../yasar_usta` (underscore) — kept, because kutay's `-e ../yasar_usta`, `start_kutai.bat`, and the registry all point there. Verified before deletion: clean tree, nothing unpushed, zero live references.
- **Dual-remote set up** on the hub (per the `[[project-per-package-repos]]` convention): `playground` = `github.com/msakirc/yasar-usta-playground` (real push), `public` = `github.com/msakirc/yasar-usta` (fetch real; **push URL bogus** `PUSH_BLOCKED__see_pre_push_hook`). Pre-push hook blocks `public` unless `YASAR_USTA_PUSH=1`. Env var standardized to `YASAR_USTA_PUSH` (the old hyphen doc used `YASAR_PUSH` — corrected).
- **Branch renamed `master`→`main`** (matches the yazbunu convention + the release procedure in `docs/git-management.md`).
- **Published:** force-pushed `main` to playground (overwrote the stale `d34b4c3`), deleted the transient remote `master`. Playground now has a single branch `main`. Hub `docs/git-management.md` + `CLAUDE.md` written (dual-remote governance, never-push-public). Pre-push hook also committed as a tracked reference `scripts/pre-push` + a fresh-clone re-arm note (git hooks don't survive clone).
- **Security:** `.env` gitignored + never tracked; full git-history secret scan clean (no Telegram tokens/keys in any commit); `registry.yaml` references env-var NAMES only.

**HUB head:** `09c6a38` (== `playground/main`, in sync). **No `public` push yet** (deferred — USER decision, §6/R6). `gh` CLI is logged out; git push works via GCM (`manager-core`) cached creds.

## 3. Commits (HUB `../yasar_usta`, branch `main`, LOCAL + playground only)
```
09c6a38 chore(git): version pre-push hook + fresh-clone setup note (durable push guard)
fd8dd7c docs: hub CLAUDE.md + dual-remote git-management (never-push-public, YASAR_USTA_PUSH)
6b39012 chore(installer): target -m yasar_usta + LOCALAPPDATA alive/marker/stopped paths   # T3.4
e80f736 fix(watchdog): verify kill-death, alert on survivor — M4b#3                        # T3.3
ec32125 fix(watchdog): hub.stopped gate (respect deliberate stop) — M4b#2                  # T3.2
5f0ec87 fix(watchdog): grace-after-kill marker (no crash-loop on slow boot) — M4b#1        # T3.1
dd30876 (Phase 2 head)
```
KUTAY (`../kutay`, branch `main`): unchanged by Phase 3 except this handoff doc.

---

## 4. PHASE 3 tail — USER live-verify (T3.5, still pending)

Cannot be done from Claude (never launch/kill the hub via Claude `!`). With the hub running:
1. One watchdog tick against the **healthy** hub — expect it kills **nothing** (fresh alive):
   ```
   ..\yasar_usta\.venv\Scripts\python.exe -m yasar_usta.watchdog --alive "%LOCALAPPDATA%\YasarUsta\hub\hub.alive"
   ```
2. Simulate a hang: hand-edit `%LOCALAPPDATA%\YasarUsta\hub\hub.alive` to an old timestamp (process still up) → the tick should **kill it + write `.watchdog_killed`**; an immediate 2nd tick must **skip (grace)**.
3. (Optional) drop a `%LOCALAPPDATA%\YasarUsta\hub\hub.stopped` file, set alive stale → the tick should **not kill** (deliberate-stop gate). Remove the file after.

Note the watchdog kills by matching `-m yasar_usta` cmdlines; running it against your live hub WILL kill it if you force a stale alive — do this only when you're ready to let Task Scheduler / the main task relaunch it (Phase 4 not yet registered, so relaunch is manual until then).

---

## 5. PHASE 4 — activation (USER-gated, not code)

From the plan §"PHASE 4":
- [ ] Run `scripts/install_yasar_autostart.ps1` **elevated** (registers `YasarUsta` at-logon main task + `YasarUstaWatchdog` 3-min watchdog task). It targets the hub venv + `-m yasar_usta --registry` + LOCALAPPDATA paths (already rewritten in T3.4).
- [ ] Configure auto-logon (`netplwiz`) for reboot-without-login (security tradeoff: stored password — your call).
- [ ] Remove any legacy `start_kutai.vbs` / startup-folder launcher that could double-launch (the singleton mutex covers a stray one, but keep it clean).
- [ ] Reboot test: PC restart → hub auto-starts, singleton holds, both tasks present (`Get-ScheduledTask YasarUsta*`).
- [ ] Optional: `os._exit(0)` → `os._exit(42)` in `_do_restart_hub` so Task Scheduler becomes the sole relauncher (the detached-Popen bridge also works; defer if unsure).
- [ ] Optional: retire legacy `guard.py` + its 2 characterization tests once nothing imports `ProcessGuard`.

---

## 6. Residuals / deferred (carried forward)

- **R1 — `hub.stopped` create-half (stopped-gate currently inert):** the watchdog respects `hub.stopped`, but nothing writes it yet. Wire the *create* on any deliberate hub-down (a future `/shutdown-hub` Telegram command) so a hand-stopped hub isn't watchdog-killed. Until then the gate is harmless but dormant. (`hub.py` unlink-on-boot is already in place.)
- **R2 — multi-target state_dir aliasing (latent):** all targets of a project share `proj.state_dir`, and `hb_paths` hardcodes `orchestrator.heartbeat` — two targets in one project would collide. Fine for single-target kutai; fix when a 2nd target is added (per-target sub-path or filename).
- **R3 — `shutdown.signal` still in Dropbox `logs/`:** hub writes `${project_root}/logs/shutdown.signal`, orchestrator reads it CWD-relative — coupled via CWD, works, but same "shared path in Dropbox" class as the pid files. Migrate with R4.
- **R4 — pid-file migration deferred (yazbunu/nerd_herd):** nerd_herd's `--pid-file` is registry-controlled (safe to move in lockstep), but **yazbunu writes its pid CWD-relative via `--log-dir ./logs`** (uncontrolled) — moving its registry `pid_file` blind would create a new split-brain. Needs a per-sidecar coupling analysis first. Low-churn, off any kill path.
- **R5 — public repo empty:** the hub has never been pushed to `public`. First public release = clean squash via `YASAR_USTA_PUSH=1 git push public main` (procedure in `docs/git-management.md`). USER decision when/whether to publish.
- **R6 — HUB remote-less locally otherwise:** only `playground`/`public` exist; there is no `origin`. Fine.
- **R7 — accepted nits (left as-is):** watchdog `_read_ts` duplicates `read_alive_ts` (matches plan verbatim); `is_pid_alive` name-shadows `lock.py`'s (different module); the installer UNDO comment only unregisters `YasarUsta`, not `YasarUstaWatchdog` (pre-existing); the `${env:}`-unset path raises `ValueError` not `SystemExit` (still fails loud).

---

## 7. Rules (resume discipline)
- **NEVER** launch the hub via Claude `!` (attaches to the transient shell → dies on close, taking the orchestrator with it). Detached only; the `🔁 Yaşar Usta` self-restart is detached by design.
- **NEVER** taskkill llama-server. Restart KutAI via Telegram `/restart` or the hub.
- **NEVER** run the full KUTAY pytest suite while the live app is up (zombie pytest holds SQLite locks → crash-loops KutAI). The HUB full suite is safe (separate venv, no live dependency).
- HUB tests: `../yasar_usta/.venv/Scripts/python.exe -m pytest <file> --timeout=60`. Git Bash resets cwd to kutay between calls — `cd` to the hub each command.
- HUB git: `git push` (bare/playground) is safe; `public` is guarded. `gh` is logged out — `gh auth login` (scope `repo`) before any repo-creation work; plain `git push` works via GCM.
- Memory file: `~/.claude/.../memory/project_yasar_relocation_alwayslive_merge_20260721.md`.
