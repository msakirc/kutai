# Handoff — m90 drive-to-completion (the long tail) — 2026-07-30

## TL;DR / DO FIRST
- **Task from USER:** "make m90 finish successfully, find any bug/residue, tackle it, don't sit idle." Drive mission 90 to completion; each phase surfaces a gate/reviewer/schema bug — root-cause + fix each. **Root is almost NEVER "weak model"** — it's a valid on-disk artifact false-rejected by a mis-plumbed gate, a confabulating reviewer, or an instruction↔schema mismatch.
- **m90 state (2026-07-30):** `completed=1164, pending=104, failed=2, waiting_human=1, skipped=19, cancelled=1` (~244 workflow steps + spawned children). Advanced from 1049 at session start.
- **git: origin/main = `8ee04d3e`** (all this session's fixes pushed). Orchestrator restarted 2026-07-29 03:05 on this code (i2p hot-reloads; mr_roboto arm changes need the restart that already happened).
- **DB (read-only for inspection!):** `sqlite3.connect('file:C:/Users/sakir/ai/kutai/kutai.db?mode=ro', uri=True)`. Workspace artifacts: `workspace/mission_90/`.
- **Process mgmt (restart/hub) is the USER's** — never restart yourself; ask. i2p config/instruction edits hot-reload (no restart); `.py` (mr_roboto arm / verifier) edits need a USER `/restart`.

## Open blockers RIGHT NOW (open DLQ: 892400/1/2 = declined stripe/tax crons; **567465, 567467, 567468** = m90)
1. **567465 [6.4 sprint_planning]** DLQ — `schema gate: sprint_0_plan.tasks: missing required field`. Model omits `tasks` in `sprint_0_plan`. Likely instruction↔schema gap (check 6.4 instruction states the required fields / the model emits a variant). Same class as the 6.6 status-enum fix (`8ee04d3e`).
2. **567467 [6.5z premortem] + 567468 [6.6 project_plan_review]** — the premortem chain, cycling. 6.6 now emits valid `status:"fail"` (my enum fix worked) flagging **"high-plausibility (5) 'Founder Failure' scenario unmapped to a monitoring rule"**. This is **OVER-STRICT**: 6.6's instruction demands a monitoring rule for EVERY `plausibility>=4` premortem scenario, but non-technical scenarios (founder/market) can't be monitored by a rule — same over-broadening class as the 4.16 real-time false-positive. **Recommended fix: scope the premortem-coverage check (6.6 instruction, hot-reload) to only require monitoring for scenarios that CAN be monitored (technical/operational), OR treat non-monitorable high-plausibility scenarios as needs_clarification-not-blocker.** Mirror the 4.16 CRITICAL-SCOPE prepend (`b7cc3b42`). 6.5z/6.6 block ~most of the 104 pending (6.6 dependents chain).
3. **567446 [4.16.git_commit_green]** — BENIGN, leave it. critic_gate false-positive vetoes the git_commit because the 4.16 review legitimately flipped fail→pass (my reviewer fix). It's a LEAF (nothing depends on it; the final phase-13 commit supersedes). DLQ annotated resolved earlier but may reappear as `failed`; ignore.
4. **1 waiting_human** — was the calendar/4.16 chain; verify which task. The **calendar founder-decision is DEFERRED** (USER hasn't decided; I applied charter-wins surgically for the 4.8 instance — see below).

## The playbook that works (through-line)
For each DLQ/stall: **read the real on-disk artifact vs what the gate received.** The bug is one of:
- **Payload starvation / path not resolved** — check payload wires no path, or wires a workspace-relative path that the verifier `open()`s against CWD (repo root, not `workspace/`). Fix = wire `mission_{mission_id}/<file>` in the i2p check payload (hot-reload) + resolve it in the mr_roboto arm via `_resolve_path_list` (RESTART-gated). Precedent: `verify_adr_register` (da3ce63a) and `verify_premortem_shape` (3d54d32b).
- **Reviewer confabulation / over-strictness** — the reviewer reads SUMMARIES (its `input_artifacts` are `*_summary`) that omit detail, then flags "missing" structure that a mechanical gate already validated; OR over-broadens a non-goal/rule. Fix = prepend a CRITICAL-SCOPE override to the reviewer instruction (hot-reload): presence=mechanical (defer to verify_*_shape), match constraints to EXACT stated scope. Precedent: 4.16 (`b7cc3b42`).
- **Instruction↔schema mismatch** — schema enum/required-field differs from what the instruction implies; model invents a value. Fix = state the exact enum/required fields in the instruction (hot-reload). Precedent: 6.6 status enum (`8ee04d3e`).
- **Requirement contradicts a non-goal** (the "required-but-impossible" class) — see `feb34ee9`; producers now get `non_goals`, 3.9b cross-checks. For an EXISTING m90 instance, reconcile charter-wins (drop the forbidden thing).

## Fixes SHIPPED this session (all pushed, all live-validated where noted)
- `feb34ee9` fix(i2p): **requirement-vs-non-goal prevention** — feed `non_goals` to 3.5 + 4.6/4.8/4.9/4.10 producers (hard-constraint instr on 3.5/4.8) + 3.9b cross-checks reqs-vs-non-goals. Live-proven: re-pend 4.8 dropped Google Calendar.
- `b7cc3b42` fix(i2p): **4.16 reviewer scope override** — presence=mechanical + exact non-goal scope. Live: 4.16 PASSED on merit.
- `3d54d32b` fix(i2p+mr_roboto): **verify_premortem_shape path wiring + arm resolution**. Live: 6.5z shape gate passes post-restart.
- `8ee04d3e` fix(i2p): **6.6 status enum pinned** (pass|approved|fail; kill invented 'needs_minor_fixes'). Live: 6.6 now emits valid status.
- (earlier this session, also pushed) `da3ce63a` ADR-register domain-coverage gate; `4506e091`/`9c6ffb95` multi-artifact-ADR + chroma-boot; surgical calendar drop on `workspace/mission_90/.adr/third_party_selections_decision.json` (backup in `workspace/mission_90/.bak_calendar_20260728/`).

## Restart-gating (CRITICAL)
- **Hot-reload (no restart):** i2p_v3.json step `description`/`instruction`/`checks`/`payload`/`input_artifacts` — coulson `_refresh_workflow_step_config` refreshes these at re-dispatch (`_CTX_FIELDS`). So instruction/payload/gate edits reach in-flight m90 tasks on re-pend.
- **RESTART-gated:** any `.py` change (mr_roboto arm, verifiers, engine). Commit it, then the USER must `/restart`; then re-pend the affected tasks. Boot warmup ~285s (sentence-transformer) before the pump starts — re-pended tasks sit `pending`, not stuck.

## Key commands
- **Re-pend a task** (clears degenerate ledger): `UPDATE tasks SET status='pending', worker_attempts=0, grade_attempts=0, result=NULL, error=NULL, task_state=NULL, sleep_state=NULL, context=json_remove(context,'$._rejection_ledger','$._schema_error','$._schema_error_for_attempt','$._prev_output') WHERE id=?`
- **Force-complete a valid-but-false-rejected artifact:** set `status='completed', error=NULL` + `UPDATE dead_letter_tasks SET resolved_at=datetime('now'), resolution='...' WHERE task_id=? AND resolved_at IS NULL`. (Store is PATH-based — editing the on-disk artifact is enough; verify shape offline first.)
- **Validate a verifier offline** (orchestrator is up → use `?mode=ro` for reads; `mr_roboto.run`/verifiers do file I/O, fine, but don't open the DB read-write). Example: `sys.path.insert(0,'packages/mr_roboto/src'); from mr_roboto.verify_X import verify_X; verify_X(path='workspace/mission_90/...')`.
- **Find failed/blocked:** query `tasks WHERE mission_id=90 AND status IN ('failed','ungraded','waiting_human')` + read `context._rejection_ledger`. Step id = `context.workflow_step_id`.
- **Monitor pattern:** a bounded/persistent Bash `Monitor` polling the tally, emitting on new-failure / completion / single-stall (poll 90-100s; make STALL emit once then `st=-1000` to avoid spam — the earlier version spammed).

## Gotchas banked
- **TZ:** log `ts` field is UTC; machine-local = UTC+3 (Turkey). A "01:10" ts = 04:10 local.
- **Current logs** are under `C:\Users\sakir\AppData\Local\YasarUsta\kutai\logs\guard.jsonl` (NOT repo `logs/` — those are stale). Heartbeat: `%LOCALAPPDATA%\YasarUsta\kutai\heartbeat`.
- **faulthandler noise (RESOLVED):** a leftover diagnostic (`logs/faulthandler.on` trigger, run.py:381-386, `dump_traceback_later(15, repeat=True)` — NO `exit`) dumped a thread-traceback every 15s to the log. It does NOT kill the process (I mis-read it as a crash-loop first; the loop was just idle in `select`). I `rm`'d `logs/faulthandler.on`. If "Timeout (0:00:15)!" dumps reappear, delete that file again.
- **Pump "stall" = tasks blocked, not dead:** orchestrator alive (heartbeat fresh) but `processing=0` means all pending are `blocked` behind failed/incomplete deps (log: "N pending, 0 ready, N blocked"). Unblock the root task → dependents become ready. NOT a crash.
- **mr_roboto.run touches the live DB** — offline repros can hang on lock contention while the orchestrator holds it; keep DB opens `?mode=ro`.
- **Zombie pytest** holds SQLite locks — if you run tests, kill stuck `pytest`+`test_X` procs via PowerShell `Stop-Process` (NEVER touch `run.py`/llama-server/nerd_herd).
- **Run tests/i2p and packages/mr_roboto/tests SEPARATELY** (conftest name clash); `git stash` to prove a red is pre-existing (known pre-existing: ~2 visual reds in mr_roboto; 4 stale reds in test_workflow_loader — version/step-count assertions).

## Open founder-DECISIONS (not code — surface, don't guess)
- **Calendar** (from phase 4): charter non_goal forbids third-party calendar integration; the app wanted Google Calendar sync. USER "has not decided calendar" but chose charter-wins policy → I dropped it surgically for m90. If the USER wants calendar, revert.
- **Premortem coverage** (567468/6.6): 'Founder Failure' (plausibility 5) has no monitoring rule. Likely over-strict (non-monitorable scenario) → fix the check scope (recommended) rather than force a founder mitigation.

## Key files
- `src/workflows/i2p/i2p_v3.json` — step defs (instructions/checks/payloads/inputs). 244 steps. Edit via targeted Edit (huge, pretty-printed, some non-ASCII → read with utf-8).
- `packages/mr_roboto/src/mr_roboto/__init__.py` — verifier dispatch arms (`_resolve_path_list` at ~299 is the workspace-root resolver; add resolution here for path-based verifiers). `verify_*.py` = the verifiers.
- `packages/coulson/src/coulson/__init__.py:459` `_refresh_workflow_step_config` + `_CTX_FIELDS` (~531) — what hot-reloads.
- `packages/general_beckman/src/general_beckman/apply.py` — grade/posthook/reviewer-verdict handling; `verify_review_verdict` grounding (Rules A/B/C in `packages/mr_roboto/src/mr_roboto/verify_review_verdict.py`).
- `src/core/threaded_heartbeat.py` — the 480s loop-wedge heartbeat (NOT the crash cause; the 15s faulthandler was the diagnostic noise).
- Memory: `project_m90_nongoal_class_and_reviewer_confab_20260728.md`, `project_m90_adr_register_gate_doa_20260727.md`, `project_reviewer_falsification_presence_mechanical_20260722.md`.

## Next-session first moves
1. Fresh snapshot (tally + failed + waiting_human + open DLQ).
2. Fix **567465 [6.4 sprint_planning]** (schema `sprint_0_plan.tasks` — read 6.4 instruction vs schema, likely instruction fix, hot-reload, re-pend).
3. Fix **567468 [6.6]** premortem-coverage over-strictness (scope the check like 4.16), re-pend 6.5z→6.6.
4. Keep grinding: re-pend, watch to terminal, root-cause the next DLQ. Commit+push each fix (i2p hot-reloads; `.py` fixes → ask USER to /restart, batch them).
5. Surface genuine founder-gates; don't guess product scope.
