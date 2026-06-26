# Handoff: DB write-consolidation follow-ups (2026-06-12)

**Status (2026-06-12 second pass):** EXECUTED — A1–A3, B1, B2, B3 implemented on branch `db-followups`, merged to main (restart-gated, NOT pushed). A4 still sqlite-gated (3.37.2). B4 deferred unchanged. B5 watch unchanged. See "Execution notes + new follow-ups" at bottom.

Original status: backlog only, nothing here implemented. Parent work is DONE: write-ownership consolidation merged `83ba7d8e`, post-review fixes merged `1467d3ea` (both on local main, NOT pushed, restart-gated). High-effort code review (7 finder angles, 16 candidates, all verified) found **zero introduced correctness bugs** — this handoff is the residue: test hygiene + pre-existing debt the review surfaced and canonicalized into db.py.

Context docs: `docs/handoff/2026-06-11-step1-write-consolidation-worklist.md` (executed worklist), `docs/handoff/2026-06-11-db-ownership-split-eval-handoff.md` (Step 2 file-split evaluation, still deferred).

---

## Tier A — test hygiene (mechanical, ~1 session, no prod behavior change)

### A1. Deduplicate the AST import-guard helper
`_ast_task_write_imports` (`packages/general_beckman/tests/test_task_write_api.py:~806`) and `_ast_mission_write_imports` (`packages/general_beckman/tests/test_mission_write_api.py:~601`) are byte-identical bodies (only name/docstring differ). Extract ONE shared helper, e.g. `_ast_db_write_imports(filepath, text, guarded_names)` in `packages/general_beckman/tests/conftest.py` (exists; currently only in-flight registry cleanup). Risk being fixed: an import-detection improvement landed in one guard but not the other leaves one table's guard silently weaker.

### A2. Shared DB fixture for the 3 write-API test files
`_reset_db`/`_close_db` are copy-pasted into `test_growth_event_api.py:~22`, `test_mission_write_api.py:~21`, `test_task_write_api.py:~22`. Additionally ~100 inline `db_module._db_connection = None` cache-busts are scattered through test bodies (61 in task file, 39 in mission file) because `get_db()` caches the connection globally. Fix: conftest fixture (e.g. autouse `fresh_db(tmp_path, monkeypatch)`) that sets DB_PATH, resets `_db_connection`/`_db_connection_path`, yields, closes in teardown. Kills all three copies + the inline resets.
**Gotcha:** combining `packages/general_beckman/tests/` and `tests/founder_actions/` in ONE pytest invocation raises pluggy "Plugin already registered under a different name" (conftest module-name collision — root `conftest.py` docstring documents this). Run suites separately when verifying.

### A3. Guard tests: 5 full-repo walks → 1 shared scan
Five guard tests (`test_growth_event_api.py:~242`, `test_mission_write_api.py:~538,~633`, `test_task_write_api.py:~735,~838`) each independently `os.walk` the repo root AND `read_text` every non-test `.py` file (~2.5k files). Five full scans per beckman test session ≈ 1–2.5s wasted I/O on this Dropbox-synced tree. Fix: session-scoped fixture in the same conftest returning `dict[Path, str]` of source texts; all five guards consume it. Combine naturally with A1/A2 in one commit.

### A4 (optional). Single-UPDATE supersede when sqlite ≥3.38
`db.supersede_growth_events()` (added 1467d3ea) kept the fetch+loop with one commit because the machine's sqlite is **3.37.2** (`json_valid`/reliable JSON1 path wants 3.38+). When Python/sqlite upgrades, replace with one `UPDATE ... json_set(...) WHERE COALESCE(json_extract(properties_json,'$.consumed'),0)=0 AND COALESCE(json_extract(properties_json,'$.superseded'),0)=0`, treating NULL properties_json as open. Closes the fetch-then-write TOCTOU window too. Tests already cover semantics incl. NULL-properties row (`test_supersede_handles_null_properties_json`).

---

## Tier B — pre-existing debt, now canonical in db.py (each needs a design decision, NOT mechanical)

These were NOT introduced by the consolidation — verified byte-identical ports — but the review flagged them and the single-owner helpers are now the right (single) place to fix each.

### B1. Dependency cascade reset uses substring LIKE
`src/infra/db.py::reset_cascade_failed_dependents` (~line 6057): `depends_on LIKE '%{task_id}%'` — task #5 matches `depends_on='[15]'` or `'[52]'`. Wrongly resets tasks that never depended on the retried task → they re-run while their real blocker is still failed. Ported verbatim from old `dead_letter.py:352`. Fix options: JSON-aware match (`'%"'||?||'"%'` if ids stored as JSON strings — CHECK actual depends_on format first: likely `[5,7]` ints, so match `'[5]'`/`'[5,'`/`',5,'`/`',5]'` four-pattern OR fetch+json.loads filter in Python). Add a red test with ids 5 vs 15 first.

### B2. Uncapped infra_resets → startup crash-loop
`src/infra/db.py::recover_startup_tasks` (~line 6000): every boot resets ALL `processing` tasks → `pending` and bumps `infra_resets`, no ceiling. A task that deterministically crashes the orchestrator mid-execution gets re-dispatched every boot forever (poison-task crash loop). Fix: cap (e.g. `infra_resets >= 5` → route to DLQ instead of pending; reuse dead_letter machinery). Decide threshold + DLQ reason string with founder; wire a test simulating the loop.

### B3. block/unblock lifecycle-column NULL fallback can split state
`packages/general_beckman/src/general_beckman/__init__.py::block_mission/unblock_mission`: guard reads `mission.get(col) or mission.get("status")` (falls back to `status` when `lifecycle_state` is NULL) but writes to `col` (= `lifecycle_state`). Legacy row with NULL lifecycle_state: block writes lifecycle_state while status stays `active` → consumers reading `status` see active, mission half-blocked; repeated unblock can no-op. Ported verbatim from old founder_actions. Real fix: one-time backfill migration (`UPDATE missions SET lifecycle_state = status WHERE lifecycle_state IS NULL`) in `init_db`, then drop the `or mission.get("status")` fallback + the PRAGMA probe entirely (`_missions_lifecycle_col` cache added 1467d3ea becomes deletable — lifecycle_state guaranteed since Z8 T1A migration, db.py ~541).

### B4. TOCTOU between founder-action count check and unblock flip
`src/founder_actions/__init__.py::unblock_mission_if_clear` checks pending-count==0 then awaits `beckman.unblock_mission` (re-fetch + flip). Concurrent founder_action insert between check and flip (telegram handler + pump share one event loop, every await yields) → mission unblocks despite pending action. Pre-existing race, marginally widened by the extra hop. Fix option: move the count check INSIDE beckman.unblock_mission (single helper re-checks immediately before flip), or a conditional UPDATE (`... WHERE NOT EXISTS (SELECT 1 FROM founder_actions WHERE mission_id=? AND status='pending')`). Low urgency — single-pump makes the window tiny; harm = blocked mission resumes one founder-action early.

### B5 (watch, no action yet). beckman `__init__.py` ≈ 1.9k LOC
26+ write delegates accreted into the package facade. Next consolidation pass (comms, when validated) should extract `general_beckman/write_api.py` re-exported from `__init__` — do it THEN, not now (avoid churning imports twice). Guard tests don't care (they police by directory).

---

## Verification commands
- `timeout 240 python -m pytest packages/general_beckman/tests/ -q` — expect 330 pass **on main checkout**; in a git worktree the 2 `test_admission_cache` tests fail (known PYTHONPATH/worktree artifact, proven not-regression — do NOT chase).
- `timeout 120 python -m pytest tests/founder_actions/ -q` — 45 pass. Run separately from beckman suite (pluggy collision, see A2 gotcha).
- Guards live in `test_task_write_api.py` + `test_mission_write_api.py` (4 tests: raw-SQL scan + AST import scan per table family).

## Execution notes + new follow-ups (2026-06-12 second pass)

Commits (worktree branch `db-followups`, merged to main):
- `0a6338cd`+`6f701b58` — A1+A2+A3: shared `_ast_db_write_imports` + `fresh_db` fixture (killed 3 fixture copies + ~115 inline `_db_connection=None` busts, which were redundant — `get_db()` re-opens on DB_PATH change) + session-scoped `repo_source_texts` scan for all 5 guards. Review fix: post-`init_db` connection no longer leaked (aiosqlite DeprecationWarnings eliminated, verified `-W error`).
- `0118560f`+`259a9cde` — B1: LIKE kept as prefilter, exact membership via `json.loads` + `str(d)==str(task_id)` (covers legacy string ids); UPDATE re-asserts failed-state guard (TOCTOU). Red-verified vs old code (`assert 4 == 2`). Warn-and-skip on unparseable depends_on.
- `bcb2ca95`+`0bce6429`+`d0be268b` — B2: `INFRA_RESET_CAP=5`; at-cap processing tasks → `status='failed'` + `quarantine_task` (canonical DLQ, same row shape as `apply._dlq_write`); `_plain_retry` grants fresh budget (`infra_resets=0`). **Review caught false-positive blocker**: sweep.py stores availability-ladder SECONDS (60..7200, `retry_reason='availability'`) in `infra_resets` — those rows are cap-exempt (marker OR value ≥ 60), re-pended untouched.
- `9490c729`+`56352211` — B3: **handoff premise was stale** — `lifecycle_state` is `NOT NULL DEFAULT 'terminal'` since Z8 T1A ALTER (retried every boot), NULL rows impossible, backfill would be a no-op. Real residue deleted instead: both PRAGMA probes (`_missions_lifecycle_col`, `_missions_lifecycle_column` + `_reset_lifecycle_cache` hook), the dead `or mission.get("status")` fallbacks; gates hard-code `lifecycle_state`. 16 test files dropped the hook call.

Tests post-merge: beckman 335 (was 330 +5 new), founder_actions 45, dead_letter 17, swept z5/z6 files 64.

New follow-ups (none blocking):
1. **infra_resets tri-semantics debt**: column carries (a) startup-reset count (recover), (b) availability-ladder seconds (sweep), (c) in-flight infra-failure count (RetryContext, terminal ≥ 3 — `to_db_fields` currently prod-dead). Real fix: move sweep ladder state to `ctx['last_avail_delay']` — would also make `accelerate_retries` effective (it already resets that ctx key, which sweep ignores; pre-existing disconnect).
2. **Residual poison gap (LOW-MOD)**: once-laddered task (`infra_resets ≥ 60` forever, never reset on completion) that later becomes a fast deterministic crasher is permanently cap-exempt → crash-loops bounded only by Yaşar Usta backoff. Fix idea: separate startup-reset counter in `ctx['_startup_resets']` for exempt rows.
3. **Quarantine partial failure**: if `quarantine_task` raises during recover, task is `failed` + cap-error but no DLQ row → invisible to `/dlq` (same failure mode as canonical `_dlq_write`; warn-logged).
4. B4 TOCTOU unblock race — still open, low urgency. A4 single-UPDATE supersede — still gated on sqlite ≥ 3.38.

## Second follow-up sweep (2026-06-13) — items #1, #2, #3, B4 EXECUTED

Branch `db-followups-2`. **MERGED to main `a3d862a0`** (no-ff, conflict-free) via a temp worktree — main was checked out nowhere (live dir is on `sp6-critic-gate`). NOT pushed — restart-gated. Adversarial review agent found NO correctness bugs; post-merge 47 beckman + 9 founder_actions lifecycle green. **LIVE-GATE CAVEAT:** the live checkout dir `C:\...\kutay` is on `sp6-critic-gate`, so a `/restart` runs sp6 code, NOT this merge — USER must `git checkout main` there (after handling the parallel sp6 session) for it to go live. Worktree note: editable packages pin to main's path; run tests with `PYTHONPATH="<worktree>;<worktree>/packages/*/src"` + main's `.venv` python so worktree code shadows the installs.

- `63dd94bf` — **#1 tri-semantics + #2 residual poison gap (both closed by one change)**. The availability-ladder seconds moved out of `infra_resets` into `context['last_avail_delay']`: sweep.py §1 reads/writes the ctx key (so `accelerate_retries`, which zeroes it, is now effective for stuck-processing rows — previously inert because sweep read the column); `recover_startup_tasks` drops the availability cap-exemption (`infra_resets` is now an unambiguous reset COUNT, so `INFRA_RESET_CAP=5` applies cleanly — this is what kills #2's permanent cap-exemption); `_LADDER_FLOOR_SECONDS` removed; one-time idempotent `init_db` migration moves legacy seconds-in-column → ctx and zeros the column. #2 needs NO separate `ctx['_startup_resets']` counter — the de-overload subsumes it.
- `4c4f6564` — **#3 quarantine partial failure**. `recover_startup_tasks` now quarantines BEFORE flipping the poison task to `failed`; on quarantine failure the row stays `processing` (re-attempted next boot, not re-dispatched since the dispatcher only picks `pending`) instead of failed-but-invisible to `/dlq`. `dead_lettered` counts only rows that reached the DLQ.
- `6eaee5ed` — **B4 unblock TOCTOU**. New `db.conditional_unblock_mission` = single atomic UPDATE (`... WHERE lifecycle_state='blocked_on_founder_action' AND NOT EXISTS(pending/in_progress founder_action)`). `beckman.unblock_mission` gains `require_actions_clear=True` (default path unchanged for other callers); `founder_actions.unblock_mission_if_clear` drops its own count check and delegates atomically.

Tests added/changed (all green via worktree PYTHONPATH): rewrote the recover availability-exemption test for count-semantics; added migration test (legacy seconds → ctx, idempotent); real-DB sweep §1 test (ladder → ctx, not column); quarantine-failure test (stays processing, no DLQ row); B4 atomic-guard test (refuses while pending, flips when clear). beckman `test_task_write_api` + `test_sweep_overcap` + `test_mission_write_api` and root `test_sweep_guards`/`test_db_migration`/`test_dead_letter`/`test_infra_vs_quality` + `tests/founder_actions/test_lifecycle` all pass.

**Pre-existing unrelated reds (NOT touched, NOT in the verification set):** `tests/test_retry_context.py` has 5 failures asserting the superseded 2026-04-07 design (`max_worker_attempts` default 6 vs current 15 from_task fallback; quality→`delayed` vs current `immediate` by deliberate `compute_retry_timing` comment). Stale tests, not a regression — `retry.py` was not edited this session. Reconcile or delete separately.

**Remaining:** A4 (sqlite ≥3.38, machine still 3.37.2 — verified). B5 (`write_api.py` extract) intentionally DEFERRED to the comms consolidation pass per the original B5 note — extract once, after comms adds its delegates.

## Standing constraints
- `get_db()` is `isolation_level=None` (autocommit) — every statement commits itself; "wrap in one commit" reasoning is invalid unless an explicit `BEGIN`/`COMMIT` region (see db.py:30 lock notes) is used.
- All mission/task/growth_events writes MUST go through general_beckman API — guards enforce; never add raw SQL outside `src/infra/db.py` (+ beckman internals).
- Datetime strings: `%Y-%m-%d %H:%M:%S` (`src/infra/times.db_now()`), never isoformat. Exception: `founder_actions.updated_at` is historically ISO via `_utc_now()` — harmless (no comparing consumer), don't "fix" piecemeal.
- Comms/CRM/email tables stay UNCONSOLIDATED by decision (fresh-built, zero missions through them) — revisit after first live comms mission.
- Use a git worktree for implementation; parallel agent sessions cross work on main.
- Everything restart-gated: main not pushed; live verify needs USER `/restart` via Telegram.
