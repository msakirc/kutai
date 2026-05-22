# Handoff — Residual reanalysis + next-session prep

**Date:** 2026-05-21
**Session theme:** re-analyze the pile of partial-progress handoffs accumulated
2026-05-16 → 2026-05-20, separate done-from-open, close the small/clear items,
and leave an accurate map for next session.

---

## §1 — What this session shipped (8 commits on `main`)

| Commit | What | Kind |
|---|---|---|
| `b88ed845` | marketing_copy writes to mission `workspace_path`, not repo root | prod fix |
| `bb71bb2c` | doc: Z9 reinforce vs Z10 calibration are deliberately separate | coherence |
| `27c11d18` | new missions start `lifecycle_state='active'` (was `'terminal'`) | **prod bug** |
| `404fc60b` | founder_actions stable ordering on same-second `created_at` ties | **prod bug** |
| `c8052a8a` | intake_todo test mocks telegram so `keyboard_sent` path runs | test |
| `9efea092` | move app-store founder-gate lock 14.8 → real upload sub-steps | design fix |
| `ae004547` | close 7 mr_roboto suite reds (safety_guard conftest + 2 more) | test/infra |
| (this) | next-session prep doc + RESOLVED banners + `data/artifacts/` gitignore | docs |

Two were **real production bugs** masked by test drift, not test-only issues:
- **Founder-action gate was dead for every post-Z0 mission.** `add_mission`
  never set `lifecycle_state`, so fresh missions inherited the Z8 column
  default `'terminal'`; the gate (`_missions_lifecycle_column` → `lifecycle_state`
  post-Z0) saw `'terminal' != 'active'` and never blocked. Also broke the
  ongoing-mission resumption query.
- **`founder_actions` "most recent first" was non-deterministic** — `ORDER BY
  created_at` only; same-second inserts tie and SQLite fell back to rowid.

---

## §2 — Prior-handoff status map (re-verified)

| Handoff | Status | Remaining |
|---|---|---|
| 2026-05-16 yalayut-demand-signals-gap | RESOLVED (05-18, `project_yalayut_demand_wiring_closed`) | — |
| 2026-05-17 z6-phase14-mobile-test-failures | RESOLVED (`7e5ecc6d` + `9efea092`) | — |
| 2026-05-17 z7-unwired-features | §1.A wired; bucket-2 mostly addressed | verify-on-mission |
| 2026-05-18 wiring-sweep z1/z6/z8/z9 | CLOSED by 05-19 | — |
| 2026-05-18 wiring-sweep z2/z3/z4/z5/z10 | CLOSED by 05-19 | — |
| 2026-05-18 z0-and-backlog | Z0 merged; Z9/Z10-loop call made (doc-split) | scoped-unbuilt (below) |
| 2026-05-19 wiring-sweep-residuals | §1+§3 done (§3 this session); §2 OPEN | 5 deferred P3s |
| 2026-05-20 mr-roboto-suite-reds | RESOLVED (`ae004547`) | — |

---

## §3 — OPEN work, carried to next session

### 3a. Deferred P3 queue (from 2026-05-19 §2) — ALL 5 RESOLVED 2026-05-22

> **STATUS 2026-05-22:** all five P3s shipped on `main`, each with host-path
> tests (red→before / green→after) and individual scope verification. Two of
> the five had **stale handoff premises** — corrected in-session (audit-call-
> sites lesson again). Combined import-check across all touched modules: clean.

1. **§2.D Z3** ✅ `545bc4c6` — `domain_layer_check` posthook kind registered
   (verb=`run_semgrep_layer_filtered`, blocker, `auto_wire_triggers=["**/domain/*.py",
   "src/domain/**/*.py"]`) + payload builder + DLQ soft-drop + full verdict
   round-trip (`_apply_domain_layer_check_verdict` reached from
   `_apply_posthook_verdict`). 21 tests.
2. **§2.E Z2** ✅ `fc4a6593` — moved `_apply_hint_from_targets` from expander
   (where the `mission_<id>` workspace never exists yet → permanent no-op) to
   `coulson.execute()` dispatch time, just before `_apply_tools_hint`. Dead
   expander call site removed. 6 tests.
3. **§2.B Z9** ✅ `e922a554` — added `model_pick_log.task_id` (idempotent ALTER
   loop) + dispatcher writes it from the `current_task_id` ContextVar (no
   plumbing change) + `_reinforce_winning_model` joins by `task_id` first
   (tier-0), title-join kept as tier-1 for legacy NULL rows, global tier-2.
   6 tests. NOTE: the **identical** fragile title-join at `db.py:~8505`
   (`_record_and_resolve_confidence`, Z10) was left untouched (scope) — now a
   trivial follow-up since the column exists.
4. **§2.A Z1** ✅ `c3a0a8f0` — **handoff premise was STALE**: the verb already
   had a production caller (`/edit_html` → upload → `handle_document` → Beckman),
   and a button can't drive it (founder must edit HTML offline first). Built the
   real missing half instead: surface the proposal into Telegram (notify_user +
   `sp_apply`/`sp_rej` short-token buttons) → accept enqueues a `coder` apply
   task (LLM judgment — no mechanical DOM-diff→spec-doc primitive exists) →
   reject discards. 8 tests.
5. **§2.C Z10** ✅ `d56cf6fd` — **NOT** the handoff's `asyncio.Event`/new-
   lifecycle design (founder steer: reuse the existing `needs_clarification`
   park state). Rewrote `_await_confirmation` to reuse the clarify park/resume
   path: first entry → `tg.request_clarification` + `update_task(waiting_human)`
   + return `Action(needs_clarification)` (orchestrator.py:316-446 already parks
   that for mechanical tasks, no orchestrator change); resume → founder reply
   sets `context.user_clarification` → approve proceeds, reject/ambiguous
   fail-closed. Fixes a real bug (founder slower than 60s = action hard-failed).
   15 tests; full mr_roboto suite 749 passed.

### 3a-followups — ALL RESOLVED 2026-05-22

- **Z10 confidence-resolver join** ✅ `073378cb` — `record_confidence_claim`
  now attributes via `model_pick_log.task_id` first (tier-0), title-join kept
  as legacy fallback. Mirrors the Z9 reinforce fix.
- **5 `test_pick_log*` failures + DB-lock hazard** ✅ `1607474b` — REAL root
  cause was NOT "missing column": `write_pick_log_row` ignores its `db_path`
  arg and writes through the singleton `get_db()` (live `kutai.db`), so the
  tests asserted against an empty temp DB AND hung on the orchestrator's WAL
  lock. Rewrote `test_pick_log.py` + `test_pick_log_provider.py` to isolate
  onto a temp DB via `init_db` + singleton patch (the `test_pick_log_task_id.py`
  pattern). 14 passed, no hang, no live-DB writes.
- **`action_confirmations` is NOT orphaned** (corrected) — the 3a.5 gate stopped
  using it, but `src/infra/cost_wiring.py:129` opens rows for the cost-decision
  gate, `mr_roboto.audit_log.pending_audit_gaps` joins it for the Z7 B9 audit
  trail, and the seeded drain cron surfaces pending rows. Nothing to clean up;
  misleading "superseded/cleanup" comment fixed in `_await_confirmation`.

### 3b. Scoped-but-unbuilt (from 2026-05-18 z0-backlog §2b)

- **`gorsel_ustasi`** — image-gen provider-abstraction package. Scoped Z1→Z2,
  built by neither. Z1 emits placeholder images only. (`project_z1_strategic_locks_20260509`
  named the image-provider abstraction a strategic lock.)
- **Web preview hosting** (C10/F1) — `emit_preview_url` verb exists; the
  cloudflared / local-port / GitHub-Pages host + viewer was never built.
- **Z8 on-call cloud impls** — `restart_service` / `scale_up` etc. fail loud
  (honest stubs); need vendor cloud-API wiring when accounts exist.

### 3c. The real reliability proof — still not done

No i2p mission has ever run end-to-end. Per every wiring sweep: static grep
closed the *wiring*, but a real prototype-tier mission run is the only thing
that surfaces what grep missed. Highest-signal, heaviest. Do after 3a.

---

## §4 — Environment / hygiene notes for next session

- **`safety_guard` editable install:** the package was shipped by Z0 but never
  `pip install -e`'d into the venv (its 31 unit tests had never run). Fixed
  in-session via `pip install -e packages/safety_guard` + the conftest
  `_PACKAGE_SRCS` add (`ae004547`). A fresh clone still needs the editable
  install — worth adding to the venv setup script.
- **`data/artifacts/` + `data/mission_*/`** now gitignored — production mission
  writes were leaking into the repo (marketing_copy default path, fixed
  `b88ed845`). If you see them in `git status`, the gitignore covers them.
- **Pre-existing env skips (NOT regressions):** `fastapi` +
  `sentence_transformers` not installed → all webhook-route + embedding tests
  fail/skip everywhere. Install or accept.
- **conftest collision:** never mix `tests/` and `packages/*/tests/` in one
  pytest invocation — the dual `conftest.py` triggers a pluggy
  "Plugin already registered" error. Run dirs separately.
- **Worktree fleet:** the ~45 stale `worktree-agent-*` worktrees were pruned
  this session (commit before `b88ed845`). Branch list is clean (`main` only).

---

## §5 — Verification gate (run before declaring next-session work done)

```
# originally-failing tests this session closed (expect all green)
.venv/Scripts/python -m pytest \
  tests/founder_actions/test_lifecycle.py \
  tests/founder_actions/test_repo.py \
  tests/i2p/test_intake_todo.py \
  tests/workflows/test_z5_t5_distribution.py \
  tests/workflows/test_z6_t6a_reversibility.py \
  tests/workflows/test_z6_polish_phase14_mobile.py \
  tests/workflows/test_i2p_v3_reversibility_tags.py -q
# -> 45 passed

# full mr_roboto suite
.venv/Scripts/python -m pytest packages/mr_roboto/tests/ -q
# -> 732 passed, 2 skipped, 0 failed

# wiring-sweep smoke gate (regression guard for the 05-18/05-19 closures)
.venv/Scripts/python -m pytest \
  tests/test_wiring_sweep_20260518.py tests/test_wiring_sweep_p2_20260518.py \
  tests/test_z3_p2_cascade_20260518.py tests/test_z8_sweep_20260518.py \
  tests/test_z6_sweep_20260518.py tests/test_z1_sweep_20260518.py -q
# -> 45 passed + 25 subtests
```
