# Handoff — Wiring-sweep residuals after 2026-05-19 closeout

> **STATUS 2026-05-21:** §1 (13 P0/P1/P2 + 4 P3s) landed pre-this-session.
> §3 pre-existing main-branch failures — **all CLOSED this session** (commits
> `27c11d18` lifecycle, `404fc60b` repo ordering, `c8052a8a` intake_todo,
> `9efea092` reversibility; two were real prod bugs, not test drift).
> §2 deferred P3s (5 items, A–E) — **still OPEN**, carried into the
> 2026-05-21 next-session doc. See `docs/handoff/2026-05-21-residual-reanalysis.md`.

**Date:** 2026-05-19
**Closes:** the open items from the two 2026-05-18 wiring-sweep handoffs
(`2026-05-18-wiring-sweep-z2-z3-z4-z5-z10.md` and
`2026-05-18-wiring-sweep-z1-z6-z8-z9.md`).
**Scope of this doc:** what landed today (P0 / P1 / P2 + 4 P3s), what
remains deferred (5 P3s that need new surface area), and pre-existing
main-branch failures flagged for other owners.

---

## §1 — What landed (20 commits on `main`, range `e082ca92..d6bb7a87`)

All 13 P0/P1/P2 items + 4 small P3s.

| Commit | Item | Sub-handoff |
|---|---|---|
| `a8a218ba` | Z10 P0 `/mission_cost` dup def dropped | z2-z3-z4-z5-z10 |
| `63c4e087` | Z4 P1 visual_review reads `a.raw` | z2-z3-z4-z5-z10 |
| `514cb7f8` | Z2 P1 `inject_lessons` registered in `POST_HOOK_REGISTRY` | z2-z3-z4-z5-z10 |
| `670fc329` | Z5 P1 `platforms_include` reads `mobile_support` + fallback | z2-z3-z4-z5-z10 |
| `1be1b7a6` | Z10 P1 `set_task_confidence` persists at react.py finalize | z2-z3-z4-z5-z10 |
| `4b476a55` `96a109cb` | Z3 P2 multifile expander cascade (4 sub-fixes) | z2-z3-z4-z5-z10 |
| `461a90d0` | Z5 P2 mobile_smoke flows + workspace auto-discovery | z2-z3-z4-z5-z10 |
| `2c29c414` | Z10 P2 `require_confirmation` auto-arm via `confirm_policy` | z2-z3-z4-z5-z10 |
| `6300270a` | Z8 P0 12 ops recipes restructured to `<name>/v1/` layout | z1-z6-z8-z9 |
| `86918ab9` | Z8 P0 `alert_triage` enqueues `oncall_agent` | z1-z6-z8-z9 |
| `f02e4fd8` | Z8 P1 `/ask` enqueues `support_tier1` | z1-z6-z8-z9 |
| `28433d5d` | Z8 P1 `synthetic_check` dispatch + 5 ops cron seeds | z1-z6-z8-z9 |
| `50762d45` | Z6 P2 `audit_completeness_check` posthook → handler module | z1-z6-z8-z9 |
| `8d480aeb` | Z1 P2 `tag_signature` + `kill_preview_url` i2p steps | z1-z6-z8-z9 |
| `6da8c834` | Z5 P3 drop `test_run` from `mobile_release_rejection` | z2-z3-z4-z5-z10 |
| `2647888f` | Z9 P3 `score_backlog` enqueues unconditionally | z1-z6-z8-z9 |
| `76b19ae2` | Z3 P3 layer-aware reflection (thread `inspect_layer`) | z2-z3-z4-z5-z10 |
| `5235dc86` | Z6 P3 `expected_output_schema` validation on `/action_done` | z1-z6-z8-z9 |

Plus 7 merge commits.

Z6 P2 `vendor_call` visibility verified existing via `TOOL_REGISTRY`
splat at `src/tools/__init__.py:1339` — handoff worry speculative, no
fix needed (test pins the merge).

Z9 P1 `metric_emit` was already closed pre-session in commit
`f7d8bd82` (analytics_digest writes `metric_emit` + `review_density_metric`).

---

## §2 — Deferred P3s (5 items)

Each one needs new surface area, not just plumbing. Sized for separate
sessions.

### §2.A — Z1 P3 — `propose_spec_patch_from_html_diff` Telegram inline button

**Where it dies:** `packages/mr_roboto/src/mr_roboto/__init__.py:1357`
(verb registered + dispatch branch present, no production caller).

**Why P3:** the C17/A20 two-way HTML-edit-reflection verb. The
`propagate:` Telegram button reaches `propagate_asset_change`, not this
verb. No spec-patch loop reaches the founder today.

**Work needed:**
1. New callback prefix (e.g. `spec_patch:<task_id>:<artifact_slug>`) handled
   in `src/app/telegram_bot.py::handle_callback` siblings to the
   `propagate:` handler.
2. New inline-button row on the result message emitted by
   `annotate_html_oids` (5.30b) and `regen_artifact` posthook
   notifications. Likely add `[Propose spec patch]` next to the existing
   `[Propagate asset change]` button when the asset is HTML.
3. Callback handler unwraps callback_data, calls
   `propose_spec_patch_from_html_diff` via `general_beckman.enqueue`
   (mechanical), surfaces the proposed patch to the founder for review
   (text message + accept/reject inline buttons).
4. Accept path enqueues `propagate_asset_change` (or a sibling) that
   actually applies the patch to spec; reject discards.

**Est. effort:** 2-4 hours including a host-path test asserting the
callback resolves and enqueues. Test pattern available in existing
`handle_callback` tests.

---

### §2.B — Z9 P3 — reinforce model-resolver join is fragile

**Where it dies:** `packages/mr_roboto/src/mr_roboto/executors/record_verdict.py:194`
joins `tasks.title = model_pick_log.task_name` (free-form strings). On
mismatch it falls back to "most recent model overall" — can reinforce
the wrong model.

**Why P3:** title-based join is the only Z9 string-key match left after
the Z9 hardening sweep. Risk is low (model_pick_log rotates fast) but
the wrong-model failure mode is silent.

**Work needed:**
1. Schema migration: `ALTER TABLE model_pick_log ADD COLUMN task_id INTEGER`
   in `src/infra/db.py`. Backfill nullable; new rows populate from the
   dispatcher.
2. Update the dispatcher write site (search for the `INSERT INTO model_pick_log`
   in `packages/fatih_hoca/`) to populate `task_id`.
3. Update `record_verdict.py:194` to join by `task_id` first, fall back to
   the title-based path for older rows.
4. Drop the title-based fallback after a release where every row has
   `task_id`.

Already-listed as an existing follow-up in MEMORY.md
(`feedback_verify_verdict_roundtrip.md`) territory.

**Est. effort:** 2-3 hours including a backfill migration test.

---

### §2.C — Z10 P3 — confirmation gate is a busy-poll skeleton

**Where it dies:** `packages/mr_roboto/src/mr_roboto/__init__.py::_await_confirmation`
busy-polls 0.5s × 120 (60s) and holds the task slot. A slow founder
makes the task `fail`. Now that Z10 P2 auto-arms the gate via
`confirm_policy`, the 60s ceiling is real exposure.

**Work needed:**
1. Replace the polling loop with an `asyncio.Event` keyed by
   `confirmation_id`. The Telegram reactor (already in `mission_event_drain`)
   sets the event when the founder replies.
2. Persist the pending confirmation row (already exists in
   `action_confirmations` table) but stop holding the worker — return
   `Action(status="waiting_confirmation")` and re-enter the gate when
   the event fires.
3. New continuation path: `general_beckman.continuations` registers a
   `confirmation_resolved` handler that re-dispatches the original task
   with `payload.require_confirmation` already satisfied.

**Est. effort:** 4-6 hours; the table + reactor exist, only the wait
machinery and continuation glue are new.

---

### §2.D — Z3 P3 — `run_semgrep_layer_filtered` has no production trigger

**Where it dies:** verb + dispatch branch exist (`packages/mr_roboto/src/mr_roboto/__init__.py:3267`)
and `rule_packs/forbidden_in_domain.yml` is authored, but nothing
triggers it. Z3 P3 layer-aware reflection (shipped today) is the soft
side — this is the hard, blocking side.

**Work needed:**
- Option A (recommended): a new posthook kind `domain_layer_check` with
  `auto_wire_triggers=["src/domain/**/*.py", "**/domain/*.py"]`,
  dispatching to `run_semgrep_layer_filtered` with
  `rule_pack_path=forbidden_in_domain.yml`.
- Option B: explicit i2p step in phase 7/8 (`*.layer_check`) after every
  coder/implementer step that emits backend code.

**Est. effort:** 2-3 hours including a host-path test that creates a
domain-layer file importing requests + asserts the posthook fires.

---

### §2.E — Z2 P3 — `_apply_hint_from_targets` no-op on fresh missions

**Where it dies:** `src/workflows/engine/expander.py::_apply_hint_from_targets`
returns early when the workspace dir doesn't exist. On a fresh mission
the workspace isn't created until a step runs — but expansion happens
at task-creation time. So the `write_file`-strip pass never fires on a
new mission; only on re-expansion (recovery).

**Work needed:**
Either (small, recommended):
- Move the call from expander to per-step dispatch (just-in-time at
  task pickup), so workspace exists.

Or (smaller, documentation-only):
- Declare it "re-expansion-only" in the docstring + add an integration
  test pinning the behaviour.

**Est. effort:** 30 minutes (doc) or 2 hours (move to per-step).

---

## §3 — Pre-existing main-branch failures (NOT closed by this session)

Verified by `git stash` + run on pristine main. All flagged in the
relevant merge commit bodies but listed here too for the next session.

| Test | Failures | Root cause |
|---|---|---|
| `tests/founder_actions/test_lifecycle.py` | 8 | Z0 `3410aeb0` added `missions.lifecycle_state` column; tests still assert the fallback path |
| `tests/workflows/test_z5_t5_distribution.py::test_step_14_8_is_metadata_anchor` | 1 | Z0 `015382a4` promoted step 14.8 reversibility full→irreversible |
| `tests/workflows/test_z6_t6a_reversibility.py::test_audit_script_never_downgrades_committed_labels` | 1 | Same Z0 reversibility promotion |
| `tests/i2p/test_intake_todo.py` | 2 | `keyboard_sent` contract mismatch + status='completed' vs 'needs_clarification' shape drift in the intake_todo executor |
| `tests/founder_actions/test_repo.py::test_list_by_mission_orders_desc` | 1 (batched only) | DB-state pollution from neighbouring tests; passes in isolation |

Owners: Z0 (lifecycle + reversibility tags), intake-todo author.

---

## §4 — Suggested order for the deferred queue

1. **§2.D Z3 P3 semgrep layer trigger** — completes the Z3 layer story
   (soft side already landed); 2-3 hours; cheapest blast radius.
2. **§2.E Z2 P3 hint-from-targets** — 30 min doc OR 2 hour move; tiny.
3. **§2.B Z9 P3 reinforce join** — 2-3 hours; cleans up the last
   string-key match in Z9.
4. **§2.A Z1 P3 spec-patch button** — 2-4 hours; user-visible UX win.
5. **§2.C Z10 P3 event-driven confirmation** — 4-6 hours; biggest
   blast radius, do last; needs to land after a release window since
   it changes worker-slot semantics.

All five are "connect existing correct code" except §2.B (one small
schema migration) and §2.C (event machinery). No rewrites needed.

---

## §5 — Verification gate before next session

The wiring-sweep handoffs all landed with host-path coverage that
defends "the unit suites passed *with* the bug". Run these as the smoke
test of last resort:

```
.venv/Scripts/python -m pytest \
  tests/test_wiring_sweep_20260518.py \
  tests/test_wiring_sweep_p2_20260518.py \
  tests/test_z3_p2_cascade_20260518.py \
  tests/test_z8_sweep_20260518.py \
  tests/test_z6_sweep_20260518.py \
  tests/test_z1_sweep_20260518.py \
  -q
```

Expected: 40+ passing, 0 failing. If any of those break, treat as a
regression on the corresponding closed item from §1.
