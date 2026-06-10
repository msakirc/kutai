# SP4b Plan 3 — design: crisis / incident / press_kit off `await_inline` (plumbing-only)

**Date:** 2026-06-08
**Branch:** `worktree-cps-sp4b-plan3` (off local HEAD).
**Predecessor handoff:** `docs/handoff/2026-06-07-cps-sp4b-plan3-handoff.md`.
**Pattern precedent:** Plan 1 reviews migration on `worktree-cps-sp4b` (commits `863fee25`, `77630383`, `d9a6a37b`, `ba7a0387`).
**Parent spec:** `docs/superpowers/specs/2026-06-05-cps-sp4b-design.md`.

## Goal

Remove the three live `await_inline=True` call sites that Plan 3 owns, so SP5 can delete the
blocking inline primitive:

| Call site | Verb |
|-----------|------|
| `mr_roboto/crisis_draft_holding.py:155` | `crisis/draft_holding` |
| `mr_roboto/incident_draft_update.py:180` | `incident/draft_update` |
| `mr_roboto/press_kit_assemble.py:109` | `press_kit/assemble` (×4 calls in a loop) |

**Scope is plumbing only.** We move each blocking LLM call onto the durable CPS continuation
substrate. We do **not** fix press_kit's grounding / spec-source design flaw, do **not** build
`degrade_on_exhaustion`, and do **not** touch the dormant demo-pipeline workspace gap. Those are
explicitly out of scope (founder decision, 2026-06-08).

## Key reframe — `degrade_on_exhaustion` is NOT needed

The handoff assumed each verb would split into a producer step and a **workflow** sink with a
`depends_on` edge. In that shape a DLQ'd producer blocks the sink → the canned fallback never
runs → regression for time-sensitive comms. That shape needs `degrade_on_exhaustion` (a Beckman
core change), which is why the handoff gated Plan 3 on "cleared to touch Beckman."

But these three verbs are **standalone mechanical dispatches** (routed by
`mr_roboto/__init__.py` `if action == "..."`), not multi-step workflow JSON steps. Nothing hangs
off them via `depends_on`; their output goes to a founder_action card. So the right primitive is
the existing **continuation substrate** (`packages/general_beckman/src/general_beckman/continuations.py`),
not a workflow sink.

Critically, `fire_for_task` (continuations.py:86–156) dispatches the `on_error` handler on terminal
`failed` status, and `reconcile_continuations` (line 240) re-fires it after a restart / TTL expiry.
**So the failure path already has a hook.** The canned fallback simply lives on the `on_error`
path (or a shared finalize that both `on_complete` and `on_error` call).

Result: **zero General Beckman core changes, zero fallback regression.** This is exactly what
Plan 1 did for the sibling `reviews/*` verbs — it does not use `degrade_on_exhaustion` either.

## The migration pattern (mirror Plan 1)

For each verb:

1. **Gut the verb module** down to its mechanical pieces only: input prep (playbook/summary
   reads, redaction), LLM-response parsing, the canned fallback, and the founder-facing emitters.
   Delete `_call_llm_draft` / `_draft_one_pager_llm` (the `await_inline` call).
2. **Add a producer enqueuer** that admits the LLM call as a normal Beckman task
   (`lane=oneshot`, `kind=overhead`, `agent_type=reviewer`/`planner`) with
   `on_complete=<resume>`, `on_error=<resume>`, and `cont_state={…the verb's mechanical inputs…}`.
   No `await_inline`.
3. **Add the continuation/sink handler** (`mr_roboto.executors.<verb>_continuations`) that receives
   `(child_task_id, result, state)`, parses the LLM result (or sees the failure), applies the canned
   fallback, and performs the founder-facing side-effect (emit card / finalize draft).
4. **Repoint the trigger** (router branch and/or cron/founder/oncall site) to enqueue the producer
   instead of dispatching the now-blocking verb.
5. **Register the new sink module in `_HANDLER_MODULES`** (continuations.py:175) so the handler is
   present after restart for `reconcile_continuations` recovery. This is a hard contract —
   forgetting it is a silent correctness bug.

## Per-verb shapes

### crisis/draft_holding
- **Keep in verb module:** `_read_playbook`, `_playbook_path`, `_VARIANT_PREFIX_RE` parse, the
  tier-labelled canned-variants fallback (currently lines 254–273), the DB summary fetch.
- **Producer cont_state:** `{tier, summary, playbook_excerpt, event_id, product_id}`.
- **Sink (`crisis.holding.resume`):** parse the JSON array of variants from the LLM result; on
  empty/failure use the canned variants; emit the founder card with the variants.
- **Open item:** the current verb *returns* variants to its caller and does not emit a card itself.
  Planning must locate the caller and confirm the card is emitted from the sink (Plan-1 shape).

### incident/draft_update
- Same shape as crisis. Keep redaction + `_fallback_draft`.
- **Wrinkle:** the founder card comes from the `incident_update_review` **post-hook**, which reads
  `result.draft`. In CPS the draft is produced by the continuation, not the prep task. The review
  gate must fire off the **sink's** result, not the prep task's. This is a repoint of where the
  posthook sees the draft — **not** a new Beckman verb and **not** `press_kit_freshness` (which we
  drop, per scope). Planning must trace `apply.py:2641` / `posthook_continuations` to wire this.

### press_kit/assemble — serial chain of 4 producers
- Decision (2026-06-08): preserve per-audience prompting; collapse the 4 `await_inline` calls into
  a **serial chain** of 4 producer→continuation hops rather than one combined call or a fan-in join.
- **Flow:** producer(investor) → `press_kit.audience.resume` drafts+stages investor one-pager,
  then enqueues producer(journalist) → … → producer(candidate) → resume stages candidate, then
  runs the existing assembly (zip per audience + manifest + founder_action).
- **cont_state carries:** `{product_id, mission_id, version, workspace_path, remaining_audiences,
  staged: {audience → one_pager_text}, source params (founder_bio, fact_sheet_md, quotes, …)}`.
- Keep the existing `_build_zip`, `_audience_extra_section`, `_emit_founder_action`,
  `_get_latest_version`, and the per-audience stub fallback (`[Draft one-pager for {audience}…]`).
- `workspace_path` derive stays the caller's responsibility (out of scope to fix here; pass through
  cont_state as today).
- **Drop** the `press_kit_freshness` post-hook from this work — it is advisory and its missing
  Beckman dispatch branch is out of scope.

## Restart recovery & landmines

- Every new sink module **must** be added to `_HANDLER_MODULES` (continuations.py:175).
- **Offline tests miss runtime wiring** (Plan 2 lesson). For each verb add a **router-level test**
  that drives the real trigger path (enqueue producer → simulate child terminal → assert the sink
  fires and produces the card/draft), plus an `on_error` path test (LLM failed → canned fallback
  shipped). Unit-testing the helpers in isolation is not sufficient.
- **No concurrent pytest** — live KutAI holds the `kutai.db` WAL. Run ONE `timeout`-prefixed
  invocation at a time; DB-touching suites hang against the live stack.
- Worktree has no `.venv` — use the main repo venv
  `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe`.
- lane = `oneshot` for the admitted producers.

## Out of scope (founder decisions, 2026-06-08)
- press_kit spec-source / grounding (the 4 planners hallucinate from `{product_id}` only).
- `degrade_on_exhaustion` and any General Beckman core change.
- `press_kit_freshness` post-hook dispatch branch.
- demo-pipeline downstream workspace plumbing.

## Open items to resolve during planning (not blockers)
1. crisis trigger / caller — where the returned variants are consumed; move card emission to sink.
2. incident post-hook repoint — where `incident_update_review` reads the draft post-split.
3. press_kit trigger — how `run(*, mission_id, …)` (kwargs, not payload dict) is currently
   dispatched, and how the serial-chain cont_state threads through the router branch.
4. Confirm each producer's `agent_type`/`difficulty` matches the verb's current spec so model
   selection is unchanged.

## Done = SP5 unblock contribution
After this lands, re-grep `await_inline\s*=\s*True` in `packages/ src/` shows the 3 Plan-3 sites
gone. Combined with Plan 1 (reviews ×2, merge-pending) that leaves only the 2 SP5 carve-outs +
the shopping `request()` shim before `await_inline` can be deleted.
