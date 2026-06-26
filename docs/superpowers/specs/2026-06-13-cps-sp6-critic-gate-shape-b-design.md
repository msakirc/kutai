# Spec — CPS SP6: critic gate shape-(b) — admitted verdict, fail-closed

**Date:** 2026-06-13
**Sprint:** CPS SP6 (the post-SP5 cleanup + the one designed item)
**Status:** DESIGN — founder approved shape-(b) over shape-(a) on 2026-06-13.
**Predecessors:** `project_sp5_await_inline_closed_20260611` (SP5 finale, await_inline deleted),
`project_cps_migration_20260527` (SP3b stood up `produce_verdict`/`confirm_gate` split).

---

## Why (the moonshot framing — plain)

The moonshot is **KutAI acting autonomously at scale** — committing code, messaging humans,
spending money — without a person watching each step. Before any such irreversible action a
*critic* LLM judges it (the "critic gate"). Today that gate has two flaws that are harmless at
2 sites but disqualifying for an autonomous fleet:

1. **Fails OPEN.** Every error path in `critic_gate.py` returns a default-`pass`. A safety gate
   that waves the action through whenever the critic call breaks. `confirm_gate` resolution rule
   3 (no verdict → default-pass), `produce_verdict` except-clause, `critic_gate` orchestrator
   except-clause — all fail-open.
2. **INVISIBLE to admission.** The gate's LLM hop runs **inline inside a mechanical executor**
   (`git_commit`/`notify_user` call `critic_gate` → `produce_verdict` → `husam.run` mid-action).
   That nested call is unaccounted by Beckman's quota look-ahead and lane accounting, and it
   violates `feedback_no_direct_dispatcher` in spirit (a mechanical drives an LLM call inside its
   own execution rather than through admission).

Shape-(b) makes the verdict a **first-class admitted task** that runs *before* the gated action,
persists its verdict, and lets the mechanical read it with **NO inline LLM call**. Consequences:
- Critic LLM fails → the producer task fails → **the gated action does not proceed** (fail-CLOSED),
  on the normal retry/DLQ rail.
- The verdict call is admitted → quota look-ahead + lanes see the load.

Shape-(a) (keep the inline call as a sanctioned exception, like `src/tools/vision.py`) was rejected
for the long term: vision is a *mid-ReAct dynamic* tool call (cannot be pre-scheduled without a
CPS-for-tools effort); critic gate sites are **static** workflow/mechanical steps whose verdict can
be produced by an admitted unit *before* the side-effect. No reason to leave an inline exception
where the structure removes the problem.

---

## Current state (verified 2026-06-13)

Three critic-gate surfaces exist:

| # | Surface | Where | Today |
|---|---------|-------|-------|
| A | **Posthook kind** `post_hooks: ["critic_gate"]` | apply.py:2357 spawns a **mechanical** child; i2p `init_mission_github_repo` step (i2p_v3.json:5707) | mechanical child's executor (`action=="critic_gate"`, `__init__.py:983`) calls `critic_gate()` → `husam.run` inline |
| B | **Inline mechanical gate** | `git_commit` (`__init__.py:810`), `notify_user` (`__init__.py:~2160`) | executor calls `critic_gate()` → `produce_verdict` → `husam.run` inline, mid-action; veto → unstage/drop + `Action(failed)` |
| C | **Standalone `critic_gate` action** | executor `action=="critic_gate"` (`__init__.py:983`) | the worker for surface A; also callable as an explicit workflow step |

Already-built shape-(b) halves (from SP3b Task 8, partial):
- `produce_verdict(action_name, payload, *, mission_id)` — runs the critic, **persists** a
  `critic_log` row, returns the verdict dict. STILL does `husam.run` inline (not admitted).
- `confirm_gate(action_name, payload, *, persisted_verdict)` — LLM-free, no dispatcher import,
  returns pass/block from a persisted verdict. **Currently fail-OPEN** (rule 3: missing verdict →
  default-pass).
- `critic_log(id, mission_id, action_name, verdict, reasons_json, redacted_payload_hash, created_at)`
  table (db.py init_db).

So the producer/confirm split is the right substrate; SP6 finishes it: make the producer an
**admitted task** and make confirm **fail-closed**.

---

## Design

### Principle
The verdict is produced by an **admitted critic task** that runs to terminal status *before* the
gated side-effect. The mechanical executor that owns the side-effect calls **`confirm_gate` only**
(no LLM, no husam) against the persisted verdict.

### Surface A — posthook (cheapest; reuse the LLM-child path)
Stop spawning a *mechanical* child that re-enters `husam.run`. Instead spawn a **raw_dispatch LLM
child directly**, exactly like grade/code_review/constrained_emit posthooks already do via
`apply._enqueue_posthook_llm_child` + `posthook_continuations`. Steps:
- In apply.py `critic_gate` posthook branch (2357): build the critic raw_dispatch spec
  (reuse `critic_gate._build_critic_spec`) and enqueue it as the posthook LLM child with
  `on_complete="posthook.critic.verdict_done"` / `on_error="posthook.critic.verdict_err"`,
  `cont_state={"source_task_id", "action_name", "payload_hash", "mission_id"}`.
- New handlers in `general_beckman/posthook_continuations.py`:
  - `verdict_done(task_id, result, state)`: parse with `critic_gate._parse_verdict`, persist
    `critic_log`, build `PostHookVerdict(kind="critic_gate", passed=(verdict!="veto"), raw={...})`,
    apply via `_apply_posthook_verdict`. **CORRECTION-2 (planning 2026-06-13): the verdict rail
    ALREADY handles `critic_gate` — no new branch needed.** `critic_gate` is in `_Z1_BLOCKER_KINDS`
    (apply.py:3949) ⊂ `_Z1_MECHANICAL_KINDS` (3958); `_apply_posthook_verdict_locked` branch 8
    (apply.py:5165 `if a.kind in _Z1_MECHANICAL_KINDS`) routes it through `_apply_z1_mechanical_verdict`
    = **single-shot DLQ on `passed==False`**, and it is NOT in `_PRODUCER_QUALITY_Z1_BLOCKERS` so it
    does NOT escalate-retry (correct for a single-shot veto). The earlier review's "must add a
    branch" was over-pessimistic — set membership covers it. The handler's only job is to build the
    right `PostHookVerdict` (`passed=False` on veto → DLQ; `passed=True` → drop from pending,
    advance). The mechanical-child verdict path used the SAME rail today, so behavior is preserved;
    only the child's TYPE changes (mechanical→raw_dispatch LLM child).
  - `verdict_err(task_id, result, state)`: **fail-CLOSED** — fail the source with
    `error="critic verdict unavailable (producer error) — action blocked"`. Goes to normal retry/DLQ.
- Delete surface C (`action=="critic_gate"` executor) once A no longer spawns a mechanical child —
  re-grep first. No PROD caller besides A, but **live test callers exist** — `test_critic_gate.py`
  (`test_router_standalone_critic_gate_pass/veto`, ~:339,360) + the `produce_verdict`/`critic_gate`
  orchestrator tests in `test_critic_gate.py:106-164` + `test_critic_gate_split.py:106-164`. These
  must be deleted/rewritten as part of the cut.

### Surface B — inline mechanical gate → two-pass self-park (waiting_human re-pend precedent)
`git_commit` and `notify_user` are mechanical executors that need the verdict *before* the
side-effect, and the **same executor must run twice** (capture → commit). The correct precedent is
the **human-confirm gate** (`__init__.py:734-789`), NOT investor_bullets.
> **CORRECTION (review 2026-06-13):** the original draft cited investor_bullets as the precedent —
> WRONG. `run_investor_bullets` (jobs/investor_bullets.py:785) **completes** and does all remaining
> work inside continuation handlers (`_finalize_bullets`); the original task never re-runs. That is
> a **kickoff-and-finalize** mechanic. Surface B needs the executor to RE-RUN (stage in pass 1,
> commit in pass 2), which is the **`needs_clarification`+`waiting_human` park-and-re-dispatch**
> mechanic the human-confirm gate already uses (orchestrator skips `on_task_finished` for
> `needs_clarification` on mechanical tasks, orchestrator.py:310-318; `waiting_human` rows are not
> re-picked by `get_ready_tasks`, db.py:4676; flip to `pending` → re-dispatch → executor re-runs).

- **Pass 1** (no verdict in `task` context):
  - Build the gated payload (for git_commit: stage `-A`, read `--cached --stat`/diff, planned msg —
    the same capture done inline today, but WITHOUT calling the critic).
  - Enqueue the critic producer as an **admitted raw_dispatch child** (reuse `_build_critic_spec`;
    do **NOT** pass `lane=` — `pick_lane("critic")` derives `oneshot`; passing `lane="overhead"`
    literally is the SP3b phantom-lane bug, `kind="overhead"` is fine and orthogonal)
    with `on_complete="mr_roboto.critic.verdict_done"` / `on_error="mr_roboto.critic.verdict_err"`,
    `cont_state={"gated_task_id": task.id, "action_name", "payload_hash", "mission_id"}`.
  - Park THIS task: `await update_task(task.id, status="waiting_human")` then return
    `Action(status="needs_clarification")` (the EXACT mechanic at `__init__.py:770,781`; the
    orchestrator skip-list matches `needs_clarification`, NOT the non-existent `needs_followup`).
    For git_commit the stage stays applied between passes (re-stage is idempotent); for notify_user
    nothing has been sent yet.
- **Resume** (`mr_roboto/critic_continuations.py`, registered in `_HANDLER_MODULES`):
  - `verdict_done`: persist `critic_log`, stamp `{verdict, reasons}` into the gated task's context,
    flip the gated task back to `status="pending"` (Beckman re-dispatches → `mr_roboto.run`
    re-runs → pass 2). Resume is gated on `context.critic_verdict` presence (mirrors the human gate
    keying on `context.user_clarification`).
  - `verdict_err`: **fail-CLOSED** — fail the gated task (`error="critic verdict unavailable — <action> blocked"`).
- **Pass 2** (verdict present in context): call `confirm_gate(..., persisted_verdict=ctx_verdict)`.
  - `pass` → perform the real side-effect (auto_commit / send notification) — unchanged tail.
  - `veto` → roll back (git: `reset`; notify: drop) + `Action(failed)` — unchanged veto tail.

### B2 carve-out — `notify_user` gated ONLY when mission-scoped (Opus review 2026-06-13)
`notify_user` is the single outbound boundary for ROUTINE non-mission alerts (VRAM/health pings,
cron/uptime, follow-up reminders, mention digests — many with no `mission_id`, firing in normal
operation). A uniform fail-closed two-pass gate would add a critic round-trip to every alert AND
**drop** the alert when the critic is unavailable — pathologically, a "VRAM 95%" alert vetoed
*because* the critic can't run under that pressure (the #261969 history is why notify_user
fail-OPEN was deliberate). The gate guards **autonomous agent-produced content** (mission comms),
not internal system health. **Rule:** `notify_user` enters the gate ONLY when
`task.mission_id is not None`; routine alerts (no mission_id) bypass entirely — send directly,
never parked, never dropped. `git_commit` stays unconditionally gated (always agent output;
reversible via `git reset`/retry so fail-closed is safe).

### `confirm_gate` → fail-CLOSED (the safety flip)
Rule 3 changes: a **missing/garbage verdict when the gate is enabled** returns `verdict="veto"`
(blocked) with reason `"no critic verdict — fail-closed"`, NOT default-pass. `KUTAI_CRITIC_GATE=off`
(rule 1) still bypasses to pass (explicit operator opt-out is the ONLY pass-without-verdict path).
This is the single change that makes "broken critic blocks the action" true everywhere.

`produce_verdict`'s internal except-clause (critic_gate.py:258-267) fakes a pass today; once A+B
migrate, the admitted-child `on_error` path governs failure (fails-closed) instead. The inline
`produce_verdict` + `critic_gate` orchestrator functions (the latter has no except of its own —
it inherits fail-open from `produce_verdict`, critic_gate.py:343-381) become **dead once A+B
migrate** — delete them (re-grep + the test cleanup noted under surface C), leaving
`_build_critic_spec` + `_parse_verdict` + `_redact_*` + `_persist` + `confirm_gate` as the live API.

---

## Out of scope / deferred
- `src/tools/vision.py` direct `husam.run` (ruling-#1) — stays shape-(a) by design (mid-ReAct tool).
  SP6 only adds a comment pointing at this decision; no behavior change.
- CPS-for-tools (a general framework for mid-ReAct LLM tool calls) — future, separate moonshot brick.

## SP6 sweep companions (light, non-design — do in the same sprint)
1. Remove the SP5 **legacy straggler shim** in `on_task_finished` (db continuation table is now the
   sole path). **Target = `__init__.py:1309-1329`** ("Legacy straggler shim (removable post-SP5)")
   — the phrase "legacy-context fallback" from the SP5 handoff is NOT in the code; grep the real
   marker `removable post-SP5`.
2. Delete the orphaned worktree dir `.claude/worktrees/sp5-deletion-sweep` (Windows file-lock survivor).
3. Reconcile the pre-existing stale tests listed in the SP5 handoff
   (`test_mission_workflow_integration.py::TestLLMClassification`, `test_agent_basic.py` ReAct-iter).
4. `vision.py` comment + a one-line note in CLAUDE.md that critic gate is now admitted+fail-closed.

---

## Validation
- **Fail-closed proof:** force the critic child to `on_error` (mock husam raise) → assert the gated
  `git_commit` task ends `failed` and **no commit lands** (real-DB pump test, RED on today's
  fail-open `confirm_gate`).
- **Admitted proof:** assert the critic verdict task appears as a real row picked by the pump
  (`lane=oneshot`, NOT a phantom lane — cf. the SP3b `lane="overhead"` regression), not an inline call.
- **Veto rail intact:** veto verdict → source fails + side-effect rolled back (git reset / no-send).
- **Opt-out intact:** `KUTAI_CRITIC_GATE=off` → pass without producing a verdict task.
- **No inline husam left:** structural test — `git_commit`/`notify_user`/surface-A executors import
  no `husam` and call no `produce_verdict`/`critic_gate` (only `confirm_gate`).
- Re-grep `await_inline=True` stays zero; re-grep `produce_verdict`/`critic_gate(` callers → zero
  before deleting them.

## Risks
- **Two-pass git_commit staging:** stage applied in pass 1 must survive to pass 2. Re-pend keeps the
  workspace; `git add -A` is idempotent. If a concurrent step mutates the tree between passes the
  pass-2 diff differs from the judged one — mitigate by re-capturing + comparing `payload_hash`;
  on mismatch, re-produce. **CORRECTION (review 2026-06-13): re-produce MUST be bounded** — with
  fail-CLOSED confirm, an unbounded re-gate livelocks (concurrent mutation → permanent mismatch →
  never commits → burns retries to DLQ). Cap re-gate at N=2 (stamp a `critic_regate_n` counter in
  context); on exhaustion, fail the task with `error="critic re-gate exhausted (tree kept changing)"`
  → DLQ with a clear reason, not an infinite spin. In practice gated commits are serialized i2p
  siblings (low concurrency) so the window is small, but the cap is mandatory. Pin in the plan.
- **Latency:** +1 task hop per gated action. Acceptable — gated actions (commit, notify) are rare
  and already heavy.
- **Park status — RESOLVED (proven precedent):** mechanicals already park-and-resume. The
  human-confirm gate (`__init__.py:734-789`) returns `Action(status="needs_clarification")` +
  `update_task(status="waiting_human")`; the orchestrator special-cases that status for mechanical
  tasks and **skips `on_task_finished`**, leaving the row parked (orchestrator.py:316-446); a later
  injector flips it back to `pending` → Beckman re-dispatches → `mr_roboto.run` re-runs on the
  resume path. The critic two-pass uses the **same park mechanic**, with ONE difference: the
  injector is the **CPS `verdict_done` continuation handler** (stamps verdict into the gated task
  context + re-pends), NOT a telegram reply. Open detail for the plan: whether to reuse
  `waiting_human` verbatim or add a `waiting_critic` status (semantically cleaner, but the
  orchestrator skip-list must include it — cheaper to reuse `waiting_human` and gate the resume on
  `context.critic_verdict` presence vs `context.user_clarification`).
