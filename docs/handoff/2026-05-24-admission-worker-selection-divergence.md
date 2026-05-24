# Handoff — admission↔worker selection divergence (mission 74, 2026-05-24)

**For:** the session implementing the RC-A fix from mission 74's `No model candidates` DLQ flood.
**Status:** root cause **proven** (paired logs below); fix design **locked** with founder; **no code written yet**.

---

## §0 — The proven root cause (founder's original question, answered)

Beckman **admits** a task with a concrete model, then the **worker** (coulson) can't find a model
seconds later. Hard evidence — researcher task #164601, paired `ADMIT` → worker failure:

| Admission (`kutai.beckman.admission`) | Worker (`fatih_hoca.selector`), seconds later |
|---|---|
| 09:20:18 ADMIT model=Qwen3.5-9B-thinking | 09:20:21 `all candidates below pressure threshold task=researcher` |
| 09:40:18 ADMIT model=Qwen3.5-9B-thinking | 09:40:28 fail |
| 09:41:48 ADMIT model=gemini/gemma-4-26b | 09:42:10 fail |
| 09:50:36 ADMIT model=gemini/gemma-4-26b | 09:51:08 fail |
| 10:00:53 ADMIT model=Qwen3.5-9B-thinking | 10:01:04 fail → DLQ (worker_attempts 5/6) |

**Two `select()` calls that don't match:**
- **Admission** — `general_beckman/__init__.py:551`: `select(task, agent_type, difficulty, urgency, est_in/out)` (est from btable).
- **Worker** — `coulson/dispatch_helpers.py:51` (`pick_for_iter`): `select(full reqs: needs_function_calling, needs_thinking, prefer_local, reqs token estimates, failures, …)`.

Different **params** + different-moment **snapshot** (GPU free at admission, busy with a sibling
seconds later; cloud model fits the small btable est but fails the worker's larger reqs est / TPM).
So admission keeps **passing** while the worker keeps **failing**, and the retry loop re-admits into
the same wall → DLQ. The comment at `__init__.py:500` ("a singular selection mechanism… only one")
is the intent — but it isn't actually one.

**Why the worker re-asks at all:** `pick_for_iter` (`dispatch_helpers.py:42-45`) reuses Beckman's
`preselected_pick` **only on iteration 0**; every iteration ≥1 re-selects fresh. Documented reason
is failure-adaptation (exclude flaky models via `failures`). But on a **no-failure** turn the re-ask
is gratuitous — it discards the model already reserved for the task (`reserve_task`; the `task-*`
in-flight slot persists through ReAct gaps because `end_call` is a no-op for it) and re-races the
live pool. That race is the `no_candidates` mechanism.

**Dead theories (each ruled out with evidence — do not revisit):**
1. Static `ONESHOT_CONCURRENCY=4` over-subscription — it's only a backstop ceiling; pressure is the real limiter.
2. Per-task local reservation — wrong model: tasks **float across models/lanes per-iteration**; there are no "local tasks"/"cloud tasks".
3. Self-veto via own in-flight slot — **disproven**: #164601 had no `reserve`/`begin_call` of its own at the fail moments; admission handed it gemma (cloud) + Qwen (local).

Aggravator (why local-only agents die while analyst survives): researcher's cloud is fully
eligibility-filtered — `needs_function_calling=True` (`requirements.py`) + groq capable models TPM
6-8k (`per_call_too_large` on ~10k calls) + gemini capable models rpd≈20 (`daily_exhausted`) +
cerebras-235b failing. So when its reserved model slips, it has no fallback.

---

## §1 — Locked fix design

Founder directives: **make `no_candidates` rare + alarming** (NOT absorbed via hold/requeue — that
option was explicitly rejected). Two changes:

### A. Worker reuses the reserved pick across no-failure iterations
`pick_for_iter` (`coulson/dispatch_helpers.py`): change the reuse gate from
`iteration == 0 and not failures` to **reuse whenever `not failures`** — but **"reuse unless the held
model is gone"**: re-select only when (a) `failures` present (failure-adaptation/escalation), OR
(b) the currently-held model is no longer servable *right now* (unloaded/swapped-out / rate-limited /
daily-exhausted). Needs a cheap **servability primitive** (see §2).

### B. Unify admission's select() params with the worker's
`general_beckman/__init__.py:551`: pass the same parameters the worker will use — at minimum the same
**token estimates** (so admission doesn't pick a model that fails the worker's real ctx/TPM), plus
`needs_thinking` / `prefer_local` / `prefer_speed`. The agent-profile floor already applies
`needs_function_calling` at admission (selector floors from `AGENT_REQUIREMENTS[task]`), so FC is
consistent — verify, don't assume. Goal: admission's pick is one the worker would also pick → the
admission hold gate becomes honest (holds when the worker would fail).

With A+B: the task keeps its reserved model turn-to-turn; admission only admits what's truly servable;
`no_candidates` collapses to the genuine rare event (reserved model died with no fallback) — exactly
when an alert is warranted.

---

## §2 — Implementation notes / open questions for the implementer

- **Servability primitive (for A's "unless gone"):** need `is_model_servable_now(pick.model)` that
  reuses the selector's existing hard-eligibility checks (`selector.py:399+` reject-reason chain:
  ctx / FC / json / vision / cost / per_request / per_call-TPM / circuit_breaker / daily_exhausted /
  rpm_cooldown / no_vram) for that ONE model against the current snapshot — WITHOUT the pool-pressure
  scalar/threshold gate (pressure is about "should I start *new* load", not "can I continue on what
  I hold"). Likely a small helper on the selector that runs `_eligibility_reason(model, snapshot)`
  and returns True when None. NOTE: the held model's own `task-*` in-flight slot must NOT veto it
  (it's continuing, not contending) — exclude the current task's slot here.
- **`pick_for_iter` has no `task_id`/snapshot today** — it gets `reqs, task, failures, iteration,
  remaining_budget`. `task` is the dict (has `id`, `preselected_pick`, `task_state.used_model`).
  Thread the current model (from `task_state.used_model` or the prior pick) + a snapshot read.
- **B's token-estimate unification:** admission uses `fatih_hoca.estimates.estimate_for(shim, btable)`
  (`__init__.py:540`); the worker's `reqs.estimated_input/output` come from `AGENT_REQUIREMENTS`.
  Decide one source of truth (prefer the btable estimate everywhere, or feed reqs from the same).
- **DO NOT** add hold-not-DLQ / requeue-without-burning-attempts (founder rejected). Instead, keep
  `no_candidates` → fail fast + ensure it's surfaced (it already writes `admission_violations` +
  Telegram). After A+B, a `no_candidates` is a real bug to investigate, not normal backpressure.

---

## §3 — Validation (mandatory before claiming done)
- **Host-path / red→green test:** reproduce "admission sets `preselected_pick=X`; worker iter≥1 with
  no failures and X still servable → reuses X (no fresh `select`, no `no_candidates`)". Then a second
  test: "X no longer servable → re-selects". Drive the real `pick_for_iter`, not a mock.
- **Equilibrium harness (sensitive machinery — required):**
  `fatih_hoca/tests/sim/run_scenarios.py` + `run_swap_storm_check.py` must still pass.
- **Regression:** `packages/mr_roboto/tests/` + the touched-package suites (run dirs separately —
  dual conftest hazard; never `pytest` without a timeout; WAL-lock hazard if orchestrator live).
- **Re-run mission 74 fresh** (runtime fix → a wrapper restart picks it up; no re-expand needed since
  this is engine/selection code, not workflow JSON). Watch `admission_violations` for `no_candidates`
  → should approach zero. `SELECT reason, COUNT(*) FROM admission_violations GROUP BY 1`.

---

## §4 — Evidence pointers
- Live logs: `logs/kutai.jsonl` (selector picks, `selector pick:` / `all candidates below pressure threshold` / admission `ADMIT`). orchestrator.jsonl is stale (April).
- DB (read-only `file:C:/Users/sakir/ai/kutai/kutai.db?mode=ro`): `admission_violations` (566 `no_candidates` across test_generator ×187, planner ×75, executor ×66, implementer ×36, reviewer ×23, researcher ×12 — systemic), `tasks` (worker_attempts/error_category), `kdv_state` (cloud quotas: gemini rpd≈20, groq tpm 6-8k).
- Scratch: `scripts/_probe_missions.py 74`.
- Memory: `memory/project_pressure_concurrency_20260524.md` (full root-cause record + dead theories).
- Mission 74 is **paused**, 234 pending behind the Phase-0/1 DLQs. Other live bugs from this run
  (separate from RC-A): RC-B1 skipped legacy steps (0.2/0.4/0.5) get schema-validated → false DLQ;
  RC-D non_goals `<id>` template not substituted; artifact_summarizer empty-result. Triage in this
  session's transcript.

---

## §5 — SHIPPED (2026-05-24)

Implemented A + B via TDD. Working tree (uncommitted): a wrapper/orchestrator restart loads it.

### A — worker reuses the reserved pick (the real `no_candidates` fix)
- `fatih_hoca.selector.Selector.is_servable(model, reqs)` + module `fatih_hoca.is_servable` —
  per-model continuation gate: runs `_check_eligibility` for ONE model vs the current snapshot,
  **no** pool-pressure gate. Loaded-local carve-out: an already-loaded local survives
  `vram_available_mb==0` (own residency ≠ contention). Tests: `tests/test_servable.py` (7).
- `coulson/dispatch_helpers.py::pick_for_iter` — reuse gate changed from `iteration==0 and not
  failures` to: **reuse the held pick whenever `not failures` AND `is_servable`**; re-select only on
  failures OR held-gone. Every fresh select stamps `task["_held_pick"]` so later no-failure iters
  reuse the *running* model, not the stale admission preselect. Tests:
  `coulson/tests/test_pick_for_iter_reuse.py` (5).

### B — estimate unification (worker ↔ admission)
**The data inverted §0's premise.** Probe (`scripts/_probe_estimate_compare.py`, 9998 real calls vs
`model_call_tokens` ground truth):

| estimator | median est | actual median in | MAE | covers actual | predicts per-call overflow |
|---|---|---|---|---|---|
| char-based (worker was) | **1000** (floored) | 8877 | 9016 | **0%** | never (missed 7069/7069 @cap6600) |
| btable `estimate_for` (admission) | 8000 | 8877 | 4525 | 30% | catches most (missed 1630) |

`tasks.description+context` is almost always short → `(len)//4` floors to 1000, while the real
assembled prompt (RAG+system+tools, added at call time) is ~9k. So the **worker** under-counted by
~9000, not over — opposite of §0's "worker's larger reqs est". Fix (founder-confirmed): worker
adopts `estimate_for`.
- `fatih_hoca/requirements_builder.py::requirements_for` — dropped char-based input; now
  `estimated_input_tokens = estimate_for(shim, btable=get_btable()).in_tokens` (lazy `get_btable`
  import — fatih_hoca→general_beckman would be circular; empty-btable fallback = agent default).
  Worker + admission now resolve input identically. Tests: `tests/test_requirements_builder_estimate.py` (2).

### Validation run
- `tests/sim/run_scenarios.py` — all equilibrium + pp1–pp8 + realistic PASS.
- `tests/sim/run_swap_storm_check.py` — PASS (0% / 0.5% / 0% swaps). **Fixed a stale harness**: its
  `_snapshot()` returned a bare `SimpleNamespace` lacking `pressure_for` (pre-existing breakage on
  main, unrelated to RC-A) → now a real `SystemSnapshot`.
- `packages/fatih_hoca/tests/` — 341 passed, 1 skipped (isolated DB).
- `packages/coulson` + `packages/general_beckman` **full** suites NOT run: KutAI orchestrator is LIVE
  (PIDs incl. wrapper 31628), suites open the live DB → WAL-lock deadlock (one hung run was killed).
  general_beckman is **unchanged** by this work; coulson change is confined to `pick_for_iter`
  (covered by the targeted suite). Run the full suites after a `/stop` if a broad sweep is wanted.

### Deferred follow-ups
1. **Flag-forwarding (B "plus")** — admission still omits `needs_thinking`/`prefer_local`/`prefer_speed`/
   `local_only`/`exclude_models` from its `select()` (ranking-only divergence, not the eligibility gate).
   To close fully, route admission through `requirements_for` (heavier: per-candidate async DB reads).
2. **Poisoned learned-p90** — `step_token_stats.in_p90` fires only 1% of calls and is inflated by
   single huge-call outliers (+48k bias on that 1%). Clamp/winsorize or prefer p50.
3. **Huge-context under-estimate** — dropping char means a genuinely large stored context now
   under-estimates (both sides equally → no divergence; degrades to a call-time 413 that A's
   failure-re-select handles). `max(estimate_for, char)` was the rejected alternative.

### Re-run mission 74 (still TODO — needs founder)
Restart the wrapper (Telegram `/restart`) so the runtime picks up the change (engine/selection code,
no re-expand). Watch: `SELECT reason, COUNT(*) FROM admission_violations GROUP BY 1` — `no_candidates`
should approach zero.
