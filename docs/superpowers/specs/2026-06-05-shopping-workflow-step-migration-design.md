# Shopping → Workflow-Step Migration (retire `request()`, unblock CPS SP5)

**Date:** 2026-06-05
**Status:** Design — approved, pending spec review
**Author:** founder + Claude (brainstorm)
**Related:** `docs/superpowers/specs/2026-05-29-cps-sp3b-design.md` §6/§7, `docs/handoff/2026-05-30-sp4-kickoff.md`, `project_shopping_state_20260505.md` (memory, corrected 2026-06-05)

---

## 1. Problem

The shopping pipeline makes LLM calls **inline inside Beckman task handlers** via the
deprecated `LLMDispatcher.request()` shim, which routes through
`general_beckman.enqueue(spec, await_inline=True)` — the blocking inline primitive.

Two consequences:

1. **Live deadlock risk.** A shopping step-task holds a `oneshot` lane slot
   (`ONESHOT_CONCURRENCY=4`) while its inline `request()` blocks awaiting a child
   raw_dispatch task that needs *another* oneshot slot. Under load this exhausts the
   lane → slot-deadlock → `INLINE_TIMEOUT` → DLQ. CPS SP3b §6 names this an explicit,
   accepted deferral: "shopping's residual deadlock risk persists until shopping
   migrates."
2. **SP5 is blocked.** CPS SP5 deletes `await_inline`/`request()`. Per SP3b §7,
   `request()` is retained *only* as a marked shopping-only shim and "dies when
   shopping migrates — **before** SP5 deletes `await_inline`." Shopping migration is a
   hard prerequisite to SP5.

CPS SP3b §6 dictates the correct shape: shopping is a **workflow definition**; each LLM
step must become **an admitted task routed to hüsam by the engine**, not a bespoke
inline-LLM worker. This spec executes that migration.

### `request()` call sites (exhaustive — all 4 must go)

| # | Site | Step | Today |
|---|------|------|-------|
| 1 | `_grouping_llm_call` (`pipeline_v2.py:363`) | 1.1 group_label_filter_gate | inline `request()` |
| 2 | `_label_llm_call` (`labels.py:22`) | 1.1 group_label_filter_gate | inline `request()` |
| 3 | `_synthesis_llm_call` (`pipeline_v2.py:487`) | 2.0/2.2 synth_one, 2.3 format_compare | inline `request()` |
| 4 | `single_shot.run` (`coulson/single_shot.py:46`) | 0.2 shopping_clarifier | inline `request()` |

`single_shot.run` is reached **only** by `shopping_clarifier` (the only profile setting
`execution_pattern=single_shot`; planner/classifier are react). Retiring it removes the
last non-pipeline `request()` caller.

---

## 2. Goal & Non-Goals

**Goal.** Decompose every inline shopping LLM call into linear workflow steps so that no
task holds a lane slot while awaiting a child. Result: zero `await_inline` in any
shopping path, `request()` fully retired, SP5 unblocked.

**Non-goals (explicitly out of scope this iteration):**
- The 3 absent intelligence layers (cross-line verdict #11, spec matrix #12, price
  intelligence #13). Tracked separately; not touched here.
- Any change to shopping's review-quality / bucketing logic. The deterministic functions
  (`step_group` SKU-bucket, `step_label` apply, `step_filter`, `step_variant_gate`,
  `format_group_card`, `step_compare_all`, `_fetch_community_reviews`,
  `step_synthesize_reviews` body) are **reused verbatim** — only their *call boundaries*
  move.
- Grading behavior. Shopping steps keep `requires_grading:false` (UX output, not graded
  artifacts).

---

## 3. The one new engine mechanism: `llm_dispatch_from`

The pump **already** routes any task with `context.llm_call.raw_dispatch == True` to
`husam.run` (`orchestrator.py:355-382`). No new routing is needed. The single missing
piece is **declarative**: a workflow step that sources its `context.llm_call` from a
prior step's output artifact.

### Step field

```json
{
  "id": "1.1b", "name": "group_dispatch", "agent": "raw_dispatch",
  "depends_on": ["1.1a"],
  "input_artifacts": ["group_llm_call"],
  "output_artifacts": ["group_llm_raw"],
  "llm_dispatch_from": "group_llm_call"
}
```

### Resolution hook — at workflow ADVANCE, before admission

When the prepare step (`1.1a`) completes and the engine advances, the
`llm_dispatch_from` resolver runs for the now-ready dispatch step (`1.1b`):

1. Read the named artifact (`group_llm_call`) — a JSON object
   `{messages, difficulty, task, response_format, needs_thinking, min_context,
   estimated_input_tokens, estimated_output_tokens, urgency, raw_dispatch:true}`.
2. Write it into the dispatch task row's `context.llm_call`.

**Why advance-time, not pre-hüsam:** `Beckman.next_task()` admission runs
`fatih_hoca.select()` to pick a model, and selection reads `estimated_input_tokens` /
`min_context` from `context.llm_call`. If materialization happened only just before
`husam.run`, admission would select against an empty prompt. Resolving at advance time
(after prepare completes, before the dispatch task is `ready`) guarantees admission sees
the real prompt size.

**Hook location (to confirm in plan):** the workflow advance path
(`general_beckman/apply.py` advance / `workflows/engine/hooks.py`) that transitions
dependents to ready. The resolver is a small, generic engine helper — not
shopping-specific — so future workflows can reuse the prepare→dispatch→parse pattern.

### The triad pattern

Every LLM call becomes three steps:

| Role | Agent | Determinism | Job |
|------|-------|-------------|-----|
| **prepare** | `shopping_pipeline_v2` | deterministic | build structured messages from artifacts → emit `<x>_llm_call` |
| **dispatch** | `raw_dispatch` | hüsam | `llm_dispatch_from` → select + one call → raw response artifact |
| **parse** | `shopping_pipeline_v2` | deterministic | parse raw JSON → domain artifact |

Message-building reuses the exact prompt assembly that lives in `_grouping_llm_call`,
`_label_llm_call`, `_synthesis_llm_call` today (the `GROUPING_PROMPT` / `LABEL_PROMPT` /
`SYNTHESIS_PROMPT` formatting). The spec-shape mirrors what `_request_kwargs_to_spec`
already produces — we are splitting one inline call into prepare(spec)+dispatch+parse.

---

## 4. `shopping_v3.json` — step layout

New plan file `src/workflows/shopping/shopping_v3.json` (`plan_id: "shopping_v3"`).
Reasons for v3 over mutating v2: the change is structural (7 → ~18 steps), and a clean
break lets v2 stay runnable until v3 is live-validated, then v2 retires. Mission creation
switches to `shopping_v3` (see §6).

### Phase 0 — Understand

- **0.1 understand_query_check_clarity** — `shopping_pipeline_v2`, unchanged.
- **0.2 ask_clarifying_questions** — **converted**: `shopping_clarifier` becomes a plain
  **react** agent (drop `execution_pattern=single_shot`), routing via `coulson.execute`.
  This removes `request()` caller #4. UX parity must be verified (single-shot vs react
  loop produces the same clarify questions).

### Phase 1 — Resolve

- **1.0 resolve_candidates** — `shopping_pipeline_v2`, unchanged (scrapers, no LLM).

### Phase 2 — Group / Label / Gate (was the fused 1.1)

Split `_handler_group_label_filter_gate` (2 inline LLM calls) into:

- **1.1a group_prep** — `shopping_pipeline_v2`: run `step_group`'s deterministic
  SKU-bucketing; if residual (un-bucketed) candidates remain, build the group-residual
  messages → emit `group_llm_call` + a `has_residuals` flag. Emit partial `groups_state`.
- **1.1b group_dispatch** — `raw_dispatch`, `llm_dispatch_from: group_llm_call`,
  `skip_when: "!has_residuals"` → `group_llm_raw`.
- **1.1c group_parse_label_prep** — `shopping_pipeline_v2`: apply group-residual result
  (or pass through if skipped), then build `step_label` messages → emit `label_llm_call`
  + updated `groups_state`.
- **1.1d label_dispatch** — `raw_dispatch`, `llm_dispatch_from: label_llm_call` →
  `label_llm_raw`.
- **1.1e label_parse_filter_gate** — `shopping_pipeline_v2`: apply labels (`labels.py`
  parse), `step_filter`, `step_variant_gate` → `gate_result` (same shape as today:
  `chosen | clarify | escalation`).

Conditional skip: if SKU bucketing already covers all candidates, 1.1b is skipped via
`skip_when` and 1.1c passes groups through unchanged. Label is always run (every group
needs a label).

### Phase 2 — Synth paths

`step_synthesize_reviews` (1 LLM call + scrape) splits into prepare/dispatch/parse.
Note the scrape (`_fetch_community_reviews` when `deep_scrape`, commerce review fetch
otherwise) is deterministic I/O and lives in **prepare**.

**Chosen path (was 2.0):**
- **2.0a synth_prep** — `shopping_pipeline_v2`: resolve chosen group, fetch reviews,
  build `SYNTHESIS_PROMPT` messages → `synth_llm_call`. `skip_when: gate.kind != chosen`.
- **2.0b synth_dispatch** — `raw_dispatch` → `synth_llm_raw`.
- **2.0c synth_parse** — `shopping_pipeline_v2`: parse → `ReviewSynthesis` →
  `format_group_card` → `synth_result`.

**Variant-pick path (was 2.2):** identical triad gated on `clarify_choice.kind ==
variant`, with `deep_scrape=true`. Steps **2.2a/2.2b/2.2c**. (Prepare/parse handlers are
shared with the chosen path; only the group-resolution differs — resolve from
`clarify_payloads[gid]`.)

- **2.1 clarify_variant** — `mechanical`, unchanged (waits for user tap → `clarify_choice`).

**Compare-all path (was 2.3) — sequential self-iterating:**

≤5 lines (`MAX_CLARIFY_OPTIONS=5`), latency-tolerant secondary path. Sequential keeps it
clean (no join, no engine fan-out). Steps:

- **2.3a compare_init** — `shopping_pipeline_v2`: list the lines from `clarify_payloads`,
  render `step_compare_all` header, init `compare_state = {lines, cursor:0, cards:[]}`.
  `skip_when: clarify_choice.kind != compare_all`.
- **2.3b compare_synth_prep** — `shopping_pipeline_v2`: build `SYNTHESIS_PROMPT` for
  `lines[cursor]` → `synth_llm_call`.
- **2.3c compare_synth_dispatch** — `raw_dispatch` → `synth_llm_raw`.
- **2.3d compare_synth_parse_loop** — `shopping_pipeline_v2`: parse → card, append to
  `compare_state.cards`, `cursor++`. **If `cursor < len(lines)`**: re-arm 2.3b for the
  next line; **else** advance to 2.3e.
- **2.3e compare_assemble** — `shopping_pipeline_v2`: join header + cards →
  `shopping_response`.

The loop (2.3b→2.3c→2.3d→2.3b…) is the **self-iterating** mechanism. Implementation
options for the loop-back, to be chosen in the plan: (a) a cursor-driven re-ready of the
prep step via the advance hook (preferred — declarative `loop_while` on 2.3d), or
(b) 2.3d enqueues the next prep iteration through the normal admitted path (no
`await_inline`). Either way each synth is its own admitted step → no slot nesting.

### Phase 3 — Deliver

- **3.0 format_response** — `shopping_pipeline_v2`, unchanged (reads `synth_result` →
  `shopping_response`). `skip_when: synth_result.cards empty`.

---

## 5. Telegram path

Today there are **two** compare-all paths: workflow step 2.3 *and* a direct bot call
(`telegram_bot.py:_run_compare_all_and_reply` → `_handler_format_compare`, bypassing the
engine and using `await_inline` in the bot loop).

**Change:** route the compare-all tap **through workflow resume** (resume the mission at
the compare path so steps 2.3a–e run in the pump). Delete the direct
`_handler_format_compare` invocation from `_run_compare_all_and_reply`. The bot keeps:
sending the variant keyboard, parsing the `vc:{mission_id}:{task_id}:{choice}` callback
(already restart-proof after the E1 fix), and rendering the final `shopping_response`.

The variant-pick tap already resumes the mission (`_resume_mission_at_step` →
`clarify_choice`), so the chosen/variant paths need no bot change beyond confirming they
no longer touch `request()`.

> **E1 (shipped this iteration, separate from the migration):** the compare-all keyboard
> re-attach used legacy `callback_data="variant_choice:{gid}"` (drops mission/task id →
> restart-fragile). Fixed to `vc:{mission_id}:{task_id}:{gid}` (`telegram_bot.py:11001`).

---

## 6. Cleanup — the literal SP5 unblock

Once all 4 callers are migrated and `shopping_v3` is live-validated:

1. **Switch mission creation** to `shopping_v3` (the shopping launch site that today
   loads `shopping_v2`). Retire `shopping_v2.json` after a validation window.
2. **Delete** `LLMDispatcher.request()` + `_request_kwargs_to_spec` (`llm_dispatcher.py`).
3. **Keep** `_task_result_to_request_response` — it is shared by SP3 grading,
   `posthook_handlers/brand_voice_lint.py`, and `src/tools/vision.py` (SP3b §7 "keep
   stable"). Verify no shopping-only assumption before leaving it; do **not** delete.
4. **Delete** `coulson/single_shot.py` (`single_shot.run`) once `shopping_clarifier` is
   react and no profile sets `execution_pattern=single_shot`. Confirm zero other callers.

After this, `await_inline`'s remaining users are SP4 (tools+mechanicals) and the SP5
carve-outs (#2 `task_classifier`, #6 `investor_bullets`) only — SP5 can proceed to delete
the primitive.

---

## 7. Risks

1. **`llm_dispatch_from` resolver hook (primary).** Must fire at advance time, before
   `next_task` admission, writing `context.llm_call` from the artifact. Wrong timing →
   model selected against empty prompt, or sentinel never fires. Mitigation: locate the
   exact advance→ready transition in `apply.py`/`hooks.py`; add a focused unit test that
   asserts `context.llm_call` is materialized *before* a mocked `fatih_hoca.select` sees
   the task.
2. **Self-iterating compare loop.** A step re-arming its predecessor is new. Verify the
   engine permits cursor-driven re-ready (or use the enqueue-next variant). Risk of
   infinite loop if `cursor` not advanced — guard with `cursor < len(lines)` and a hard
   cap of `MAX_CLARIFY_OPTIONS`.
3. **`shopping_clarifier` → react parity.** Single-shot vs react-loop may change clarify
   output (extra iterations, different prompt framing). Verify the clarify questions and
   the `may_need_clarification` pause behavior are unchanged via a live test.
4. **Conditional group-residual skip.** `skip_when:"!has_residuals"` must evaluate against
   the `has_residuals` flag emitted by 1.1a. Confirm `conditions.py` supports the negation
   form, else emit an explicit boolean and use `skip_when: has_residuals == false`.
5. **Artifact size.** `synth_llm_call` carries the full review-snippet pile (up to
   `_MAX_SNIPPETS_PER_PRODUCT=80` × deep multiplier) inside `messages`. Confirm the
   artifact store handles the payload (it already stored `search_results` candidate
   blobs); no truncation (per the NO-TRUNCATION rule).

---

## 8. Testing

- **Per-step units:** each prepare builds the correct `messages`/`response_format`
  (compare against the current inline-call kwargs); each parse handles a representative
  raw hüsam response (success + malformed-JSON + insufficient_data).
- **Engine-glue test:** a synthetic 3-step prepare→dispatch→parse plan; assert
  `context.llm_call` materialized at advance time, `raw_dispatch` sentinel routes to a
  stubbed `husam.run`, parse consumes the raw result. RED before the hook, GREEN after.
- **Deadlock regression:** drive a real `enqueue → next_task → dispatch` for a shopping
  dispatch step with the oneshot lane saturated; assert it admits without blocking on a
  held slot (the inline path would have deadlocked).
- **Live Telegram missions (founder's standing rule — every increment):** chosen path
  (single line), variant-pick path, compare-all (≥2 lines), and a vague query exercising
  `shopping_clarifier`. Test live after each path migrates; do not rely on pytest alone.
- **Suite discipline:** `timeout 120 pytest` targeted per package; never concurrent runs
  (SQLite lock).

---

## 9. Migration order (for the plan)

1. Engine: `llm_dispatch_from` resolver + tests (substrate; nothing shopping yet).
2. `shopping_v3.json` skeleton + the prepare/parse handler splits (no behavior change vs
   v2 when run through the new steps).
3. Group + label triad (1.1a–e); live-test chosen + clarify.
4. Synth triad (2.0/2.2); live-test chosen + variant-pick.
5. Compare-all sequential loop (2.3a–e) + telegram resume re-route; live-test compare.
6. `shopping_clarifier` → react; live-test vague query.
7. Switch mission creation to v3; validation window; retire v2.
8. Delete `request()` + `_request_kwargs_to_spec` + `single_shot.run`. SP5 unblocked.

Each step: own commit, tests green, live-verified before the next.
