# Shopping → Workflow-Step Migration (retire `request()`, unblock CPS SP5)

**Date:** 2026-06-05
**Status:** Design — REVISED to align with SP4 carrier patterns (rev2)
**Author:** founder + Claude (brainstorm)
**Aligns with:** `docs/handoff/2026-06-05-shopping-llm-dispatch-handoff.md` (the carrier decision tree written for this session), `docs/superpowers/specs/2026-05-29-cps-sp3b-design.md` §6/§7, `docs/handoff/2026-06-05-cps-sp4b-kickoff.md` (the producer-step split pattern), memories [[no-direct-dispatcher-from-mechanical]], [[feedback_singular_dispatcher_caller]].

> **rev2 change:** rev1 invented a new `llm_dispatch_from` engine mechanism
> (prepare→raw_dispatch-from-artifact→parse with a custom resolver hooked at
> `hooks.py:1480`). That was **non-uniform** and collided with the just-landed
> `materialize_produces` at the same hook. rev2 replaces it with the **established
> producer-agent-step carrier** (SP4b `demo_storyboard` shape): each LLM hop is an
> admitted `agent:<type>` step whose prompt lives in the agent profile and whose data
> comes from a prior step's artifact via the agent's context builder. **No engine
> feature is added.** The deterministic prep/apply stay as `shopping_pipeline_v2`
> handler steps.

---

## 1. Problem

The shopping pipeline makes LLM calls **inline inside Beckman task handlers** via the
deprecated `LLMDispatcher.request()` shim, which routes through
`general_beckman.enqueue(spec, await_inline=True)` — the blocking inline primitive.

Two consequences:

1. **Live deadlock risk.** A shopping step-task holds a `oneshot` lane slot
   (`ONESHOT_CONCURRENCY=4`) while its inline `request()` blocks awaiting a child task
   that needs *another* oneshot slot. Under load the lane exhausts → slot-deadlock →
   `INLINE_TIMEOUT` → DLQ. CPS SP3b §6 names shopping's deadlock an explicit, accepted
   deferral.
2. **SP5 is blocked.** CPS SP5 deletes `await_inline`/`request()`. Per SP3b §7,
   `request()` is retained *only* as a shopping-only shim and "dies when shopping
   migrates — **before** SP5 deletes `await_inline`." Shopping migration is a hard SP5
   prerequisite.

### The one rule (from the shopping-llm-dispatch handoff)

**No code calls `request()` / the dispatcher / husam directly. Every LLM hop is an
admitted Beckman task.** The carrier is chosen by job-shape:

- **multi-step / fan-out pipeline → WORKFLOW STEP** (`agent:<type>` step; engine
  sequences via `depends_on` + artifacts; prompt in the agent profile). ← shopping_v2 is
  this.
- single interactive hop (user waiting) → interactive CPS resume / husam-inline.
- single fire-and-surface-later hop → CPS continuation → mechanical sink.

The shopping_v2 workflow is unambiguously the **workflow-step** carrier. This spec
migrates it accordingly.

### `request()` call sites — full kill-list (5 live + 2 dead)

| # | Site | Status | Handled by |
|---|------|--------|-----------|
| 1 | `single_shot.run` (`coulson/single_shot.py:46`), reached only by `shopping_clarifier` | LIVE | §4 step 0.2 → react |
| 2 | `_grouping_llm_call` (`pipeline_v2.py:363`) | LIVE | §4 group producer agent |
| 3 | `_synthesis_llm_call` (`pipeline_v2.py:487`) | LIVE | §4 synth producer agent |
| 4 | `_label_llm_call` (`labels.py:22`) | LIVE | §4 label producer agent |
| 5 | `src/shopping/intelligence/_llm.py:42` | LIVE | **OUT OF SCOPE — §7 separate spec** |
| 6 | `coulson/reflection.py:89` (self_reflect) | DEAD-LEGACY | §6 delete outright |
| 7 | `constrained_emit.py:147` | DEAD-LEGACY | §6 delete outright |

`single_shot.run` is reached **only** by `shopping_clarifier` (the only live profile
setting `execution_pattern=single_shot`; `src/agents/shopping_clarifier.py:21`).
`#6`/`#7` are superseded by CPS post-hook children (dispatcher docstring confirms) — zero
prod callers.

---

## 2. Goal & Non-Goals

**Goal.** Migrate the `src/workflows/shopping/` (`shopping_v2`) pipeline so every LLM call
is an admitted `agent:<type>` workflow step — no `await_inline`, deadlock-free by
construction. Retire `request()` (sites #1–#4, #6, #7). Unblock SP5.

**Non-goals:**
- **`src/shopping/` ShoppingPipeline + `intelligence/_llm.py` (site #5).** A different,
  25.9k-LOC subsystem; per the handoff it "gets its own brainstorm + spec." Tracked as a
  §7 follow-up. SP5 cannot delete `await_inline` until that spec also lands — both
  shopping subsystems must migrate.
- The 3 absent intelligence layers (cross-line verdict / spec matrix / price
  intelligence). Separate iteration.
- Grading. Shopping steps keep `requires_grading:false`.
- Shopping's bucketing / synthesis / review-quality logic. The deterministic functions
  are **reused verbatim**; only the LLM-call boundary moves.

---

## 3. The carrier: producer agent steps (no engine change)

Each inline LLM call becomes a **triad** of workflow steps:

| Role | Agent | Determinism | Job |
|------|-------|-------------|-----|
| **prep** | `shopping_pipeline_v2` | deterministic | gather data the prompt needs (candidate `view` JSON / scraped snippets) → emit `<x>_input` artifact |
| **producer** | `agent:<shopping_*>` | LLM (react, one pass) | system prompt = the template; reads `<x>_input` as context; emits JSON `final_answer` → `<x>_raw` artifact |
| **apply** | `shopping_pipeline_v2` | deterministic | read `<x>_raw`, parse JSON, build domain objects → emit domain artifact |

**Why this is uniform.** It is exactly the SP4b `demo_storyboard` split (producer
`agent:` step → mechanical/deterministic sibling via `depends_on` + artifact passing).
The engine already: stores an agent step's `final_answer` under its `output_artifacts`
name (`hooks.py:1253-1480`, `store.store`), and feeds it to the next step's
`input_artifacts`. **No new engine mechanism.** `single_shot` dies because the producers
are react agents (the SP-wide direction).

### Three new producer agent profiles

New `BaseAgent` subclasses in `src/agents/`, registered in the agent registry +
classifier coverage. Each is **prompt-only** (no tools, no subtasks), `execution_pattern`
defaults to `react_loop` (one pass to `final_answer`). Each prompt is lifted verbatim
from the current `GROUPING_PROMPT` / `LABEL_PROMPT` / `SYNTHESIS_PROMPT`
(`prompts_v2.py`) into the agent's `get_system_prompt`, adjusted to satisfy the 3 prompt
invariants (`tests/agents/test_prompt_quality.py`: first line `You are …`; body has
must/always + don't/never; body contains `final_answer` + a fenced ` ```json ` schema).

| Agent | `name` | difficulty | selection hints | Input artifact (its context) | Emits |
|-------|--------|-----------|-----------------|------------------------------|-------|
| Grouper | `shopping_grouper` | 3 | `needs_thinking=False` | `group_input` (candidate `view` JSON) | `groups` JSON |
| Labeler | `shopping_labeler` | 4 | — | `label_input` (group `view` JSON + query) | `groups` taxonomy JSON |
| Synthesizer | `shopping_synthesizer` | 6 | `needs_thinking=False`, `min_context=max(8192, est+2048)`, `estimated_output_tokens=1200` | `synth_input` (representative_title + snippet pile) | aspects/praise/… JSON |

The selection hints (`difficulty`, `needs_thinking`, `min_context`) move from the
inline `request()` kwargs onto the agent profile / step (`difficulty` per-step;
`needs_thinking`/`min_context` via the agent's requirements). The synthesizer's
`min_context` must still floor on the real prompt size — the prep step computes
`est = len(prompt)//3` and the producer step carries `difficulty:"hard"` + a
`min_context` hint so admission's `fatih_hoca.select` floors correctly (the prep artifact
is available before the producer is admitted, since prep completes first).

> **The data-in-context detail.** The agent's user message is the `<x>_input` artifact
> (built deterministically in prep), not a static step instruction — because the grouping/
> synthesis prompts embed runtime-scraped data (candidates, ≤80 review snippets) that only
> exists after a prior step. The agent's context builder reads `input_artifacts`; prep
> writes the exact `view`/snippet JSON the template expects. The template's static rules +
> output schema live in `get_system_prompt`.

---

## 4. `shopping_v3.json` — step layout

New plan file `src/workflows/shopping/shopping_v3.json` (`plan_id: "shopping_v3"`).
Clean break (7 → ~18 steps); v2 stays runnable until v3 is live-validated, then retires.
The loader resolves `shopping_v3` → `src/workflows/shopping/shopping_v3.json`
automatically (suffix-strip, `loader.py:100`). One launch site flips
(`telegram_bot.py:8490-8498`, §6).

### Phase 0 — Understand
- **0.1 understand_query_check_clarity** — `shopping_pipeline_v2`, unchanged.
- **0.2 ask_clarifying_questions** — `shopping_clarifier`, **drop
  `execution_pattern=single_shot`** so it routes via `coulson.execute` → react (kills
  `request()` caller #1). The clarifier already emits
  `{"action":"needs_clarification"|"final_answer"}` JSON that react's `parse_action`
  consumes identically; the `may_need_clarification` pause is unchanged. UX parity
  verified live (§7 risk).

### Phase 1 — Resolve
- **1.0 resolve_candidates** — `shopping_pipeline_v2`, unchanged (scrapers, no LLM).

### Phase 2 — Group / Label / Gate (was fused 1.1)
- **1.1a group_prep** — `shopping_pipeline_v2`: deterministic SKU-bucketing (`step_group`
  body). If residual sku-less candidates remain, build the grouping `view`
  (`[{index,title,site,price,sku,category_path}]`) → emit `group_input` + a
  `has_residuals` boolean; else emit the bucketed `groups_state` and mark no residuals.
- **1.1b group_producer** — `agent:shopping_grouper`, input `group_input` → `group_raw`.
  `skip_when: has_residuals == 'false'` (see §7 risk 4 on the skip operator).
- **1.1c group_apply_label_prep** — `shopping_pipeline_v2`: parse `group_raw` into
  residual `ProductGroup`s (the `_llm_group_residuals` parse body, with
  `_per_site_top1_fallback` on parse error), merge with the bucketed groups, then build
  the label `view` (`[{group_id,title,category_path,member_count}]` + query) → emit
  `label_input` + `groups_state`.
- **1.1d label_producer** — `agent:shopping_labeler`, input `label_input` → `label_raw`.
- **1.1e label_apply_filter_gate** — `shopping_pipeline_v2`: apply labels to groups (the
  `step_label` parse body, `_fallback_labels` on error), `step_filter`,
  `step_variant_gate` → `gate_result` (shape unchanged: `chosen|clarify|escalation`).

### Phase 2 — Synth paths
`step_synthesize_reviews` (1 LLM call + scrape) → prep/producer/apply. The scrape
(`_fetch_community_reviews` when `deep_scrape`, commerce snippets otherwise) is
deterministic I/O → **prep**.

**Chosen path (was 2.0):**
- **2.0a synth_prep** — `shopping_pipeline_v2`: resolve chosen group, gather snippets
  (cap `_MAX_SNIPPETS_PER_PRODUCT*2`), build `SYNTHESIS_PROMPT` data → `synth_input`
  (carries `representative_title` + snippet JSON + computed `est_input_tokens`).
  `skip_when: gate.kind != 'chosen'`.
- **2.0b synth_producer** — `agent:shopping_synthesizer`, input `synth_input` →
  `synth_raw`.
- **2.0c synth_apply** — `shopping_pipeline_v2`: parse `synth_raw` → `ReviewSynthesis`
  (the `step_synthesize_reviews` parse body, `_insufficient()` on error) →
  `format_group_card` → `synth_result`.

**Variant-pick path (was 2.2):** same triad gated `clarify_choice.kind == 'variant'`,
`deep_scrape=true` (prep resolves the group from `clarify_payloads[gid]` and taps
community sources). Steps **2.2a/2.2b/2.2c** (prep/apply handlers shared with chosen path,
parameterized by group source).

- **2.1 clarify_variant** — `mechanical`, unchanged (waits for tap → `clarify_choice`).

**Compare-all path (was 2.3) — sequential self-iterating (≤5 lines,
`MAX_CLARIFY_OPTIONS=5`):**
- **2.3a compare_init** — `shopping_pipeline_v2`: list lines from `clarify_payloads`,
  render `step_compare_all` header, init `compare_state={lines,cursor:0,cards:[]}`.
  `skip_when: clarify_choice.kind != 'compare_all'`.
- **2.3b compare_synth_prep** — `shopping_pipeline_v2`: build `synth_input` for
  `lines[cursor]`.
- **2.3c compare_synth_producer** — `agent:shopping_synthesizer` → `synth_raw`.
- **2.3d compare_synth_apply_loop** — `shopping_pipeline_v2`: parse → card, append,
  `cursor++`. If `cursor < len(lines)`: re-arm 2.3b; else → 2.3e.
- **2.3e compare_assemble** — `shopping_pipeline_v2`: header + cards → `shopping_response`.

The loop (2.3b→c→d→b) is self-iterating; each synth is its own admitted producer step →
no slot nesting. Loop-back mechanism chosen in the plan: cursor-driven `loop_while` on
2.3d, or 2.3d re-enqueues the next prep via the normal admitted path (no `await_inline`).
Hard cap `MAX_CLARIFY_OPTIONS` guards runaway.

### Phase 3 — Deliver
- **3.0 format_response** — `shopping_pipeline_v2`, unchanged (`synth_result` →
  `shopping_response`). `skip_when: synth_result.cards empty`.

---

## 5. Telegram path

Two compare-all paths exist today: workflow step 2.3 **and** a direct bot call
(`telegram_bot.py:_run_compare_all_and_reply` → `_handler_format_compare`, bypassing the
engine and using `await_inline` in the bot loop).

**Change:** route the compare-all tap **through workflow resume** (resume the mission so
steps 2.3a–e run in the pump). Delete the direct `_handler_format_compare` invocation from
`_run_compare_all_and_reply`. The bot keeps: variant keyboard, the restart-proof
`vc:{mission_id}:{task_id}:{choice}` callback parse, and rendering the final
`shopping_response`. The variant-pick tap already resumes the mission
(`_resume_mission_at_step` → `clarify_choice`); confirm it no longer touches `request()`.

> **E1 (shipped this iteration, commit `23e98554`):** the compare-all keyboard re-attach
> used legacy `callback_data="variant_choice:{gid}"` (restart-fragile). Fixed to
> `vc:{mission_id}:{task_id}:{gid}`.

---

## 6. Cleanup — the literal SP5 unblock

After all in-scope callers migrate and `shopping_v3` is live-validated:

1. **Switch mission creation** to `shopping_v3` at `telegram_bot.py:8490-8498` (3 string
   flips: the `deep_research`/`research` map values + the default). Repoint
   `quick_search_v2.json:8` `escalation_target` → `shopping_v3`. Update
   `tests/shopping/test_workflow_json.py:5` + `tests/integration/test_e2e_llm_pipeline.py:142`.
   Retire `shopping_v2.json` after a validation window.
2. **Delete** the dead-legacy callers #6 (`reflection.py:89`) and #7
   (`constrained_emit.py:147`) outright (zero prod path).
3. **Delete** `coulson/single_shot.py` (`single_shot.run`) once `shopping_clarifier` is
   react and no profile sets `execution_pattern=single_shot`. Confirm zero other callers.
4. **Delete** `LLMDispatcher.request()` + `_request_kwargs_to_spec` (request-only).
   `_task_result_to_request_response` is **no longer shared on main** — vision /
   brand_voice_lint already migrated off it in SP4a; `request()` is its only live caller
   today. So it can be deleted too — but **re-grep `_task_result_to_request_response` at
   migration time** (grading/SP3 may re-introduce a dependency on the branch base) before
   removing.

> **SP5 gating:** SP5 deletes `await_inline` only after **both** shopping subsystems
> migrate — this spec (`src/workflows/shopping/`) **and** the §7 follow-up
> (`src/shopping/`). After both, `await_inline`'s remaining users are SP4b producers
> (in flight) + the SP5 carve-outs (`task_classifier`, `investor_bullets`).

---

## 7. Out of scope — `src/shopping/` (separate spec)

`src/shopping/intelligence/_llm.py:42` is a `request()` caller in the **other** shopping
subsystem (`ShoppingPipeline`, the simple-query / two-tier path). Per the handoff it
warrants its own brainstorm + spec: its carrier may be **interactive CPS resume /
husam-inline** (a user waiting synchronously in Telegram), NOT a workflow step — the
opposite of this spec's path. Do not fold it in; do not reuse a workflow-step or async
sink for a synchronous product answer. Flag it as the second SP5 prerequisite.

---

## 8. Risks

1. **Producer agent prompt quality.** The 3 new profiles must pass
   `tests/agents/test_prompt_quality.py` (3 invariants) and be covered by the classifier
   + its cluster disambiguation (CLAUDE.md). Lifting `GROUPING_PROMPT` etc. verbatim
   fails the `You are …` / must+never invariants — the prompts need light reshaping
   without changing their JSON-output contract. Mitigation: unit-test the reshaped prompt
   still elicits the same JSON shape (golden-output test against a stub).
2. **React single-pass vs single_shot.** Producers run through react (`coulson.react`),
   which adds the tool-loop/finish-gate machinery for what is one structured emit. Verify
   a no-tools react agent emits `final_answer` in one iteration (planner/classifier
   precedent) and that the JSON survives react's envelope-unwrap (`hooks.py` final_answer
   handling). Same applies to `shopping_clarifier`'s react conversion (UX parity on the
   `needs_clarification` path — live test).
3. **Agent context = data artifact.** Confirm the producer's context builder surfaces the
   `<x>_input` artifact as the user message (the data-heavy JSON). If `_build_context`
   truncates or reformats large artifacts, the snippet pile could be cut (NO-TRUNCATION
   rule). Mitigation: test the synth producer receives the full ≤80-snippet pile.
4. **`skip_when` operator.** `should_skip_workflow_step` (`hooks.py:1047`) supports only
   `<artifact>.<path> == '<literal>'` / `!= '<literal>'` — **no `!field` / `== false`**.
   For `has_residuals`, store the string `"false"`/`"true"` and use
   `group_input.has_residuals == 'false'`, OR extend the evaluator's regex to accept the
   boolean form (port `conditions.py:120-126`). Decide in plan.
5. **Selection floor for synth.** Admission's `fatih_hoca.select` reads token estimates
   from context (`general_beckman.next_task`). The synth producer step must carry the
   `min_context`/`difficulty` hints (set on the step + agent profile) so a long snippet
   pile isn't routed to an 8K-ctx model. Test: a large `synth_input` admits with
   `min_context ≥ est+2048`.
6. **Materializer interaction.** The post-execute hook (`hooks.py`) just gained
   `materialize_produces` (live, currently under error triage). Shopping agent steps emit
   plain JSON artifacts (no declared file `produces`), so `materialize_produces` is a
   no-op for them — but confirm shopping steps declare no `.md`/`.json` *file* produces
   that would route them through the materializer.

---

## 9. Testing

- **Per-step units:** each prep builds the correct `<x>_input` (compare against current
  inline-call message data); each apply handles a representative producer `final_answer`
  (success + malformed-JSON → fallback + insufficient_data).
- **Producer prompt golden test:** each new agent prompt, fed a fixture input, elicits the
  expected JSON shape from a stub model; passes `test_prompt_quality`.
- **Agent-step wiring test:** a synthetic prep→`agent:shopping_grouper`→apply mini-plan;
  assert the agent's `final_answer` lands as `group_raw` and the apply step consumes it.
- **Deadlock regression:** drive a real `enqueue → next_task → dispatch` for a producer
  step with the oneshot lane saturated; assert it admits without blocking on a held slot.
- **Live Telegram missions (founder's standing rule — every increment):** chosen path,
  variant-pick path, compare-all (≥2 lines), vague query (`shopping_clarifier` react).
  Test live after each path migrates.
- **Suite discipline:** `timeout 120 pytest` targeted; never concurrent runs (SQLite
  lock); `tests/` and `packages/*/tests/` in **separate** invocations (colliding conftest
  roots).

---

## 10. Migration order (for the plan)

1. Three producer agent profiles (`shopping_grouper`/`labeler`/`synthesizer`) + prompt-
   quality + classifier coverage + golden tests. No wiring yet.
2. `shopping_v3.json` skeleton + the prep/apply handler splits (behavior-equivalent to v2
   when run through the new steps).
3. Group + label triad (1.1a–e); live-test chosen + clarify.
4. Synth triad (2.0/2.2); live-test chosen + variant-pick.
5. Compare-all sequential loop (2.3a–e) + telegram resume re-route; live-test compare.
6. `shopping_clarifier` → react (drop `single_shot`); live-test vague query.
7. Switch mission creation to v3; validation window; retire v2.
8. Delete dead callers #6/#7, `single_shot.run`, `request()` + `_request_kwargs_to_spec`
   (+ `_task_result_to_request_response` after re-grep). This spec's part of SP5 unblock
   done; `src/shopping/` (§7) remains before SP5 deletes `await_inline`.

Each step: own commit, tests green, live-verified before the next.
