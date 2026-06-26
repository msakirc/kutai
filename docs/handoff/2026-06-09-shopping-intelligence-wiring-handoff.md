# Handoff — Shopping intelligence wiring (the deferred Group 2 session that unblocks SP5)

**Date:** 2026-06-09
**For:** the dedicated shopping-oriented session that wires the intelligence subsystem live, with every LLM call Beckman-admitted.
**From:** the SP5 triage session (Group 1 deleted, Group 2/3 characterized).
**Read first:** memory `project_shopping_sp5_group_findings_20260608`, prior handoff `docs/handoff/2026-06-08-shopping-sp5-remaining-scope-handoff.md` (esp. the "UPDATE 2026-06-08 (triage session 2)" block), memory `project_shopping_sp5_coupling_20260605`.

---

## Founder rulings (binding — do not relitigate)

1. **Every LLM call MUST be Beckman-admitted via husam/coulson.** No `await_inline`. No direct `husam.run()`. No deterministic-strip of the tool capability. ("non-authorized shortcuts not permitted")
2. **Do NOT delete the dead intelligence modules** — "they are meant to be wired, not deleted."
3. This is the session to **wire the intelligence subsystem**. Mid-ReAct CPS loop-hacking is OUT (see why below).

---

## TL;DR — what this session must produce

Re-home the shopping intelligence LLM calls so `LLMDispatcher.request()` has zero live callers, WITHOUT deleting modules and WITHOUT a shortcut. That requires a **brainstorm + spec + plan** for the carrier (the founder explicitly wanted this designed, not improvised). Then implement, then the SP5 sweep deletes `await_inline`.

**SP5 is NOT ready.** Deletion order:
1. **(this session)** wire Group 2 intelligence → admitted producers/steps → `_llm.py request()` reach gone.
2. delete Group 3 (`reflection.self_reflect` :88, `constrained_emit.maybe_apply` :147 — confirmed dead) + `single_shot.py` (+ rewrite 5 test-entangled patches) + `LLMDispatcher.request()` + `_request_kwargs_to_spec` (+ `_task_result_to_request_response` after 0-caller re-grep).
3. **THEN** SP5 deletes `await_inline`.

---

## State of the request() blockers (verified 2026-06-08)

```
Group 1 — DELETED + merged main (−933 LOC, NOT pushed, restart-gated). Dead pre-v3 fused
          handlers + inline helpers gone. 749 shopping tests pass.
Group 2 — src/shopping/intelligence/_llm.py:42  _llm_call → request()   ← THIS SESSION
          DORMANT but still wired. Single chokepoint. Real fix = wire intelligence.
Group 3 — coulson/reflection.py:88, constrained_emit.py:147             ← confirmed DEAD
          (SP3b Task 7 moved both to Beckman posthook children; zero live callers).
          Delete in step 2 above, NOT this session.
single_shot.py — dead, TEST-ENTANGLED (5 tests patch coulson._single_shot_run). Step 2.
```

`request()` cannot be deleted until BOTH Group 2 is re-homed AND Group 3/single_shot are deleted.

---

## Group 2 — the exact surface to wire

**Chokepoint:** `src/shopping/intelligence/_llm.py::_llm_call` → `get_dispatcher().request(OVERHEAD, task="shopping_advisor", ...)`. Every intelligence module funnels through it. Catches all exceptions → returns `""` (graceful rule-based degrade today).

**Live reach (exercised path):** 4 LLM-backed tools handed to the `shopping_advisor` ReAct agent (`src/agents/shopping_advisor.py` allowed_tools), wired in `src/tools/__init__.py:470-691`:
| Tool | module fn | _llm_call site |
|---|---|---|
| `shopping_reviews` | review_synthesizer.synthesize_reviews | :229 |
| `shopping_timing` | timing.advise_timing | :214 |
| `shopping_alternatives` | alternatives.generate_alternatives | :139 |
| `shopping_compare` | delivery_compare.compare_delivery | :284 |
(`shopping_constraints` imports `_llm_call` but never invokes it — rule-based.)

**Dead-but-meant-to-be-wired modules** (zero live caller — relics of the retired v1 ShoppingPipeline / two-tier path; DO NOT delete): `query_analyzer`, `search_planner`, `combo_builder`, `return_analyzer`, `used_market`, `installment_calculator`, intelligence `substitution`/`complementary`.

**Liveness evidence (why mid-ReAct CPS is moot):** `model_pick_log` — `shopping_advisor` OVERHEAD picks (the `_llm_call` path) last fired **2026-04-18**, while the agent ran **116 main_work** picks through 2026-06-07. The agent (gemini-2.5-flash) does not call its LLM-backed tools in practice. Live shopping = the v3 workflow producers (`shopping_grouper`/`labeler` fired 06-08), which DON'T touch `_llm_call`.
```
SELECT call_category,COUNT(*),MAX(timestamp) FROM model_pick_log
WHERE task_name='shopping_advisor' GROUP BY call_category;
-- main_work 116 (last 2026-06-07) | overhead 20 (last 2026-04-18)
```
DB: `C:\Users\sakir\ai\kutai\kutai.db` (open read-only `?mode=ro&immutable=1`; live bot holds WAL — readers OK). Use `PYTHONIOENCODING=utf-8`.

---

## The design problem (brainstorm THIS first — don't improvise)

A mid-ReAct agent tool that needs an LLM sub-call cannot, today, get one in a Beckman-admitted way:
- `await_inline` (current `request()` path) is the blocking primitive SP5 deletes — forbidden as the destination.
- Direct `husam.run()` (what `src/tools/vision.py` does) is "through husam" but NOT Beckman-admitted → **ruling #1 rejects it.** (vision.py is itself non-compliant — fold it into this fix.)
- True mid-ReAct CPS = build suspend/resume into the coulson ReAct loop. The CPS migration explicitly scoped the ReAct loop OUT. Large. AND the path is dormant, so building it for these tools is unjustified.

**Likely-correct shape (validate in brainstorm):** these capabilities are NOT mid-ReAct tools — they should be **admitted producer steps / sibling tasks** (shopping_v3 prep→producer-agent→apply triads), the same pattern the live workflows already use. The dead modules get wired as workflow steps too. This removes the mid-ReAct LLM entirely → no loop hack, fully admitted. But it overlaps the agent-vs-workflow question — that's the brainstorm.

**Carrier options to weigh (founder picks):**
- A) intelligence capabilities → v3-style producer triads (admitted; reuses existing machinery; shopping becomes workflow-driven).
- B) keep agent tools but each LLM-backed tool enqueues an admitted child + agent suspends/resumes via CPS (needs the coulson loop primitive — big).
- C) per-module: some belong as workflow steps, some as agent tools.

Reuse the built triad machinery: producers `shopping_grouper`/`shopping_labeler`/`shopping_synthesizer` (in `src/agents/__init__.py`), `_STEP_HANDLERS_V2` prep/apply handlers in `pipeline_v2.py`, reference `src/workflows/shopping/shopping_v3.json`.

---

## Steps

1. `superpowers:brainstorming` on the carrier (A/B/C) given founder ruling #1 — produce a spec at `docs/superpowers/specs/`.
2. Map every `_llm_call` consumer in `src/shopping/` (13 modules) → its intended live role (workflow step vs agent tool). Decide which dead modules wire where.
3. `superpowers:writing-plans` → implement. Each LLM call = an admitted task → husam/coulson. Keep rule-based fallbacks as `on_error`.
4. Validate: re-route `_llm_call` (or its callers) so `rg "\.request\(" src/shopping` = 0 live. Live-test via Telegram (/shop, /price, /watch, /research_product) — confirm admitted children in DB, none orphaned.
5. Hand back to the SP5 sweep (step 2 of deletion order above).

---

## Build/test discipline (carry over)
- `timeout 120 pytest` always; NEVER concurrent pytest (SQLite WAL lock crash-loops live KutAI); `tests/` and `packages/*/tests/` in SEPARATE invocations.
- Bare `python -c` misses editable packages (fatih_hoca, dogru_mu_samet, husam) — validate via pytest.
- Live KutAI runs from `main`; loads new code on founder's `/restart` (Telegram). NEVER `taskkill`. Group 1 deletion is on main but NOT pushed.
- Work on a **git worktree** — parallel agent sessions cross-sweep `main` (repeated hazard). Avoid `git stash`. Atomic commits.
- Telegram "✅" ticks show for SKIPPED steps too — inspect `tasks.result` / `blackboards.data.artifacts` in DB to tell run-vs-skip.
