# Shopping intelligence CPS migration — retire `LLMDispatcher.request()` reach from `src/shopping`

**Date:** 2026-06-09
**Status:** design / approved-shape
**Predecessors:** `docs/handoff/2026-06-09-shopping-intelligence-wiring-handoff.md`, memory `project_shopping_sp5_group_findings_20260608`, `project_shopping_sp5_coupling_20260605`.
**Successor:** SP5 deletion sweep (deletes `request()` itself + Group 3 + `single_shot.py` + `await_inline`).

---

## Problem

`LLMDispatcher.request()` is a deprecated shopping-only shim. To delete it (and ultimately `await_inline` in the SP5 sweep) it must have **zero callers**. Across all of `src/shopping/` it has exactly **one** caller: `src/shopping/intelligence/_llm.py:42` (`_llm_call → get_dispatcher().request(OVERHEAD, ...)`). Thirteen intelligence modules funnel through that one helper.

Founder rulings (binding, from the handoff):
1. Every LLM call must be Beckman-admitted via husam/coulson. No `await_inline`, no direct `husam.run()`, **no deterministic-strip of a live tool capability**.
2. Do **not** delete the intelligence modules — "they are meant to be wired, not deleted."
3. Mid-ReAct CPS (suspend/resume in the coulson loop) is OUT — large and unjustified for a dormant path.

## Key liveness fact (decides the whole design)

Every `_llm_call` invocation dispatches as `task="shopping_advisor"`, `CallCategory.OVERHEAD`. Per `model_pick_log`, the `shopping_advisor`/OVERHEAD bucket **last fired 2026-04-18**, while the `shopping_advisor` agent itself ran 116 `main_work` picks through 2026-06-07. The live shopping path is the **v3 producer-triad workflow** (`shopping_grouper`/`labeler`/`synthesizer`, fired 2026-06-08) — which never touches `_llm_call`.

Conclusion: **the entire `_llm_call` / `request()` path is dormant.** Removing it strips no live capability. The one live LLM-shopping capability (review synthesis) already runs admitted as the `shopping_synthesizer` producer step in `shopping_v3.json`.

```sql
SELECT call_category, COUNT(*), MAX(timestamp) FROM model_pick_log
WHERE task_name='shopping_advisor' GROUP BY call_category;
-- main_work 116 (last 2026-06-07) | overhead 20 (last 2026-04-18)
```
DB: `C:\Users\sakir\ai\kutai\kutai.db` (read-only `?mode=ro&immutable=1`; live bot holds WAL — readers OK; `PYTHONIOENCODING=utf-8`).

## Decision (carrier A; minimal — inert seam only)

The live LLM-shopping capability already runs admitted via the **producer-triad pattern** (`shopping_synthesizer`). No new primitive is built and **no mid-ReAct LLM is admitted** — the dormant `_llm_call` path is simply made inert. All callers fall back to their existing rule paths.

> An earlier draft also dropped 4 dormant LLM agent tools ("mechanism A1"). Adversarial review showed those tools are live in `deal_analyst`/`product_researcher` → dropping = a ruling-#1 strip. **Dropping is rejected** (see change #2). Inerting `_llm_call` alone is sufficient and touches nothing else.

Scope **excludes** wiring the 7 dead modules into new live workflow phases (founder: "no need to wire them now").

## Changes

### 1. `src/shopping/intelligence/_llm.py::_llm_call` — remove the `request()` body

`_llm_call` becomes a documented **admitted-producer seam**. It no longer imports or calls `get_dispatcher().request()`. It returns `""` so every caller falls through to its existing rule-based path (all 13 modules already catch `""`/exception and degrade — see Appendix A).

Docstring must state:
- This path is intentionally inert; the LLM home for shopping intelligence is an **admitted v3 producer step** (reference `shopping_synthesizer` + the `synth_dispatch` triad in `shopping_v3.json`).
- **Never reintroduce `dispatcher.request()` / `await_inline` / direct `husam.run()` here.**
- To make a capability live, wire it as a producer triad (prep handler → producer agent → apply handler), not as a mid-ReAct call.

The function signature is preserved (callers unchanged), so the 13 modules stay intact (ruling #2). Nothing is deleted.

### 2. Touch NO agent tool lists, drop NO tools — `_llm_call` (change #1) alone is sufficient

**This is the only change needed.** Inerting `_llm_call` removes the single `request()` caller in `src/shopping` (acceptance #1). No agent, tool registry, or `allowed_tools` is edited.

> **Rejected (was A1 "drop tools"):** an earlier draft dropped `shopping_reviews`/`shopping_compare`/`shopping_timing`/`shopping_alternatives` from the registry + `shopping_advisor.allowed_tools`. Adversarial review found this **violates ruling #1.** The dormancy evidence (2026-04-18) covers only the `shopping_advisor` OVERHEAD/`_llm_call` path — it says nothing about the other two agents. The four tools are also in **live** agents `deal_analyst` (`src/agents/deal_analyst.py:27-30`, prompt `:49,:51`) and `product_researcher` (`src/agents/product_researcher.py:28-29`), reachable via `/compare` (`telegram_bot.py:8869` → `combo_research.json`) plus `exploration`/`gift_recommendation`/`price_watch`. Tool resolution (`coulson/context.py:62-66`) intersects `allowed_tools` against the registry by name, so dropping a registry entry **silently strips it from those live agents** — exactly the "deterministic-strip of a live tool capability" ruling #1 forbids. Dropping is therefore out.

After change #1, those four tools keep working — they call `_llm_call`, get `""`, and return their **rule-based** result. That is honest rule-based behavior (the tool still does its job), not a false "LLM-backed" claim, and not a strip (the capability — review/compare/timing/alternatives output — is still produced).

### 3. The rule-based-effective tools (unchanged, noted)

`shopping_search` (wraps `query_analyzer.analyze_query` → `_fallback_analyze()` on `""`), `shopping_constraints` (rule-based; imports `_llm_call` but never invokes it), `shopping_fetch_reviews`, `shopping_user_profile`, `shopping_price_watch` all degrade cleanly and keep working.

### 4. Keep all 13 intelligence modules (ruling #2)

No module deleted. Each retains its rule-based fallback (verified — Appendix A). Each remains re-homeable as a producer triad when a real live trigger is wanted.

## Ruling #1 compliance argument

After the correction (change #2 = no tool drops), the migration touches **exactly one inert, dormant code path** — `_llm_call`'s `request()` body:
- The single live LLM-shopping capability — **review synthesis — is preserved**, running admitted through `shopping_synthesizer` in the v3 workflow. Not stripped.
- All 13 modules and all agent tool lists are **untouched**. Every tool that called `_llm_call` continues to return a result (now always rule-based). No capability is removed; the LLM *enrichment* of a path that has not fired since 2026-04-18 is what goes inert.
- Nothing is silently downgraded **while still claimed as LLM-backed** to the user — these are tool outputs, not user-facing "powered by LLM" claims; and the modules' fallbacks are their designed degrade path.

If the founder wants any of these capabilities to use an LLM **live**, the correct response is to **wire it as a producer triad** (a follow-up, out of this scope) — never to reintroduce the `request()` path.

## Out of scope (residuals, documented not done)

- **7 dead modules** (`search_planner`, `return_analyzer`, `installment_calculator`, `combo_builder`, `substitution`, `special/complementary`, `special/used_market`): not wired into new live phases. Kept intact, inert LLM path, rule fallback live. Founder deferred.
- **`src/tools/vision.py`** calls `husam.run()` directly (non-compliant with ruling #1). It does **not** touch `request()`, so it does **not** block the SP5 deletion. Tracked as a separate compliance fix; deferred. Documented here so it is not forgotten.

## Acceptance

1. `rg "\.request\(" src/shopping` → **0 matches** (was 1: `_llm.py:42`).
2. `rg "_llm_call" src/shopping` still resolves (helper kept; 13 importers intact) — no `ImportError`. `_llm_call` no longer imports `get_dispatcher`/`CallCategory`.
3. `timeout 120 pytest tests/` shopping suites green. **No tool is dropped, so no tool-existence test breaks** — `tests/test_shopping_tools.py::TestShoppingCompareSerialization` (reads `_optional_tools["shopping_compare"]`) stays green. The `_llm_call`-mocking tests (`tests/shopping/test_intelligence.py`, `test_output_quality.py:817`, `test_scenarios.py`) patch the module-level symbol, so inerting the *body* is invisible to them — they keep passing untouched. No agent `allowed_tools` changes → no agent test churn.
4. No agent tool list or prompt is edited (change #2 = no drops). Nothing to reconcile.
5. Live smoke (founder, post-restart): `/shop`, `/price`, `/watch`, `/compare`, `/research_product` still return results (rule-based + v3 producers); no orphaned children; v3 producers still admitted in DB.

## Hand-back

On green: hand to the **SP5 deletion sweep** (handoff deletion order step 2) — delete Group 3 (`reflection.self_reflect`, `constrained_emit.maybe_apply`), `single_shot.py` (+ 5 test patches), then `LLMDispatcher.request()` + `_request_kwargs_to_spec` (+ `_task_result_to_request_response` after 0-caller re-grep), then SP5 deletes `await_inline`.

## Build/test discipline

- Work on the `shopping-cps-migration` git **worktree** (parallel sessions cross-sweep `main` — repeated hazard). Atomic commits. Avoid `git stash`.
- `timeout 120 pytest` always; **never** concurrent pytest (SQLite WAL lock crash-loops live KutAI); `tests/` and `packages/*/tests/` in separate invocations.
- Bare `python -c` misses editable packages — validate via pytest.
- Live KutAI runs from `main`, loads new code on founder `/restart`. **Never `taskkill`.**

## Appendix A — `_llm_call` callers and their rule fallbacks (verified)

All sites verified by adversarial review: each guards the parse (`if response:` / `if not response: return ...`) so `""` cleanly degrades — no unguarded `json.loads("")`. No module is dropped; tools listed for context only.

| Module | `_llm_call` site | Guard + rule fallback on `""` | Tool (all KEPT) |
|---|---|---|---|
| `review_synthesizer.synthesize_reviews` | :229 | `if llm_response:` (:234) → rating-based sentiment | `shopping_reviews` (live LLM home = `shopping_synthesizer` producer) |
| `timing.advise_timing` | :214 | `if llm_response:` (:218) → neutral/calendar aggregate | `shopping_timing` |
| `alternatives.generate_alternatives` | :139 | `if not response: return []` (:140) → static maps | `shopping_alternatives` |
| `delivery_compare.compare_delivery` | :284 | `if llm_resp:` (:289) → store defaults | `shopping_compare` |
| `query_analyzer.analyze_query` | :227 | `if llm_response:` (:233) + `except` → `_fallback_analyze()` (:275) | `shopping_search` |
| `constraints.check_constraints` | (imports only) | rule-based; never calls `_llm_call` | `shopping_constraints` |
| `search_planner.generate_search_plan` | :159 | `if not response: return None` (:164) → `_build_rule_based_plan` | dead — module kept |
| `return_analyzer.analyze_return_policy` | :248 | `if llm_resp:` (:252) → static policies | dead — module kept |
| `installment_calculator.calculate_installments` | (imports only) | default rates; never calls `_llm_call` | dead — module kept |
| `combo_builder.build_combos` | :270 | `if llm_resp:` (:274) → rule compat | dead — module kept |
| `substitution.suggest_substitutes` | :157 | `if not response: return []` (:158) → static map | dead — module kept |
| `special/complementary.suggest_complements` | :445 | `if not raw: return []` (:446) → **empty list** | dead — module kept |
| `special/used_market.assess_used_viability` | :417 | `raw.find("{")` guard + `try/except` (:417-426) → rule lookup | dead — module kept |
