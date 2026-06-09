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

## Decision (carrier A, mechanism A1)

Re-home via the **producer-triad pattern** that the live workflow already uses. For the dormant mid-ReAct tools, **mechanism A1**: drop the tools rather than build a mid-ReAct CPS primitive. Capability that has live value (review synth) is already served by an admitted producer; the rest are dormant and explicitly deferred.

Scope **excludes** wiring the 7 dead modules into new live workflow phases (founder: "no need to wire them now").

## Changes

### 1. `src/shopping/intelligence/_llm.py::_llm_call` — remove the `request()` body

`_llm_call` becomes a documented **admitted-producer seam**. It no longer imports or calls `get_dispatcher().request()`. It returns `""` so every caller falls through to its existing rule-based path (all 13 modules already catch `""`/exception and degrade — see Appendix A).

Docstring must state:
- This path is intentionally inert; the LLM home for shopping intelligence is an **admitted v3 producer step** (reference `shopping_synthesizer` + the `synth_dispatch` triad in `shopping_v3.json`).
- **Never reintroduce `dispatcher.request()` / `await_inline` / direct `husam.run()` here.**
- To make a capability live, wire it as a producer triad (prep handler → producer agent → apply handler), not as a mid-ReAct call.

The function signature is preserved (callers unchanged), so the 13 modules stay intact (ruling #2). Nothing is deleted.

### 2. Drop the 4 dormant LLM-backed agent tools

Remove from `src/tools/__init__.py` (the `_tool_*` fns + their `_optional_tools[...]` registry entries, ~lines 538–691):
- `shopping_reviews` (`_tool_shopping_reviews`)
- `shopping_compare` (`_tool_shopping_compare`)
- `shopping_timing` (`_tool_shopping_timing`)
- `shopping_alternatives` (`_tool_shopping_alternatives`)

Remove the same four names from `src/agents/shopping_advisor.py::allowed_tools`.

Rationale: review synthesis is served live by the `shopping_synthesizer` producer; the other three are dormant (last fired 2026-04-18) and, once `_llm_call` is inert, would advertise an "LLM synthesis" they no longer perform — dishonest to keep. This removes **dormant non-admitted paths**, not live capability.

### 3. Keep the rule-based-effective tools

`shopping_search`, `shopping_constraints`, `shopping_fetch_reviews`, `shopping_user_profile`, `shopping_price_watch`, `web_search`, blackboard, `api_*` stay on `shopping_advisor`. `shopping_search` wraps `query_analyzer.analyze_query`, which calls `_llm_call` but has a complete `_fallback_analyze()` rule path — it degrades cleanly and keeps working. (It too has been rule-based-effective since 2026-04-18.)

### 4. Keep all 13 intelligence modules (ruling #2)

No module deleted. Each retains its rule-based fallback. Each remains re-homeable as a producer triad when a real live trigger is wanted.

## Ruling #1 compliance argument (the one sensitive spot — for founder veto)

Dropping the four tools + inerting `_llm_call` removes **only dormant, non-admitted LLM paths**:
- The single live LLM-shopping capability — **review synthesis — is preserved**, running admitted through `shopping_synthesizer` in the v3 workflow. Not stripped.
- The other paths have not fired since 2026-04-18; the live agent does not call them. Removing a dead path is not "deterministic-strip of a live capability."
- No capability is silently downgraded to rules **while still claimed as LLM-backed** — the tools that claimed LLM synthesis are removed outright; the modules that stay are honest about their rule-based fallback.

If the founder considers any of the four a capability that *should* be live, the correct response is to **wire it as a producer triad** (a follow-up, out of this scope), not to keep the `request()` path.

## Out of scope (residuals, documented not done)

- **7 dead modules** (`search_planner`, `return_analyzer`, `installment_calculator`, `combo_builder`, `substitution`, `special/complementary`, `special/used_market`): not wired into new live phases. Kept intact, inert LLM path, rule fallback live. Founder deferred.
- **`src/tools/vision.py`** calls `husam.run()` directly (non-compliant with ruling #1). It does **not** touch `request()`, so it does **not** block the SP5 deletion. Tracked as a separate compliance fix; deferred. Documented here so it is not forgotten.

## Acceptance

1. `rg "\.request\(" src/shopping` → **0 matches** (was 1: `_llm.py:42`).
2. `rg "_llm_call" src/shopping` still resolves (helper kept; 13 importers intact) — no `ImportError`.
3. `timeout 120 pytest tests/` for shopping suites green (749 shopping tests from the Group 1 baseline). Tests asserting the four dropped tools exist/synthesize via LLM are updated to assert rule-based behavior or removed if they only covered the dropped tool surface.
4. `shopping_advisor.allowed_tools` no longer lists the four; agent prompt's "Available Shopping Tools" section reconciled (no dangling references to removed tools).
5. Live smoke (founder, post-restart): `/shop`, `/price`, `/watch`, `/research_product` still return results (rule-based + v3 producers); no orphaned children; v3 producers still admitted in DB.

## Hand-back

On green: hand to the **SP5 deletion sweep** (handoff deletion order step 2) — delete Group 3 (`reflection.self_reflect`, `constrained_emit.maybe_apply`), `single_shot.py` (+ 5 test patches), then `LLMDispatcher.request()` + `_request_kwargs_to_spec` (+ `_task_result_to_request_response` after 0-caller re-grep), then SP5 deletes `await_inline`.

## Build/test discipline

- Work on the `shopping-cps-migration` git **worktree** (parallel sessions cross-sweep `main` — repeated hazard). Atomic commits. Avoid `git stash`.
- `timeout 120 pytest` always; **never** concurrent pytest (SQLite WAL lock crash-loops live KutAI); `tests/` and `packages/*/tests/` in separate invocations.
- Bare `python -c` misses editable packages — validate via pytest.
- Live KutAI runs from `main`, loads new code on founder `/restart`. **Never `taskkill`.**

## Appendix A — `_llm_call` callers and their rule fallbacks (verified)

| Module | `_llm_call` site | Has rule fallback on `""`? | Live tool? |
|---|---|---|---|
| `review_synthesizer.synthesize_reviews` | :229 | yes (rating-based) | `shopping_reviews` → DROP (live home = `shopping_synthesizer` producer) |
| `timing.advise_timing` | :214 | yes (calendar/seasonal) | `shopping_timing` → DROP |
| `alternatives.generate_alternatives` | :139 | yes (static maps) | `shopping_alternatives` → DROP |
| `delivery_compare.compare_delivery` | :284 | yes (store defaults) | `shopping_compare` → DROP |
| `query_analyzer.analyze_query` | :227 | yes (`_fallback_analyze`) | `shopping_search` → KEEP (rule-effective) |
| `constraints.check_constraints` | (imports only) | n/a (rule-based) | `shopping_constraints` → KEEP |
| `search_planner.generate_search_plan` | :159 | yes (`_build_rule_based_plan`) | dead — KEEP module |
| `return_analyzer.analyze_return_policy` | :248 | yes (static policies) | dead — KEEP module |
| `installment_calculator.calculate_installments` | (imports only) | yes (default rates) | dead — KEEP module |
| `combo_builder.build_combos` | :270 | yes (rule compat) | dead — KEEP module |
| `substitution.suggest_substitutes` | :157 | yes (static map) | dead — KEEP module |
| `special/complementary.suggest_complements` | :445 | yes (static map) | dead — KEEP module |
| `special/used_market.assess_used_viability` | :417 | yes (rule lookup) | dead — KEEP module |
