# Next Session Prompt — Intelligence Modules

Paste this to start the next session:

---

Read `docs/superpowers/specs/2026-04-15-intelligence-modules.md` — it's a full survey of 17 shopping intelligence modules that exist in `src/shopping/intelligence/` but aren't wired into the pipeline.

The shopping pipeline currently does: search → relevance filter → product matcher → format. It returns raw scraper data with no scoring, no review synthesis, no delivery comparison. The format step picks the winner by taking each site's #1 result (trusting site ranking), which works but is crude.

**Your task**: Wire the intelligence modules into the pipeline, starting with the highest-impact pure-Python ones for quick_search (value_scorer, installment_calculator), then the LLM-requiring ones for the full shopping workflow.

Priority order:
1. `value_scorer.score_products()` in `_step_search` — score products after matching, include scores in output
2. Update `_step_format` to rank by value_score instead of price, and use `summary.format_recommendation_summary()` for richer output
3. `installment_calculator.calculate_installments()` in format step for winner product
4. Community data relevance filter (Şikayetvar returns garbage — prison complaints for coffee machine queries)
5. Wire `delivery_compare`, `review_synthesizer`, `timing` into the full shopping workflow steps

Key constraint: quick_search must stay zero-LLM and fast (< 10s). Only pure Python modules there. LLM modules go in the full "shopping" workflow (steps 2.1-4.1).

Start by reading the spec and the current pipeline code (`src/workflows/shopping/pipeline.py`), then implement in priority order.
