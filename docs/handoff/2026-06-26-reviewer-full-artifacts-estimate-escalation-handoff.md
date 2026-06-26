# Handoff — Review steps fetch FULL artifacts; input estimate escalates accordingly

**Date:** 2026-06-26
**Status:** ✅ FIXED — TDD green, regression green. **RESTART-GATED, NOT committed.**
**Trigger:** mission-90 `[1.13] research_quality_review` (task 567399) halted the mission with 7/9 **hallucinated** blockers (charter "lacks solutions", positioning "lacks sections", prior-art "lacks coverage" — all present in the full artifacts).

## Root cause (diagnosed from the real prompt in `tasks.task_state`)
The reviewer was fed **lossy `_summary` stubs** of all 6 input artifacts, then ran 17 line-by-line checks against them and reported as "missing" the content the summaries dropped.
- `coulson/context.py::fetch_deps` prefers `<name>_summary` for every workflow step (general context-saving). The prompt confirmed it: every block labeled `### <name> (summary):`. Charter summary cut after 3/5 solutions, prior-art missing most of 20 entries, positioning missing `## Notes`, market report missing `Recommendations`.
- It was told *"fetch full via `read_blackboard`"* — but step 1.13 has `tools_hint: []` → **no tools** → could not fetch.
- Input estimate `estimated_input_tokens = 3958` (btable, learned from prior summary-fed runs) — far below the ~10k tok of full artifacts, so selection could pick a small-ctx model.
- (Compounding, not fixed here: the reviewer also gets the generic **code-reviewer** system prompt; model was `groq/llama-3.3-70b-versatile` after `cerebras/zai-glm-4.7` failed; single-LLM, no 2nd opinion.)

## Dry run (does full inject become huge?) — NO
6 input artifacts full = **46,594 chars / 10,112 tok**. Full prompt ≈ scaffold (~3.9k) + artifacts (10.1k) = **~14k tok = ~11% of the 128k window** the reviewer's cloud models have.

## Fix (2 parts, both gated on the reviewer role)
1. **`coulson/context.py::fetch_deps`** — when `profile.name == "reviewer"`, fetch the **FULL** artifact (skip the `_summary` preference). Every other agent unchanged.
2. **`fatih_hoca/requirements_builder.py`** — after the base `estimate_for`, for reviewer steps with `input_artifacts`, escalate `estimated_input_tokens` to `min(4000 + Σ full-artifact tokens, 64000)` (new helper `_estimate_full_artifact_tokens` reads the artifact store). `effective_context_needed` then floors model selection on the escalated value (never lands on an 8k local). Only the reviewer fetches full, so only the reviewer escalates.

## End-to-end verification
- Reviewer active context layers (after `apply_heuristics`) = `{deps, board}` → **deps gets 5/7 of the budget**. For a 128k model: deps budget = **93,620 chars ≫ 46,594 full artifacts** → all 6 injected full, **zero truncation**. (64k model: 73k chars, still fits. Only a 34–41k-ctx model drops the single largest artifact with an honest omission note — graceful, far better than all-summaries.)
- **Tests:** 4 new (2 fetch_deps full-vs-summary, 2 estimate escalation) RED→GREEN. Regression: **coulson 141, fatih_hoca 494** passed. Import smoke OK.

## Deploy
Restart-gated. The reviewer (567399) is frozen `waiting_human`. After `/restart`: re-pend/re-run 1.13 → it now reviews full artifacts → the bogus blockers should clear. (To unblock m90 immediately you can also override the halt — the artifacts are already good; see `2026-06-25` review-halt audit.)

## CORRECTION (2026-06-26, after live re-reset) — the budget was NOT enough
The first verification re-run **exposed a deeper root my dry-run missed.** I computed the deps budget against a 128k model_ctx — but prod resolves **`model_ctx = 4096`** for every step, because `BaseAgent._build_context` (and `coulson.context.build_context`) called `self._get_context_window(loaded)` — a **DEAD method that never existed** → the `try` always raised → 4096 default. So `compute_layer_budgets(4096)` gave deps ≈1.2k tok ≈4.7k chars. With full-fetch on, the reviewer's full artifacts were then **truncated** (`_[5 earlier result(s) omitted to fit budget]_`) — it saw *less* than with summaries. (Also: escalation returned 0 silently → weak `gemini-flash-lite` picked → code-reviewer system prompt → garbage → re-pend loop.)

**Real fix (this is what makes "ctx populated via actual deps" true — no static floors):**
- `src/agents/base.py::_build_context` + `packages/coulson/src/coulson/context.py::build_context` — resolve `model_ctx` from the **actual selected model** (`ctx.generating_model` → `coulson.window.context_window_for`, litellm-info → registry → difficulty fallback), cloud OR local. Budget now scales with real capacity (bounded by `CONTEXT_ABS_CAP=32768`; `trim_if_needed` protects the per-call model). For a cloud reviewer: available 32768 → deps 5/7 ≈ 23,405 tok ≈ **93,620 chars ≫ 46,594** → all 6 full, zero truncation.
- `requirements_builder._estimate_full_artifact_tokens` — made observable (`[req-escalation]` logs) + summary-form fallback, so the next run pinpoints why the store-read returned 0.

Tests after correction: coulson 141, fatih 18, context cluster 29 + context_policy 19, targeted 9 — green. **Needs another RESTART** (running orchestrator had old 4096 code), then re-reset 567399.

## Verdict-quality fixes (DONE — `packages/finch/src/finch/profiles/reviewer.yaml` + `packages/fatih_hoca/src/fatih_hoca/requirements.py`)
- **Reviewer system prompt rewritten** — was 100% code-reviewer ("read source, run pytest, git_diff, file_tree"), wrong for the ~15 artifact-review steps (research/requirements/architecture/legal/docs/checklist) and confused weak models into "I will read the files". Now a general WORK/ARTIFACT reviewer that reviews the artifacts already provided in-prompt, only uses tools when the task is code-about AND tools are present, with an **anti-hallucination evidence rule** (every issue MUST quote/cite the artifact; truncated-from-view ≠ missing; never raise generic "lacks a clear analysis" without evidence; prefer pass/needs_minor_fixes over fail). Output schema generic + defers to the task's stated format — verified compatible with `verify_review_verdict` (needs only verdict/status + issues/findings w/ severity + target_artifact/problem; the dropped route_to/file/line aren't required). Tools kept in the profile for the few code/security steps; artifact steps strip them via `tools_hint:[]`.
- **Model capability** — reviewer `difficulty 6→7 + prefer_quality=True` so a mission-halting gate stops picking weak high-ctx models (gemini-flash-lite) and prefers a capable one.
- Tests: prompt_quality + requirements 101, finch 43, sensitivity/estimates 19 — green. Restart-gated.

## Still deferred
- **Single-LLM review** has no 2nd opinion (`verify_review_verdict` mechanical) — add an adversarial/confirm pass so one bad verdict can't halt a mission.
