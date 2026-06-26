# Handoff — remaining mission-87 failures (content/quality + routing class)

**Date:** 2026-06-21
**Context:** Overnight run of mission 87 ("improved todo app → habit builder"). The **availability** failures ("No model candidates", rate-limit storms, capability-404) were diagnosed and fixed in the same session (see "Already fixed" below). What remains is a **different class**: tasks that DO get a model and execute, then fail on **artifact content / schema gates** or on **reviewer/graph routing**. These need a separate investigation.

> Do NOT re-chase availability/selection. After the 4 fixes below + `/restart`, tasks reach execution. The remaining errors are produced AFTER a successful model call.

---

## Already fixed this session (local `main`, restart-gated, NOT pushed)
| commit | fix |
|---|---|
| `8f7baede` | react checkpoint no longer freezes `reqs` (selection deadlock) |
| `d2038c67` | leaked control tokens (`<longcat_*>`) → degenerate → re-select |
| `2205f969` | OpenRouter `free-models-per-day` → `daily_exhausted` (kills 2-min rate storm) |
| `4680ef9a` | capability-404 `no endpoints found that support tool use` → permanent (kills 4h `1.0c` loop) |

Deploy order: `/restart` (loads all four) → re-pend mission 87 → then assess the failures below on FRESH attempts.

---

## Remaining failures to investigate

### A. Schema-validation / shape-gate failures (DOMINANT remaining class)
The agent returns a final answer; the deterministic shape/schema gate (`mr_roboto/verify_*_shape.py`, `general_beckman/apply.py` schema validation) rejects it. Repeated across many i2p steps:

| Task | Step | Exact gate error |
|---|---|---|
| 524360 (DLQ) | `0.6a.draft` non_goals_draft | `'non_goals' missing sections: ['Non-goals']` |
| — | `1.4a` competitive_positioning_lock | `missing sections: ['Landscape','Value Thesis','Strengths','Our Differentiators','Switching…']` |
| 524377 (DLQ) | `1.11a` compliance_overlay | `compliance_overlay.required_documents: empty placeholder value` |
| — | `1.0a` prior_art_query_plan | `'prior_art_queries' missing content about: ['queries','domain_keywords']` |
| — | `1.3` direct_competitor_identification | `'direct_competitors_list' has ~0 list items, need >= 3` |
| — | `1.0c` prior_art_synthesize | `'prior_art_report' missing content about: search_summary, attempted_solutions, key_lessons, verdict` |

**Common shape:** the model emits prose that does NOT contain the literal section headers / keys the gate scans for, OR emits placeholder values (`<…>` / empty). Leads to chase:
- Is the gate scanning for EXACT header strings the prompt never instructs the model to emit? (prompt↔gate contract drift). Check the step's `artifact_schema` / `produces` vs the agent prompt's required-output spec.
- Is the model producing a valid-but-differently-headed artifact that a stricter-than-necessary gate rejects? (false-reject) vs genuinely-incomplete output (true-reject).
- Cross-check with the `<longcat_*>` control-token fix (`d2038c67`): some "empty placeholder"/`<…>` rejections overnight were the owl-alpha leak — re-verify these reproduce AFTER restart (the leak fix may resolve a subset).
- Which model produced each failing artifact? (`model_call_tokens` by task_id). If owl-alpha/cloaked → may be model-quality, not gate.

### B. "Agent reported completion but output indicates failure"
| Task | Step |
|---|---|
| — | `1.0a` prior_art_query_plan |
| — | `1.11a` compliance_overlay |
Agent self-declares `final_answer` but a post-check flags the output as a failure. Find the emitter of that exact string (grep) and what signal it keys on (empty result? error marker in content? schema?). Likely overlaps with class A.

### C. Reviewer / workflow-graph routing
| Task | Step | Error |
|---|---|---|
| 524380 (DLQ) | `1.13` research_quality_review | `reviewer rejected artifact but workflow graph unavailable` |
A reviewer verdict = reject, but the re-pend-the-producer path couldn't find the workflow graph (cf. reviewer-failure routing `e756355c`). Investigate why the graph was unavailable at that point (mission paused? graph not loaded on this path?).

---

## Diagnostic method that worked this session (reuse it)
- **`admission_violations` table** (`site='coulson_pool_empty'`, `'kdv_pre_call_refusal'`, `'daily_exhausted_at_call'`): `snapshot_summary` + `extra_json.diag.filter_reasons` give the exact per-reason selection histogram. (For content failures, look instead at the task `error` / `error_category` + the artifact.)
- **`model_call_tokens`** by `task_id` / `agent_type`: which models ran, prompt/completion sizes, success. (tz: this table + Telegram are local UTC+3; `logs/*.jsonl` `ts` is UTC.)
- **`tasks.task_state`** (JSON checkpoint): `reqs`, `messages` (the actual conversation — see what the model was told vs what it produced), `iteration`, `used_model`.
- Read-only DB: `sqlite3.connect("file:C:/Users/sakir/ai/kutai/kutai.db?mode=ro", uri=True)`.
- Pkg tests: `.venv/Scripts/python.exe -m pytest <path> -o addopts="" -p no:aiohttp` (foreground + timeout; never background — orphan holds the prod SQLite lock).

## Open policy item (declined this session, owner decision)
**Free-exhausted → prefer-local / back-off.** The autonomous night run drains free cloud daily quotas; once drained, selection thrashes broken free cloud. The point-fixes (`2205f969`, `4680ef9a`) push selection off broken models onto local, but a deliberate policy ("when all free cloud daily-exhausted, prefer local or pause until reset") would address the root pattern. Decide before the next overnight run.

## #3 insufficient-credits — ruled adequate, logged for completeness
OpenRouter `Insufficient credits` → `auth_failure` → `mark_dead(cause="auth")` = never-revive → each paid model fails once then excluded (self-limiting; free models need no credits). A provider-wide credit-kill would converge faster but is the mechanism that caused the prior 33-model outage (`caller.py:1106`) — only revisit with an explicit policy decision.
