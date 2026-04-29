# Empirical Token Distribution — KutAI (2026-04-28)

Derived from streaming `kutai.jsonl*` log files + `tasks` DB rows. Approximation: tokens = chars / 3.5. Lower bound for cloud thinking-models (server-side reasoning is stripped).

## TL;DR for the pool-pressure / TPM design decision

1. **Sample is i2p-only and small (n=66).** The char-count log lines only fire from `BaseAgent.run` on fresh starts. Mechanical, grader, artifact_summarizer, shopping_pipeline_v2, executor, coder, implementer, fixer, test_generator → **0 % coverage**. Any TPM model built on this data applies to i2p ReAct agents only; other paths need separate instrumentation.
2. **Step_id is NOT a usable signal as-is.** Each i2p mission visits each step exactly once, so every `step_id` bucket has n=1 and CV is undefined. The hypothesis "tokens correlate more with step_id than agent_type" can't be tested on this sample without more missions.
3. **Phase is the strongest signal we have.** Weighted output-CV: `agent_type only = 1.51`, `phase only = 1.20`, `(agent_type, phase) = 1.13`. Joint key wins, but only by ~25 % — output is fundamentally heavy-tailed (top tasks output 30–50× the median for the same agent).
4. **`AGENT_REQUIREMENTS.estimated_output_tokens` is dramatically wrong on the heavy tail.** Measured p90/estimate ratios: writer ×3.9, analyst ×8.3, architect ×5.5, reviewer ×3.2 (only researcher comes in at ×1.5). The estimates roughly fit *p50* but under-reserve TPM headroom by 3–8× for the steps that actually drive 429s.
5. **Real outliers are step-driven, not agent-driven.** Top 10 by output: `4.5b openapi_spec` (architect, 100k), `5.4b forms_and_states` (analyst, 92k), `3.5 integration_requirements` (researcher, 58k), `4.15a1 backend_core_design` (writer, 44k). These are the steps to budget for, not generic "architect = 3.5k".

**Recommendation for pool-pressure / TPM:** budget per-call on `(agent_type, phase)`-keyed p90 rather than per-agent constants, and add a **step-level override table** for the ~10 known-heavy steps (`4.5b`, `5.4b`, `3.5`, `4.15a1`, `5.11b`). Re-collect once a few more i2p missions complete to get n>=3 per step.

## 1. Sample Summary

- Log files: `kutai.jsonl, kutai.jsonl.1, kutai.jsonl.2, kutai.jsonl.3, kutai.jsonl.4, kutai.jsonl.5`
- Total bytes scanned: 258,221,329
- JSONL lines scanned: 1,157,034
- Char-count log lines matched: 2,253
- Unique tasks with at least one char-count line in logs: 181
- Tasks in DB (full table): 626
- Joined sample (in both logs+DB with chars data): **66**
- Log timestamp range: `2026-04-26T15:10:11.628941+00:00` → `2026-04-28T05:23:27.300368+00:00`

### Coverage gap — what's missing

`BaseAgent.run` (`src/agents/base.py:2098`) is the *only* place that emits `[Task #N] System prompt/User context/Raw response (X chars)` lines. It only fires for **fresh starts** (not checkpoint resumes). Mechanical executor (`salako`), graders, `artifact_summarizer`, and `shopping_pipeline_v2` bypass `BaseAgent.run` entirely and emit no char data. Of 626 DB tasks, log-emitting agents only ran on the i2p ReAct path: **analyst, architect, writer, researcher, reviewer**. Other agent types (grader, mechanical, artifact_summarizer, etc.) are absent from this analysis.

**Per-agent coverage in joined sample:**

| agent_type | DB_count | joined_count | coverage |
|---|---|---|---|
| mechanical | 241 | 0 | 0.0% |
| grader | 155 | 0 | 0.0% |
| artifact_summarizer | 61 | 0 | 0.0% |
| analyst | 57 | 36 | 63.2% |
| executor | 20 | 0 | 0.0% |
| reviewer | 18 | 5 | 27.8% |
| writer | 14 | 8 | 57.1% |
| shopping_pipeline_v2 | 12 | 0 | 0.0% |
| architect | 12 | 10 | 83.3% |
| researcher | 11 | 7 | 63.6% |
| coder | 10 | 0 | 0.0% |
| implementer | 6 | 0 | 0.0% |
| fixer | 5 | 0 | 0.0% |
| test_generator | 3 | 0 | 0.0% |
| summarizer | 1 | 0 | 0.0% |

## 2. Distribution by `agent_type`

`out_tokens` are *summed across ReAct iterations + retries* per task. `avg_iters` shows how many `Raw response` lines fired per task in this agent.

| agent_type | category | n | avg_iters | in_p50 | in_p90 | in_p99 | out_p50 | out_p90 | out_p99 | in_CV | out_CV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| analyst | i2p | 36 | 7.1 | 7,597 | 11,478 | 27,411 | 4,510 | 25,016 | 92,542 | 0.618 | 1.643 |
| architect | i2p | 10 | 11.8 | 6,558 | 11,440 | 12,129 | 8,566 | 19,328 | 100,556 | 0.427 | 1.613 |
| writer | i2p | 8 | 5.8 | 6,970 | 10,842 | 19,420 | 14,116 | 15,458 | 44,855 | 0.574 | 0.849 |
| researcher | i2p | 7 | 23.4 | 7,262 | 7,514 | 9,254 | 1,500 | 4,363 | 58,752 | 0.146 | 1.954 |
| reviewer | i2p | 5 | 11.6 | 6,652 | 21,171 | 21,171 | 3,057 | 7,911 | 7,911 | 0.602 | 0.732 |

## 3. Distribution by i2p `step_id` (top 30 by frequency)

| step_id | agent_types | n | in_p50 | in_p90 | in_p99 | out_p50 | out_p90 | out_p99 | in_CV | out_CV |
|---|---|---|---|---|---|---|---|---|---|---|
| 4.15a1 | writer | 1 | 19,420 | 19,420 | 19,420 | 44,855 | 44,855 | 44,855 | 0.0 | 0.0 |
| 4.9 | architect | 1 | 11,440 | 11,440 | 11,440 | 19,328 | 19,328 | 19,328 | 0.0 | 0.0 |
| 3.5 | researcher | 1 | 9,254 | 9,254 | 9,254 | 58,752 | 58,752 | 58,752 | 0.0 | 0.0 |
| 5.5 | reviewer | 1 | 5,472 | 5,472 | 5,472 | 797 | 797 | 797 | 0.0 | 0.0 |
| 5.8 | analyst | 1 | 10,823 | 10,823 | 10,823 | 8,558 | 8,558 | 8,558 | 0.0 | 0.0 |
| 4.5b | architect | 1 | 9,021 | 9,021 | 9,021 | 100,556 | 100,556 | 100,556 | 0.0 | 0.0 |
| 5.9 | analyst | 1 | 11,123 | 11,123 | 11,123 | 0 | 0 | 0 | 0.0 | 0.0 |
| 5.10 | reviewer | 1 | 21,171 | 21,171 | 21,171 | 1,166 | 1,166 | 1,166 | 0.0 | 0.0 |
| 5.11a | analyst | 1 | 12,788 | 12,788 | 12,788 | 24,670 | 24,670 | 24,670 | 0.0 | 0.0 |
| 5.11b | analyst | 1 | 27,411 | 27,411 | 27,411 | 42,966 | 42,966 | 42,966 | 0.0 | 0.0 |
| 4.7 | architect | 1 | 12,129 | 12,129 | 12,129 | 2,754 | 2,754 | 2,754 | 0.0 | 0.0 |
| 4.12 | architect | 1 | 2,952 | 2,952 | 2,952 | 1,758 | 1,758 | 1,758 | 0.0 | 0.0 |
| 4.11 | architect | 1 | 3,693 | 3,693 | 3,693 | 3,211 | 3,211 | 3,211 | 0.0 | 0.0 |
| 5.4b | analyst | 1 | 5,870 | 5,870 | 5,870 | 92,542 | 92,542 | 92,542 | 0.0 | 0.0 |
| 3.9b | analyst | 1 | 8,493 | 8,493 | 8,493 | 3,607 | 3,607 | 3,607 | 0.0 | 0.0 |
| 3.10b | analyst | 1 | 16,800 | 16,800 | 16,800 | 7,650 | 7,650 | 7,650 | 0.0 | 0.0 |
| 3.11 | reviewer | 1 | 11,189 | 11,189 | 11,189 | 7,911 | 7,911 | 7,911 | 0.0 | 0.0 |
| 3.6 | analyst | 1 | 11,478 | 11,478 | 11,478 | 26,797 | 26,797 | 26,797 | 0.0 | 0.0 |
| 4.1 | architect | 1 | 4,964 | 4,964 | 4,964 | 2,925 | 2,925 | 2,925 | 0.0 | 0.0 |
| 4.2 | architect | 1 | 7,825 | 7,825 | 7,825 | 8,566 | 8,566 | 8,566 | 0.0 | 0.0 |
| 4.3 | architect | 1 | 6,558 | 6,558 | 6,558 | 11,108 | 11,108 | 11,108 | 0.0 | 0.0 |
| 4.4 | architect | 1 | 10,934 | 10,934 | 10,934 | 11,711 | 11,711 | 11,711 | 0.0 | 0.0 |
| 4.5a | analyst | 1 | 10,189 | 10,189 | 10,189 | 25,016 | 25,016 | 25,016 | 0.0 | 0.0 |
| 4.6 | architect | 1 | 4,802 | 4,802 | 4,802 | 12,979 | 12,979 | 12,979 | 0.0 | 0.0 |
| 5.7 | analyst | 1 | 5,338 | 5,338 | 5,338 | 23,386 | 23,386 | 23,386 | 0.0 | 0.0 |
| 2.6 | analyst | 1 | 7,905 | 7,905 | 7,905 | 7,598 | 7,598 | 7,598 | 0.0 | 0.0 |
| 2.7 | analyst | 1 | 6,919 | 6,919 | 6,919 | 13,261 | 13,261 | 13,261 | 0.0 | 0.0 |
| 2.8 | analyst | 1 | 3,567 | 3,567 | 3,567 | 6,780 | 6,780 | 6,780 | 0.0 | 0.0 |
| 2.10 | analyst | 1 | 4,355 | 4,355 | 4,355 | 1,915 | 1,915 | 1,915 | 0.0 | 0.0 |
| 2.11b | writer | 1 | 10,842 | 10,842 | 10,842 | 13,158 | 13,158 | 13,158 | 0.0 | 0.0 |

## 3b. Distribution by i2p `workflow_phase` (more samples per group)

Most `step_id` buckets above are n=1 because each i2p mission visits each step exactly once. Phase-level rollups have more samples per group.

| phase | n | in_p50 | in_p90 | in_p99 | out_p50 | out_p90 | out_p99 | out_CV |
|---|---|---|---|---|---|---|---|---|
| phase_0 | 6 | 2,797 | 4,181 | 4,636 | 1,693 | 4,510 | 14,116 | 1.186 |
| phase_1 | 10 | 6,787 | 7,267 | 7,514 | 3,314 | 6,522 | 15,458 | 0.918 |
| phase_2 | 12 | 6,704 | 9,069 | 10,842 | 6,780 | 13,261 | 14,887 | 0.789 |
| phase_3 | 13 | 9,254 | 11,478 | 16,800 | 7,402 | 26,797 | 58,752 | 1.278 |
| phase_4 | 12 | 9,021 | 12,129 | 19,420 | 11,711 | 44,855 | 100,556 | 1.317 |
| phase_5 | 13 | 7,562 | 21,171 | 27,411 | 2,172 | 42,966 | 92,542 | 1.635 |

## 4. CV (Coefficient of Variation) — Which grouping is tighter?

Lower CV = tighter distribution = stronger predictive grouping. We compute the *sample-size-weighted* mean CV across groups (only groups with n>=3).

Restricted to tasks with `workflow_step_id` set (n=66); buckets with n<3 dropped.

| grouping | groups | n_total | weighted_CV_in | weighted_CV_out |
|---|---|---|---|---|
| agent_type only | 5 | 66 | 0.532 | 1.506 |
| phase only | 6 | 66 | 0.406 | 1.204 |
| step_id only | 0 | 0 | 0.0 | 0.0 |
| (agent_type, phase) | 6 | 50 | 0.399 | 1.131 |
| (agent_type, step_id) | 0 | 0 | 0.0 | 0.0 |

*Lower weighted CV in the output column wins for our use-case (output tokens drive TPM pressure & cost).*

## 5. `AGENT_REQUIREMENTS.estimated_output_tokens` vs Measured

| agent_type | estimated | n | measured_p50 | measured_p90 | ratio (measured/est) |
|---|---|---|---|---|---|
| analyst | 3000 | 36 | 4,510 | 25,016 | p50×1.5 / p90×8.34 |
| architect | 3500 | 10 | 8,566 | 19,328 | p50×2.45 / p90×5.52 |
| classifier | 400 | — | — | — | — |
| coder | 4000 | — | — | — | — |
| deal_analyst | 2500 | — | — | — | — |
| executor | 1500 | — | — | — | — |
| grader | 800 | — | — | — | — |
| implementer | 4000 | — | — | — | — |
| planner | 2000 | — | — | — | — |
| product_researcher | 2500 | — | — | — | — |
| researcher | 3000 | 7 | 1,500 | 4,363 | p50×0.5 / p90×1.45 |
| reviewer | 2500 | 5 | 3,057 | 7,911 | p50×1.22 / p90×3.16 |
| shopping_advisor | 3000 | — | — | — | — |
| shopping_clarifier | 800 | — | — | — | — |
| summarizer | 2000 | — | — | — | — |
| test_generator | 3000 | — | — | — | — |
| writer | 4000 | 8 | 14,116 | 15,458 | p50×3.53 / p90×3.86 |

*Ratio < 1 = current estimate is too high (over-reserves TPM headroom). Ratio > 1 = under-reserves and risks 429s.*

## 6. Outliers & Surprises

### Steps where output_p90 ≥ 2× the dominant agent's output_p90

*(none found at 2× threshold)*

### Top 10 single tasks by output_tokens (after summing iterations)

| task_id | agent | step_id | in_tok | out_tok | step_name |
|---|---|---|---|---|---|
| 4423 | architect | 4.5b | 9,021 | 100,556 | openapi_spec |
| 4441 | analyst | 5.4b | 5,870 | 92,542 | forms_and_states |
| 4409 | researcher | 3.5 | 9,254 | 58,752 | integration_requirements |
| 4433 | writer | 4.15a1 | 19,420 | 44,855 | backend_core_design |
| 4449 | analyst | 5.11b | 27,411 | 42,966 | design_handoff_document |
| 4410 | analyst | 3.6 | 11,478 | 26,797 | platform_and_accessibility_requirements |
| 4422 | analyst | 4.5a | 10,189 | 25,016 | api_resource_model |
| 4448 | analyst | 5.11a | 12,788 | 24,670 | design_system_handoff |
| 4444 | analyst | 5.7 | 5,338 | 23,386 | component_specs |
| 4411 | analyst | 3.7 | 9,604 | 22,533 | business_rules_extraction |

### Category roll-up (shopping vs i2p vs other)

| category | n | in_p50 | in_p90 | in_p99 | out_p50 | out_p90 | out_p99 |
|---|---|---|---|---|---|---|---|
| i2p | 66 | 6,970 | 11,440 | 21,171 | 4,510 | 24,670 | 92,542 |

## 7. Caveats

- **Thinking tokens missing**: cloud thinking models (Claude, GPT-OSS, Qwen-thinking) produce server-side reasoning that is stripped before logging. Output token counts for those models are a *lower bound* — actual billed output is higher.
- **chars/3.5 is approximate**: KutAI prompts mix English + Turkish + JSON + code. Real `tiktoken` would shift numbers ±10–20 %. Useful for *relative* comparisons, less reliable as an absolute TPM budget input.
- **Sample sizes per group**: see `n` column in each table. Some `step_id` buckets have <5 tasks; their p99 is essentially the single worst sample. Do not over-fit.
- **Log gaps**: sys/user prompts use MAX (per-call constants), raw responses use SUM (across ReAct iterations + retries). If a task crashed before a Raw response line was emitted, its output is undercounted.
- **DB ↔ log join**: tasks created BEFORE the oldest rotated log file (kutai.jsonl.5) appear in the DB (560 unmatched) but have no char data. These are silently excluded from distributions.
- **Iteration sum vs per-call**: `out_chars` is *cumulative across all ReAct iterations + retries* for a task. For per-call TPM budgeting, divide by the average iteration count for that step.
