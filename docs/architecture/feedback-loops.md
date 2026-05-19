# Feedback Loops — Z9 Reinforce vs Z10 Calibration

KutAI has **two** outcome-driven feedback loops. They were each described as
"the learning loop" in their own zone doc (Z9 growth, Z10 cross-cutting),
which invites the assumption that they are one mechanism split in two, or
that they should be unified. **They should not be.** They are orthogonal —
different subjects, different signals, different consumers, different
decisions. They read and write different tables *correctly*. This document
exists so a future session does not try to merge them.

## Loop A — Z9 Reinforce (model selection)

**Question it answers:** *Which model should Fatih Hoca pick?*

| | |
|---|---|
| Subject | a **model** (e.g. `qwen-7b`, `claude-sonnet-4-6`) |
| Signal | a growth **hypothesis verdict of `confirmed`** — a feature the model built moved a real business metric |
| Writer | `mr_roboto/executors/record_verdict.py::_reinforce_winning_model` → `record_reinforce_nudge()` |
| Storage | `model_pick_log` rows, `call_category='reinforce'`, `+0.05` in the `reinforce` column |
| Reader | `fatih_hoca/grading.py::reinforce_bonus()` — time-decayed sum (50% / 30d, capped), folded into the model's `perf_score` |
| Effect | nudges model **selection** scoring |

Outcome-based: a model whose work produced a *confirmed business win* earns a
small, decaying selection boost.

## Loop B — Z10 Calibration (confidence trust)

**Question it answers:** *How much should we trust an agent's self-reported
confidence claim?*

| | |
|---|---|
| Subject | a **(model, task_kind, confidence_band)** triple |
| Signal | did the agent's **claimed confidence match actual correctness** |
| Writer | confidence claims resolved into `confidence_outcomes` |
| Aggregator | `general_beckman/cron.py::_confidence_calibration_recompute` → `recompute_reliability_scores()` → `confidence_reliability_scores` |
| Reader | `coulson/context.py::_lookup_reliability_cached` — emits a prompt nudge ("your 'high' confidence claims have correlated 0.XX with downstream success") |
| Effect | calibrates how the prompt **frames / trusts** a confidence claim |

Calibration-based: tracks whether an agent's confidence is *honest*, and feeds
that back so the agent (and the system) calibrate.

## Why they are NOT merged

| | Loop A — Reinforce | Loop B — Calibration |
|---|---|---|
| Subject | model | model × task_kind × confidence_band |
| Signal | business-metric verdict | confidence-vs-correctness |
| Decision affected | model selection | confidence trust in prompts |
| Table | `model_pick_log` | `confidence_outcomes` / `confidence_reliability_scores` |

A well-calibrated confidence claim is **not** a business win, and a business
win says **nothing** about whether a confidence claim was honest. Folding them
into one table or one loop would conflate two unrelated quantities and degrade
both signals. The split is deliberate. Keep it.

The only thing the two share is timing — both fire after an outcome is known.
That is not a reason to unify them.
