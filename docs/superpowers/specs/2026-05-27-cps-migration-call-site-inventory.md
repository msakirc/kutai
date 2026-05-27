# CPS migration — call-site inventory (companion to `2026-05-27-cps-migration-design.md`)

**Date:** 2026-05-28
**Purpose:** capture the per-caller knowledge needed to migrate every `await_inline=True`
site off the primitive, so SP2–SP5 specs/plans can be written *cheaply and correctly* from
this + the built SP1 — without re-deriving each caller's behavior. This is the perishable
context from the 2026-05-27 debug session, persisted to disk.

**Status caveat:** annotations are current as of 2026-05-28. Each is a *starting point* —
re-confirm the call region before migrating it (code drifts; `feedback_audit_call_sites`).

---

## Deadlock-risk key (why grouping matters)

The `await_inline` deadlock needs **all three**: the caller is **(a)** a *dispatched task*,
**(b)** *cap-counted* (`ONESHOT_CONCURRENCY`), **(c)** *non-mechanical*. So:

- **Edge** (Telegram handler, cron/scheduler) → holds **no** lane slot → no deadlock. Migrated
  only because of full-deletion; lowest risk.
- **Mechanical executor** (mr_roboto, runs as `agent_type=mechanical`) → **cap-exempt**
  (`ea0d5b2d`) → no deadlock. Migrated for full-deletion; low risk.
- **In-task agent / tool** (grader/code_reviewer agent, or a tool called mid-ReAct inside an
  agent task) → cap-counted, non-mechanical → **deadlocks**. The real bug surface (SP3/SP4).

Common shape across nearly all sites: each builds a `spec` with a `context.llm_call`
(`raw_dispatch` for the LLM-call ones), `enqueue(await_inline=True)`, then reads
`task_result` (`.status`, `.result`/`.raw`/`.output`) and parses content/JSON. **Almost all
already have a try/except or status-check fallback → those fallbacks are the natural
`on_error` handlers.**

---

## Inventory

| # | Site | SP | Caller ctx | Deadlock? | Result use | State to thread | on_error | Gotcha |
|---|------|----|-----------|-----------|-----------|----------------|----------|--------|
| 1 | `src/app/telegram_bot.py:120` `_enqueue` | 2 | **edge** (bot handler) | no | parse JSON → return to bot cmd | `chat_id`, reply target | send error msg to user | **Root bot primitive — many commands call it.** CPS inverts it: handler returns, resume *sends the reply*. Touch this first/carefully in SP2; it sets the pattern for all edge migrations. |
| 2 | `src/core/task_classifier.py:283` | 2 | **edge** (incoming-msg classify) | no | parsed JSON classification → routes msg | message + chat ctx | raises `ModelCallFailed(availability)` | Routing decision moves into the resume (classification is async). |
| 3 | `src/app/interview.py:253` | 2 | edge (interactive) | no | `task_result.output` → JSON | `note_id` | try/except → regex-rescue | uses `.output` (not `.result`); verify field. |
| 4 | `src/app/meetings.py:398` | 2 | edge (cmd/cron) | no | `.status`+content | meeting ctx | returns `[],[]` | — |
| 5 | `src/app/jobs/faq_regen.py:159` | 2 | edge (cron job) | no | extract text | faq ctx | warn + skip | — |
| 6 | `src/app/jobs/investor_bullets.py:204` | 2 | edge (cron job) | no | hypothesis text | metric_name | return `""` | — |
| 7 | `src/core/grading.py:373` | 3 | **in-task agent** (grader) | **YES** | verdict `{passed}` | `source_task_id`, attempt, exclusions | auto-fail grade | The confirmed deadlock culprit. 2 *sequential* attempts (not join). Resume = apply verdict via post-hook path. |
| 8 | `src/core/code_review.py:179` | 3 | **in-task agent** (code_reviewer) | **YES** | review verdict | `source_task_id` (`graded_task_id`) | as grading | Same pattern/shape as grading — migrate together. |
| 9 | `src/workflows/engine/hooks.py:84` `_llm_summarize` | 3 | post-hook (likely cap-counted) | likely | `.result` content → summary | `artifact_name`, source | returns `None` | Confirm whether the post-hook runs cap-counted before assuming deadlock. |
| 10 | `src/core/llm_dispatcher.py:273` `dispatcher.request` shim | 3 | **fan-out** (many internal callers) | depends on caller | returns LLM result to inline code | per-caller | per-caller | **Riskiest single migration (design rev2).** Contract = "return value to sync caller"; CPS can't. Scope its callers explicitly; may need its own sub-spec. Bounded by `feedback_singular_dispatcher_caller` (only Beckman calls `request`). |
| 11 | `src/tools/vision.py:93` | 4 | **tool mid-ReAct** (inside agent) | **YES** | content + `dogru_mu_samet` quality check | `_parent_id`, image ctx | `"Error: vision call failed"` | **Hardest.** A tool call is synchronous *within a ReAct iteration*; CPS-ing it means the agent iteration must suspend/resume — i.e. touches the ReAct loop, which is otherwise out of scope. May warrant keeping a bounded direct path or a tool-level exception. Flag for design before SP4. |
| 12 | `…/posthook_handlers/brand_voice_lint.py:412` | 4 | posthook | low | lint findings list | `source_task_id` | returns "skipped" finding | — |
| 13 | `…/posthook_handlers/copy_compliance_review.py:498` | 4 | posthook | low | `.raw` → result/answer/content | `task_id` | fence-strip + fallback | reads `.raw` dict shape; verify envelope. |
| 14 | `…/mr_roboto/crisis_draft_holding.py:155` | 4 | **mechanical** (cap-exempt) | no | drafts | crisis ctx | return `[]` | — |
| 15 | `…/mr_roboto/demo_storyboard.py:168` | 4 | **mechanical** | no | `.result.content` | `parent_task_id` | `{ok:false}` | uses a local `_enqueue_storyboard_llm` wrapper. |
| 16 | `…/mr_roboto/incident_draft_update.py:180` | 4 | **mechanical** | no | draft str | incident ctx | return `""` | — |
| 17 | `…/mr_roboto/press_kit_assemble.py:109` | 4 | **mechanical** | no | `.result.content` | `audience`, `prompt` | fallback text | uses `agent_type=planner`, not raw_dispatch. |
| 18 | `…/mr_roboto/reviews_classify.py:97` | 4 | **mechanical** | no | classification | review body/rating | `_heuristic_classify` fallback | strong fallback already exists. |
| 19 | `…/mr_roboto/reviews_draft_reply.py:123` | 4 | **mechanical** | no | reply draft | platform/author/rating | `_fallback_draft` | — |
| 20 | `…/yalayut/discovery/synthesize.py:119` | 4 | **edge** (cron/autonomous) | no | `dict.result` → JSON manifest | prompt ctx | empty manifest | "synthesis must never crash cron"; `model_hint:sonnet`. |

(`packages/mr_roboto/.../reviews_classify.py:221` is explicitly fire-and-forget — *not*
`await_inline` — so it's already CPS-shaped; no migration.)

---

## Cross-cutting notes for SP2–SP5 authors

- **Two existing `on_complete` callers** (`analytics_digest`, `classify_signals`) are the only
  current durable-continuation users; SP1 updates them to the `(task_id, result, state)`
  handler signature. They're the framework's first real exercise.
- **Edge migrations (SP2)** share one shape: handler enqueues + returns; resume delivers the
  outcome (send Telegram reply / write job output / store note). State = the delivery target.
  `telegram_bot._enqueue` (#1) is the keystone — get it right and #2–#6, #20 follow.
- **`dispatcher.request` (#10)** and **`vision` (#11)** are the two that don't decompose
  mechanically. Both deserve a design pass *before* their SP, not just a plan. `vision` is the
  one place full-deletion presses against the out-of-scope ReAct loop.
- **Result-field inconsistency** is a latent trap: sites read `.output` (#3), `.result` (most),
  `.raw` (#13), `dict["result"]` (#20). The resume handler receives the normalized `result`
  dict from `on_task_finished`; confirm each site's expected shape maps onto it during its SP.
- **`on_error` is cheap to wire** because almost every site already has a fallback branch —
  lift that branch into the registered `on_error` handler verbatim.
