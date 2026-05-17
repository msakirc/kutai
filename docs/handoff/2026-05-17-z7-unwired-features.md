# Handoff — Z7 unwired features (post fix-pass review)

**Date:** 2026-05-17
**Context:** After the Z7 humanish-layers fix pass (`docs/handoff/2026-05-16-z7-fix-pass.md`) closed
all SYSTEMIC + critical + minor bugs, a 5-agent read-only wiring audit was run to hunt scaffolds.
It found two buckets.

- **Bucket 1 — genuine fix-pass defects — FIXED 2026-05-17.** C7 completeness-check correlation
  (`596509eb`), A6 paused-forever lockout (`b3a21ae4`), 3 dead verbs in `EXTERNAL_PUBLISH_VERBS`
  (folded into `596509eb`).
- **Bucket 2 — pre-existing Z7-BUILD hollowness — this document.** ~8 features whose fix-pass
  repairs are *internally correct* but whose feature has **no production trigger**. The fix pass
  was scoped to fix specific bugs, not to wire features; these gaps predate it. Wiring them needs
  new commands / crons / i2p steps and, for two of them, data-model decisions. Deferred by the
  founder as a separate effort — recorded here so it isn't lost.

Each item below: what works, the exact broken link (file:line), and what wiring would close it.

---

## 1. C9 — announcement / changelog blast has no trigger

**Works:** `lifecycle_email.trigger_sequence(broadcast=True)` + `list_subscribed_tokens` genuinely
fan out one `email_sends` row per subscriber; the send job picks them up and sends. Verified end-to-end.

**Broken link:** the broadcast path's *only* caller is the `changelog/publish` mr_roboto verb
(`packages/mr_roboto/src/mr_roboto/__init__.py:4246`), and **nothing enqueues that verb**:
- no `i2p_v3.json` step has `agent: mechanical` / `action: changelog/publish` (step 11.4 produces a
  `changelog` artifact but never publishes it);
- no `/changelog` Telegram command exists;
- the `changelog_freshness` posthook (`changelog_freshness.py:107`) only emits a founder_action whose
  *text* says "Run changelog/publish" — it does not enqueue anything.

**Wiring needed:** a `/changelog publish` Telegram command, or an i2p mechanical step, or make the
`changelog_freshness` founder_action machine-actionable (an approval handler that enqueues the verb).

---

## 2. FAQ flywheel — approved FAQs are never indexed

**Works:** `support_docs_en` / `support_docs_tr` collections are created at `init_store()`;
`documentation_gap_detect` reads them; `faq_regen._reindex_collection` → `embed_and_store` writes
them with the correct signature.

**Broken link:** `faq_regen._apply_faq_approval` (`src/app/jobs/faq_regen.py:240`) — the only writer
of the per-language collections — has **zero non-test callers**. `faq_regen` emits a founder_action
with `expected_output_schema={"_faq_approval_pending": True}` (`faq_regen.py:301`), but **nothing
consumes `_faq_approval_pending`** — `src/founder_actions/` has no FAQ-apply hook. Consequence: the
`support_docs_*` collections stay permanently empty, so `documentation_gap_detect` always misses
and reports a gap for every question.

**Wiring needed:** a founder-action approval handler that, when a `_faq_approval_pending` card is
approved, routes the approved FAQ back into `_apply_faq_approval`.

---

## 3. A6 — outreach deliverability check never runs

**Works (post Bucket-1):** `outreach_deliverability_check` writes an `outreach_pauses` row;
`outreach_send` Gate 2b honors it; `/outreach resume <list>` clears it (`b3a21ae4`).

**Broken link:** `outreach_deliverability_check` itself is never invoked. Its `PostHookSpec`
(`packages/general_beckman/src/general_beckman/posthooks.py:795`) has `auto_wire_triggers=[]`, so the
expander never attaches it; no workflow JSON declares it as a step; nothing enqueues the mr_roboto
`outreach_deliverability_check` action. The pause is never *written* in production.

**Wiring needed:** a cron seed (periodic deliverability sweep) or a non-empty `auto_wire_triggers`
on the posthook spec, or an explicit i2p step.

---

## 4. A7 — cold-outreach send path has no entry point

**Works:** `run_outreach_send` internals — warmup day-1 seed, quota math, Gate 2b pause check — are
correct.

**Broken link:** `run_outreach_send` is reachable only via the mr_roboto `outreach/send` action, and
nothing creates that task. `/outreach upload` (`src/app/telegram_bot.py:12097`) is a **pure stub** —
it replies "queued for founder approval / a card will surface with 3 draft previews" but creates no
list row, no founder_action, no task, no batch. There is no list storage and no approval flow.

**Wiring needed:** make `/outreach upload` actually persist the prospect list, create the founder
approval card with draft previews, and on approval dispatch `outreach/send` tasks.

---

## 5. A7 — reply handling not connected; `follow_up` template_id inert

**Works:** `run_outreach_draft` is a real verb that enqueues a real task; the `replied_at` dedup
guard added in the fix pass is correct.

**Broken links (two):**
- `outreach_handle_reply` (mr_roboto action `outreach/handle_reply`) has **no inbound-reply caller** —
  no ESP reply webhook dispatches it. The module docstring claims "when the ESP fires a reply-event"
  but no such webhook handler exists.
- `template_id='follow_up'` passed by `outreach_handle_reply` is an **ignored free-form string** —
  `run_outreach_draft` drops it into `spec.context.template_id` and nothing branches on it. There is
  no template registry. The follow-up draft is structurally identical to a cold draft (it does carry
  `prospect_data.is_follow_up=True` + `reply_body`, so the agent *could* infer it — but `template_id`
  itself changes nothing).

**Wiring needed:** an ESP inbound-reply webhook route → `outreach/handle_reply`; and a template
registry (or a prompt branch) that actually consumes `template_id`.

---

## 6. A11 — mention monitor is never scheduled

**Works:** the Critical-6 Twitter-gate fix is correct (Twitter genuinely off by default); the
`mention_polls/<source>` verbs work if dispatched; the corrective `mentions` UNIQUE migration ran.

**Broken links:**
- `src/workflows/mention_monitor.json` declares trigger `/mention_monitor add` and a 60-min
  recurring cadence, but **no `/mention_monitor` Telegram command exists** and `mention_monitor` is
  **not in `cron_seed.INTERNAL_CADENCES`**. Nothing loads or runs the workflow.
- `mentions.acted_on` is *written* by `mention_polls.py:212` but **read by nothing** — its intended
  reader is the `mention_monitor.json` digest step, which is itself broken: `skip_when:
  "no_pending_digest_mentions"` is a bare token the only `skip_when` evaluator can't parse, and
  there is no `mention_digest` notify-template handler.

**Wiring needed:** a `/mention_monitor` command + a cron seed (or workflow loader); fix the digest
step's `skip_when` to the `<artifact>.<path> == '<literal>'` shape and add the `mention_digest`
template; then `acted_on` has a real consumer.

---

## 7. A9 — investor bullets is scheduled but starved of input data

**Works:** the job IS cron-seeded (`cron_seed.py:356`, 30-day). The fix pass's product-scoped JOINs
and `sigma=None` guard are internally correct.

**Broken links (data-model gaps — need decisions):**
- `missions.product_id` is a nullable placeholder **never populated by any code** (`db.py:803`
  comment: "Z0 may take ownership later"). The product-scoped JOINs filter `WHERE m.product_id=?`
  against a column that is always NULL → zero rows.
- `growth_events` metric kinds `metric_emit` and `review_density_metric` are **never written by any
  production code** (`review_density_metric` appears only as a string literal inside
  `investor_bullets.py` itself). So `_fetch_z6_metrics` / `_fetch_review_density` /
  `_fetch_support_metrics` always return `{}` → bullets render empty → `_call_llm_anomaly_hypothesis`
  never actually fires.

**Decisions needed:** who/what sets `missions.product_id`; who emits the metric `growth_events`.
Until then A9 produces empty founder cards.

---

## 8. `briefing_compose` recovered-lessons query realistically empty

**Works:** the fix pass replaced a brittle `LIKE '%"mission_id": 42%'` (false-matched `421`) with
`json_extract(source_ref,'$.mission_id') = ?` — correct SQL.

**Broken link:** the dominant production `mission_lessons` writers omit the `mission_id` key in
`source_ref`: `apply.py:490` (posthook_fail) writes `{source_task_id,kind,attempts}`;
`mission_lessons.py:253` (dlq_pattern) writes `{dlq_ids:[...]}`; `record_verdict.py:162`
(hypothesis_verdict) writes `{feature,metric,verdict}`. Only `mr_roboto/__init__.py:278`
(bisect_break) and `telegram_bot.py:8211` (verdict) include `mission_id`. So the `## Recovered
Failures` briefing section finds rows only for that narrow subset — exactly the posthook_fail /
dlq_pattern recoveries it most wants to surface return nothing. Also `briefing_compose`'s posthook
spec is opt-in (`auto_wire_triggers=[]`) and no i2p step was found declaring it — it may never fire.

**Wiring needed:** make the `mission_lessons` writers include `mission_id` in `source_ref`
consistently; declare the `briefing_compose` posthook on a real workflow step.

---

## Out-of-scope items noticed during the audit (not Z7, flag only)

- `src/core/orchestrator.py:99,122` — yalayut discovery / source-scout tasks are enqueued with
  `lane="mechanical"`, but no `"mechanical"` lane exists (`lanes.py` has only `LANE_ONESHOT` /
  `LANE_ONGOING`). The pump only selects `oneshot`, so these tasks are silently orphaned. Predates Z7.
- `demo/distribute` mr_roboto verb is dispatchable but is **not** a step in `i2p_v3.json` phase_13
  (the other 5 `demo/*` verbs are). Orphan verb; predates the fix pass.

---

## Summary table

| # | Feature | Status | Wiring gap |
|---|---------|--------|------------|
| 1 | C9 announcement blast | engine works, no trigger | no `/changelog` cmd / i2p step |
| 2 | FAQ flywheel write | collections empty | `_faq_approval_pending` consumed by nothing |
| 3 | A6 deliverability check | pause read/write works | check never invoked (`auto_wire_triggers=[]`) |
| 4 | A7 cold outreach send | send internals correct | `/outreach upload` is a stub |
| 5 | A7 reply handling | draft verb works | no ESP reply webhook; `template_id` inert |
| 6 | A11 mention monitor | verbs work | no command, no cron, digest step broken |
| 7 | A9 investor bullets | scheduled, queries correct | `missions.product_id` + metric events never written |
| 8 | briefing_compose lessons | `json_extract` correct | writers omit `mission_id` key; posthook unwired |

Bucket 1 (C7 / A6 / dead verbs) is fixed and on `main`. Bucket 2 above is the remaining Z7 wiring
debt — a deliberate, decision-bearing effort, not a bug fix.
