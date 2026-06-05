# SP4b Design — extract the LLM out of the 6 mr_roboto executors

**Date:** 2026-06-05
**Status:** design (awaiting founder review)
**Parent:** `docs/superpowers/specs/2026-05-27-cps-migration-design.md` (umbrella rev2) + `docs/superpowers/specs/2026-06-05-cps-sp4a-design.md` (§6 names this).
**Kickoff:** `docs/handoff/2026-06-05-cps-sp4b-kickoff.md`.
**Predecessors merged to local `main`:** SP1/SP1.1, SP2, SP3, SP3b, SP4a (`19f23aa1`).

---

## 1. Founder ruling (the why)

[[no-direct-dispatcher-from-mechanical]]: **mr_roboto is mechanical-only, non-LLM. Any LLM execution does not belong to it.** Six executors make an LLM call inside their `run()` (`await_inline=True`) — LLM tasks wearing a mechanical costume. SP4b extracts the LLM out of mr_roboto entirely.

Founder refinement during brainstorming (2026-06-05): **every LLM hop must be a clean admitted Beckman task — either a post-hook/continuation child or a workflow (i2p) step. Never buried inside a mechanical verb.** Both carriers are admitted pump tasks (full Beckman admission, model selection, retry/grade, cost, telemetry).

This is shape-(b) (producer + mechanical confirm), the end state the SP3b review named — NOT SP4a's husam-inline (shape-a), which is only for non-mechanical consumers.

## 2. The pattern already exists — in both a right and a wrong form

- **Right (launch_playbook.json):** `launch_drafts/*` steps **enqueue an LLM task async (no `await_inline`)**; a separate `2.founder_review_*` step surfaces the draft. Producer step → consumer step. This is the target shape.
- **Wrong (the 6):** block inline on an LLM *inside* a mechanical executor (`await_inline=True`).
- **Sink template (`incident_update_review` post-hook):** a mechanical handler with NO LLM — reads `result.draft`, emits a `founder_action`. Already the shape-(b) sink for the workflow case.
- **Producer template (`critic_gate.produce_verdict`):** builds a verb-specific prompt + `raw_dispatch` spec. SP4b relocates this shape OUTSIDE mr_roboto (founder ruling) and routes it through the **pump**, not husam-inline.

## 3. Principle: one LLM = one admitted pump task; carrier matches job-shape

Every extracted LLM call becomes an admitted Beckman task. The **carrier** is chosen by the job's natural grain, not by taste:

- **Workflow split** when the job is inherently multi-step or fan-out (the engine sequences producer→consumer via `depends_on`, passes artifacts, and runs post-hooks).
- **CPS continuation** when the job is a single producer→sink hop fired imperatively (router button / cron) — a workflow/mission would be the wrong grain (esp. the reviews cron: ≤100 reviews/run = 100 missions if each were a workflow).

mr_roboto keeps ONLY the **mechanical sink**: DB reads/writes, redaction, asset copy/zip, enum validation, versioning, `founder_action` emission, and the **never-auto-post** contract — taking already-produced text as input. It makes NO LLM call and imports NO dispatcher/husam.

## 4. Per-verb matrix (the decisions)

| Verb | File | Invocation | LLM carrier | Mechanical residue (the sink) |
|------|------|-----------|-------------|-------------------------------|
| demo_storyboard | `demo_storyboard.py` | i2p step `13.demo_storyboard` (+ router 4151) | **workflow split** (existing step) | parse+normalize scenes, write `storyboard.json` |
| incident_draft_update | `incident_draft_update.py` | `incident_comms.json` step `2.draft_update` (+ router 4417) | **workflow split** (existing step) | redaction (pre **and** post), DB fetch, carry draft; keep `incident_update_review` post-hook |
| press_kit_assemble | `press_kit_assemble.py` | router 4274 | **new workflow** (`press_kit.json`) | per-audience file writes, asset copy, zip, versioning, `founder_action`; `press_kit_freshness` post-hook |
| crisis_draft_holding | `crisis_draft_holding.py` | router 4539 | **new workflow** (`crisis_comms.json`, B6) | playbook read is producer input; sink returns variants → `founder_action` |
| reviews_classify | `reviews_classify.py` | router 4642 **+ cron** `reviews_poll_daily.py:64` | **CPS continuation** | enum-validate, `UPDATE external_reviews`, route 2 side-effects (founder_action + bug enqueue), heuristic fallback |
| reviews_draft_reply | `reviews_draft_reply.py` | router 4653 | **CPS continuation** | never-auto-post contract, hand draft to caller/founder_action |

**Note vs kickoff:** the kickoff listed incident_draft_update as router-only; it is in fact also a workflow step (`incident_comms.json 2.draft_update`). demo_storyboard has 3 dependents in i2p_v3.json (:9582/:9602/:9654) that `depends_on 13.demo_storyboard` — repoint them at the mechanical sink step.

## 5. The mechanical sink comes in 3 tiers (drives the split shape)

The residue is NOT uniform. Where the LLM sits relative to the mechanism differs per verb:

1. **Thin sink** — reviews_draft_reply, crisis_draft_holding: producer does ~all the work; sink mostly hands the produced text to a `founder_action` (+ never-auto-post for reply).
2. **Real sink** — reviews_classify: genuine deterministic work AROUND the LLM (enum-validate → `UPDATE` → side-effect routing).
3. **Wrap / fan-in** — incident (redaction wraps the LLM on **both** sides), press_kit (4 producers fan into one assemble).

### 5.1 incident redaction ordering (wrap)
Redaction is mechanical and security-critical, and it must run **before** the producer (the redacted alert feeds the prompt) **and after** (final pass over the LLM output). So `2.draft_update` splits into **three** sub-steps:

- `2a.redact_alert` (`agent:mechanical`) — DB fetch incident + `_redact_alert(alert_details)` → produces `safe_alert_details`.
- `2b.draft_update` (`agent:reviewer`) — producer; drafts from `safe_alert_details`.
- `2c.finalize_draft` (`agent:mechanical`) — `redact_internal`/`redact_secrets`/`redact_user_pii` over the LLM output; carries `draft`; `post_hooks:["incident_update_review"]` (unchanged) emits the founder_action.

`2.publish` repoints its `depends_on` to `2c.finalize_draft`.

### 5.2 press_kit fan-in
- 4 producer steps `1.draft_onepager_{investor,journalist,partner,candidate}` (`agent:planner`), each produces `one_pager_{audience}.md`. Prompts = today's `_AUDIENCE_PROMPTS`, moved into the step JSON.
- 1 mechanical step `2.assemble` (`agent:mechanical`) `depends_on` all 4 → reads the 4 one-pagers, copies logo/screenshots, builds the per-audience zips, versions, emits the sign-off `founder_action`. Verified: the engine resolves a list `depends_on` to N task ids (fan-in supported).

### 5.3 reviews (CPS)
- **draft_reply (button):** router branch enqueues a producer (`agent:reviewer`, prompt built by a thin producer module) with `on_complete="reviews_draft_reply_sink"`, `cont_state={review_id, platform, product_id}`. Sink enforces never-auto-post and routes the draft into a `founder_action`.
- **classify (cron + button):** `reviews_poll_daily` (a legit orchestration layer, not a mechanical) drives, per unclassified review, a producer + `on_complete="reviews_classify_sink"`. Sink validates the enum, `UPDATE`s `external_reviews`, and routes the two existing side-effects. The fire-and-forget `_enqueue_bug_investigation` (NOT `await_inline`) is already a separate admitted task — leave it untouched.

## 6. Fork #2 — where the prompt lives (falls out of §3)

The prompts are highly verb-specific (platform reply conventions, crisis tiers, redaction framing, 4 audience prompts) — NOT generic agent behavior, so they must NOT be folded into the reviewer/planner agent system prompts (that leaks verb knowledge into generic agents).

- **Workflow-carried** (demo, incident, press_kit, crisis): prompt lives in the **step JSON** (the `agent:`-typed step's `description`/instructions, built by the expander). Out of mr_roboto by construction.
- **CPS-carried** (reviews ×2): no JSON step to hold the prompt → a **thin producer module outside mr_roboto** (builds the verb-specific prompt + the `raw_dispatch` overhead spec + enqueues with the continuation). Home: coulson-adjacent or a small new package — decide in the plan; it must NOT live in `packages/mr_roboto`.

Either way the prompt and the LLM leave mr_roboto.

## 7. Fallback / failure path (the key risk)

Each verb has a fallback today (`_fallback_draft`, `_heuristic_classify`, canned crisis variants). Fallback is **mechanical** and belongs in the **sink**, fed a degraded signal when the producer fails:

- **CPS (reviews):** natural — wire `on_error="…_sink"` (or the same sink with a `degraded` flag in `cont_state`) so the sink runs with the heuristic when the producer DLQs.
- **Workflow (demo/incident/crisis/press_kit):** ⚠️ **producer DLQ blocks the dependent sink step** ([[feedback_dlq_blocks_phases]]) — which would kill the fallback. The producer must therefore **emit a degraded artifact on retry-exhaustion** (so it "completes" with a sentinel rather than hard-DLQ, keeping DLQ-blocks-phase intact) and the sink falls back when it sees the sentinel / missing draft. Exact mechanism (degraded-emit vs the engine's `fallback`/`fb` step support seen in `loader.py:343`) resolved in the plan after reading the expander's fallback handling. For a time-sensitive crisis/incident, a canned holding statement is preferable to a blocked mission; surface the degradation in the `founder_action`.

## 8. Non-goals

- Coulson self-reflection loop (umbrella non-goal).
- The `launch_drafts/*` verbs — already async-enqueue (no `await_inline`), already the right shape; out of scope.
- SP5 carve-outs (`task_classifier`, `investor_bullets`) and the shopping `request()` shim — SP5 deletes the primitive after shopping migrates.

## 9. Landmines (carry-over from SP3b/SP4a)

- **NEVER `lane="overhead"`** or any lane ∉ {`oneshot`,`ongoing`} — phantom lane the pump never selects → silent orphaning. Use `lane="oneshot"` for admitted producers. (`add_task` persists any lane verbatim; the `db.py:4464` validation may still be a deferred item.)
- **`ongoing` may also be unpumped** (deferred #9) — the pump only calls `next_task("oneshot")`. Stick to oneshot.
- **No concurrent pytest** — two at once deadlock shared SQLite + crash-loop live KutAI. `tests/` and `packages/*/tests/` are colliding conftest roots — run in SEPARATE `timeout`-prefixed invocations.
- **Worktree conftest is a HARDCODED package list** (`conftest.py` `_PACKAGE_SRCS` + eviction set) — any NEW package SP4b adds (if the CPS producer module becomes a package) must be appended to BOTH.
- **Worktree:** packages are `pip install -e` against MAIN; root conftest injects worktree srcs onto `sys.path`. Use the main venv python from inside the worktree. `worktree.baseRef=head`.
- **husam.run RAISES** — but SP4b producers go through the PUMP (admitted enqueue + continuation), so the failure surface is Beckman's retry/DLQ ladder + the continuation, not a try/except.

## 10. ⛔ HARD GATE — verify the substrate BEFORE building (founder, via Telegram)

SP4b's producers run ON THE PUMP — the post-hook/admission substrate that has never been confirmed live in prod (SP3b found post-hook children orphaned on a phantom `lane="overhead"`; fixed in code, unexercised live). Do not build SP4b on an unproven base:

1. `.venv\Scripts\pip install -e packages\husam` (also required by SP4a's `import husam` sites). Manual; no install script.
2. `/restart`.
3. Run ONE real graded mission end-to-end; confirm `constrained_emit→self_reflect→grade` post-hook children **dispatch on `lane=oneshot`**, rewrite, and complete: `SELECT lane, status, COUNT(*) FROM tasks GROUP BY 1,2` — want oneshot children completing, not a pile of pending.

If the substrate is broken, fix it first. SP4b is moot until admitted producer tasks actually run.

## 11. What "done" looks like

- None of the 6 executors makes an LLM call or touches the dispatcher/husam. Each LLM hop is an admitted task on the pump; mr_roboto holds only the mechanical sink.
- demo_storyboard + incident split in their existing workflow JSONs (producer agent step + mechanical sink; dependents/`publish` repointed). press_kit + crisis become new workflows. reviews ×2 use CPS producer + sink.
- Producer prompts live in step JSON (workflow) or a thin producer module outside mr_roboto (CPS).
- Fallback preserved (sink-owned, degraded-signal path covered).
- Tests pin each split (producer admitted, sink mechanical-only + dispatcher/husam-free, contract preserved, fallback covered). Suites green, sequential runs.
- Remaining `await_inline=True` users = SP5 carve-outs + shopping shim only.

## 12. Open questions for the plan phase

1. Exact home for the CPS producer module(s) (coulson-adjacent file vs small new package). If a package, update worktree conftest lists.
2. Workflow fallback mechanism: producer degraded-emit vs engine `fallback`/`fb` step support — read `expander.py`/`loader.py:343` handling.
3. crisis_comms.json trigger + minimal step graph (likely `/crisis` + mention_monitor B6 escalation): playbook-read/event-fetch (mechanical) → producer → sink.
4. Do the router branches for crisis/press_kit become "launch the workflow" verbs, or do their `/`-commands launch the workflow directly (bypassing the mechanical router action entirely)?
