# SP4b Plan 2 kickoff — workflow-split the 4 remaining LLM-bearing mr_roboto verbs

**For:** a session building **SP4b Plan 2** (the workflow-engine half of SP4b).
**Date:** 2026-06-06.
**Author:** SP4b Plan 1 ship session.
**Spec (already written — your design source):** `docs/superpowers/specs/2026-06-05-cps-sp4b-design.md` §3–§7 + §12.
**Plan 1 (DONE, reference pattern):** `docs/superpowers/plans/2026-06-05-cps-sp4b-reviews.md` + handoff `docs/handoff/2026-06-05-cps-sp4b-plan1-reviews-shipped.md`.

---

## Where this sits

SP4b = "extract the LLM out of the 6 mr_roboto executors" (founder ruling [[no-direct-dispatcher-from-mechanical]]: mechanical verbs make NO LLM call). It split into **two plans by subsystem**:

- **Plan 1 — reviews CPS (DONE, branch `worktree-cps-sp4b`, NOT merged):** `reviews_classify` + `reviews_draft_reply` → CPS producer + mechanical sink. The *continuation* substrate.
- **Plan 2 — workflow splits (THIS, not started):** the 4 verbs whose carrier is the **workflow engine**, not CPS continuations.

Carrier was chosen by job-shape (spec §3). The deciding axis: **"is the consumer waiting?"** + "is the job multi-step/fan-out?". These 4 are workflow-shaped.

## The 4 verbs (re-confirm with a fresh grep — line numbers drift)

`rg -n "await_inline\s*=\s*True" packages/mr_roboto/src` — after Plan 1 merges, only these 4 + the SP5 carve-outs remain.

| Verb | File | Carrier | Split shape (spec §5) |
|------|------|---------|------------------------|
| **demo_storyboard** | `demo_storyboard.py` | **split existing i2p step** `13.demo_storyboard` (`i2p_v3.json`) | `agent:reviewer` producer step → `agent:mechanical` writer (parse/normalize/write storyboard.json). **Repoint 3 dependents** at `i2p_v3.json:9582/:9602/:9654` (they `depends_on 13.demo_storyboard`). |
| **incident_draft_update** | `incident_draft_update.py` | **split existing step** `2.draft_update` (`src/workflows/incident_comms.json`) | **3 sub-steps** (redaction wraps the LLM both sides): `2a.redact_alert` (mechanical → `safe_alert_details`) → `2b.draft_update` (`agent:reviewer`) → `2c.finalize_draft` (mechanical post-redact + carry draft; keep `post_hooks:["incident_update_review"]`). Repoint `2.publish` `depends_on` → `2c`. |
| **press_kit_assemble** | `press_kit_assemble.py` | **new workflow** `press_kit.json` | **4→1 fan-in**: 4 `agent:planner` producer steps (one per audience: investor/journalist/partner/candidate, each `produces one_pager_{aud}.md`; prompts = today's `_AUDIENCE_PROMPTS`) → 1 mechanical `assemble` step `depends_on` all 4 (write/copy/zip/version/founder_action). Engine resolves list `depends_on` to N task ids (verified). Keep `press_kit_freshness` post-hook. |
| **crisis_draft_holding** | `crisis_draft_holding.py` | **new workflow** `crisis_comms.json` (B6) | mechanical pre-step (fetch event + read tier playbook → producer input) → `agent:reviewer` producer (2 variants) → mechanical sink (return variants → founder_action). Trigger: `/crisis` + mention_monitor B6 negative-cluster escalation. |

## ⛔ Open questions you MUST resolve first (spec §12) — brainstorm + investigate before coding

These are workflow-engine mechanics Plan 1 (pure CPS) never touched. Don't write the plan until these are answered against the code:

1. **How does an `agent:`-typed workflow step become an LLM call?** Read the worker/expander path: where the step's `description`/agent_type build the messages. The producer step's prompt must come from the **step JSON**, NOT a code module (spec §6). Confirm the expander wires this.
2. **Producer-DLQ-blocks-sink (the §7 fallback risk — CRITICAL).** [[feedback_dlq_blocks_phases]]: a DLQ'd producer step blocks its dependent sink → kills the fallback (canned crisis/incident draft, etc.). The producer must **emit a degraded artifact on retry-exhaustion** (complete with a sentinel, not hard-DLQ) so the sink runs and falls back. Read `src/workflows/engine/expander.py` + `loader.py:343` (`fb`/fallback-step support) to pick the mechanism. For a time-sensitive crisis, a canned holding statement beats a blocked mission — surface degradation in the founder_action.
3. **Fan-in artifact passing** (press_kit): how the mechanical `assemble` step reads the 4 producers' `one_pager_{aud}.md` artifacts. Check `produces` + context_injection + the materializer ([[project_deterministic_materializer_20260605]]).
4. **Workflow launch from a `/`-command / router** (crisis, press_kit): does `/crisis` / `/press_kit` launch the new workflow directly, or does the router verb become a thin "launch workflow" action? Read how `/launch`, `/incident` spawn their workflows. The router branches at `mr_roboto/__init__.py` ~4274 (press_kit), ~4539 (crisis), ~4149 (demo) currently `await <verb>.run()` synchronously — that synchronous `Action` contract is what breaks (same as Plan 1's router branches).
5. **incident redaction lives in the sink steps, not the producer** — pre-redact (2a) feeds the prompt; post-redact (2c) cleans LLM output. The mechanical `redact_internal`/`_redact_alert` helpers stay in mr_roboto (mechanical).

## Reference pattern from Plan 1 (reuse the shape, not the carrier)

- mr_roboto keeps ONLY the mechanical sink; the LLM + its prompt leave mr_roboto. For workflow-carried verbs the prompt lives in the **step JSON** (vs Plan 1's `src/reviews/producers.py` module for CPS).
- `incident_update_review` post-hook (`packages/general_beckman/.../posthook_handlers/`) is the template mechanical sink (reads `result.draft` → founder_action, no LLM). It already exists and stays.
- TDD per task; drive the mechanical step/sink directly in tests; assert producer step is `agent:`-typed and the sink makes no LLM call.

## Landmines (carry over)

- **lane = "oneshot" only** (Plan 1 producers use it; workflow steps are admitted the same way). Never `overhead`/`ongoing`.
- **No concurrent pytest across sessions** — Plan 1 hit a 16-zombie deadlock on the shared `kutai.db` WAL (mine + the parallel shopping session). Run one `timeout`-prefixed invocation at a time. `tests/` and `packages/*/tests/` are colliding conftest roots → separate invocations.
- **Env hazard (2026-06-05):** TWO KutAI wrappers were running (PID 31628 venv + 40528 global) + 0 llama-server. If still true, reconcile to one stack before live-testing.
- **Worktree:** `EnterWorktree` name `cps-sp4b-plan2`; baseRef=head. Branch from **Plan 1's tip** (`worktree-cps-sp4b`) if Plan 1 isn't merged yet, OR from main after Plan 1 merges — Plan 2 touches mr_roboto/__init__.py (different router branches than Plan 1) + workflow JSONs + the 4 verb modules; mostly disjoint from Plan 1's reviews files. Use main venv python from inside the worktree.

## Workflow

1. **Brainstorm** (`superpowers:brainstorming`) the §12 open questions — settle the producer-prompt-in-step mechanism, the degraded-emit fallback, and the workflow-launch fork, per-verb.
2. **Write Plan 2** (`superpowers:writing-plans`) → `docs/superpowers/plans/YYYY-MM-DD-cps-sp4b-workflow-splits.md`. TDD task breakdown, one verb at a time (suggest order: demo_storyboard simplest → incident → crisis → press_kit fan-in hardest).
3. **Execute** (subagent-driven or inline-TDD per coupling), worktree, sequential pytest, merge `--no-ff`, re-verify.

## Done =

None of the 4 verbs makes an LLM call or touches dispatcher/husam. demo+incident split in their existing workflow JSONs (dependents/publish repointed); press_kit+crisis are new workflows. Producer prompts live in step JSON. Fallback preserved (degraded-emit so producer DLQ never blocks the sink). After Plan 2, the only `await_inline=True` users left = SP5 carve-outs (`task_classifier`, `investor_bullets`) + the shopping `request()` shim → SP5 deletes the primitive.
