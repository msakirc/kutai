# SP4b Plan 2 Design — workflow-split the 4 LLM-bearing mr_roboto verbs

**Date:** 2026-06-07
**Status:** design (awaiting founder review)
**Parent:** `docs/superpowers/specs/2026-06-05-cps-sp4b-design.md` (full SP4b design; this resolves its §12 open questions for the 4 workflow-carried verbs).
**Kickoff:** `docs/handoff/2026-06-06-cps-sp4b-plan2-workflow-splits-kickoff.md`.
**Sibling:** Plan 1 (reviews CPS) — branch `worktree-cps-sp4b`, NOT yet merged. Plan 2 branches from Plan 1's tip.

This doc records the **code-grounded answers** to the parent §12 open questions and the one new mechanism Plan 2 introduces. It does not re-derive the per-verb split shapes (parent §4–§5).

---

## 1. Scope

The 4 verbs whose LLM carrier is the **workflow engine** (not CPS continuations) split by whether they need a dead-producer→canned-fallback path:

| Verb | Carrier | Split | Plan 2? |
|------|---------|-------|---------|
| demo_storyboard | split existing i2p step `13.demo_storyboard` | AI producer → mechanical writer; repoint 3 dependents | ✅ ship now |
| press_kit_assemble | new workflow `press_kit.json` | 4 `agent:planner` one-pagers → 1 mechanical `assemble` (fan-in) | ✅ ship now |
| incident_draft_update | split existing step `2.draft_update` in `incident_comms.json` | `2a.redact_alert`(mech) → `2b.draft_update`(reviewer) → `2c.finalize_draft`(mech); repoint `2.publish`→`2c` | ⛔ DEFER (Plan 3) |
| crisis_draft_holding | new workflow `crisis_comms.json` | mechanical pre-step (event fetch + tier playbook) → `agent:reviewer` producer → mechanical sink | ⛔ DEFER (Plan 3) |

**Plan 2 = demo_storyboard + press_kit only.** Both lack an inline canned fallback today (verified — no `_fallback`/canned in `demo_storyboard.py`, `press_kit_assemble.py`), so splitting them cannot regress a fallback: a DLQ'd producer blocks the sink = same failure-shape as today's `ok:false`. Neither needs the `degrade_on_exhaustion` mechanism (§3), which would require touching General Beckman — **out of bounds this plan** (founder ruling 2026-06-07).

**incident + crisis DEFERRED to Plan 3.** Both ship canned text today when the LLM fails (`_fallback_draft`, canned crisis variants — parent §7). Splitting them *without* `degrade_on_exhaustion` would regress that (DLQ'd producer → blocked sink → nothing ships), strictly worse than today. They wait until a plan is allowed to touch Beckman.

End state (Plan 2): demo_storyboard + press_kit make no LLM call and import no dispatcher/husam. Producer prompts live in step JSON.

## 2. Resolved §12 questions (grounded in code)

### 2.1 How an `agent:`-typed step becomes an LLM call (parent §6, kickoff Q1)
**Resolved — no new wiring.** The expander maps step `instruction` → task `description` **verbatim** (`src/workflows/engine/expander.py:494`) and step `agent` → `agent_type` (`:495`). The task is enqueued to Beckman; the orchestrator dispatches it to `get_agent(agent_type).execute(task)` (`src/core/orchestrator.py:392`), which uses `description` as the prompt. **No code module rewrites the prompt.** `reviewer` and `planner` are already in `AGENT_REGISTRY` (`src/agents/__init__.py:33-63`). A mechanical step (`agent:"mechanical"` + `executor:"mechanical"` + `payload`) routes to `mr_roboto.run(task)` (`orchestrator.py:343-355`); the expander propagates `executor`+`payload` through (`expander.py:480-484`).

➡️ Adding a producer step = pure workflow-JSON work. Prompts move from `_AUDIENCE_PROMPTS` / verb modules into the step `instruction`.

### 2.2 Fan-in: mechanical step reads N producers' artifacts (kickoff Q3)
**Resolved — supported, with precedent.** A list `depends_on: ["1.a","1.b","1.c","1.d"]` resolves to N task ids (`runner.py:resolve_dependencies` 49-63, wired 470-471). Producers write their `produces` artifacts to the **mission blackboard** (`artifacts.py:39-69`); a downstream step declares `input_artifacts` to have them loaded/injected into its context (`context_injection.py:86-149`). Real precedent: `shopping_v3.json` step `3.0` fans in `2.0c`+`2.2c` via `input_artifacts`. The mechanical `assemble` step receives full task context + `mission_id`, so it reads the 4 one-pagers from the blackboard.

➡️ `press_kit.json` `2.assemble` declares `depends_on` the 4 producers + `input_artifacts: [one_pager_investor, …]`; reads them in the mechanical verb instead of from payload params.

### 2.3 Workflow launch from a `/`-command (kickoff Q4)
**Resolved — router verb becomes a thin launcher.** Entry point: `WorkflowRunner.start(workflow_name, initial_input, title, chat_id)` (`runner.py:332`); precedent `_create_mission_with_workflow` (`telegram_bot.py:1619`). Current router branches (`mr_roboto/__init__.py` ~4274 press_kit, ~4539 crisis, ~4149 demo) `await <verb>.run()` synchronously and return an `Action` — that synchronous contract is what breaks.

➡️ The `/press_kit` and `/crisis` router branches become: call `WorkflowRunner.start("press_kit"|"crisis_comms", initial_input=…)` and return `Action(status="completed", result={"mission_id": mission_id})`. demo_storyboard's router branch (4149) is already inside i2p; it follows the existing i2p step (no separate launcher).

### 2.4 incident redaction stays mechanical (kickoff Q5)
**Resolved (parent §5.1).** `_redact_alert` (pre) feeds the producer's prompt via `2a`; `redact_internal`/`redact_secrets`/`redact_user_pii` (post) clean the LLM output in `2c`. Both stay in mr_roboto. `2c` keeps `post_hooks:["incident_update_review"]`.

## 3. DEFERRED mechanism — `degrade_on_exhaustion` (Plan 3, NOT this plan)

> **Status: deferred.** Requires touching General Beckman (`retry.py` + `apply.py`), out of bounds per founder ruling 2026-06-07. Recorded here because incident + crisis (Plan 3) depend on it, and the analysis below is the resolution of parent §12.2. No code in Plan 2 touches this.

### 3.1 Why the parent's first idea (engine `fallback`/`fb` steps) does NOT work
**Disproven in code.** `conditional_groups.fallback_steps` are **condition-driven, not failure-driven**. `resolve_group` (`src/workflows/engine/conditions.py:143-177`) evaluates a `condition_check` against an artifact value at expansion time and includes `if_true` OR `(if_false + fallback_ids)`. It is the *else-branch of a declarative condition* (e.g. `has_existing_codebase == true`), chosen up-front. There is **no runtime "producer DLQ'd → activate fallback step" path**. The `loader.py:343` `fb` support is branch-validation, not failure recovery.

### 3.2 The problem
A producer (AI) step feeds a mechanical sink via `depends_on`. On retry-exhaustion the producer is dead-lettered (`status='failed'`, `retry.py:146` `DLQAction`), and a failed dependency **permanently blocks** the dependent sink (`src/infra/db.py:4622`). For time-sensitive comms (crisis, incident) the sink IS the fallback — it ships a canned holding statement. A blocked sink ⇒ nothing ships ⇒ worst outcome.

### 3.3 The mechanism (chosen: producer completes-with-sentinel)
A step may carry `degrade_on_exhaustion: true`. When such a step exhausts retries, instead of `DLQAction` it terminally **completes with a marked sentinel result** (e.g. `{"degraded": true, "reason": …}`). The dependent sink then runs normally (a `completed` dep resolves — `db.py` ready-check unchanged) and falls back to its canned text on seeing the sentinel / missing real draft, surfacing the degradation in the `founder_action`.

**Why this over a "soft-dependency" variant (producer stays `failed`, sink runs anyway):** KutAI has heavy DLQ-cascade / phase-blocking machinery ([[feedback_dlq_blocks_phases]]). Keeping the producer's state `completed` confines the change to the producer's terminal-verdict path; every downstream blocking/grading/mission-health rule sees an ordinary completed task — no cascade audit needed. A `failed` state would leak into those paths.

**Grade short-circuit is automatic.** The grader only runs on steps in the `ungraded` post-hook waiting-room (`apply.py`). Retry-exhaustion is a separate path (`decide_retry` → terminal verdict) that never enters that room, so a sentinel completion is never graded. No bolt-on needed.

### 3.4 Touch points (producer terminal path only)
- `packages/general_beckman/src/general_beckman/retry.py` — `decide_retry`: when the failing task's context carries `degrade_on_exhaustion`, on exhaustion return a new decision instead of `DLQAction`.
- `src/core/result_router.py` (re-export) + the action type — a new terminal action (e.g. `CompleteDegraded`) carrying the sentinel.
- `packages/general_beckman/src/general_beckman/apply.py` — apply it: `status='completed'`, write/materialize the sentinel result, **no** grade enqueue. (Verify against `_apply_complete` 206 — likely a sibling `_apply_complete_degraded`.)
- `src/workflows/engine/expander.py` — carry the step's `degrade_on_exhaustion` flag into task context (alongside the existing `executor`/`produces` propagation).

### 3.5 Scope — where the flag is set (NOT blanket)
- **crisis** + **incident** producer steps → `degrade_on_exhaustion: true`. Time-sensitive; canned statement must ship.
- **demo_storyboard** + **press_kit** → NO flag. Not time-critical; plain DLQ/retry is acceptable. (press_kit may revisit "ship the N audiences that succeeded" later — out of scope for Plan 2.)

## 4. Implementation order (simplest → hardest)
1. **demo_storyboard** — split existing i2p step `13.demo_storyboard` into AI producer + mechanical writer; repoint 3 dependents at `i2p_v3.json:9582/:9602/:9654`. Simplest (no new workflow, no degrade, no launcher).
2. **press_kit_assemble** — new `press_kit.json`, 4→1 fan-in (§2.2) + `/press_kit` thin launcher (§2.3). Hardest of the two (new workflow + fan-in artifact passing).

Deferred to Plan 3: incident_draft_update, crisis_draft_holding (need §3).

## 5. Landmines (carry-over)
- **lane = `oneshot` only.** Never `overhead`/`ongoing` (parent §9).
- **No concurrent pytest** — shared `kutai.db` WAL deadlock (Plan 1 hit a 16-zombie lock). One `timeout`-prefixed invocation at a time; `tests/` and `packages/*/tests/` are separate conftest roots.
- **Worktree:** `EnterWorktree` name `cps-sp4b-plan2`, `baseRef=head`, branch from `worktree-cps-sp4b` (Plan 1 tip). Main venv python from inside the worktree.
- **Env hazard (2026-06-05):** two KutAI wrappers + 0 llama-server were running. Reconcile to one stack before live-testing.

## 6. What "done" looks like (Plan 2)
- demo_storyboard + press_kit_assemble make no LLM call and import no dispatcher/husam. Producer prompts live in step JSON.
- demo split in `i2p_v3.json` (producer step + mechanical writer; 3 dependents repointed). press_kit is a new `press_kit.json` (4-producer fan-in → mechanical assemble) with a thin `/press_kit` launcher.
- Zero changes to General Beckman.
- Tests pin each split (producer `agent:`-typed, sink mechanical + dispatcher/husam-free, dependents repointed, fan-in reads the 4 one-pagers). Suites green, sequential runs.
- `await_inline=True` remaining after Plan 2: incident + crisis (Plan 3) + SP5 carve-outs (`task_classifier`, `investor_bullets`) + shopping `request()` shim.
