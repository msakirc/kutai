# SP4b Plan 3 handoff — press_kit (rebuild) + incident + crisis

**For:** a session building **SP4b Plan 3**.
**Date:** 2026-06-07.
**Author:** SP4b Plan 2 session.
**Spec:** `docs/superpowers/specs/2026-06-07-cps-sp4b-plan2-design.md` (esp. §3 degrade mechanism), parent `docs/superpowers/specs/2026-06-05-cps-sp4b-design.md` §4–§7.
**Plan 2 branch (reference):** merged to `main` `19066139`. Plan 1 (reviews CPS) = `worktree-cps-sp4b` (still unmerged at time of writing).

---

## Why this matters — Plan 3 is the last gate before SP5

**SP5 = delete the `await_inline=True` primitive** (the blocking inline LLM call on `beckman.enqueue`). It can only happen when the ONLY remaining callers are the SP5 carve-outs + the shopping shim (which SP5 itself migrates/deletes). Live `await_inline=True` call sites on `main` today (re-grep — `rg -n "await_inline\s*=\s*True" packages/ src/`):

| Call site | Owner | Status |
|-----------|-------|--------|
| `mr_roboto/crisis_draft_holding.py:155` | **Plan 3** | this handoff |
| `mr_roboto/incident_draft_update.py:180` | **Plan 3** | this handoff |
| `mr_roboto/press_kit_assemble.py:109` | **Plan 3** | this handoff |
| `mr_roboto/reviews_classify.py:97` | **Plan 1** | branch `worktree-cps-sp4b`, NOT merged |
| `mr_roboto/reviews_draft_reply.py:123` | **Plan 1** | branch `worktree-cps-sp4b`, NOT merged |
| `src/core/task_classifier.py:284` | SP5 carve-out | SP5 migrates |
| `src/app/jobs/investor_bullets.py:211` | SP5 carve-out | SP5 migrates |
| `src/core/llm_dispatcher.py:273` (`request()` shim) | shopping | T11 delete, evidence-gated on 1 live shopping_v3 run post-restart ([[project_shopping_sp5_coupling_20260605]]) |

**To unblock SP5:** (1) merge Plan 1 (reviews ×2 off the primitive), (2) finish Plan 3 (these 3 verbs off the primitive), (3) retire the shopping `request()` shim (T11). Then SP5 migrates the 2 carve-outs and deletes `await_inline` + the shim. Plan 2 (demo) already removed one caller. **Plan 3 clears 3 of the 5 mechanical-verb callers.**

---

## What Plan 2 actually shipped (scope was narrowed mid-flight)

Plan 2 set out to split 4 LLM-bearing mr_roboto verbs (demo_storyboard, incident, crisis, press_kit). A thorough runtime review collapsed it to **demo_storyboard only**. Shipped:

- **demo_storyboard**: LLM removed from the mechanical verb. New `agent:reviewer` producer step `13.demo_storyboard_draft` in `i2p_v3.json` drafts the storyboard (prompt in step JSON); the existing mechanical `13.demo_storyboard` sink reads the producer's materialized file, normalizes, writes `demo/storyboard.json`. The 3 downstream demo steps keep depending on the sink id (no repoint). Producer→sink workspace wiring is correct (canonical `get_mission_workspace(mission_id)` derive + mission-prefixed `produces`).
- ZERO General Beckman changes.

Everything else was **deferred to Plan 3** and BACKED OUT of the branch (press_kit code restored to base). Plan 3 rebuilds press_kit from scratch and adds incident + crisis.

---

## Why each was deferred (the blocking facts — verify still true)

### press_kit — 3 blockers
1. **No spec input (design flaw).** `/press_kit <product_id>` launches a fresh mission with `initial_input={"product_id": <id>}` — just an id string. The 4 `agent:planner` producers have `depends_on:[]`, no `input_artifacts`, an EMPTY workspace (no codebase), and no products table to resolve the id against. The planner sees only `{"product_id":"x"}` in "Additional Context" → **hallucinates the entire one-pager.** Evidence: `coulson/context.py:500-608` (artifact injection needs `input_artifacts`), `context_injection.py:131-146` (workspace_snapshot only for non-empty tree), `runner.py:375-400` (initial_input → blackboard, not a spec file).
   - **Plan 3 must decide press_kit's spec source.** Options: (a) `/press_kit` operates on an existing i2p mission — resolve product_id → that mission's `.charter/product_charter.md`, copy/inject it as an `input_artifact` to the producers; (b) a mechanical pre-step gathers a product record; (c) founder supplies spec text at launch. This is a **brainstorming-level decision**, not mechanical.
2. **`press_kit_freshness` post-hook raises.** Registered (`posthooks.py:786`) + handler (`posthook_handlers/press_kit_freshness.py`) + mr_roboto verb (`__init__.py:4325`) all exist, BUT `apply._posthook_agent_and_payload` has **no dispatch branch** → `raise ValueError("unknown posthook kind")` (`apply.py:2682`) when a step fires it. Fix = add a branch mirroring `incident_update_review` (`apply.py:2641`). **This is in General Beckman** → only allowed once Plan 3 is cleared to touch Beckman. (Or drop the post-hook — it's advisory freshness.)
3. **Mechanical-step payload wiring** (shared with demo, see below): `2.assemble` needs `product_id` + `workspace_path`; the expander does NOT inject them into mechanical payloads. The sink must derive `workspace_path = get_mission_workspace(mission_id)` itself (canonical pattern), and `product_id` must come from `initial_input` — which means the expander/launcher must thread it, OR the assemble verb resolves it from the mission row. Decide alongside #1.

The Plan 2 press_kit implementation (gutted verb reading 4 one-pager files + `press_kit.json` + `/press_kit` launcher + fan-in) was correct in shape and is in branch history (commits ccb1a7ae, bc9f32d1, 05d06a35, b409a28c) before the backout commit `50eb1744` — **reuse it as the starting skeleton**, but solve #1/#2/#3 first.

### incident + crisis — need `degrade_on_exhaustion` (Beckman)
Both ship a canned fallback today (`incident_draft_update._fallback_draft`, `crisis_draft_holding` canned variants). Splitting them into producer+sink WITHOUT a fallback mechanism regresses that: a DLQ'd producer blocks the mechanical sink → nothing ships (worst case for time-sensitive comms). The fix is the **`degrade_on_exhaustion`** mechanism (spec §3): on retry-exhaustion the producer completes-with-sentinel instead of DLQ, so the sink runs and falls back. **This touches General Beckman** (`retry.py` decide_retry, `apply.py`, a new terminal action) → deferred until Plan 3 is cleared for Beckman.
- Engine `fallback_steps` do NOT solve this — they are condition-driven (`conditions.py:143-177`), not failure-driven. Confirmed dead end.
- Grade short-circuit is automatic (exhaustion path never enters the `ungraded` grader gate).
- Split shapes already designed: incident = `2a.redact_alert`(mech) → `2b.draft_update`(reviewer) → `2c.finalize_draft`(mech), repoint `2.publish`→`2c` (parent §5.1); crisis = new `crisis_comms.json`, mech pre-step (event+playbook) → reviewer producer → mech sink (parent §4).

---

## Pre-existing gap surfaced (NOT introduced by Plan 2) — whole demo pipeline workspace plumbing

The demo pipeline (`demo/record`, `demo/edit`, `demo/caption`, `demo/accessibility_pass`) is **dormant — no evidence it has ever run end-to-end through the engine.** Every demo step reads `payload.get("workspace_path") or ""`, but the expander never fills `workspace_path` for mechanical steps and there is no `chdir` into the mission workspace. So those verbs read/write relative to the orchestrator CWD, not the mission workspace.

Plan 2 fixed this for the `13.demo_storyboard` sink only (it now derives `get_mission_workspace(mission_id)`). The **downstream** demo steps still have the gap, so the full demo pipeline still won't flow end-to-end. **Follow-up (separate from SP4b):** migrate `demo_record`/`demo_edit`/`demo_caption`/`demo_accessibility` to derive `workspace_path = get_mission_workspace(int(mission_id))` (the canonical pattern; ~1 line each) + ensure their router branches pass `mission_id=payload.get("mission_id") or task.get("mission_id")`. Then a real `public_demo` mission can validate the whole chain. Note these verbs shell out to Playwright/ffmpeg → need an integration/manual run, not just unit tests.

---

## Landmines (carry over — all bit during Plan 2)

- **No concurrent pytest.** Two pytest invocations (or one + live KutAI) deadlock on the shared `kutai.db` WAL. Plan 2 hit this — 4 zombie pytest had to be killed. Run ONE `timeout`-prefixed invocation at a time. `tests/` and `packages/*/tests/` are separate conftest roots. DB-touching suites (`tests/workflows/`) will hang against the live stack — stop KutAI first or skip.
- **Live KutAI stack present** (PIDs 31628 + 40528 venv/global, 2 wrappers; the documented env hazard). Never kill these or llama-server; only kill pytest python processes (filter by `CommandLine -like '*pytest*'`).
- **Worktree has no `.venv`** — use the main repo venv: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe`. pytest from the worktree picks up worktree srcs via root conftest.
- **lane = `oneshot` only** for admitted producers.
- **Offline tests miss runtime wiring.** Plan 2's per-task TDD passed while the live producer→sink path was broken, because tests called `run()` with explicit args. ALWAYS add a router-level dispatch test (drive `mr_roboto.run(task)` with the real workflow payload shape) AND trace the materializer write-path vs the sink read-path. The materializer (`hooks.py:314`) writes `produces` to `WORKSPACE_DIR/<entry>` and SKIPS mechanical steps; mission isolation comes ONLY from a `mission_{mission_id}/` prefix in the `produces`/payload string (expanded by `_substitute_payload`, expander.py:756).

---

## Suggested Plan 3 order
1. Brainstorm press_kit spec-input (blocker #1) — founder decision.
2. (If cleared for Beckman) build `degrade_on_exhaustion` (spec §3) — foundational for incident+crisis; add `press_kit_freshness` dispatch branch (#2).
3. press_kit rebuild (reuse Plan 2 skeleton + spec-input + workspace derive).
4. incident split (degrade flag on producer).
5. crisis new workflow (degrade flag on producer).
6. (Optional, separate) demo-pipeline downstream workspace migration.
