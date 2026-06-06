# SP4b Kickoff — extract LLM out of the 6 mr_roboto executors

**For:** a session designing/building **SP4b** of the CPS migration.
**Date:** 2026-06-05
**Author:** SP4a ship session.
**Parent:** `docs/superpowers/specs/2026-05-27-cps-migration-design.md` (umbrella, rev2) + `docs/superpowers/specs/2026-06-05-cps-sp4a-design.md` (SP4a, §6 names this).
**Predecessors (merged to local `main`):** SP1/SP1.1, SP2, SP3, SP3b, **SP4a** (merge `19f23aa1` — husam-inline for the 4 legit LLM consumers).

---

## ⛔ HARD GATE — do this BEFORE building SP4b (read first)

**SP4b's producers run ON THE PUMP — the exact post-hook/admission substrate that has NEVER dispatched in prod** (the SP3b review found post-hook children were orphaned on a phantom `lane="overhead"`; fixed in code but unexercised live). Do not pour SP4b onto an unproven base. Founder must, via Telegram (never `taskkill`):

1. `.venv\Scripts\pip install -e packages\husam` (also required by SP4a's new `import husam` sites). No install script — manual.
2. `/restart`.
3. **Run ONE real graded mission end-to-end** and confirm `constrained_emit→self_reflect→grade` post-hook children **dispatch on `lane=oneshot`**, rewrite the result, and complete. Check `SELECT lane, status, COUNT(*) FROM tasks GROUP BY 1,2` — want oneshot children completing, not a pile of pending.

If the substrate is broken, fix it first. SP4b is moot until admitted producer tasks actually run. (SP4a shipped independently precisely because husam-inline does NOT use the pump — SP4b does.)

## Mission

The founder ruling (2026-06-05, [[no-direct-dispatcher-from-mechanical]]): **"mr_roboto is mechanical-only, non-LLM. Any LLM execution doesn't belong to it."** The six mr_roboto executors below make LLM calls inside their `run()` — they are LLM tasks wearing a mechanical costume. SP4b **extracts the LLM out of mr_roboto entirely**:

- **LLM producer leaves mr_roboto** → an **admitted agent step** (the agent_type already exists; see matrix) → pump → worker (the sanctioned admitted path — full Beckman admission, model selection, retry/grade, cost, telemetry).
- **mr_roboto keeps ONLY the mechanical sink** — DB reads, redaction, persist the draft, enforce the "never auto-post" contract — taking the **already-produced** text as input. It makes NO LLM call and imports NO dispatcher/husam.

This is shape-(b) (producer + confirm), the sanctioned end state the SP3b review named — NOT SP4a's husam-inline (shape-a), which is only acceptable for non-mechanical consumers.

## The 6 executors (re-confirm with a fresh grep — `feedback_audit_call_sites`)

`rg -n "await_inline\s*=\s*True" packages/mr_roboto/src` and classify each. Current facts (verify line numbers — they drift):

| Executor | File | LLM helper | agent_type | `run()` returns | Invoked from |
|----------|------|-----------|-----------|-----------------|--------------|
| reviews_draft_reply | `reviews_draft_reply.py` | `_call_llm_draft_reply` (:64) | `reviewer` | `{status, reply_draft, auto_posted:False}` | router action `reviews/draft_reply` (`__init__.py:4653`) + (indirectly) reviews flows |
| reviews_classify | `reviews_classify.py` | `_call_llm_classify` (:51) | `reviewer` | `{status, sentiment, theme_tag}` | router action `reviews/classify` (`__init__.py:4642`) **+ cron** `src/app/jobs/reviews_poll_daily.py:64` (direct `classify_run`) |
| crisis_draft_holding | `crisis_draft_holding.py` | `_call_llm_draft` (:87) | `reviewer` | `{status, draft}` | router (`__init__.py:4539`) |
| incident_draft_update | `incident_draft_update.py` | `_call_llm_draft` (:116) | `reviewer` | `{status, draft, redaction_applied}` | router (`__init__.py:4417`) |
| press_kit_assemble | `press_kit_assemble.py` | `_draft_one_pager_llm(spec_text, audience)` (:84) | `planner` | `{status, ...}` | router (`__init__.py:4274`) |
| demo_storyboard | `demo_storyboard.py` | `_enqueue_storyboard_llm` (:78) | `reviewer` | `{status, ...}` | **i2p workflow step** `13.demo_storyboard` (i2p_v3.json:9546, `agent:mechanical`) + router (`__init__.py:4151`) |

**Two invocation shapes → two split mechanisms:**
- **demo_storyboard is the ONLY i2p workflow step.** Split it in `i2p_v3.json`: a `agent`-typed producer step (the storyboard LLM, agent_type `reviewer`) feeding a `mechanical` confirm/persist sibling via `depends_on` + artifact passing. The workflow engine already sequences steps + passes artifacts. (Mind its 3 dependents at :9582/:9602/:9654 — they `depends_on 13.demo_storyboard`; repoint them at the confirm step.)
- **The other 5 are router-action / cron invoked, NOT i2p steps.** No step graph to edit. Here the **caller** must orchestrate: enqueue an admitted agent producer task + a CPS `on_complete` continuation → mechanical persist (the SP3b post-hook CPS pattern, `enqueue(on_complete=, cont_state=)`). For reviews_classify the **cron** (`reviews_poll_daily`) is a legit orchestration layer (not a mechanical) and can drive the producer→continuation chain directly. The `__init__.py` router branches that currently `await <executor>.run(payload)` synchronously must change: either the action becomes the mechanical-sink-only verb (producer admitted upstream), or the branch enqueues producer+continuation. **This is the central design fork — brainstorm it.**

## Design forks to brainstorm (don't skip — `superpowers:brainstorming`)

1. **Producer-step (i2p / declarative) vs CPS-continuation (router/cron / imperative).** demo_storyboard is clearly the former. The 5 router/cron callers: do you (a) lift them into small workflow graphs so the engine sequences producer→confirm, or (b) keep them imperative and have each caller enqueue producer + `on_complete` mechanical confirm? (b) is less invasive; (a) is more uniform. The router's synchronous `Action(status, result)` contract (it currently blocks on `run()`) is the thing CPS breaks — same character as `dispatcher.request` was for SP3b and vision was for SP4a.
2. **Where does the producer's prompt-building live?** Today it's inside the mr_roboto module (`_call_llm_*`). After extraction it belongs OUTSIDE mr_roboto — either as an agent-prompt the worker builds from a real agent_type, or a thin producer module (coulson-adjacent, like `critic_gate.produce_verdict` but located outside mr_roboto). Decide the home; mr_roboto must not retain the prompt.
3. **"never auto-post" contract** (reviews_draft_reply, crisis, incident) — this is mechanical and STAYS in the sink. Confirm the sink, not the producer, owns the no-side-effect guarantee.
4. **reviews_classify has a fire-and-forget child** (`_enqueue_bug_investigation`, :218, NOT await_inline) — that's already correct (a separate admitted task). Don't disturb it; only the `_call_llm_classify` await_inline call is in scope.

## Landmines (carry over from SP3b/SP4a)

- **NEVER `lane="overhead"`** or any lane ∉ {`oneshot`,`ongoing`} — phantom lane the pump never selects → silent orphaning. Use `lane="oneshot"` for admitted producers. (`add_task` persists any lane verbatim — no validation; the defensive validation in `db.py:4464` was a deferred item, may still be open.)
- **`ongoing` lane may also be unpumped** (deferred-items #9) — the orchestrator pump only ever calls `next_task("oneshot")`. If you enqueue producers on `ongoing`, verify something pumps it. Live DB historically shows ALL tasks `lane=oneshot`. Stick to oneshot.
- **No concurrent pytest** — two pytest at once deadlocks shared SQLite + crash-loops live KutAI. One `timeout`-prefixed invocation at a time. `tests/` and `packages/*/tests/` are **colliding conftest roots** — run them in SEPARATE invocations (mixing them errors at collection: "Plugin already registered under a different name"). SP4a hit this.
- **Worktree conftest is a HARDCODED package list** (`conftest.py` `_PACKAGE_SRCS` + eviction set) — already includes general_beckman/yalayut/husam/mr_roboto. Any NEW package SP4b adds must be appended to BOTH.
- **Worktree mechanics:** packages are `pip install -e` against the MAIN checkout; the root conftest injects worktree srcs onto sys.path so worktree edits ARE used. Use the main venv python `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe` from inside the worktree. `worktree.baseRef=head` → EnterWorktree branches from local HEAD (good — local main is ~48 ahead of origin).
- **husam.run RAISES** (ModelCallFailed/RuntimeError), does NOT return a failed TaskResult — but SP4b producers go through the PUMP (admitted enqueue + continuation), not husam-inline, so the failure surface is Beckman's retry/DLQ ladder + the `on_error` continuation, not a try/except. Each executor today has a `_fallback_*` (generic draft / default classification) on LLM failure — decide where that fallback lives now (the mechanical sink's `on_error` continuation is the natural home).

## The substrate you have (use it; don't reinvent)

- **post-hook CPS** — `_enqueue_posthook_llm_child` spawns a raw_dispatch child + `on_complete`/`on_error` continuation → `_apply_posthook_verdict`. Ordered chain + per-source verdict lock + `reconcile_stranded_posthook_chains`.
- **durable continuations** (SP1) — `enqueue(on_complete=, on_error=, cont_state=)`, claim-then-fire CAS, restart-reconcile.
- **husam** (`packages/husam/`) — the admitted single-call worker the pump routes `raw_dispatch` tasks to. (SP4b uses the PUMP path, not inline.)
- **critic_gate** (`mr_roboto/critic_gate.py`) — `produce_verdict` (husam-inline, shape-a) + unused `confirm_gate` (shape-b mechanical seam). The confirm_gate is the **unused template** for SP4b's mechanical sink.

## Read order

1. Umbrella spec rev2 (SP4 bullet + coulson-loop non-goal) + SP4a design §6 (the deferral rationale).
2. The 6 executors + their `__init__.py` router branches (~4151–4663) + `reviews_poll_daily.py`.
3. i2p_v3.json `13.demo_storyboard` + its 3 dependents.
4. SP3 design (`docs/superpowers/specs/2026-05-29-cps-sp3-design.md`) — the post-hook CPS pattern SP4b reuses for the 5 imperative callers.
5. `critic_gate.confirm_gate` (the shape-b mechanical-sink seam) + `produce_verdict`.
6. Memories: [[no-direct-dispatcher-from-mechanical]], [[project_cps_sp4a_shipped_20260605]], [[feedback_singular_dispatcher_caller]], [[project_cps_migration_20260527]].
7. Deferred-items log `docs/handoff/2026-05-30-cps-deferred-and-remaining.md` (#4 critic_gate shape-b, #5 add_task lane validation, #9 ongoing-lane).

## Workflow

1. **Brainstorm** (`superpowers:brainstorming`) — settle fork #1 (producer-step vs CPS) and #2 (producer home). Decide per-caller in a matrix.
2. **Write SP4b spec** `docs/superpowers/specs/YYYY-MM-DD-cps-sp4b-design.md`.
3. **Write SP4b plan** (`superpowers:writing-plans`), TDD task breakdown.
4. **Execute subagent-driven** (`superpowers:subagent-driven-development`) in a fresh worktree (`EnterWorktree` name `cps-sp4b`). One implementer at a time; two-stage review per task; NO concurrent pytest; merge `--no-ff` + re-verify on merged tree.

## What "done" looks like

- None of the 6 mr_roboto executors makes an LLM call or touches the dispatcher/husam. Each LLM hop is an admitted agent task on the pump; mr_roboto holds only the mechanical sink (DB + persist + never-auto-post).
- demo_storyboard split in i2p_v3.json (producer agent step + mechanical confirm; dependents repointed). The 5 router/cron callers orchestrate producer + CPS confirm per fork #1.
- Tests pin each split (producer admitted, sink mechanical-only, contract preserved, fallback path covered). Suites green, sequential runs.
- After SP4b, remaining `await_inline=True` users = SP5 carve-outs (`task_classifier`, `investor_bullets`) + the shopping `request()` shim only → shopping migration, then **SP5 deletes the primitive**.

## Working environment

- Windows, Python 3.10, venv `.venv`. Worktree-absolute python inside worktrees. `rtk git ...`. Push to `main` directly; merge `--no-ff`, re-verify on the merged tree. Live KutAI runs from the main checkout; founder restarts via Telegram.
