# Handoff — Prompt Simplification + Debugging (next session)

User: sakircimen@gmail.com
Caveman mode: drop articles/filler/pleasantries. Fragments OK.
Code/commits/security: write normal.

Anchor: 2026-04-30T19:13Z

---

## Session arc

User reported observation: **single LLM call hit 90k input tokens**.
Goal — shrink prompt without losing quality. Earlier handoff at
`docs/handoff/2026-04-28-prompt-simplification.md` framed the problem.

### What shipped (chronological)

| Hash | Branch | What |
|------|--------|------|
| `fa760ba` | main | per-section + per-iteration prompt size telemetry in `_build_context` and ReAct loop |
| `1171667` | main | `_fetch_deps` artifact-granular fetch with `_summary` preference (workflow-step branch added before legacy task-id path) |
| `6bec6dd` | main | P1 — Additional Context markdown render (drops json.dumps wrapper) + tool-name list trim in INPUT ARTIFACTS / deps headers |
| `ee62b3b` | `prompt-simplification-p2` | P2 — Additional Context deny-list (drop 17 meta keys) + remove system-prompt INPUT ARTIFACTS warning |

`fa760ba` and `1171667` got into origin/main via parallel-session
push (not cleanly visible in this session's reflog — see git
reflog for evidence of `rebase: checkout origin/main` at HEAD@{13}
that absorbed them).

P2 is **branch-only**. Validation procedure at
`docs/research/2026-04-30-p2-validation-procedure.md`.

### Prompt-section inventory (do NOT relearn)

`_build_context` user prompt blocks, in emit order:

1. `## Task (PRIMARY)` — title + description
2. `## Current Workspace State` (if `task_context.workspace_snapshot`)
3. `## Prior Tool Result` (if `task_context.tool_result`)
4. `## User Clarification` (if applicable)
5. `## Missing Input Artifacts` (when input_artifacts declared but
   absent from store)
6. `## Additional Context` — markdown render of survivors after
   `_skip` + `_drop_meta` (P2 only) + `_*` filters
7. `## Results from Previous Steps` — `_fetch_deps`. **Workflow-aware
   branch** prefers `<name>_summary` over full, falls back to legacy
   task-id path for non-workflow tasks. Budget unbounded scaling
   confirmed: `CONTEXT_FRACTION = 0.40` × model_ctx × weight/total
   weight = 21k tokens for coder/128k.
8. `## Results from Prior Steps (Inline)` — implementer profile only
9. `## Recent Conversation` — first 2 raw exchanges, capped 400c each
10. Ambient / project profile / blackboard / skills / api / RAG /
    prefs / memory — gated layers per `context_policy.py`
11. **TAIL** `## Required Output Format` — schema example
12. **TAIL** `## IMPORTANT: Previous Output Was Invalid` (retry)
13. **TAIL** `## Your Previous Output` — `_prev[:4000]` hard cap

System prompt blocks:

A. Persona — DB row from `prompt_versions` OR hardcoded fallback
B. Tools catalog — TOOL_REGISTRY filtered by `allowed_tools`
C. Iteration count line
D. INPUT ARTIFACTS warning (P1 trim, P2 drop)
E. TOOL USE warning
F. SECURITY warning

ReAct loop appends `assistant content + user tool_result` per
iteration. Pruner runs `_prune_tool_results_to_fit`. Compressor at
80% ctx threshold. Checkpoint persists full list. Retry with
`_schema_error` SKIPS checkpoint → fresh build (line 2027 in current
base.py).

`_maybe_constrained_emit` is a separate post-execution OVERHEAD call
with its own messages. Draft capped 30k chars + schema. Bounded.

### Empirical anchors (from `docs/research/2026-04-28-token-distribution.md`)

- Input p50 ~7k, p90 ~12k, p99 ~27k tokens (n=66, i2p only,
  2026-04-26 → 2026-04-28).
- Top output: 4423 architect 4.5b openapi_spec = 100k out / 9k in.
  Output dominates cost.
- 90k input never observed in that scan — likely newer or on a
  non-BaseAgent path (constrained_emit / grader / summarizer have
  0% coverage in the report).
- `AGENT_REQUIREMENTS.estimated_output_tokens` is wildly off on
  heavy tail: analyst ×8.34, writer ×3.86, architect ×5.52.

### What's been ruled out

- `_schema_error` / `_prev_output` leak into Additional Context: NO,
  `_*` prefix excludes them.
- `input_artifacts` listed as JSON dump key in extra: NO, in `_skip`.
- ReAct messages accumulating across retries: NO, retry with
  `_schema_error` skips checkpoint.
- Hard caps on deps / RAG: REJECTED — silent quality loss
  unacceptable.
- Pull-on-demand (D): REJECTED — math shows iteration multiplier
  makes total billed input WORSE except in best-case selective
  fetching, which local 8b models can't reliably perform.

### What's still open

| ID | Idea | Status |
|----|------|--------|
| C | Per-step custom shapes (phase-keyed budgets, agent skip-list) | Designed but not started |
| C-RAG | Skip RAG for agent_types where it doesn't help | Designed but not started |
| C-Cache | Hoist stable system prompt + workflow_context to prefix-cache zone | Designed but not started — cloud only |
| Smart-Summary | Make `_structural_summary` target the fields the consuming step actually needs (currently fixed `target=1500`) | Designed but not started |
| Empirical | Re-run 2026-04-28 token-distribution scan after telemetry has accumulated 24h | Pending data |
| Pull-on-demand (D) | RESEARCH-ONLY, not in scope |

---

## Debugging focus for next session

### Tier 1 — verify P1 + ship-or-revert P2

1. **Pull telemetry distribution** (per-section size). Command in the
   P2 validation procedure doc.
2. **Compare sections**: confirm `Additional Context` is bounded
   below ~3kB on workflow steps. Confirm deps block is hitting the
   summary form on the heavy steps (4423 / 4441 / 4409 / 4433).
3. **Decide on P2**: per the validation criteria. If green,
   `git checkout prompt-simplification-p2`, fast-forward main, push.
4. **Decide on next quality-neutral lever**: Smart-Summary or C-RAG
   skip-list.

### Tier 2 — verify the artifact-granular fetch is firing as designed

Look for `_fetch_deps artifact-mode: X/Y resolved (...)` log lines.
If `Y > X` consistently → many declared input_artifacts are missing
from the store, fetcher is falling through to legacy. That's a
post-hook bug, not a fetcher bug. Trace via post-hook execution
order.

If artifact-mode never fires at all → workflow steps aren't getting
`is_workflow_step=True` + `input_artifacts` set. Check expander.

### Tier 3 — chase the actual 90k call

Empirical scan didn't catch a 90k input. Possibilities:

- **Constrained-emit input** — schema_text + draft (capped 30k chars
  + schema JSON). Schema for multi-artifact steps can be 5-10k
  tokens. Worst case ~12k tokens. Not 90k.
- **ReAct tool result accumulation** — initial 25k context + 4
  iterations of 15kB tool results = 85k accumulated. New
  `messages state iter` log line catches this. **This is the most
  likely culprit.**
- **Checkpoint resume** — restored messages list from prior attempt's
  ReAct state. `BaseAgent.run` only logs context size on fresh starts;
  resume doesn't log. The new `messages state iter` log fires on
  every iteration regardless of checkpoint state, so this is now
  visible.
- **Specific high-input agent** — e.g. analyst on 5.11b had 27k
  input in scan. Could climb on retry.

Action: grep `messages state iter` lines for `total > 60000c` after
24h. The role breakdown will show whether bloat is `user` (tool
results) or `assistant` (agent's own output bouncing back).

---

## Architecture facts (don't relearn)

- **DB_PATH**: `C:\Users\sakir\ai\kutai\kutai.db` per `.env`.
  `data/kutai.db` is orphan.
- **Workflow JSON** has mtime cache. Edits propagate without restart.
- **base.py code edits** need `/restart` via Telegram.
- **Workflow loader** is `src/workflows/engine/loader.py`.
- **Artifact storage**: `ArtifactStore` (`src/workflows/engine/artifacts.py`).
- **`_summary` post-hook** auto-fires when output > 3000 chars
  (`hooks.py:1043`). Stores `<name>_summary` via
  `_structural_summary(target=1500)`.
- **base.py step-refresh** at dispatch resyncs description /
  done_when / input_artifacts / output_artifacts / artifact_schema
  / tools_hint / difficulty + free-form context.
- **ReAct loop**: each iteration appends `{role:assistant} +
  {role:user, content:tool_result}`. Pruner trims when input estimate
  > 80% ctx. Checkpoint preserves full list. Retry-with-schema-error
  skips checkpoint.
- **CONTEXT_FRACTION = 0.40** × model_ctx for available layer
  budget. Distributed by `LAYER_WEIGHTS` (deps 5, prior 4, skills 3,
  rag 3, convo 2, board 2, profile 1, ambient 1, api 1, memory 1,
  prefs 1).

---

## DON'T

- DON'T hard-cap deps or RAG. Quality loss unacceptable per user.
- DON'T merge P2 to main without empirical validation per the
  validation doc.
- DON'T strip the schema/example block (load-bearing — 0856cd5).
- DON'T drop `_schema_error` retry hint or per-artifact checklist.
- DON'T mass-rewrite all input_artifacts at once — surface offenders,
  fix surgically. (Largely subsumed by the artifact-mode fetcher
  preferring `_summary` automatically — manual JSON audit is now
  rarely needed.)
- DON'T `call_model()` directly — `LLMDispatcher.request()` only.
- DON'T `taskkill llama-server`. Use `/restart` via Telegram.
- DON'T `pytest` without timeout: `timeout 60 pytest tests/...`

---

## Memory entries

- `project_session_20260428_arc.md` (today, write after ship)
- (this handoff covers session-end)

Update memory after next session if architectural conclusions change.

---

## Where to start (fresh session)

1. **Pull telemetry distribution** — per-section sizes from
   `kutai.jsonl` since `fa760ba` ship.
2. **Decide P2 ship-or-revert** per validation criteria.
3. **Pick next quality-neutral lever**: Smart-Summary or C-RAG.
4. **Investigate 90k path** via `messages state iter` log lines.

User prefers: surface findings + propose specific cuts before any
destructive change. No unilateral compression.
