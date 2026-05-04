# Handoff — Runtime Extraction Phase A Complete

Anchor: 2026-05-04. User: sakircimen@gmail.com.
Caveman mode in body. Code/commits: normal.

## TL;DR

Killed BaseAgent. **base.py 4092 → 179 LOC (96% reduction)** in one session. Multi-call orchestration is now its own architectural peer at `src/runtime/`, sitting between Orchestrator and Dispatcher. Phase A done. Phases B/C/D pending.

## Read first (in this order)

1. `docs/superpowers/specs/2026-05-04-runtime-extraction-design.md` — architecture: 3-lane dispatch (mechanical / direct / react), runtime as new peer, hoca attaches to admission + (future) runtime, dispatcher slim-down.
2. `docs/superpowers/plans/2026-05-04-runtime-extraction.md` — phased tasks. Phase A items A.1-A.13 all checked off.
3. `docs/handoff/2026-05-04-base-py-audit.md` — original 4092-LOC audit. Categorization that drove the extraction.
4. `docs/handoff/2026-05-04-kill-agents.md` — predecessor handoff that started this thread.

## What shipped (commit chain on `main`)

| SHA | Phase | What |
|---|---|---|
| `b44f713` | A.1-A.7 | parsing.py, window.py, validation.py, guards.py, escalation.py, reflection.py, checkpoint.py, tools.py, context.py — 9 modules, 1621 LOC moved |
| `363cdf3` | A.8 | react.py — 1300 LOC ReAct loop moved verbatim with profile substitutions |
| `7cb1fac` | A.9-A.10 | single_shot.py + runtime entry (`src/runtime/__init__.py::execute`) |
| `dd6c779` | A.11 | runtime self-sufficient — direct imports replace profile._X callbacks; 24 unused delegates deleted from base.py |
| `05796e6` | A.12 | `_maybe_constrained_emit` → `src/workflows/engine/constrained_emit.py` |
| `b69c7a6` | A.13 | `_build_model_requirements` → `packages/fatih_hoca/src/fatih_hoca/requirements_builder.py` |

## Final state

```
src/agents/base.py  (179 LOC)            # profile interface only
   - profile attrs (name, allowed_tools, max_iterations, ...)
   - get_system_prompt (subclass override)
   - execute (delegate to runtime)
   - _build_context wrapper (skill-injection per-execution copy plumbing)
   - re-exports (SIDE_EFFECT_TOOLS, GuardCorrection, unwrap_final_answer, ...)

src/agents/*.py     (20 subclass files, untouched)   # prompt suppliers

src/runtime/        (12 modules, 4082 LOC)
   __init__.py            (288)  public execute() entry + setup phase
   react.py              (1366)  ReAct loop
   context.py             (992)  prompt + RAG + skills + retry-hint assembly
   parsing.py             (332)  ReAct DSL parser
   guards.py              (260)  sub-iter guards
   window.py              (227)  ctx-window count/trim/prune
   tools.py               (142)  tool VM helpers (partition, perm, schema, idem)
   checkpoint.py          (107)  state save/restore + log
   reflection.py          (104)  self-reflect post-final review
   single_shot.py         (102)  one-call execute path
   escalation.py           (83)  message trim on mid-task escalate
   validation.py           (82)  refusal/length/empty + samet wrappers

src/workflows/engine/constrained_emit.py  (208)   workflow post-hook

packages/fatih_hoca/src/fatih_hoca/
   requirements_builder.py  (200)   build ModelRequirements next to AGENT_REQUIREMENTS
```

## Bug fixes caught + applied during port

1. **`tools.build_litellm_tools`** — first port of A.4 removed the protective filter that strips `final_answer`/`clarify` from the exclude set. Without it, `exclude={'final_answer'}` would strip the loop-termination pseudo-tool. Re-added the filter.
2. **`context.build_user_context`** — original mutated `self.allowed_tools.append(...)` for skill-injected tools. Class-attr leak across tasks. Fix: `build_user_context` returns `(ctx_str, list_of_injected_tools)`; `BaseAgent._build_context` wrapper applies them to a per-execution mutable copy via the existing `_original_allowed_tools` snapshot pattern (the same one `tools_hint` and auto-strip use).

## Architecture decisions that survived multiple challenges

User pushed back on every loose framing. Final agreed shape:

- **Runtime is a missing peer**, not a "kill agent class" rename. Beckman owns multi-task. Dispatcher owns one-task-one-call. Runtime owns one-task-many-calls. Filling a real architectural gap.
- **Dispatcher stays as choreography layer**, not "dumb pipe." Substance: ask Hoca + load DaLLaMa + KDV bracket + Hallederiz + retry. Removing it means runtime absorbs Hoca/DaLLaMa/KDV/Hallederiz imports — same god-class problem one floor down.
- **Hoca attaches to dispatcher (today) and runtime (planned Phase C)**. Total Hoca touchpoints stay at 2 (Beckman admission + dispatcher's per-iter retry, OR runtime's per-iter selection). Option C in the design doc moves dispatcher's slot to runtime.
- **Runtime never imports `ModelRequirements` semantics** even after Phase C — narrows to `select(task, failures) → Pick` query interface. Runtime emits failures via `failures` arg; Hoca interprets. No `reqs.escalate()` in runtime.
- **Three lanes (`task.runner`)** explicit polymorphism. Beckman doesn't pick lane; task carries it.
- **Subclasses stay as prompt suppliers.** User direction: "agents can serve as prompt suppliers no worries." Don't kill them.

## What's NOT done

- **Phase B — package extraction.** `src/runtime/` → `packages/<turkish-name>/src/<name>/`. Naming undecided (`yapan_eleman`, `mesai`, `is_eri`, `agent_runtime` all on table — user said "forget about naming"). Mechanical: pyproject.toml, src layout, editable install, tests/, README. `src/runtime/__init__.py` becomes thin shim.
- **Phase C — dispatcher slim-down.** Move Hoca call out of `_do_dispatch`, runtime owns per-iter selection. Single retry surface (collapses today's two retry layers). Dispatcher becomes `dispatcher.execute(pick, messages, task_ref)` ~150 LOC primitive.
- **Phase D — `task.runner` field.** Schema migration + producer updates + orchestrator dispatch by `task.runner`.

Per plan estimate: B = ~1 day, C = ~3 days, D = ~half day.

## Test state

19/19 pass across:
- `tests/test_artifact_summarizer_agent.py`
- `tests/test_grader_agent.py`
- `tests/test_constrained_emit.py` (updated to call `maybe_apply` directly)
- `tests/test_missing_artifact_note.py`
- `tests/test_per_artifact_checklist_envelope.py`
- `tests/test_prompt_noise_reduction.py`

NO trace-replay was run yet — Phase A.11 plan called for one but session capacity ran out. **Do this before Phase B.** Capture 10 production traces from `task_state` table, replay through new pipeline, diff TaskResult dicts. If discrepancies surface, they trace to A.5 (context builder, biggest mover) or A.8 (ReAct loop) — those are the two extractions with the most subtle behavior to preserve.

## Known follow-ups inside what shipped

- `_build_context` wrapper in BaseAgent still bridges skill-injection. Could move into runtime entry alongside other setup phase logic. ~30 LOC. Optional cosmetic.
- `_TOOL_SCHEMAS_BY_NAME` re-export at top of base.py is unused — A.11 substitutions made every reference go direct. Delete.
- `record_model_call` likely double-counted (dispatcher + runtime both call). Audit during Phase C dispatcher slim-down.
- `safe_log_conversation_p` wrapper in react.py (binds profile.name). Tiny. Could fold into checkpoint.py if signature changes.

## Subagent lessons

- Caveman tone briefings + clear file paths + explicit verification commands worked cleanly.
- Subagent on A.4 (tools) silently flipped a behavior (final_answer excludable) and reported success. **Always inspect the diff** before accepting subagent claims, especially around protective filters and short-circuits. The "all tests pass" report doesn't catch behavior the tests don't exercise.
- Parallel dispatch fine for non-overlapping module extractions (A.2/A.3/A.6 ran in parallel safely; A.7 trio same). Sequential for big ones (A.4 then A.5).

## Quick-resume command for next session

```bash
git log --oneline -8       # see commit chain ending at b69c7a6
wc -l src/agents/base.py src/runtime/*.py   # confirm 179 LOC base + 12 runtime modules
```

Then read the 4 docs in order at top of this memo, pick up at Phase B.
