# Runtime Extraction — Architecture Spec

Anchor: 2026-05-04. User: sakircimen@gmail.com.
Continues `docs/handoff/2026-05-04-kill-agents.md` and `docs/handoff/2026-05-04-base-py-audit.md`.

## Problem

`src/agents/base.py` is 4092 LOC. Owns ReAct loop, tool exec, context build, parse, ctx-window, sub-iter guards, escalation, checkpoint, self-reflect, constrained-emit, model-requirements building, response normalization. Mix of multi-call orchestration concerns + selection-vocabulary leaks + workflow-step concerns. Two retry layers stacked (dispatcher in-call + runtime escalation) — every cross-cutting bug for the past month landed at this seam.

## Decision

Extract a new architectural peer: **Runtime**. Sits between Orchestrator and Dispatcher. Owns multi-call DSL interpretation + tool VM + state. Dispatcher becomes a thin one-attempt execute primitive.

## Layering

```
Beckman.next_task()                                 # admit, queue, lifecycle, retry
   │  - calls fatih_hoca.preview(task) → Pick       #   for pool-pressure gate
   │  - attaches preselected_pick to task spec
   ↓
Orchestrator pump dispatches by task.runner
   │
   ├── runner=mechanical → salako.run(task)
   │
   ├── runner=direct → dispatcher.execute(preselected_pick, messages)
   │      (graders, structured_emit, classifier, raw_dispatch overhead)
   │      on CallError: bubble to Beckman lifecycle → retry-admit with failure
   │
   └── runner=react → runtime.execute(profile, task)
              │
              ↓ per iter
        ┌────────────────────────────────────────────────────────────┐
        │ runtime.iter:                                              │
        │   pick = preselected_pick if iter==0 else                  │
        │          fatih_hoca.select(task, failures)                 │
        │   call_result = await dispatcher.execute(pick, messages)   │
        │   if CallError: append to failures, retry or give up       │
        │   parse content → action verb (DSL)                        │
        │   tool_call    → tools.exec(name, args)  → loop            │
        │   multi_tool   → tools.exec_parallel(...) → loop           │
        │   final_answer → samet quality, return TaskResult          │
        │   clarify      → return needs_clarification                │
        │   sub-iter guards (format/hallucination/search) — re-prompt│
        │     SAME messages, no new pick, doesn't burn outer budget  │
        │   on every iter: heartbeat bump, checkpoint save           │
        └────────────────────────────────────────────────────────────┘
```

## Layer responsibilities

| Layer | Owns | Does NOT own |
|---|---|---|
| **Beckman** | admission, queue eligibility, pool-pressure gate (calls Hoca preview), lifecycle, retry on terminal failure | how a task runs internally |
| **Runtime** | ReAct DSL parse, tool VM, multi-call state machine, per-iter Hoca query, transport-failure retry, sub-iter guards, escalation (signal-based), checkpoint, self-reflect, context assembly | model selection internals, transport, single-call shapes |
| **Dispatcher** | one attempt: load + bracket + transport. Pure execute primitive. | selection, retry, ReAct vocabulary, tools |
| **Hallederiz** | one transport attempt (litellm + streaming + samet mid-stream kill) | which model, when to retry |
| **Fatih Hoca** | given (task, failures) → Pick. Failure-adaptive scoring, swap budget. | how the call is made |
| **DaLLaMa** | local model process (load, swap, idle, health) | selection |
| **KDV** | cloud capacity (rate limits, in-flight) | selection |
| **Samet** | output quality verdict (degenerate/repetitive heuristics, salvage) | what to do about it |

## Three lanes — task.runner

New required field on tasks at admission:

```python
class Runner(StrEnum):
    MECHANICAL = "mechanical"
    DIRECT     = "direct"        # single LLM call, raw content out
    REACT      = "react"         # multi-call ReAct loop with tools
```

Set at task creation by the producer:
- Mechanical executors (salako sub-tasks): `runner=mechanical`
- Single-call OVERHEAD (graders, structured_emit, classifier, raw_dispatch): `runner=direct`
- Agent multi-call dispatches: `runner=react`

Orchestrator pump dispatches by `task.runner` — no other lane decision logic.

Backward-compat: tasks without `runner` default to `react` if `kind=main_work` else `direct`. Migration field-fill is one-shot.

## Dispatcher's new shape (target ~150 LOC)

```python
async def execute(pick: Pick, messages: list[dict], task_ref: dict, **call_opts) -> CallResult:
    """One attempt. No selection, no retry."""
    if pick.is_local:
        await dallama.ensure(pick.model)
    if pick.is_cloud:
        await kdv.begin_call(pick.provider)
    try:
        prepared = prepare_messages(messages, pick)  # secret redact, thinking adapt
        result = await hallederiz.call(pick, prepared, **call_opts)
        await record_model_call(model=pick.model, cost=result.cost, latency=result.latency)
        return result
    finally:
        if pick.is_cloud:
            await kdv.end_call(pick.provider)
```

NO Hoca call. NO retry loop. NO ReAct awareness. Pure attempt primitive.

`dispatcher.request()` (legacy alias) stays as Beckman shim for backward compat; internally enqueues a `runner=direct` task.

## Runtime's narrow Hoca surface

Runtime imports from fatih_hoca:
- `select(task, failures: list[Failure]) -> Pick`
- `Pick` (treated opaque; `pick.model` accessed only as telemetry string)

Forbidden imports:
- `ModelRequirements`
- `AGENT_REQUIREMENTS`
- `escalate()`
- `exclude_models` accumulation
- scoring weights, swap budget, capability curves

Reqs are built ONCE at admission (Beckman + classifier or fatih_hoca.requirements_for(task)) and stored on the task row. Runtime reads task as opaque dict; never constructs or mutates reqs.

## Failure-signal interface (replaces `reqs.escalate`)

Runtime emits structured failures, Hoca interprets:

```python
@dataclass
class Failure:
    kind: str                    # "transport" | "tool_exec" | "format_parse" | "quality_low" | "hallucination_guard" | "search_required" | "validation"
    iteration: int               # which outer iter
    model_used: str              # which Pick produced it
    consecutive: int             # n-in-a-row of same kind
    detail: str                  # short cause
```

Runtime appends Failures to `failures` list, passes to next `fatih_hoca.select(task, failures)`. Hoca's failure-adaptation logic decides: exclude that model, raise difficulty floor, swap budget, etc. Runtime never says "use bigger model."

## Single retry surface

Today: dispatcher hides per-call retries from runtime. Runtime sees opaque "call failed" without knowing which models were tried.

After: runtime's per-iter loop is the ONLY retry layer. Each transport failure is a Failure. Each Hoca pick is a logical attempt. Runtime decides when to give up (iteration cap, time budget, exhaustion).

Transport-failure retry within one logical iter: runtime calls Hoca again with updated failures, dispatcher.execute again. Runtime decides budget — typically 2-3 transport retries before counting toward outer iteration budget.

## What moves out of base.py (final destinations)

| Block | Destination | Reason |
|---|---|---|
| ReAct loop, sub-iter guards, escalation, checkpoint, self-reflect, validation, partial-recovery | **Runtime package** | Multi-call orchestration |
| Tool execution (single + multi + idempotency + permission + audit + metrics + cache) | **Runtime package** (tools submodule) | In-process tool VM |
| Context builder (`_build_context`, `_fetch_deps`, `_build_full_system_prompt`, `_get_available_tools_prompt`, format helpers) | **Runtime package** (context submodule) | Multi-call prompt assembly |
| Response parsing (action vocab, alias map, envelope unwrap, function-call response normalize) | **Runtime package** (parsing submodule) | ReAct DSL |
| Context window mgmt (count/trim/prune) | **Runtime package** (window submodule) | Multi-call accumulator mgmt |
| `_build_model_requirements` (200 LOC) | **Fatih Hoca** (`requirements_for(task) -> ModelRequirements`) | Selection vocabulary |
| `_maybe_constrained_emit` | **Workflow engine post-hooks** (`workflows/engine/post_hooks/constrained_emit.py`) | Workflow-step concern |
| `record_model_call`, `record_cost`, `log_conversation` per-call | **Dispatcher** (single owner per call) | Per-call accounting |

## What stays in `src/agents/`

```
src/agents/
├── __init__.py           # AGENT_REGISTRY unchanged
├── base.py               # ~80 LOC: Profile attrs + execute() shim → runtime.execute(self, task)
└── *.py                  # 20 subclass files unchanged (prompt suppliers)
```

`base.py` becomes:

```python
class BaseAgent:
    """Profile interface for the runtime. Subclasses customize attributes only."""
    name: str = "base"
    description: str = "..."
    allowed_tools: list[str] | None = None
    max_iterations: int = MAX_AGENT_ITERATIONS
    execution_pattern: str = "react_loop"
    enable_self_reflection: bool = False
    min_confidence: int = 0
    can_create_subtasks: bool = False

    def get_system_prompt(self, task: dict) -> str: ...

    async def execute(self, task: dict, progress_callback=None) -> dict:
        from runtime_pkg import execute
        return await execute(self, task, progress_callback)
```

4092 → ~80 LOC. Subclasses untouched.

## Boundary checks (CI-enforceable)

After extraction:
```bash
# Runtime imports nothing from selection/transport internals
grep -rE "from (dallama|kdv|hallederiz_kadir|nerd_herd)" packages/runtime/src/   # → empty
grep -rE "from fatih_hoca" packages/runtime/src/                                   # → only `select, Pick`
grep -rE "ModelRequirements|escalate|exclude_models" packages/runtime/src/         # → empty

# Dispatcher knows nothing about tools or ReAct
grep -rE "TOOL_REGISTRY|execute_tool|action.*final_answer" src/core/llm_dispatcher.py packages/dispatcher/   # → empty

# BaseAgent is profile-only
wc -l src/agents/base.py     # → ≤ 100
```

## Invariants preserved

1. **Iteration budget vs checkpoint orthogonality** (modularization doc, task #1174 lesson). Runtime iterates `range(effective_max_iterations)` fresh per attempt; checkpoint restores context, never control flow.
2. **Streaming partial-content recovery on cancel** — runtime catches `CancelledError`, persists `_partial_content`, re-raises.
3. **Heartbeat bump cadence** — orchestrator's no-progress watchdog dependency. Runtime bumps each iter.
4. **Doğru mu Samet checks** at checkpoint-restore + self-reflection. Runtime imports `dogru_mu_samet`.
5. **DB-versioned prompt override** (`prompt_versions` table) — runtime loads at start of execute(), takes precedence over profile's static prompt.
6. **Skill-injection** mutates per-execution allowed_tools, NOT class attr. Today's mutation of class attr is a bug; runtime fixes via per-execution mutable copy.

## Phased migration

Each phase shippable, no behavior change until last. See `docs/superpowers/plans/2026-05-04-runtime-extraction.md`.

**Phase A — in-tree to `src/runtime/`** (mirrors orchestrator Phase 1).
1. Pure leaf modules (parsing, window, validation, samet integration). BaseAgent delegates.
2. Tool VM extraction (single + multi + idempotency + permission + audit/metrics).
3. Context builder extraction.
4. Sub-iter guards + escalation + checkpoint + self-reflect.
5. ReAct loop + single-shot loop. BaseAgent shim ~80 LOC.
6. `_build_model_requirements` → fatih_hoca; `_maybe_constrained_emit` → workflow_engine.
7. Trace-replay verification (10+ production traces, diff results).

**Phase B — package extraction to `packages/<name>/`** (mirrors beckman / fatih_hoca).
- src layout, pyproject.toml, editable install, tests/, README.
- `src/runtime/__init__.py` becomes thin shim re-exporting from package.
- Naming: TBD (Turkish-named per project convention; not naming-precious).

**Phase C — dispatcher slim-down**.
- Move Hoca selection out of `_do_dispatch` (move to Beckman admission + runtime per-iter).
- Move per-call retry out (runtime owns).
- `dispatcher.execute(pick, messages)` becomes the one primitive.
- `dispatcher.request()` stays as Beckman alias for back-compat (single-call OVERHEAD callers).

**Phase D — task.runner field**.
- Schema migration: add `runner` column to tasks.
- Producers set `runner` at creation.
- Orchestrator pump dispatches by runner.
- Default fill: existing rows get `runner=react` if `kind=main_work` else `direct`.

## Risks / hotspots

- **5800 LOC of mixed concerns.** Trace-replay before flipping the BaseAgent shim. Capture 10 production traces, replay through new path, diff TaskResult. No assertion-only verification.
- **Skill-injection mutating class attr** today. Confirmed bug. Fix during context-builder extraction (per-execution mutable copy).
- **Two retry layers in flight today.** Single-retry-layer architecture is the WIN of this extraction. During migration, dispatcher's in-call retry stays until runtime's transport-retry lands; cutover is atomic flip.
- **`record_model_call` double-counting** suspected today. Audit during dispatcher slim-down — one owner per call.
- **Step-config refresh** runs in two places (BaseAgent.execute + `_build_model_requirements`). Dedupe at port: one shared `runtime.workflow_step.refresh_from_json(task)` called once at iter setup.
- **Constrained_emit interactions.** Today fires AFTER react returns. Move to workflow_engine post-hook means workflow_engine calls it after the runtime task lands `status=completed`. Same timing, different owner.
- **Beckman lifecycle for `runner=direct` failures.** Today dispatcher retries internally; tomorrow Beckman lifecycle handler retries via re-admission with `failures` accumulated. Verify Beckman.apply already supports failure-aware retry shape.

## Open questions for plan/implementation

1. Runtime package name — Turkish convention (e.g. `yapan_eleman`, `is_eri`, `mesai`) or English (`runtime`, `react_runtime`). Decide at Phase B.
2. `Failure` dataclass home — runtime defines, Hoca consumes? Or fatih_hoca defines and runtime imports? Probably fatih_hoca (it's the consumer; the producer's vocab follows the consumer).
3. Transport-retry budget within one iter — fixed (3) or per-task config? Start fixed, expose later if needed.
4. Default profile when subclass missing field — handle in BaseAgent base attrs (already the pattern). No new mechanism.
5. Dual-path safety net during cutover — feature flag in BaseAgent.execute? Probably yes for one cycle, then delete.
