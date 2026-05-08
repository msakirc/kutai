# Runtime Extraction — Phased Plan

Anchor: 2026-05-04. Spec: `docs/superpowers/specs/2026-05-04-runtime-extraction-design.md`.

Each phase shippable in isolation. No behavior change until cutover step in Phase A.

---

## Phase A — in-tree extraction to `src/runtime/`

Mirrors orchestrator Phase 1: extract within `src/`, no package yet, BaseAgent delegates to new modules.

### A.1 — Response parsing module (`src/runtime/parsing.py`)

Move from `base.py`:
- `_unwrap_final_answer` (helper, lines 68-109)
- `_parse_agent_response` (1409-1475)
- `_try_parse_json` (1477-1490)
- `_normalize_action` (1492-1612, 120-LOC alias map)
- `_parse_function_call_response` (1845-1895)

Public API:
```python
def parse_action(content: str) -> dict | None: ...
def parse_function_call(tool_calls: list[dict]) -> dict | None: ...
def unwrap_final_answer(content: str) -> str: ...
```

BaseAgent methods become 1-line delegates. No behavior change.

**Verification:** `timeout 60 pytest tests/agents/ -k parse` (existing parse tests).

### A.2 — Context window module (`src/runtime/window.py`)

Move from `base.py`:
- `_count_tokens` (1617-1623)
- `_get_context_window` (1625-1661)
- `_trim_messages_if_needed` (1664-1736)
- `_prune_tool_results_to_fit` (1738-1812)

Public API:
```python
def count_tokens(messages: list[dict], model: str) -> int: ...
def context_window_for(model: str, reqs=None) -> int: ...
def trim_if_needed(messages: list[dict], model: str, reqs=None) -> list[dict]: ...
def prune_tool_results(messages: list[dict], ctx_window: int, est_output: int, task_id) -> list[dict]: ...
```

BaseAgent delegates.

**Verification:** `timeout 60 pytest tests/agents/ -k 'window or context'`.

### A.3 — Output validation + samet integration (`src/runtime/validation.py`)

Move:
- `_validate_response` (1900-1934, refusal/length/empty checks)
- Samet calls scattered in BaseAgent (checkpoint-restore, self-reflect)

Public API:
```python
def validate_final_answer(result: str, task: dict) -> str | None: ...   # error msg or None
def is_degenerate(text: str) -> tuple[bool, str | None]: ...             # wraps samet.assess
def salvage_or_drop(text: str) -> str | None: ...                        # wraps samet.salvage
```

### A.4 — Tool VM module (`src/runtime/tools.py`)

Move from `base.py`:
- `_partition_tool_calls` (112-123)
- `SIDE_EFFECT_TOOLS`, `CACHEABLE_READ_TOOLS` (47-62)
- `_check_tool_permission` (598-607)
- `_build_litellm_tools` (1817-1843)
- `_tool_idempotency_key` (4011-4019)
- Single-tool exec block (2839-3102)
- Multi-tool exec block (3104-3277)
- read_file artifact intercept (2856-2880)
- audit + metrics + post-tool reindex calls

Public API:
```python
@dataclass
class ToolExecResult:
    output: str
    failed: bool
    cached: bool

async def execute_tool_call(
    tool_name: str, tool_args: dict,
    *, agent_name: str, allowed_tools: list[str] | None,
    completed_ops: dict[str, str], iter_seen: set[str],
    task: dict, task_ctx: dict, search_depth: str,
    timeout_floor: int = 60,
) -> ToolExecResult: ...

async def execute_tool_calls_batch(
    tool_list: list[dict], *, ...same context...,
) -> list[tuple[str, dict, ToolExecResult]]: ...

def build_litellm_tools(allowed: list[str] | None, exclude: set[str]) -> list[dict] | None: ...
def idempotency_key(tool: str, args: dict) -> str: ...
```

State (completed_ops, iter_seen) passed in by caller — module is stateless.

**Verification:** new `tests/runtime/test_tools.py` with 5+ scenarios (single, multi-parallel, idempotency hit, side-effect cache invalidate, permission deny). Plus existing agent integration tests.

### A.5 — Context builder module (`src/runtime/context.py`)

Move from `base.py`:
- `_build_full_system_prompt` (331-389)
- `_get_available_tools_prompt` (233-330)
- `_build_context` (683-1224, 540 LOC — the megafunction)
- `_truncate_to_tokens` (1226-1231)
- `_fetch_deps` (1233-1369)
- `_format_prior_steps` (1370-1385)
- `_format_conversation` (1386-1408)
- Skill-injection logic (line 1098-1130) — **fix the class-attr mutation bug here**: produce a per-execution allowed_tools copy

Public API:
```python
async def build_system_prompt(profile, task: dict) -> str: ...
async def build_user_context(profile, task: dict, *, task_ctx: dict, model_ctx: int) -> str: ...
async def fetch_deps(task: dict, max_tokens: int) -> str: ...
```

`profile` is the duck-typed Profile (BaseAgent or any Profile-shaped object).

**Verification:** snapshot 5 production prompts (one per agent type), assert byte-identical to pre-extraction.

### A.6 — Sub-iter guards module (`src/runtime/guards.py`)

Move:
- `_check_sub_iteration_guards` (474-592)
- `_is_action_task` (391-448)
- `_get_search_depth` (450-463)
- `GuardCorrection` dataclass (140-148)
- `MAX_SUB_CORRECTIONS`, `MAX_FORMAT_CORRECTIONS` constants

Public API:
```python
@dataclass
class GuardCorrection:
    name: str
    message: str

def check_sub_iter_guards(parsed: dict, *, profile, iteration: int,
                          tools_used: bool, tools_used_names: set[str],
                          task: dict, search_depth: str,
                          suppress_guards: bool) -> GuardCorrection | None: ...
```

### A.7 — Escalation + reflection + checkpoint modules

Three small modules:

`src/runtime/escalation.py`:
- `_trim_for_escalation` (609-671)
- Failure-signal builder (NEW — replaces reqs.escalate)

`src/runtime/reflection.py`:
- `_self_reflect` (3933-4005)

`src/runtime/checkpoint.py`:
- `_save_checkpoint` (4024-4062)
- `_clear_checkpoint_safe` (4064-4073)
- `ExecutionState` dataclass (NEW — collects messages, total_cost, used_model, tools_used, completed_tool_ops, format_corrections, etc.)

### A.8 — ReAct loop module (`src/runtime/react.py`)

Move:
- `_execute_react_loop` (2156-3454, ~1300 LOC)
- Empty-response retry (2567-2580)
- Cancel + partial-content recovery (3373-3395)
- Heartbeat bumps (2358-2362)
- Progress callback (2364-2408)
- Exhaustion classification (3397-3454)

Public API:
```python
async def run(profile, task: dict, *, progress_callback=None) -> dict: ...
```

This module imports tools, parsing, window, validation, guards, escalation, reflection, checkpoint, context. It's the orchestrator of all the leaves. Largest extraction.

**During this step:** runtime's dispatcher calls stay as `dispatcher.request(...)` (Phase A keeps current dispatcher). Per-iter Hoca move is Phase C.

### A.9 — Single-shot module (`src/runtime/single_shot.py`)

Move `execute_single_shot` (3845-3931). Becomes thin:

```python
async def run(profile, task: dict) -> dict: ...
```

Used internally by runtime when `profile.execution_pattern == "single_shot"`.

### A.10 — Runtime entry (`src/runtime/__init__.py`)

```python
async def execute(profile, task: dict, progress_callback=None) -> dict:
    """Public entry. Routes by profile.execution_pattern, runs constrained_emit if applicable."""
    # tools_hint override + auto-strip + step-refresh + DB prompt override (setup phase)
    if profile.execution_pattern == "single_shot":
        result = await single_shot.run(profile, task)
    else:
        result = await react.run(profile, task, progress_callback=progress_callback)
    # constrained_emit moves to workflow_engine in Phase A.12 — temp keep here
    if hasattr(profile, '_maybe_constrained_emit'):
        result = await profile._maybe_constrained_emit(task, result)
    return result
```

Setup phase (lines 1953-2128 of base.py): DB prompt load, tools_hint override, auto-strip (file/web/write tools by step config), workflow step-config refresh.

### A.11 — BaseAgent shim cutover

`src/agents/base.py` final shape (~80 LOC):

```python
class BaseAgent:
    # profile attrs
    name = "base"
    description = "..."
    allowed_tools: list[str] | None = None
    max_iterations: int = MAX_AGENT_ITERATIONS
    execution_pattern: str = "react_loop"
    enable_self_reflection: bool = False
    min_confidence: int = 0
    can_create_subtasks: bool = False
    default_tier = "cheap"
    min_tier = "cheap"

    def get_system_prompt(self, task): ...

    async def execute(self, task, progress_callback=None):
        from src.runtime import execute as _runtime_execute
        return await _runtime_execute(self, task, progress_callback)
```

20 subclasses untouched.

**Verification gate (REQUIRED before merge):**
1. `timeout 120 pytest tests/` — full suite passes.
2. **Trace replay**: capture 10 production task traces (5 main_work, 5 overhead) from `task_state` table. Replay each through new pipeline. Diff TaskResult dicts. Allowed differences: timestamps, request IDs. Forbidden: result content, status, model, cost (within 1%), iterations, tools_used_names.
3. Run a real mission end-to-end (i2p_v3 simple ticket). Compare to prior production run.

### A.12 — Move `_maybe_constrained_emit` to workflow engine

Create `packages/workflow_engine/post_hooks/constrained_emit.py`. Move 200-LOC method as-is. Workflow engine post-hook scheduler calls it after task lands `status=completed` for steps with constrainable artifact_schema.

`src/runtime/__init__.py::execute` removes the constrained_emit call. Workflow engine fires it from outside.

### A.13 — Move `_build_model_requirements` to fatih_hoca

Create `packages/fatih_hoca/src/fatih_hoca/requirements_builder.py`:

```python
async def requirements_for(task: dict, task_ctx: dict | None = None) -> ModelRequirements: ...
```

Move 200 LOC from BaseAgent. BaseAgent loses the method entirely. Runtime never calls it — Beckman admission calls it once at task creation, stores result on task. Runtime reads task as opaque.

Step-config refresh (lines 3562-3601) deduped with the BaseAgent.execute version (2017-2128) — single shared `runtime.workflow_step.refresh_from_json(task)` called at iter setup.

---

## Phase B — Package extraction to `packages/<name>/`

Mirrors beckman / fatih_hoca pattern.

### B.1 — Package skeleton

```
packages/<name>/
├── pyproject.toml
├── README.md
├── src/<name>/
│   ├── __init__.py
│   ├── runner.py        # public execute()
│   ├── react.py
│   ├── single_shot.py
│   ├── tools.py
│   ├── context.py
│   ├── parsing.py
│   ├── window.py
│   ├── guards.py
│   ├── escalation.py
│   ├── reflection.py
│   ├── checkpoint.py
│   ├── validation.py
│   ├── state.py         # ExecutionState dataclass
│   └── protocol.py      # Profile typing.Protocol + TaskResult dataclass
└── tests/
```

Editable install via requirements.txt.

### B.2 — Move modules from `src/runtime/` to package

`src/runtime/__init__.py` becomes thin shim:
```python
from <name> import execute, Profile, TaskResult
__all__ = ["execute", "Profile", "TaskResult"]
```

### B.3 — Update `src/agents/base.py` import

Change `from src.runtime import execute` → `from <name> import execute`.

### B.4 — Tests, README, requirements.txt registration

Naming decision: defer until B.1 — Turkish convention. Candidates: `mesai`, `is_eri`, `yapan_eleman`. Pick at the time.

---

## Phase C — Dispatcher slim-down

### C.1 — Add `dispatcher.execute(pick, messages, task_ref)` primitive

New method, no Hoca call, no retry. Prepares messages, runs hallederiz once, returns CallResult or raises CallError. ~150 LOC.

### C.2 — Move per-iter Hoca call to runtime

In `src/runtime/react.py`:
- iter 0: use `task.preselected_pick` if present
- iter N>0 OR transport failure: `await fatih_hoca.select(task, failures)`
- Call `dispatcher.execute(pick, messages, task)`
- On CallError: append Failure to failures, decide whether to retry (small budget) or count toward outer iter and continue

Runtime imports `fatih_hoca.select` and `Pick`. Single retry surface.

### C.3 — Delete `_do_dispatch`'s retry loop

Remove dispatcher's in-call retry + re-select. Runtime owns it now. `_do_dispatch` becomes a thin wrapper around `dispatcher.execute` (or merged).

### C.4 — Keep `dispatcher.request` as Beckman alias

Single-call OVERHEAD callers (graders, structured_emit, classifier) keep calling `dispatcher.request(...)` which enqueues `runner=direct` task. Beckman admission picks via Hoca, attaches preselected_pick. Orchestrator pump dispatches `runner=direct` tasks straight to `dispatcher.execute(preselected_pick, messages)`.

### C.5 — Verify single retry surface

Acceptance:
- `grep -rE "fatih_hoca.select|hoca.select" src/core/llm_dispatcher.py packages/dispatcher/` returns nothing.
- All retry logic for LLM calls lives in runtime (per-iter) OR Beckman lifecycle (terminal).
- `record_model_call` called exactly once per attempt; audit shows no duplicates.

---

## Phase D — `task.runner` field

### D.1 — Schema migration

Add `runner` column to `tasks`. Default `react`. Migration fills existing rows: `runner = 'direct' if kind = 'overhead' else 'react'` (mechanical tasks already have `agent_type='mechanical'` — preserve via secondary check).

### D.2 — Producers set runner at task creation

- `mr_roboto` mechanical executors: emit `runner=mechanical`
- `dispatcher.request` Beckman alias: emit `runner=direct`
- Workflow expander: emit `runner=react` for agent steps
- Orchestrator dispatches by `task.runner`

### D.3 — Remove `agent_type=mechanical` lane decision

Replace with `runner=mechanical`. `agent_type` becomes informational only (still useful for telemetry, prompt selection, fatih_hoca's task-profile lookup).

### D.4 — Verification

- All producers updated.
- `agent_type` column kept as informational; no orchestrator dispatch logic reads it for lane decision.

---

## Verification gates

**After Phase A** (most risk):
1. Full pytest passes with timeout.
2. Trace replay: 10 production traces, byte-equal TaskResult (modulo timestamps/IDs).
3. Real mission end-to-end (simple i2p_v3 ticket).

**After Phase B**:
1. Import from package works.
2. Editable install configured.
3. Boundary check: `grep -rE "from (dallama|kdv|hallederiz_kadir|nerd_herd)" packages/<name>/src/` empty.
4. Boundary check: `grep -rE "from fatih_hoca" packages/<name>/src/` returns only `select, Pick`.

**After Phase C**:
1. Single retry surface verified: grep dispatcher for Hoca call → empty.
2. `record_model_call` called once per attempt (audit log).
3. Mission end-to-end, perf within 5% of pre-Phase-C.

**After Phase D**:
1. All tasks carry `runner`.
2. Orchestrator dispatch logic reads only `task.runner` for lane decision.

## Rollback strategy

Each phase commits to a separate branch. Phase A is one big merge with trace-replay gate. If post-merge issues, revert the merge commit; BaseAgent shim is the only change to revert.

Phase B-D are smaller, each rollback is git revert + pip uninstall package.

## Out of scope

- Killing agent classes / profile registry rewrite. Subclasses stay as prompt suppliers per user direction (2026-05-04).
- Telegram bot refactor.
- Workflow engine restructure beyond moving constrained_emit.
- Beckman lifecycle changes beyond consuming new failure-signal interface.

## Open questions

1. Package name (Phase B). Defer.
2. Failure dataclass home — fatih_hoca defines, runtime imports? Probably yes (consumer owns vocab).
3. Transport-retry budget within one iter — fixed (3) or per-task config? Start fixed.
4. Dual-path safety net during cutover — feature flag in BaseAgent.execute? Probably yes for one cycle, then delete.

## Estimated work

- Phase A: 8-10 PRs across modules. ~2-3 days focused work per phase. Total ~2 weeks.
- Phase B: 1-2 PRs. ~1 day.
- Phase C: 2-3 PRs. ~3 days.
- Phase D: 1 PR. ~half day.

Total: ~3 weeks of focused work, shippable per-phase.
