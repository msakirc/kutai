# Orchestrator Phase 1 — In-Tree Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Untangle `src/core/orchestrator.py` (3,865 lines) in-tree so clean package boundaries become visible. This plan does **not** extract packages — that's Phase 2a (Mechanical Dispatcher) and Phase 2b (Task Master), each a separate plan written after this one lands.

**Architecture:** Target end-state after all three phases:

```
Task Master (brain)           — owns queue, scheduling, gates, context, retry, missions, workflow recipes
  ├─ intakes: telegram, cron, agent subtasks, workflow starts
  ├─ consults (pure libs): classifier, workflow interpreter
  ├─ records failures into task context as learning substrate
  └─ emits: Dispatch(task, executor, payload) | NotifyUser(...)

Orchestrator (router + I/O)   — ~200 lines at end-state
  ├─ loop + signal handling + Telegram adapter
  ├─ routes Dispatch by executor: llm → LLM Dispatcher, mechanical → Mechanical Dispatcher
  └─ wakes on: in_flight completion, capacity event, tick

LLM Dispatcher (existing)     — ask/load/call/retry, iteration failures feed Fatih Hoca
Mechanical Dispatcher (new)   — sibling to LLM Dispatcher, no LLM, no model selection
```

Phase 1 (this plan) produces in-tree modules with the right shape — `task_context`, `task_gates`, `result_router`, `mechanical/`, `scheduled_jobs`, a Decision vocabulary, and a sliced `process_task`. Nothing gets extracted to `packages/` yet; that waits until Phase 2.

**Tech Stack:** Python 3.10, asyncio, pytest, aiosqlite. Existing patterns: `get_logger("component.name")`, `src/infra/db.py` for DB, `self.telegram.*` for Telegram.

**Invariants to preserve throughout:**
- Every existing test continues to pass — `timeout 120 pytest tests/` at the end of each task that touches shared code.
- No behavioral change visible from Telegram. Classifier, dispatcher, agents all unchanged.
- KutAI restart via `/restart` still works.

---

## File Structure

Files created in this plan:

- `src/core/decisions.py` — Decision and GateDecision dataclasses (pure types)
- `src/core/task_context.py` — centralized task context parse/serialize
- `src/core/task_gates.py` — human + risk + clarification gates; returns GateDecision
- `src/core/result_router.py` — agent result → next action (complete, subtasks, clarification, review, exhausted)
- `src/core/mechanical/__init__.py` — mechanical executor module
- `src/core/mechanical/git_commit.py` — git commit executor (absorbs `_auto_commit`)
- `src/core/mechanical/workspace_snapshot.py` — workspace hashing + git snapshot
- `src/app/scheduled_jobs.py` — proactive job entry points (todos, price watches, digest, API discovery)
- Test files mirroring each of the above under `tests/`

Files modified:

- `src/core/orchestrator.py` — shrinks as modules absorb responsibilities
- `src/security/risk_assessor.py` — `assess_risk` becomes async

---

## Task 1: Decision Vocabulary

**Files:**
- Create: `src/core/decisions.py`
- Test: `tests/test_decisions.py`

The decision types future task master will emit. Introduced up front so later tasks can refer to the same types. `GateDecision` is used immediately (Task 5); `Dispatch` / `NotifyUser` are introduced but not yet wired (wiring happens in Phase 2b).

- [ ] **Step 1: Write the failing test**

`tests/test_decisions.py`:

```python
from src.core.decisions import Dispatch, NotifyUser, GateDecision, Allow, Block, Cancel


def test_dispatch_carries_task_and_executor():
    d = Dispatch(task_id=42, executor="llm", payload={"agent_type": "executor"})
    assert d.task_id == 42
    assert d.executor == "llm"
    assert d.payload["agent_type"] == "executor"


def test_notify_user_carries_chat_and_text():
    n = NotifyUser(chat_id=1001, text="done")
    assert n.chat_id == 1001
    assert n.text == "done"


def test_gate_decision_allow_has_no_reason():
    g: GateDecision = Allow()
    assert isinstance(g, Allow)


def test_gate_decision_block_carries_reason():
    g: GateDecision = Block(reason="awaiting_approval")
    assert isinstance(g, Block)
    assert g.reason == "awaiting_approval"


def test_gate_decision_cancel_carries_reason():
    g: GateDecision = Cancel(reason="risk_rejected")
    assert isinstance(g, Cancel)
    assert g.reason == "risk_rejected"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 python -m pytest tests/test_decisions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.core.decisions'`

- [ ] **Step 3: Write minimal implementation**

`src/core/decisions.py`:

```python
"""Decision types emitted by task master to orchestrator.

Phase 1: introduced but minimally wired. Phase 2b wires task master to emit these.

Rule: a Decision exists only if orchestrator must call a different package/subsystem.
Internal state changes (spawning subtasks, marking complete, suspending) are not decisions.
"""

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass(frozen=True)
class Dispatch:
    """Run a task. Orchestrator routes to executor based on `executor` tag."""
    task_id: int
    executor: str  # "llm" | "mechanical"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NotifyUser:
    """Send a message to a user. Orchestrator routes to Telegram."""
    chat_id: int
    text: str


# Gate decisions — used by task_gates module (Task 5).
@dataclass(frozen=True)
class Allow:
    """Gate passed; task may proceed."""


@dataclass(frozen=True)
class Block:
    """Task is suspended (awaiting approval, clarification, etc.)."""
    reason: str


@dataclass(frozen=True)
class Cancel:
    """Task is cancelled (rejected at gate). No retry."""
    reason: str


GateDecision = Union[Allow, Block, Cancel]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 python -m pytest tests/test_decisions.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/decisions.py tests/test_decisions.py
git commit -m "feat(core): add Decision and GateDecision vocabulary

Phase 1 of orchestrator refactor. Decisions are introduced up front
so later tasks can refer to typed return values. GateDecision is used
immediately by task_gates; Dispatch/NotifyUser are Phase 2b targets."
```

---

## Task 2: Centralize Task Context Parsing

**Problem:** `task_ctx = json.loads(task.get("context", "{}"))` appears 4+ times across `orchestrator.py` (lines 345–352, 1483–1490, 2618–2625, 3203–3210). Each site has slightly different fallback handling. DRY.

**Files:**
- Create: `src/core/task_context.py`
- Test: `tests/test_task_context.py`
- Modify: `src/core/orchestrator.py` (replace 4 occurrences — do this last in this task)

- [ ] **Step 1: Write the failing test**

`tests/test_task_context.py`:

```python
import json
import pytest
from src.core.task_context import parse_context, set_context


def test_parse_context_dict_passthrough():
    task = {"context": {"classification": {"agent_type": "executor"}}}
    ctx = parse_context(task)
    assert ctx == {"classification": {"agent_type": "executor"}}


def test_parse_context_json_string():
    task = {"context": json.dumps({"chat_id": 1001})}
    ctx = parse_context(task)
    assert ctx == {"chat_id": 1001}


def test_parse_context_missing_returns_empty_dict():
    task = {"id": 1}
    ctx = parse_context(task)
    assert ctx == {}


def test_parse_context_empty_string_returns_empty_dict():
    task = {"context": ""}
    ctx = parse_context(task)
    assert ctx == {}


def test_parse_context_malformed_json_returns_empty_dict():
    task = {"context": "{not valid json"}
    ctx = parse_context(task)
    assert ctx == {}


def test_parse_context_non_dict_json_returns_empty_dict():
    task = {"context": json.dumps(["list", "not", "dict"])}
    ctx = parse_context(task)
    assert ctx == {}


def test_set_context_serializes_to_string():
    task = {"id": 1, "context": "{}"}
    updated = set_context(task, {"chat_id": 42})
    parsed = json.loads(updated["context"])
    assert parsed == {"chat_id": 42}


def test_set_context_does_not_mutate_input():
    task = {"id": 1, "context": "{}"}
    set_context(task, {"chat_id": 42})
    # Original task unchanged
    assert task["context"] == "{}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 python -m pytest tests/test_task_context.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`src/core/task_context.py`:

```python
"""Centralized parse/serialize for the `context` field of a task row.

Task context is stored as a JSON string in SQLite. Before this module, each
call site had slightly different fallback handling; this centralizes the
parse-with-fallback and the serialize-back pattern.
"""

import json
from typing import Any


def parse_context(task: dict) -> dict:
    """Return task context as a dict. Empty dict on any parse failure or type mismatch."""
    raw = task.get("context")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def set_context(task: dict, ctx: dict) -> dict:
    """Return a shallow copy of task with `context` field set to serialized ctx."""
    updated = dict(task)
    updated["context"] = json.dumps(ctx)
    return updated
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 python -m pytest tests/test_task_context.py -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Replace call sites in orchestrator**

Replace each of the following patterns in `src/core/orchestrator.py`:

```python
# BEFORE (appears ~4 times)
task_ctx = task.get("context", "{}")
if isinstance(task_ctx, str):
    try:
        task_ctx = json.loads(task_ctx)
    except (json.JSONDecodeError, TypeError):
        task_ctx = {}
if not isinstance(task_ctx, dict):
    task_ctx = {}

# AFTER
from src.core.task_context import parse_context
task_ctx = parse_context(task)
```

And where we write the context back:

```python
# BEFORE
task["context"] = json.dumps(task_ctx)

# AFTER
from src.core.task_context import set_context
task = set_context(task, task_ctx)
```

Find all occurrences with:

```bash
grep -n 'json.loads(task\(_ctx\)?.get("context"' src/core/orchestrator.py
grep -n 'task\[.context.\] = json.dumps' src/core/orchestrator.py
```

Replace each. Import `parse_context` and `set_context` at the top of `orchestrator.py`.

- [ ] **Step 6: Run full orchestrator test suite**

Run: `timeout 120 python -m pytest tests/test_orchestrator_routing.py tests/test_lifecycle_fixes.py tests/test_human_gates.py -v`
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/core/task_context.py tests/test_task_context.py src/core/orchestrator.py
git commit -m "refactor(core): centralize task context parsing

DRY the json.loads(task['context']) pattern that appeared 4x in orchestrator.
New module src/core/task_context.py handles dict/string/malformed/empty cases."
```

---

## Task 3: Make Risk Assessment Async

**Problem:** `src/core/orchestrator.py:1600` calls `assess_risk()` synchronously inside the async `process_task`. This blocks the event loop during a potentially slow regex + heuristic sweep.

**Files:**
- Modify: `src/security/risk_assessor.py` — make `assess_risk` async
- Test: `tests/test_risk_assessor_async.py` (new)
- Modify: `src/core/orchestrator.py` — await the call

- [ ] **Step 1: Inspect current implementation**

Run: `timeout 10 grep -n 'def assess_risk\|def format_risk_assessment' src/security/risk_assessor.py`

Confirm current signatures. If `assess_risk` is currently `def` (sync), proceed. If already async, skip this task.

- [ ] **Step 2: Write the failing test**

`tests/test_risk_assessor_async.py`:

```python
import asyncio
import pytest
from src.security.risk_assessor import assess_risk


@pytest.mark.asyncio
async def test_assess_risk_is_awaitable():
    result = await assess_risk(
        task_title="delete all files",
        task_description="rm -rf /",
    )
    assert isinstance(result, dict)
    assert "score" in result
    assert "needs_approval" in result


@pytest.mark.asyncio
async def test_assess_risk_low_risk_task():
    result = await assess_risk(
        task_title="list files in current directory",
        task_description="ls",
    )
    assert isinstance(result, dict)
    assert result["needs_approval"] is False
```

- [ ] **Step 3: Run test to verify it fails**

Run: `timeout 30 python -m pytest tests/test_risk_assessor_async.py -v`
Expected: FAIL (TypeError: object dict can't be used in 'await' expression — because assess_risk is still sync)

- [ ] **Step 4: Make `assess_risk` async**

In `src/security/risk_assessor.py`, change:

```python
# BEFORE
def assess_risk(task_title: str, task_description: str = "") -> dict:
    ...
```

to:

```python
# AFTER
async def assess_risk(task_title: str, task_description: str = "") -> dict:
    ...
```

The body stays the same — it's pure-CPU regex work, no I/O — but the `async` keyword makes it awaitable and defers it through the event loop. If the body has any call to another sync function from the same module, leave those sync. Only the top-level entry point changes.

- [ ] **Step 5: Run test to verify it passes**

Run: `timeout 30 python -m pytest tests/test_risk_assessor_async.py -v`
Expected: both tests PASS

- [ ] **Step 6: Update the caller in orchestrator**

In `src/core/orchestrator.py` around line 1600:

```python
# BEFORE
risk = assess_risk(
    task_title=task.get("title", ""),
    task_description=task.get("description", ""),
)

# AFTER
risk = await assess_risk(
    task_title=task.get("title", ""),
    task_description=task.get("description", ""),
)
```

Also search for any other callers:

```bash
grep -rn 'assess_risk(' src/ tests/ --include='*.py'
```

Every call site must now be awaited.

- [ ] **Step 7: Run full test suite**

Run: `timeout 120 python -m pytest tests/ -x`
Expected: all tests PASS. If any caller was missed, this reveals it as a TypeError.

- [ ] **Step 8: Commit**

```bash
git add src/security/risk_assessor.py src/core/orchestrator.py tests/test_risk_assessor_async.py
git commit -m "refactor(security): make assess_risk async

Previously assess_risk was sync and called inline from process_task.
No event-loop yield between claim and dispatch could starve other coroutines."
```

---

## Task 4: Extract Gate Logic to `task_gates.py`

**Goal:** Pull human approval gate (lines 1574–1592) and risk gate (lines 1593–1623) out of `process_task` into a module that returns `GateDecision`. Orchestrator (for now) still handles the Telegram I/O — the module returns `Block(reason)` and orchestrator maps that to a Telegram approval request + update. This is the first step toward inverting the coupling.

**Files:**
- Create: `src/core/task_gates.py`
- Test: `tests/test_task_gates.py`
- Modify: `src/core/orchestrator.py` — call `run_gates` instead of inline gate code

- [ ] **Step 1: Write the failing test**

`tests/test_task_gates.py`:

```python
import pytest
from unittest.mock import AsyncMock
from src.core.task_gates import run_gates, GateContext
from src.core.decisions import Allow, Block, Cancel


@pytest.mark.asyncio
async def test_no_gates_returns_allow():
    """Task with no gate flags and low-risk content passes through."""
    task = {"id": 1, "title": "list files", "description": ""}
    ctx = {}
    approval = AsyncMock(return_value=True)
    decision = await run_gates(task, ctx, approval_fn=approval)
    assert isinstance(decision, Allow)
    approval.assert_not_called()


@pytest.mark.asyncio
async def test_human_gate_approved_returns_allow():
    task = {"id": 1, "title": "x", "description": "y", "tier": "auto"}
    ctx = {"human_gate": True}
    approval = AsyncMock(return_value=True)
    decision = await run_gates(task, ctx, approval_fn=approval)
    assert isinstance(decision, Allow)
    approval.assert_called_once()


@pytest.mark.asyncio
async def test_human_gate_rejected_returns_cancel():
    task = {"id": 1, "title": "x", "description": "y"}
    ctx = {"human_gate": True}
    approval = AsyncMock(return_value=False)
    decision = await run_gates(task, ctx, approval_fn=approval)
    assert isinstance(decision, Cancel)
    assert "human" in decision.reason.lower()


@pytest.mark.asyncio
async def test_risk_gate_for_dangerous_task_requires_approval():
    task = {"id": 1, "title": "rm -rf /", "description": "wipe the disk"}
    ctx = {}
    approval = AsyncMock(return_value=False)
    decision = await run_gates(task, ctx, approval_fn=approval)
    assert isinstance(decision, Cancel)
    assert "risk" in decision.reason.lower()


@pytest.mark.asyncio
async def test_workflow_step_skips_risk_gate():
    """Workflow steps are pre-approved via the workflow definition; risk gate skipped."""
    task = {"id": 1, "title": "rm temp files", "description": ""}
    ctx = {"is_workflow_step": True}
    approval = AsyncMock(return_value=False)
    decision = await run_gates(task, ctx, approval_fn=approval)
    # Even if content looks risky, workflow steps pass
    assert isinstance(decision, Allow)
    approval.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 python -m pytest tests/test_task_gates.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/core/task_gates.py`:

```python
"""Pre-dispatch gates: human approval, risk assessment, clarification validation.

Returns GateDecision (Allow | Block | Cancel) instead of directly sending Telegram.
Orchestrator remains responsible for the I/O side (approval_fn injected).

Phase 1 keeps the injection seam here; Phase 2b will move this entire module
into the task master package and replace approval_fn with a RequestApproval
decision emission.
"""

from dataclasses import dataclass
from typing import Awaitable, Callable

from src.core.decisions import Allow, Block, Cancel, GateDecision
from src.infra.logging_config import get_logger

logger = get_logger("core.task_gates")


@dataclass
class GateContext:
    """Data needed to evaluate gates. Built by process_task from task + task_ctx."""
    task: dict
    task_ctx: dict


# Signature: (task_id, title, description, tier, mission_id) -> bool
ApprovalFn = Callable[..., Awaitable[bool]]


async def run_gates(
    task: dict,
    task_ctx: dict,
    approval_fn: ApprovalFn,
) -> GateDecision:
    """Evaluate gates in order. First blocking gate wins.

    Returns:
        Allow() — task may proceed to dispatch
        Cancel(reason) — task is rejected; orchestrator marks cancelled
        Block(reason) — reserved for Phase 2b (awaiting-approval-resume flow)
    """
    is_workflow = task_ctx.get("is_workflow_step", False)

    # ── Human approval gate (explicit) ──
    if task_ctx.get("human_gate"):
        logger.info("human approval gate triggered", task_id=task.get("id"))
        approved = await approval_fn(
            task.get("id"),
            task.get("title", ""),
            task.get("description", "")[:200],
            tier=task.get("tier", "auto"),
            mission_id=task.get("mission_id"),
        )
        if not approved:
            return Cancel(reason="human_gate_rejected")
        # Approved — fall through to risk check

    # ── Risk assessment gate ──
    # Skipped for workflow steps (pre-approved in recipe) and tasks that already
    # passed the human gate above (double approval wastes user time).
    if is_workflow or task_ctx.get("human_gate"):
        return Allow()

    try:
        from src.security.risk_assessor import assess_risk, format_risk_assessment
        risk = await assess_risk(
            task_title=task.get("title", ""),
            task_description=task.get("description", ""),
        )
    except Exception as e:
        logger.debug("risk assessment failed (open-circuit allow): %s", e)
        return Allow()

    if not risk.get("needs_approval"):
        return Allow()

    logger.info(
        "risk gate triggered",
        task_id=task.get("id"),
        risk_score=risk.get("score"),
        factors=risk.get("risk_factors"),
    )
    approved = await approval_fn(
        task.get("id"),
        task.get("title", ""),
        format_risk_assessment(risk),
        tier=task.get("tier", "auto"),
        mission_id=task.get("mission_id"),
    )
    if not approved:
        return Cancel(reason="risk_gate_rejected")
    return Allow()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 python -m pytest tests/test_task_gates.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Wire into `process_task`**

In `src/core/orchestrator.py`, replace lines ~1574–1623 (human gate + risk gate blocks) with:

```python
# ── Gates: human approval + risk assessment ──
from src.core.task_gates import run_gates
from src.core.decisions import Cancel

gate_decision = await run_gates(
    task=task,
    task_ctx=task_ctx,
    approval_fn=self.telegram.request_approval,
)
if isinstance(gate_decision, Cancel):
    logger.info("gate rejected", task_id=task_id, reason=gate_decision.reason)
    await update_task(task_id, status="cancelled")
    return
```

Remove the now-unused imports of `assess_risk` and `format_risk_assessment` from the top of `orchestrator.py` — they're used inside `task_gates` now.

- [ ] **Step 6: Run full gate test suite**

Run: `timeout 60 python -m pytest tests/test_human_gates.py tests/test_task_gates.py -v`
Expected: all tests PASS. Existing human-gate tests must still pass.

- [ ] **Step 7: Commit**

```bash
git add src/core/task_gates.py tests/test_task_gates.py src/core/orchestrator.py
git commit -m "refactor(core): extract task gates into task_gates module

Human approval and risk gates now live in src/core/task_gates.py and
return GateDecision. Telegram I/O stays in orchestrator (injected via
approval_fn). This is the first step toward inverting the Telegram
coupling and the foundation for Phase 2b task master extraction."
```

---

## Task 5: Extract Result Router to `result_router.py`

**Goal:** The branching in `process_task` that handles agent results by status (complete / subtasks / clarification / review / exhausted / failed — lines ~2165–2598) becomes a pure function `route_result(task, agent_result) -> list[Action]`. Orchestrator still executes the actions, but the state machine is now a testable module.

**Note:** This is the biggest chunk of `process_task`. We extract *what happens* for each status; the side effects (DB writes, Telegram sends) stay in orchestrator for Phase 1. Phase 2b converts these to Decision emissions.

**Files:**
- Create: `src/core/result_router.py`
- Test: `tests/test_result_router.py`
- Modify: `src/core/orchestrator.py` — call `route_result` and execute returned actions

- [ ] **Step 1: Read the existing result-handling branches**

Run: `timeout 10 grep -n 'if result\|status == "complete"\|status == "subtasks"\|status == "clarification"\|status == "review"\|status == "exhausted"\|status == "failed"' src/core/orchestrator.py`

Note the line numbers. You'll map each branch to a case in the router.

- [ ] **Step 2: Write the failing test**

`tests/test_result_router.py`:

```python
import pytest
from src.core.result_router import route_result, Complete, SpawnSubtasks, RequestClarification, RequestReview, Exhausted, Failed


def test_complete_result_produces_complete_action():
    task = {"id": 1, "title": "t"}
    agent_result = {"status": "complete", "result": "done", "iterations": 3}
    actions = route_result(task, agent_result)
    assert len(actions) == 1
    assert isinstance(actions[0], Complete)
    assert actions[0].result == "done"


def test_subtasks_result_produces_spawn_action():
    task = {"id": 1, "title": "t"}
    agent_result = {"status": "subtasks", "subtasks": [{"title": "s1"}, {"title": "s2"}]}
    actions = route_result(task, agent_result)
    assert len(actions) == 1
    assert isinstance(actions[0], SpawnSubtasks)
    assert len(actions[0].subtasks) == 2


def test_clarification_result_produces_clarification_action():
    task = {"id": 1, "title": "t", "chat_id": 42}
    agent_result = {"status": "clarification", "question": "which one?"}
    actions = route_result(task, agent_result)
    assert len(actions) == 1
    assert isinstance(actions[0], RequestClarification)
    assert actions[0].question == "which one?"


def test_review_result_produces_review_action():
    task = {"id": 1, "title": "t"}
    agent_result = {"status": "review", "summary": "please review"}
    actions = route_result(task, agent_result)
    assert len(actions) == 1
    assert isinstance(actions[0], RequestReview)


def test_exhausted_result_produces_exhausted_action():
    task = {"id": 1, "title": "t"}
    agent_result = {"status": "exhausted", "error": "max iterations"}
    actions = route_result(task, agent_result)
    assert len(actions) == 1
    assert isinstance(actions[0], Exhausted)


def test_failed_result_produces_failed_action():
    task = {"id": 1, "title": "t"}
    agent_result = {"status": "failed", "error": "llm timeout"}
    actions = route_result(task, agent_result)
    assert len(actions) == 1
    assert isinstance(actions[0], Failed)
    assert actions[0].error == "llm timeout"


def test_unknown_status_treated_as_failed():
    task = {"id": 1, "title": "t"}
    agent_result = {"status": "???"}
    actions = route_result(task, agent_result)
    assert len(actions) == 1
    assert isinstance(actions[0], Failed)


def test_none_result_treated_as_failed():
    task = {"id": 1, "title": "t"}
    actions = route_result(task, None)
    assert len(actions) == 1
    assert isinstance(actions[0], Failed)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `timeout 30 python -m pytest tests/test_result_router.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write the implementation**

`src/core/result_router.py`:

```python
"""Pure function: agent result -> list of actions orchestrator must take.

Phase 1: Action types are lightweight dataclasses consumed by orchestrator's
existing side-effect code (update_task, spawn subtasks, telegram.send, etc.).
Phase 2b: these become Decision emissions consumed by orchestrator's switch.
"""

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass(frozen=True)
class Complete:
    task_id: int
    result: str
    iterations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpawnSubtasks:
    parent_task_id: int
    subtasks: list[dict]  # list of subtask dicts as returned by the agent


@dataclass(frozen=True)
class RequestClarification:
    task_id: int
    question: str
    chat_id: int | None = None


@dataclass(frozen=True)
class RequestReview:
    task_id: int
    summary: str


@dataclass(frozen=True)
class Exhausted:
    task_id: int
    error: str


@dataclass(frozen=True)
class Failed:
    task_id: int
    error: str


Action = Union[Complete, SpawnSubtasks, RequestClarification, RequestReview, Exhausted, Failed]


def route_result(task: dict, agent_result: dict | None) -> list[Action]:
    """Map (task, agent_result) -> actions orchestrator must execute."""
    task_id = task["id"]

    if agent_result is None:
        return [Failed(task_id=task_id, error="no_result_returned")]

    status = agent_result.get("status")

    if status == "complete":
        return [Complete(
            task_id=task_id,
            result=agent_result.get("result", ""),
            iterations=agent_result.get("iterations", 0),
            metadata=agent_result.get("metadata", {}),
        )]

    if status == "subtasks":
        subtasks = agent_result.get("subtasks", [])
        return [SpawnSubtasks(parent_task_id=task_id, subtasks=subtasks)]

    if status == "clarification":
        return [RequestClarification(
            task_id=task_id,
            question=agent_result.get("question", ""),
            chat_id=task.get("chat_id"),
        )]

    if status == "review":
        return [RequestReview(
            task_id=task_id,
            summary=agent_result.get("summary", ""),
        )]

    if status == "exhausted":
        return [Exhausted(
            task_id=task_id,
            error=agent_result.get("error", "max_iterations_reached"),
        )]

    # "failed" or unknown status
    return [Failed(
        task_id=task_id,
        error=agent_result.get("error", f"unknown_status:{status}"),
    )]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `timeout 30 python -m pytest tests/test_result_router.py -v`
Expected: all 8 tests PASS

- [ ] **Step 6: Use `route_result` inside `process_task`**

In `src/core/orchestrator.py`, after the agent returns a result, replace the big `if result.get('status') == ...` chain with:

```python
from src.core.result_router import (
    route_result, Complete, SpawnSubtasks, RequestClarification,
    RequestReview, Exhausted, Failed,
)

actions = route_result(task, result)
for action in actions:
    if isinstance(action, Complete):
        await self._handle_complete(task, action.result, action.iterations, action.metadata)
    elif isinstance(action, SpawnSubtasks):
        await self._handle_subtasks(task, action.subtasks)
    elif isinstance(action, RequestClarification):
        await self._handle_clarification(task, action.question, action.chat_id)
    elif isinstance(action, RequestReview):
        await self._handle_review(task, action.summary)
    elif isinstance(action, Exhausted):
        await self._handle_exhausted(task, action.error)
    elif isinstance(action, Failed):
        await self._handle_failed(task, action.error)
```

The existing `_handle_*` methods stay unchanged (they own the DB writes + Telegram sends). If any branch didn't have a helper method before (e.g. exhausted/failed were inline), extract them as thin `_handle_*` methods too. **Do not modify the side-effect logic** — copy it verbatim into the handler method.

- [ ] **Step 7: Run full orchestrator test suite**

Run: `timeout 120 python -m pytest tests/test_orchestrator_routing.py tests/test_lifecycle_fixes.py tests/test_grading.py tests/test_resilience_approvals.py -v`
Expected: all tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/core/result_router.py tests/test_result_router.py src/core/orchestrator.py
git commit -m "refactor(core): extract result routing state machine

Status -> Action mapping is now a pure function in src/core/result_router.py.
Orchestrator retains I/O handlers (_handle_complete, _handle_subtasks, etc.).
The switch structure is now testable without spinning up the full orchestrator."
```

---

## Task 6: Introduce Mechanical Executor Module

**Goal:** Create `src/core/mechanical/` as the in-tree home for non-LLM task executors. Phase 2a will promote this to `packages/mechanical_dispatcher/`. For Phase 1, absorb `_auto_commit` (currently called from orchestrator line 427) and `compute_workspace_hashes + save_workspace_snapshot` (currently lines 1625–1642).

**Files:**
- Create: `src/core/mechanical/__init__.py`
- Create: `src/core/mechanical/git_commit.py`
- Create: `src/core/mechanical/workspace_snapshot.py`
- Test: `tests/core/mechanical/test_workspace_snapshot.py`
- Modify: `src/core/orchestrator.py` — import from new module, disconnect old auto_commit call site

- [ ] **Step 1: Read the existing `_auto_commit` code**

Run: `timeout 10 grep -n '_auto_commit\|def auto_commit' src/core/orchestrator.py src/tools/ 2>/dev/null`

Find the current implementation. Copy it verbatim into the new module. Do **not** rewrite it — the user explicitly said "don't delete, just disconnect."

- [ ] **Step 2: Write the failing test for workspace snapshot**

`tests/core/mechanical/test_workspace_snapshot.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch
from src.core.mechanical.workspace_snapshot import snapshot_workspace


@pytest.mark.asyncio
async def test_snapshot_returns_dict_with_hashes_and_sha():
    """snapshot_workspace wraps the three underlying calls and returns a flat dict."""
    with patch("src.core.mechanical.workspace_snapshot.compute_workspace_hashes") as mock_hashes, \
         patch("src.core.mechanical.workspace_snapshot.get_commit_sha", new_callable=AsyncMock) as mock_sha, \
         patch("src.core.mechanical.workspace_snapshot.get_current_branch", new_callable=AsyncMock) as mock_branch, \
         patch("src.core.mechanical.workspace_snapshot.save_workspace_snapshot", new_callable=AsyncMock) as mock_save:
        mock_hashes.return_value = {"a.py": "abc"}
        mock_sha.return_value = "deadbeef"
        mock_branch.return_value = "main"
        result = await snapshot_workspace(mission_id=1, task_id=42, workspace_path="/tmp/ws")
        assert result["hashes"] == {"a.py": "abc"}
        assert result["commit_sha"] == "deadbeef"
        assert result["branch"] == "main"
        mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_snapshot_returns_none_on_exception():
    """Snapshot failures are non-fatal — return None, caller continues."""
    with patch("src.core.mechanical.workspace_snapshot.compute_workspace_hashes",
               side_effect=OSError("disk gone")):
        result = await snapshot_workspace(mission_id=1, task_id=42, workspace_path="/nonexistent")
        assert result is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `timeout 30 python -m pytest tests/core/mechanical/test_workspace_snapshot.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Create the module structure**

`src/core/mechanical/__init__.py`:

```python
"""In-tree home for non-LLM task executors.

Phase 1: absorbs _auto_commit, workspace snapshots, and other mechanical work
currently scattered across the orchestrator.

Phase 2a: promote this whole directory to packages/mechanical_dispatcher/.
At that point it becomes a sibling to the LLM dispatcher and is selected by
orchestrator based on task.executor tag.
"""
```

`src/core/mechanical/workspace_snapshot.py`:

```python
"""Workspace snapshot executor: compute file hashes + git SHA + branch, save to DB.

Absorbed from src/core/orchestrator.py:1625-1642 (pre-task snapshot block).
"""

from src.infra.db import save_workspace_snapshot
from src.tools.git_tools import get_commit_sha, get_current_branch
from src.tools.workspace import compute_workspace_hashes
from src.infra.logging_config import get_logger

logger = get_logger("core.mechanical.workspace_snapshot")


async def snapshot_workspace(
    mission_id: int,
    task_id: int,
    workspace_path: str,
    repo_path: str | None = None,
) -> dict | None:
    """Snapshot workspace state. Returns dict with hashes/branch/commit_sha, or None on failure.

    Failure is non-fatal (logged at debug). Caller should proceed.
    """
    try:
        hashes = compute_workspace_hashes(workspace_path)
        sha = await get_commit_sha(path=repo_path or workspace_path)
        branch = await get_current_branch(path=repo_path or workspace_path)
        await save_workspace_snapshot(
            mission_id=mission_id,
            file_hashes=hashes,
            task_id=task_id,
            branch_name=branch,
            commit_sha=sha,
        )
        return {"hashes": hashes, "commit_sha": sha, "branch": branch}
    except Exception as e:
        logger.debug(f"snapshot skipped task={task_id}: {e}")
        return None
```

`src/core/mechanical/git_commit.py`:

```python
"""Git auto-commit executor. Dormant in Phase 1 — i2p refactor will re-wire.

Moved from src/core/orchestrator.py _auto_commit(). Behavior unchanged;
only the call site was removed from the orchestrator main loop.

Phase 2a: invoked via Dispatch(executor='mechanical', payload={'action': 'git_commit', ...}).
"""

# TODO: paste the _auto_commit function body here verbatim, renamed to auto_commit.
# Convert any references to `self.*` to module-level or param-passed state.
# If the original _auto_commit reached into self.* for anything (e.g. self.telegram,
# self.db_path), refactor to accept those as parameters instead. DO NOT add new behavior.

async def auto_commit(paths: list[str] | None = None, message: str | None = None) -> bool:
    """Auto-commit a set of paths with a message. Returns True on success."""
    # Implementation pasted from orchestrator._auto_commit, with self.* removed.
    raise NotImplementedError("Paste body from orchestrator._auto_commit during this task")
```

**NOTE to implementer:** open `src/core/orchestrator.py` at the current `_auto_commit` location, copy the body into `auto_commit` above, replace the `raise NotImplementedError` line. Remove any `self.` references by inlining or accepting a parameter.

- [ ] **Step 5: Paste the `_auto_commit` body**

Using `Read` + `Edit`, transfer the function body. Run `grep -n '_auto_commit' src/core/orchestrator.py` to find it.

- [ ] **Step 6: Disconnect the orchestrator call site**

Find every call to `self._auto_commit(...)` in orchestrator and **comment it out** with:

```python
# PHASE 1 DISCONNECTED: auto-commit moved to src/core/mechanical/git_commit.py.
# Next i2p workflow refactor will re-wire this as an explicit workflow step
# or agent tool. Do not delete — code preserved in the mechanical module.
# await self._auto_commit(...)
```

Leave the `_auto_commit` method definition in orchestrator in place (dead but harmless) with a comment pointing to `src/core/mechanical/git_commit.py` as the current home.

- [ ] **Step 7: Replace the inline workspace snapshot block**

In `src/core/orchestrator.py` around lines 1625–1642, replace:

```python
# ── Phase 6: Snapshot workspace before coder/pipeline tasks ──
mission_id = task.get("mission_id")
if mission_id and agent_type in ("coder", "pipeline", "implementer", "fixer"):
    try:
        ws_path = get_mission_workspace(mission_id)
        hashes = compute_workspace_hashes(ws_path)
        ... (lines 1625-1642)
    except Exception as e:
        logger.debug(f"[Task #{task_id}] Snapshot skipped: {e}")
```

with:

```python
# ── Snapshot workspace before coder/pipeline tasks ──
mission_id = task.get("mission_id")
if mission_id and agent_type in ("coder", "pipeline", "implementer", "fixer"):
    from src.core.mechanical.workspace_snapshot import snapshot_workspace
    ws_path = get_mission_workspace(mission_id)
    repo_path = get_mission_workspace_relative(mission_id)
    await snapshot_workspace(
        mission_id=mission_id,
        task_id=task_id,
        workspace_path=ws_path,
        repo_path=repo_path,
    )
```

- [ ] **Step 8: Run mechanical tests**

Run: `timeout 30 python -m pytest tests/core/mechanical/ -v`
Expected: PASS

- [ ] **Step 9: Run full suite to catch any missed reference**

Run: `timeout 120 python -m pytest tests/ -x`
Expected: all tests PASS

- [ ] **Step 10: Commit**

```bash
git add src/core/mechanical/ tests/core/mechanical/ src/core/orchestrator.py
git commit -m "refactor(core): introduce mechanical executor module

New directory src/core/mechanical/ is the in-tree home for non-LLM task
executors. Phase 2a will promote to packages/mechanical_dispatcher/.

Absorbed:
  - workspace_snapshot.py (formerly inline in process_task lines 1625-1642)
  - git_commit.py (formerly orchestrator._auto_commit, now dormant;
    call site disconnected per user direction — i2p refactor will re-wire)"
```

---

## Task 7: Move Proactive Jobs to `scheduled_jobs.py`

**Goal:** Todo suggestions, price watch polling, API discovery, and daily digest currently live as methods on Orchestrator (lines 1229–1394 and 3184–3236). They're cron-triggered, not task-orchestration. Extract to a standalone module that orchestrator calls on a tick.

**Files:**
- Create: `src/app/scheduled_jobs.py`
- Test: `tests/test_scheduled_jobs.py`
- Modify: `src/core/orchestrator.py` — remove the methods, call the new module

- [ ] **Step 1: Inventory the methods to move**

Run: `timeout 10 grep -n 'def _check_todo_reminders\|def _start_todo_suggestions\|def _generate_suggestions\|def _check_api_discovery\|def check_scheduled_tasks\|def daily_digest' src/core/orchestrator.py`

Record the line ranges for each. You'll move them verbatim, not rewrite.

- [ ] **Step 2: Write the failing test**

`tests/test_scheduled_jobs.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.app.scheduled_jobs import ScheduledJobs


@pytest.mark.asyncio
async def test_scheduled_jobs_construction():
    """ScheduledJobs takes a telegram adapter (optional) and exposes tick methods."""
    telegram = MagicMock()
    jobs = ScheduledJobs(telegram=telegram)
    assert jobs is not None
    # Each tick method is callable
    assert callable(jobs.tick_todos)
    assert callable(jobs.tick_api_discovery)
    assert callable(jobs.tick_digest)
    assert callable(jobs.tick_price_watches)


@pytest.mark.asyncio
async def test_tick_todos_noop_when_not_due(monkeypatch):
    """Todo tick returns quietly if the 2h interval hasn't elapsed."""
    telegram = MagicMock()
    jobs = ScheduledJobs(telegram=telegram)
    # Force "last run was 30 seconds ago"
    jobs._last_todo_run = pytest.approx(__import__("time").time() - 30)
    result = await jobs.tick_todos()
    # Should not raise; should return falsy "nothing done"
    assert result is None or result is False
```

- [ ] **Step 3: Run test to verify it fails**

Run: `timeout 30 python -m pytest tests/test_scheduled_jobs.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Move the methods**

Create `src/app/scheduled_jobs.py` with this skeleton:

```python
"""Scheduled / proactive jobs that don't fit the task-orchestration main loop.

Todos reminders (every 2h), price watch re-scrape (daily noon), API discovery,
daily digest. Orchestrator calls tick_* methods on its main loop heartbeat.

Phase 2b target: each tick becomes a scheduled CronEvent that task master
accepts as intake, producing mechanical tasks that go through the dispatch
loop like anything else. For now they stay as direct coroutines.
"""

import time
from src.infra.logging_config import get_logger

logger = get_logger("app.scheduled_jobs")


class ScheduledJobs:
    """Proactive / cron-triggered jobs."""

    TODO_INTERVAL_SECONDS = 2 * 60 * 60  # 2h
    API_DISCOVERY_INTERVAL_SECONDS = 24 * 60 * 60  # daily
    DIGEST_HOUR_UTC = 20  # approx evening digest
    PRICE_WATCH_HOUR_UTC = 12  # noon UTC

    def __init__(self, telegram=None):
        self.telegram = telegram
        self._last_todo_run = 0.0
        self._last_api_discovery_run = 0.0
        self._last_digest_date = None
        self._last_price_watch_date = None

    async def tick_todos(self):
        """Check and fire todo reminders + suggestions if the 2h interval elapsed."""
        now = time.time()
        if now - self._last_todo_run < self.TODO_INTERVAL_SECONDS:
            return None
        self._last_todo_run = now
        # TODO: paste body of orchestrator._check_todo_reminders +
        # _start_todo_suggestions + _generate_suggestions here. Replace
        # `self.telegram.*` references with `self.telegram.*` (same attribute).
        raise NotImplementedError("Paste from orchestrator during this task")

    async def tick_api_discovery(self):
        """Daily-ish API discovery sweep."""
        # TODO: paste body of orchestrator._check_api_discovery here.
        raise NotImplementedError("Paste from orchestrator during this task")

    async def tick_digest(self):
        """Fire the daily digest once per day."""
        # TODO: paste body of orchestrator.daily_digest here.
        raise NotImplementedError("Paste from orchestrator during this task")

    async def tick_price_watches(self):
        """Fire the noon-UTC price watch check once per day."""
        # TODO: paste body of orchestrator.check_scheduled_tasks price-watch block here.
        raise NotImplementedError("Paste from orchestrator during this task")
```

Now paste each method body into the appropriate `tick_*`:
- `_check_todo_reminders` + `_start_todo_suggestions` + `_generate_suggestions` → `tick_todos` body (chained in order)
- `_check_api_discovery` → `tick_api_discovery`
- `daily_digest` → `tick_digest`
- Price-watch section of `check_scheduled_tasks` → `tick_price_watches`

Replace `self.telegram.*` references with `self.telegram.*` (no change — it's the same attribute on ScheduledJobs now).

Replace any `self.*` references to attributes that live only on Orchestrator (e.g. `self.db_path` if present) by importing from `src.app.config` or passing them in at construction.

- [ ] **Step 5: Remove the old methods from orchestrator**

Delete the method definitions that were moved. Replace their calls in orchestrator's main loop with calls on `self.scheduled_jobs`:

```python
# In Orchestrator.__init__:
from src.app.scheduled_jobs import ScheduledJobs
self.scheduled_jobs = ScheduledJobs(telegram=self.telegram)

# Where _check_todo_reminders was called:
await self.scheduled_jobs.tick_todos()

# Where _check_api_discovery was called:
await self.scheduled_jobs.tick_api_discovery()

# Where daily_digest was called:
await self.scheduled_jobs.tick_digest()

# In check_scheduled_tasks where price-watch block lived:
await self.scheduled_jobs.tick_price_watches()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `timeout 30 python -m pytest tests/test_scheduled_jobs.py -v`
Expected: all tests PASS

- [ ] **Step 7: Run full suite**

Run: `timeout 120 python -m pytest tests/ -x`
Expected: all tests PASS. Telegram flow tests in particular must keep working.

- [ ] **Step 8: Smoke-test via restart**

Restart KutAI via Telegram `/restart`. Send `/todos`. Expected: todo reminder fires as before.

- [ ] **Step 9: Commit**

```bash
git add src/app/scheduled_jobs.py tests/test_scheduled_jobs.py src/core/orchestrator.py
git commit -m "refactor(app): move proactive jobs to scheduled_jobs module

Todo reminders, API discovery, daily digest, and price watch ticks no
longer live as orchestrator methods. They're in src/app/scheduled_jobs.py
with tick_* entry points orchestrator calls on the main loop heartbeat.

Phase 2b: these ticks become CronEvent intakes into task master."
```

---

## Task 8: Split `process_task` into Stages

**Goal:** By now `process_task` has most of its inline logic replaced by module calls (gates from Task 4, result routing from Task 5, snapshots from Task 6). This task organizes what remains into three named stages, each a private method:

- `_prepare(task)` — claim, inject chain context, classify, enrich, workspace snapshot, gate check → returns (prepared_task, agent_type, timeout) | None
- `_dispatch(task, agent_type, timeout)` — invoke agent / pipeline, return agent_result
- `_record(task, agent_result)` — call route_result, execute actions via existing `_handle_*` methods

**Files:**
- Modify: `src/core/orchestrator.py`
- No new files, no new tests (existing test suite is the contract)

- [ ] **Step 1: Read the current `process_task` end-to-end**

Run: `timeout 10 grep -n 'async def process_task\|async def _handle' src/core/orchestrator.py`

Understand the boundaries. Everything before the agent invocation is `_prepare`; the agent call is `_dispatch`; everything after is `_record`.

- [ ] **Step 2: Extract `_prepare`**

Add a new method before `process_task`:

```python
async def _prepare(self, task: dict) -> tuple[dict, str, int] | None:
    """Run all pre-dispatch work.

    Returns (prepared_task, agent_type, timeout_seconds) or None if the task
    should not proceed (cancelled, rejected at gate, fast-resolved, etc.).
    """
    task_id = task["id"]

    # Claim
    claimed = await claim_task(task_id)
    if not claimed:
        logger.info("task already claimed", task_id=task_id)
        return None

    # Cancellation check
    fresh = await get_task(task_id)
    if fresh and fresh.get("status") == "cancelled":
        return None

    # Chain context + context parsing
    task = await self._inject_chain_context(task)
    task_ctx = parse_context(task)
    agent_type = task.get("agent_type", "executor")

    # Classification (same as current process_task lines 1492-1507)
    if "classification" not in task_ctx and agent_type == "executor":
        from .task_classifier import classify_task as classify
        classification = await classify(task["title"], task.get("description", ""))
        task_ctx["classification"] = dataclasses.asdict(classification)
        if classification.confidence >= 0.7 and agent_type == "executor":
            task["agent_type"] = classification.agent_type
            agent_type = classification.agent_type
        if classification.agent_type == "shopping_advisor" and classification.shopping_sub_intent:
            task_ctx["shopping_workflow"] = classification.shopping_sub_intent
        task = set_context(task, task_ctx)

    # Fast-path (lines 1509-1523)
    _skip_fast_path = agent_type in ("pipeline", "shopping_pipeline")
    try:
        from ..core.fast_resolver import try_resolve
        fast_result = None if _skip_fast_path else await try_resolve(task)
        if fast_result:
            await update_task(task_id, status="completed", result=fast_result, completed_at=db_now())
            if self.telegram and task.get("chat_id"):
                await self.telegram.send_notification(fast_result)
            return None
    except Exception as exc:
        logger.debug("fast-path check failed: %s", exc)

    # Shopping intent fallback (lines 1525-1543)
    if agent_type not in ("shopping_advisor", "product_researcher", "deal_analyst", "shopping_clarifier"):
        from ..workflows.engine.dispatch import should_start_shopping_workflow
        shopping_wf = should_start_shopping_workflow(task["title"])
        if shopping_wf:
            agent_type = "shopping_advisor"
            task["agent_type"] = "shopping_advisor"
            task_ctx["shopping_workflow"] = shopping_wf
            task = set_context(task, task_ctx)

    # Workflow pre-hook (lines 1545-1560)
    from ..workflows.engine.hooks import pre_execute_workflow_step, is_workflow_step
    if is_workflow_step(task_ctx):
        task = await pre_execute_workflow_step(task)
        from ..workflows.engine.pipeline_bridge import should_delegate_to_pipeline
        template_step_id = task_ctx.get("workflow_step_id", "")
        if should_delegate_to_pipeline(template_step_id, agent_type):
            agent_type = "pipeline"
            task["agent_type"] = "pipeline"

    # API enrichment (lines 1562-1572)
    try:
        from ..core.fast_resolver import enrich_context
        enrichment = None if _skip_fast_path else await enrich_context(task)
        if enrichment:
            task_ctx["api_enrichment"] = enrichment
            task = set_context(task, task_ctx)
    except Exception as exc:
        logger.debug("context enrichment failed: %s", exc)

    # Gates
    from src.core.task_gates import run_gates
    from src.core.decisions import Cancel
    gate_decision = await run_gates(task, task_ctx, approval_fn=self.telegram.request_approval)
    if isinstance(gate_decision, Cancel):
        await update_task(task_id, status="cancelled")
        return None

    # Workspace snapshot
    mission_id = task.get("mission_id")
    if mission_id and agent_type in ("coder", "pipeline", "implementer", "fixer"):
        from src.core.mechanical.workspace_snapshot import snapshot_workspace
        ws_path = get_mission_workspace(mission_id)
        repo_path = get_mission_workspace_relative(mission_id)
        await snapshot_workspace(mission_id=mission_id, task_id=task_id,
                                 workspace_path=ws_path, repo_path=repo_path)

    # Internet check
    classification = task_ctx.get("classification", {})
    if classification.get("search_depth", "none") != "none":
        if not await _check_internet():
            await update_task(task_id, started_at=None, status="pending")
            return None

    # Timeout
    timeout_seconds = task.get("timeout_seconds") or AGENT_TIMEOUTS.get(agent_type, 240)

    return task, agent_type, timeout_seconds
```

- [ ] **Step 3: Extract `_dispatch`**

```python
async def _dispatch(self, task: dict, agent_type: str, timeout_seconds: int):
    """Invoke the agent/pipeline for this task. Returns the agent's result dict."""
    task_id = task["id"]

    if agent_type == "pipeline":
        from ..workflows.pipeline import CodingPipeline
        pipeline = CodingPipeline()
        return await asyncio.wait_for(pipeline.run(task), timeout=timeout_seconds)

    if agent_type == "shopping_pipeline":
        from ..workflows.shopping.pipeline import ShoppingPipeline
        pipeline = ShoppingPipeline()
        return await asyncio.wait_for(pipeline.run(task), timeout=timeout_seconds)

    agent = get_agent(agent_type)
    _task_start_time = time.time()
    _attempt_num = (task.get("worker_attempts") or 0) + 1

    async def _progress_cb(tid, iteration, max_iter, summary):
        if self.telegram:
            elapsed = int(time.time() - _task_start_time)
            attempt_tag = f" | attempt {_attempt_num}" if _attempt_num > 1 else ""
            msg = (f"🔄 *Task #{tid}* — iteration {iteration}/{max_iter} "
                   f"({elapsed}s elapsed{attempt_tag})\n{summary[:200]}")
            try:
                await self.telegram.send_notification(msg)
            except Exception:
                pass

    # Task-started notification
    try:
        task_ctx = parse_context(task)
        if not task_ctx.get("silent"):
            chat_id = task_ctx.get("chat_id")
            if chat_id and self.telegram:
                await self.telegram.app.bot.send_message(
                    chat_id=chat_id,
                    text=f"🚀 Task #{task_id} assigned to {agent_type}, starting...",
                )
    except Exception:
        pass

    agent._task_timeout = timeout_seconds
    return await asyncio.wait_for(
        agent.execute(task, progress_callback=_progress_cb),
        timeout=timeout_seconds,
    )
```

- [ ] **Step 4: Extract `_record`**

```python
async def _record(self, task: dict, agent_result: dict | None):
    """Route the agent result to the appropriate handler."""
    from src.core.result_router import (
        route_result, Complete, SpawnSubtasks, RequestClarification,
        RequestReview, Exhausted, Failed,
    )
    actions = route_result(task, agent_result)
    for action in actions:
        if isinstance(action, Complete):
            await self._handle_complete(task, action.result, action.iterations, action.metadata)
        elif isinstance(action, SpawnSubtasks):
            await self._handle_subtasks(task, action.subtasks)
        elif isinstance(action, RequestClarification):
            await self._handle_clarification(task, action.question, action.chat_id)
        elif isinstance(action, RequestReview):
            await self._handle_review(task, action.summary)
        elif isinstance(action, Exhausted):
            await self._handle_exhausted(task, action.error)
        elif isinstance(action, Failed):
            await self._handle_failed(task, action.error)
```

- [ ] **Step 5: Rewrite `process_task` as orchestration of the three stages**

```python
async def process_task(self, task: dict):
    """Process a single task: prepare -> dispatch -> record."""
    task_id = task["id"]
    try:
        prepared = await self._prepare(task)
        if prepared is None:
            return
        task, agent_type, timeout_seconds = prepared

        try:
            result = await self._dispatch(task, agent_type, timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning("task timeout", task_id=task_id, timeout=timeout_seconds)
            result = {"status": "failed", "error": f"timeout_after_{timeout_seconds}s"}
        except Exception as e:
            logger.error("dispatch error", task_id=task_id, error=str(e))
            result = {"status": "failed", "error": str(e)}

        await self._record(task, result)
    except Exception as e:
        logger.error("process_task fatal", task_id=task_id, error=str(e))
        try:
            await update_task(task_id, status="failed", error=str(e)[:500])
        except Exception:
            pass
```

Target: `process_task` shrinks from 1,143 lines to ~25 lines.

- [ ] **Step 6: Handle any `_handle_*` methods that don't yet exist**

If `_handle_exhausted` or `_handle_failed` don't exist as helper methods in the current orchestrator, extract them from the inline code in the pre-refactor `process_task`. Each just does DB update + Telegram notify:

```python
async def _handle_exhausted(self, task, error):
    await update_task(task["id"], status="exhausted", error=error[:500])
    # Notify user if chat_id present — mirror existing inline behavior

async def _handle_failed(self, task, error):
    await update_task(task["id"], status="failed", error=error[:500])
    # Notify user if chat_id present — mirror existing inline behavior
```

Look in the current `process_task` for what was done inline for each status and move it verbatim.

- [ ] **Step 7: Run full suite**

Run: `timeout 180 python -m pytest tests/ -x`
Expected: all tests PASS. This is the moment of truth — if any behavior drifted, this catches it.

- [ ] **Step 8: Line count check**

Run: `wc -l src/core/orchestrator.py`
Expected: significantly reduced from 3,865. Target is roughly 2,000 or below after Tasks 4–8 land.

- [ ] **Step 9: Smoke test via Telegram**

Restart KutAI. Run: simple `/task hello world`, `/shop coffee machine`, and a workflow that triggers a human gate. Each must behave identically.

- [ ] **Step 10: Commit**

```bash
git add src/core/orchestrator.py
git commit -m "refactor(core): split process_task into prepare/dispatch/record stages

process_task shrinks from 1,143 lines to ~25. Each stage is a private
method with a single responsibility:

  _prepare  — claim, classify, enrich, gate, snapshot (returns None if blocked)
  _dispatch — invoke agent or pipeline, wait with timeout
  _record   — route the result via result_router

All side effects preserved verbatim. This is a pure refactor — no
behavioral change. The shape now matches Phase 2b's task master interface:
prepare corresponds to intake+gate, dispatch corresponds to orchestrator
routing an LLM Dispatch, record corresponds to task_master.record_result."
```

---

## Task 9: Split `watchdog()` into Focused Functions

**Goal:** The 541-line `watchdog()` method mixes task claiming races, stuck task cleanup, load management, decay checks, and mission advancement. Split into standalone async functions (or module-level helpers) so each is individually understandable.

**Files:**
- Create: `src/core/watchdog.py`
- Modify: `src/core/orchestrator.py` — thin wrapper methods that delegate

- [ ] **Step 1: Inventory what watchdog does**

Run: `timeout 10 python -c "
import re
with open('src/core/orchestrator.py') as f:
    content = f.read()
start = content.index('async def watchdog')
end = content.index('async def ', start + 10)
print(content[start:end][:3000])
"`

Identify the distinct sub-concerns. Expected themes (based on earlier analysis):
- Stuck task detection and requeue
- Load management (too many in flight? model swap?)
- Decay checks (old pending tasks)
- Mission advancement (phase transitions, completion checks)

- [ ] **Step 2: Create the module**

`src/core/watchdog.py`:

```python
"""Watchdog functions: periodic maintenance called from orchestrator's main loop.

Previously all inside orchestrator.watchdog() (541 lines). Split by concern
so each is individually testable and extractable later.
"""

from src.infra.logging_config import get_logger

logger = get_logger("core.watchdog")


async def check_stuck_tasks(telegram=None):
    """Find tasks in 'in_progress' longer than the timeout threshold and requeue them.

    Pasted verbatim from the stuck-check section of orchestrator.watchdog.
    """
    # TODO: paste stuck-task block from orchestrator.watchdog during this task
    raise NotImplementedError("Paste during Task 9 step 3")


async def check_load():
    """Query system load and apply backpressure / model-swap decisions.

    Pasted verbatim from the load-management section of orchestrator.watchdog.
    """
    # TODO: paste load-management block
    raise NotImplementedError("Paste during Task 9 step 3")


async def check_decay():
    """Age-out or reprioritize stale pending tasks.

    Pasted verbatim from the decay-check section of orchestrator.watchdog.
    """
    # TODO: paste decay block
    raise NotImplementedError("Paste during Task 9 step 3")


async def advance_missions(telegram=None):
    """Advance mission phases; trigger completion checks.

    Pasted verbatim from the mission-advancement section of orchestrator.watchdog.
    """
    # TODO: paste mission-advancement block
    raise NotImplementedError("Paste during Task 9 step 3")
```

- [ ] **Step 3: Paste each block verbatim**

Open `src/core/orchestrator.py`, locate `watchdog()`. For each of the four sections above, copy the relevant code into the matching module function. Replace `self.telegram` with the `telegram` parameter. Replace any `self.*` references with imports or parameter passing.

- [ ] **Step 4: Replace `orchestrator.watchdog()` with a thin dispatcher**

```python
async def watchdog(self):
    """Periodic maintenance. Delegates to src/core/watchdog module."""
    from src.core import watchdog as wd
    try:
        await wd.check_stuck_tasks(telegram=self.telegram)
    except Exception as e:
        logger.error("watchdog stuck_tasks error: %s", e)
    try:
        await wd.check_load()
    except Exception as e:
        logger.error("watchdog load error: %s", e)
    try:
        await wd.check_decay()
    except Exception as e:
        logger.error("watchdog decay error: %s", e)
    try:
        await wd.advance_missions(telegram=self.telegram)
    except Exception as e:
        logger.error("watchdog missions error: %s", e)
```

- [ ] **Step 5: Run full suite**

Run: `timeout 180 python -m pytest tests/ -x`
Expected: all tests PASS. Watchdog is exercised indirectly via any test that runs the orchestrator loop.

- [ ] **Step 6: Smoke test**

Let KutAI run for 30 minutes with a few tasks. Confirm no stuck-task lies around and watchdog logs fire normally.

- [ ] **Step 7: Commit**

```bash
git add src/core/watchdog.py src/core/orchestrator.py
git commit -m "refactor(core): split watchdog into focused module functions

541-line watchdog() method replaced by a thin delegator. The four concerns
(stuck tasks, load, decay, missions) are now standalone async functions
in src/core/watchdog.py. Each is individually testable and Phase 2b will
promote stuck_tasks + decay into task master (they're task-domain logic)."
```

---

## Task 10: Smoke Test + Line Count + Final Commit

**Goal:** Verify the whole refactor holds up, document the outcome, and commit a closing marker.

- [ ] **Step 1: Run the full test suite with timeouts**

Run: `timeout 300 python -m pytest tests/ -v`
Expected: all tests PASS, no new warnings, no new xfails.

- [ ] **Step 2: Measure line counts**

Run: `wc -l src/core/orchestrator.py src/core/task_gates.py src/core/task_context.py src/core/result_router.py src/core/watchdog.py src/core/decisions.py src/core/mechanical/*.py src/app/scheduled_jobs.py`

Record the numbers. Orchestrator should be well under half its starting 3,865.

- [ ] **Step 3: Manual Telegram smoke test**

Restart KutAI. Exercise:
- Simple task: `/task what time is it?`
- Shopping intent: `/shop coffee machine under 500 TL`
- Workflow: trigger i2p with a small prompt
- Clarification flow: trigger a task that asks a question
- Human gate: trigger a high-risk task (e.g. `/task delete old log files`)

Each must behave identically to pre-refactor.

- [ ] **Step 4: Update architecture doc**

Open `docs/architecture-modularization.md`. Add a short section at the end:

```markdown
## Phase 1 In-Tree Refactor (2026-04-17)

Orchestrator untangled in-tree without package extraction. New in-tree modules:

- `src/core/decisions.py` — Dispatch, NotifyUser, GateDecision types (Phase 2b targets)
- `src/core/task_context.py` — centralized context parse/serialize
- `src/core/task_gates.py` — human + risk gates, returns GateDecision
- `src/core/result_router.py` — agent result → Action state machine
- `src/core/watchdog.py` — focused maintenance functions
- `src/core/mechanical/` — in-tree home for non-LLM executors (Phase 2a target)
- `src/app/scheduled_jobs.py` — proactive jobs (todos, price watch, digest)

`process_task` shrank from 1,143 lines to ~25; `orchestrator.py` overall from 3,865 to <2,000.

Phase 2a (`packages/mechanical_dispatcher/`) and Phase 2b (`packages/gorev_ustasi/`)
are separate plans, written after Phase 1 revealed the real seams.
```

- [ ] **Step 5: Save auto-memory note**

Append to `MEMORY.md`:

```
- [Orchestrator Phase 1 Refactor](project_orchestrator_phase1_20260417.md) — in-tree untangle: decisions/gates/router/mechanical/scheduled_jobs modules, process_task 1143→25 lines, orchestrator 3865→<2000 (2026-04-17)
```

Create the memory file content per the memory system rules.

- [ ] **Step 6: Final commit**

```bash
git add docs/architecture-modularization.md MEMORY.md memory/project_orchestrator_phase1_20260417.md
git commit -m "docs: record Phase 1 orchestrator refactor outcome

Phase 1 complete. Phase 2a (Mechanical Dispatcher package) and
Phase 2b (Task Master package) to be planned separately once the
in-tree seams are proven in daily use."
```

---

## Phase 1 Definition of Done

- [ ] All existing tests pass (`timeout 300 pytest tests/`)
- [ ] Manual Telegram smoke checks pass
- [ ] `orchestrator.py` < 2,000 lines
- [ ] `process_task` < 50 lines
- [ ] No behavioral change visible to users
- [ ] Architecture doc updated
- [ ] Memory note saved
- [ ] All commits reviewed before merging to main

## Out of Scope (Phase 2)

These are intentionally deferred:

- Extracting `src/core/mechanical/` to `packages/mechanical_dispatcher/` (Phase 2a)
- Extracting gates/router/context/watchdog to `packages/gorev_ustasi/` (Phase 2b)
- Converting `_handle_*` methods into Decision emissions
- Wiring the capacity-push event from Nerd Herd/KDV
- Inverting Telegram coupling fully (gates still take `approval_fn`; Phase 2b replaces with `RequestApproval` decision)

Each Phase 2 extraction gets its own plan written after Phase 1 lands and has been in use for at least a week.
