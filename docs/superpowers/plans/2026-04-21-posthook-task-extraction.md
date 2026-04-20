# Post-Hook Task Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract grading and per-task LLM summary from orphaned drain loops into first-class Beckman-scheduled tasks. Clean two layer violations (dispatcher doing grading, agent self-transitioning to `ungraded`). Restore the summarization path broken since Task 13.

**Architecture:** Source task returns plain `completed`; Beckman's rewrite emits `Complete + RequestPostHook` actions; apply enqueues a `grader` (or `artifact_summarizer`) agent task and parks the source in `ungraded`. Post-hook task runs through the normal dispatch pipe (Fatih Hoca picks model, dispatcher is a pipe). On completion, its `posthook_verdict` payload becomes a `PostHookVerdict` action that unblocks or retries the source. Grade is always first; summaries spawn per-large-artifact after grade passes.

**Tech Stack:** Python 3.10, asyncio, aiosqlite, pytest-asyncio. Worktree `.worktrees/posthook-extraction` on branch `feat/posthook-extraction`. Shared venv at `../../.venv/Scripts/python.exe`. Never `pip install -e` from worktree paths.

**Spec:** `docs/superpowers/specs/2026-04-21-posthook-task-extraction-design.md`

**Mandatory test invocation** (every task's "run tests" step uses this prefix; Windows path separator in `PYTHONPATH` is `;`):

```bash
DB_PATH="$PWD/worktree_test.db" timeout 60 .venv/Scripts/python.exe -m pytest <tests> -v
rm -f worktree_test.db
```

**Do not run the full pytest suite** — some paths eagerly spawn llama-server. Stick to targeted lists in each task.

---

## File Structure

**New files:**
- `src/agents/grader.py` — `GraderAgent` wrapping `grade_task()`.
- `src/agents/artifact_summarizer.py` — `ArtifactSummarizerAgent` wrapping `_llm_summarize()`.
- `packages/general_beckman/src/general_beckman/posthooks.py` — `determine_posthooks` policy + verdict-apply helpers.
- `tests/test_beckman_posthooks.py` — new Beckman post-hook pipeline tests.
- `tests/test_grader_agent.py` — GraderAgent unit tests.
- `tests/test_artifact_summarizer_agent.py` — ArtifactSummarizerAgent unit tests.
- `tests/test_migration_ungraded_to_posthooks.py` — boot sweep test.

**Modified files:**
- `packages/general_beckman/src/general_beckman/result_router.py` — new `RequestPostHook`, `PostHookVerdict` dataclasses.
- `packages/general_beckman/src/general_beckman/rewrite.py` — emit RequestPostHook; skip-list extension.
- `packages/general_beckman/src/general_beckman/apply.py` — `_apply_request_posthook`, `_apply_posthook_verdict` handlers.
- `packages/general_beckman/src/general_beckman/__init__.py` — remove ungraded short-circuit, add `on_model_swap`, migration sweep, progress-ping skip-list extension.
- `src/agents/__init__.py` — register two new agents.
- `src/agents/base.py` — remove `transition_task("ungraded")` block.
- `src/core/llm_dispatcher.py` — delete `on_model_swap` method.
- `src/models/local_model_manager.py` — redirect swap event to Beckman.
- `src/workflows/engine/hooks.py` — delete `drain_pending_summaries`, `queue_llm_summary`, remove call site in `post_execute_workflow_step`.
- `src/core/grading.py` — delete `drain_ungraded_tasks`.

**Deleted (tests referencing removed functions):**
- Any test calling `drain_ungraded_tasks` / `drain_pending_summaries`.

---

## Task 1: New action dataclasses

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/result_router.py`
- Test: `tests/test_beckman_posthooks.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_beckman_posthooks.py`:

```python
"""Post-hook pipeline tests: actions, policy, apply, migrations."""
import pytest
from general_beckman.result_router import (
    Action, RequestPostHook, PostHookVerdict,
)


def test_request_posthook_is_action():
    a = RequestPostHook(source_task_id=1, kind="grade", source_ctx={})
    assert isinstance(a, RequestPostHook)
    # Action is a Union; isinstance check works via dataclass identity.
    assert a.source_task_id == 1
    assert a.kind == "grade"


def test_posthook_verdict_is_action():
    v = PostHookVerdict(
        source_task_id=2, kind="grade", passed=True, raw={"score": 0.9},
    )
    assert v.passed is True
    assert v.raw == {"score": 0.9}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_beckman_posthooks.py -v
rm -f worktree_test.db
```

Expected: `ImportError: cannot import name 'RequestPostHook'`.

- [ ] **Step 3: Implement the dataclasses**

Add to `packages/general_beckman/src/general_beckman/result_router.py` after the existing `MissionAdvance` class:

```python
@dataclass(frozen=True)
class RequestPostHook:
    """Spawn a post-hook task (grader or artifact_summarizer) for a source.

    `kind` is either "grade" or "summary:<artifact_name>" (one spawn per
    large output artifact after a grade pass).
    """
    source_task_id: int
    kind: str
    source_ctx: dict


@dataclass(frozen=True)
class PostHookVerdict:
    """Apply the result of a completed post-hook task back to the source."""
    source_task_id: int
    kind: str
    passed: bool
    raw: dict
```

Extend the `Action` union at the bottom of the file:

```python
Action = Union[
    Complete, SpawnSubtasks, RequestClarification, RequestReview,
    Exhausted, Failed, MissionAdvance, CompleteWithReusedAnswer,
    RequestPostHook, PostHookVerdict,
]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_beckman_posthooks.py -v
rm -f worktree_test.db
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/result_router.py tests/test_beckman_posthooks.py
git commit -m "feat(beckman): add RequestPostHook and PostHookVerdict actions"
```

---

## Task 2: `determine_posthooks` policy function

**Files:**
- Create: `packages/general_beckman/src/general_beckman/posthooks.py`
- Modify: `tests/test_beckman_posthooks.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_beckman_posthooks.py`:

```python
from general_beckman.posthooks import determine_posthooks


def test_mechanical_task_needs_no_posthooks():
    task = {"agent_type": "mechanical"}
    assert determine_posthooks(task, {}, {}) == []


def test_shopping_pipeline_task_needs_no_posthooks():
    task = {"agent_type": "shopping_pipeline"}
    assert determine_posthooks(task, {}, {}) == []


def test_grader_task_needs_no_posthooks():
    task = {"agent_type": "grader"}
    assert determine_posthooks(task, {}, {}) == []


def test_artifact_summarizer_task_needs_no_posthooks():
    task = {"agent_type": "artifact_summarizer"}
    assert determine_posthooks(task, {}, {}) == []


def test_llm_agent_task_needs_grade_by_default():
    task = {"agent_type": "writer"}
    assert determine_posthooks(task, {}, {}) == ["grade"]


def test_requires_grading_false_opts_out():
    task = {"agent_type": "writer"}
    ctx = {"requires_grading": False}
    assert determine_posthooks(task, ctx, {}) == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_beckman_posthooks.py -v -k posthook
rm -f worktree_test.db
```

Expected: ImportError for `posthooks`.

- [ ] **Step 3: Implement the policy**

Create `packages/general_beckman/src/general_beckman/posthooks.py`:

```python
"""Policy and apply helpers for post-hook tasks (grading, artifact summary).

`determine_posthooks` decides which post-hooks to spawn *immediately* after
a source task completes. Summary spawning happens later — after grade
passes — and is driven by `_apply_posthook_verdict` in `apply.py`.
"""
from __future__ import annotations

# Agent types that never need post-hooks:
# - mechanical: not LLM output, nothing to grade/summarise
# - shopping_pipeline: deterministic pipeline, not LLM output
# - grader, artifact_summarizer: the post-hook runners themselves
_NO_POSTHOOKS_AGENT_TYPES: frozenset[str] = frozenset({
    "mechanical",
    "shopping_pipeline",
    "grader",
    "artifact_summarizer",
})


def determine_posthooks(
    task: dict, task_ctx: dict, result: dict,
) -> list[str]:
    """Return the list of post-hook kinds to spawn immediately.

    Summary is NOT in the immediately-spawned list — it's deferred until
    the grade task passes (see apply._apply_posthook_verdict).
    """
    agent_type = task.get("agent_type", "")
    if agent_type in _NO_POSTHOOKS_AGENT_TYPES:
        return []
    if task_ctx.get("requires_grading") is False:
        return []
    return ["grade"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_beckman_posthooks.py -v -k posthook
rm -f worktree_test.db
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/posthooks.py tests/test_beckman_posthooks.py
git commit -m "feat(beckman): determine_posthooks policy function"
```

---

## Task 3: Rewrite rule — emit RequestPostHook on mission Complete

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/rewrite.py`
- Test: `tests/test_beckman_rewrite.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_beckman_rewrite.py`:

```python
from general_beckman.result_router import Complete, RequestPostHook, MissionAdvance
from general_beckman.rewrite import rewrite_actions


def test_writer_task_complete_emits_request_grade_posthook():
    task = {"id": 100, "mission_id": 5, "agent_type": "writer"}
    ctx = {}
    complete = Complete(task_id=100, result="out", iterations=1, metadata={}, raw={})
    out = rewrite_actions(task, ctx, [complete])
    # Expect: Complete, MissionAdvance, RequestPostHook(grade).
    kinds = [type(a).__name__ for a in out]
    assert "RequestPostHook" in kinds
    posthook = next(a for a in out if isinstance(a, RequestPostHook))
    assert posthook.kind == "grade"
    assert posthook.source_task_id == 100


def test_mechanical_task_complete_emits_no_posthook():
    task = {"id": 200, "mission_id": 5, "agent_type": "mechanical"}
    ctx = {"payload": {"action": "workflow_advance"}}
    complete = Complete(task_id=200, result="out", iterations=1, metadata={}, raw={})
    out = rewrite_actions(task, ctx, [complete])
    kinds = [type(a).__name__ for a in out]
    assert "RequestPostHook" not in kinds
    # workflow_advance mechanical also skips MissionAdvance (pre-existing guard).
    assert "MissionAdvance" not in kinds
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_beckman_rewrite.py -v -k posthook
rm -f worktree_test.db
```

Expected: FAIL (rewrite doesn't emit RequestPostHook).

- [ ] **Step 3: Update rewrite.py**

Modify `packages/general_beckman/src/general_beckman/rewrite.py` — replace the imports at top and the `_rewrite_one` body. New imports:

```python
from general_beckman.result_router import (
    Action, Complete, SpawnSubtasks, RequestClarification,
    Failed, MissionAdvance, CompleteWithReusedAnswer,
    RequestPostHook,
)
from general_beckman.posthooks import determine_posthooks
```

Replace `_rewrite_one` so rule 1 also emits post-hook requests:

```python
def _rewrite_one(task: dict, task_ctx: dict, a: Action) -> list[Action]:
    # Rule 1: mission-task clean completion → emit MissionAdvance (unless
    # bookkeeping) and RequestPostHook (unless policy says no).
    payload_action = (task_ctx.get("payload") or {}).get("action")
    agent_type = task.get("agent_type", "")
    is_bookkeeping = (
        payload_action == "workflow_advance"
        or agent_type in {"grader", "artifact_summarizer"}
    )

    if isinstance(a, Complete) and task.get("mission_id") and not is_bookkeeping:
        result_actions: list[Action] = [a]
        result_actions.append(
            MissionAdvance(
                task_id=a.task_id,
                mission_id=task["mission_id"],
                completed_task_id=a.task_id,
                raw=a.raw,
            )
        )
        for kind in determine_posthooks(task, task_ctx, a.raw):
            result_actions.append(
                RequestPostHook(
                    source_task_id=a.task_id,
                    kind=kind,
                    source_ctx=dict(task_ctx),
                )
            )
        return result_actions

    # Rule 2: workflow step tried to decompose
    if isinstance(a, SpawnSubtasks) and _is_workflow_step(task_ctx):
        return [Failed(
            task_id=a.parent_task_id,
            error="Workflow step tried to decompose instead of producing artifact",
            raw=a.raw,
        )]
    # Rules 3–5: clarification rewrites (unchanged)
    if isinstance(a, RequestClarification):
        if task_ctx.get("silent"):
            return [Failed(
                task_id=a.task_id,
                error="Insufficient info (silent task, no clarification)",
                raw=a.raw,
            )]
        if task_ctx.get("may_need_clarification") is False:
            return [Failed(
                task_id=a.task_id,
                error="Agent requested clarification on no-clarification step",
                raw=a.raw,
            )]
        history = task_ctx.get("clarification_history")
        if history:
            body = _format_history(history) or task_ctx.get("user_clarification", "")
            return [CompleteWithReusedAnswer(
                task_id=a.task_id, result=body, raw=a.raw,
            )]
    return [a]
```

- [ ] **Step 4: Run full rewrite test suite**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_beckman_rewrite.py -v
rm -f worktree_test.db
```

Expected: all pre-existing tests + the two new ones pass.

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/rewrite.py tests/test_beckman_rewrite.py
git commit -m "feat(beckman): emit RequestPostHook on mission-task completion"
```

---

## Task 4: `_apply_request_posthook` — enqueue grader task, park source

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/apply.py`
- Test: `tests/test_beckman_posthooks.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_beckman_posthooks.py`:

```python
import json
import tempfile
import os
import pytest
from general_beckman.result_router import RequestPostHook
from general_beckman.apply import _apply_one


@pytest.mark.asyncio
async def test_apply_request_posthook_grade_enqueues_grader_and_parks_source(tmp_path, monkeypatch):
    # Set up a throwaway DB.
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, get_task
    await init_db()
    source_id = await add_task(
        title="source work",
        description="",
        agent_type="writer",
        mission_id=1,
        context=json.dumps({"generating_model": "qwen-7b"}),
    )

    source_task = await get_task(source_id)
    action = RequestPostHook(
        source_task_id=source_id,
        kind="grade",
        source_ctx=json.loads(source_task["context"]),
    )
    await _apply_one(source_task, action)

    refreshed = await get_task(source_id)
    assert refreshed["status"] == "ungraded"
    ctx = json.loads(refreshed["context"])
    assert ctx["_pending_posthooks"] == ["grade"]

    # Grader task exists.
    from src.infra.db import get_db
    db = await get_db()
    cursor = await db.execute(
        "SELECT id, agent_type, mission_id, context FROM tasks "
        "WHERE agent_type = 'grader'"
    )
    rows = list(await cursor.fetchall())
    assert len(rows) == 1
    grader_ctx = json.loads(rows[0]["context"])
    assert grader_ctx["source_task_id"] == source_id
    assert grader_ctx["generating_model"] == "qwen-7b"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_beckman_posthooks.py::test_apply_request_posthook_grade_enqueues_grader_and_parks_source -v
rm -f worktree_test.db
```

Expected: FAIL — `_apply_one` doesn't handle `RequestPostHook`.

- [ ] **Step 3: Add the handler**

Modify `packages/general_beckman/src/general_beckman/apply.py`:

1. Update imports:

```python
from general_beckman.result_router import (
    Action, Complete, SpawnSubtasks, RequestClarification, RequestReview,
    Exhausted, Failed, MissionAdvance, CompleteWithReusedAnswer,
    RequestPostHook, PostHookVerdict,
)
```

2. Extend `_apply_one` dispatch:

```python
async def _apply_one(task: dict, a: Action) -> None:
    if isinstance(a, (Complete, CompleteWithReusedAnswer)):
        await _apply_complete(task, a)
    elif isinstance(a, SpawnSubtasks):
        await _apply_subtasks(task, a)
    elif isinstance(a, RequestClarification):
        await _apply_clarify(task, a)
    elif isinstance(a, RequestReview):
        await _apply_review(task, a)
    elif isinstance(a, Exhausted):
        await _apply_exhausted(task, a)
    elif isinstance(a, Failed):
        await _apply_failed(task, a)
    elif isinstance(a, MissionAdvance):
        await _apply_mission_advance(task, a)
    elif isinstance(a, RequestPostHook):
        await _apply_request_posthook(task, a)
    elif isinstance(a, PostHookVerdict):
        await _apply_posthook_verdict(task, a)
    else:
        logger.warning("unknown action type", action=type(a).__name__)
```

3. Append the handler near the other `_apply_*` functions:

```python
async def _apply_request_posthook(task: dict, a: RequestPostHook) -> None:
    """Park the source in `ungraded`, enqueue a post-hook task row."""
    import json as _json
    from src.infra.db import add_task, get_task, update_task

    source = await get_task(a.source_task_id)
    if source is None:
        logger.warning("posthook: source missing", source_id=a.source_task_id)
        return

    ctx = _parse_ctx(source)
    pending = list(ctx.get("_pending_posthooks") or [])
    if a.kind not in pending:
        pending.append(a.kind)
    ctx["_pending_posthooks"] = pending

    await update_task(
        a.source_task_id,
        status="ungraded",
        context=_json.dumps(ctx),
    )

    agent_type, payload = _posthook_agent_and_payload(a, source, ctx)
    await add_task(
        title=_posthook_title(a, source),
        description="",
        agent_type=agent_type,
        mission_id=source.get("mission_id"),
        depends_on=[],
        context=_json.dumps(payload),
    )


def _posthook_agent_and_payload(
    a: RequestPostHook, source: dict, source_ctx: dict,
) -> tuple[str, dict]:
    if a.kind == "grade":
        return ("grader", {
            "source_task_id": a.source_task_id,
            "generating_model": source_ctx.get("generating_model", ""),
            "excluded_models": list(source_ctx.get("grade_excluded_models") or []),
        })
    if a.kind.startswith("summary:"):
        artifact_name = a.kind.split(":", 1)[1]
        return ("artifact_summarizer", {
            "source_task_id": a.source_task_id,
            "artifact_name": artifact_name,
        })
    raise ValueError(f"unknown posthook kind: {a.kind!r}")


def _posthook_title(a: RequestPostHook, source: dict) -> str:
    if a.kind == "grade":
        return f"Grade task #{a.source_task_id}"
    if a.kind.startswith("summary:"):
        name = a.kind.split(":", 1)[1]
        return f"Summarize '{name}' for #{a.source_task_id}"
    return f"Posthook {a.kind} for #{a.source_task_id}"


async def _apply_posthook_verdict(task: dict, a: PostHookVerdict) -> None:
    """Placeholder; implemented in later tasks."""
    raise NotImplementedError("_apply_posthook_verdict lands in Task 5")
```

- [ ] **Step 4: Run test**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 60 .venv/Scripts/python.exe -m pytest tests/test_beckman_posthooks.py tests/test_beckman_apply.py -v
rm -f worktree_test.db
```

Expected: new test + existing apply tests pass.

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/apply.py tests/test_beckman_posthooks.py
git commit -m "feat(beckman): _apply_request_posthook parks source and enqueues post-hook task"
```

---

## Task 5: `_apply_posthook_verdict` — grade pass with small output

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/apply.py`
- Test: `tests/test_beckman_posthooks.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_beckman_posthooks.py`:

```python
@pytest.mark.asyncio
async def test_grade_verdict_pass_small_output_completes_source(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, get_task, update_task
    await init_db()
    source_id = await add_task(
        title="source",
        description="",
        agent_type="writer",
        mission_id=1,
        context=json.dumps({
            "_pending_posthooks": ["grade"],
            "output_artifacts": ["short_out"],
        }),
    )
    await update_task(source_id, status="ungraded", result="short result (<3KB)")

    from general_beckman.result_router import PostHookVerdict
    verdict = PostHookVerdict(
        source_task_id=source_id, kind="grade", passed=True,
        raw={"score": 0.9},
    )
    from general_beckman.apply import _apply_one
    grade_task_row = {"id": 999, "mission_id": 1, "agent_type": "grader"}
    await _apply_one(grade_task_row, verdict)

    refreshed = await get_task(source_id)
    assert refreshed["status"] == "completed"
    ctx = json.loads(refreshed["context"])
    assert ctx["_pending_posthooks"] == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_beckman_posthooks.py::test_grade_verdict_pass_small_output_completes_source -v
rm -f worktree_test.db
```

Expected: FAIL (`NotImplementedError`).

- [ ] **Step 3: Implement `_apply_posthook_verdict`**

Replace the `NotImplementedError` stub in `apply.py`:

```python
async def _apply_posthook_verdict(task: dict, a: PostHookVerdict) -> None:
    """Apply a post-hook verdict back to the source task."""
    import json as _json
    from src.infra.db import get_task, update_task, add_task
    from src.workflows.engine.artifacts import ArtifactStore

    source = await get_task(a.source_task_id)
    if source is None:
        logger.debug("posthook verdict: source missing",
                     source_id=a.source_task_id)
        return
    if source.get("status") != "ungraded":
        logger.debug(
            "posthook verdict: source no longer ungraded, dropping",
            source_id=a.source_task_id, status=source.get("status"),
        )
        return

    ctx = _parse_ctx(source)
    pending = list(ctx.get("_pending_posthooks") or [])

    if a.kind == "grade" and not a.passed:
        # Reject: retry the source with updated exclude list.
        attempts = int(source.get("worker_attempts") or 0) + 1
        excluded = list(ctx.get("grade_excluded_models") or [])
        gen_model = ctx.get("generating_model") or ""
        if gen_model and gen_model not in excluded:
            excluded.append(gen_model)
        ctx["grade_excluded_models"] = excluded
        ctx["_pending_posthooks"] = []
        await update_task(
            a.source_task_id,
            status="pending",
            worker_attempts=attempts,
            error=str(a.raw)[:500],
            context=_json.dumps(ctx),
        )
        return

    if a.kind == "grade" and a.passed:
        # Remove "grade" from pending; spawn summary tasks for large artifacts.
        pending = [k for k in pending if k != "grade"]
        new_summary_kinds = _summary_kinds_for_source(source, ctx)
        for kind in new_summary_kinds:
            pending.append(kind)
            await add_task(
                title=f"Summarize '{kind.split(':',1)[1]}' for #{a.source_task_id}",
                description="",
                agent_type="artifact_summarizer",
                mission_id=source.get("mission_id"),
                depends_on=[],
                context=_json.dumps({
                    "source_task_id": a.source_task_id,
                    "artifact_name": kind.split(":", 1)[1],
                }),
            )
        ctx["_pending_posthooks"] = pending
        if not pending:
            await update_task(
                a.source_task_id, status="completed",
                context=_json.dumps(ctx),
            )
        else:
            await update_task(
                a.source_task_id, context=_json.dumps(ctx),
            )
        return

    if a.kind.startswith("summary:"):
        artifact_name = a.kind.split(":", 1)[1]
        if a.passed:
            summary_text = a.raw.get("summary", "") if isinstance(a.raw, dict) else ""
            if summary_text:
                store = ArtifactStore()
                await store.store(
                    source.get("mission_id"),
                    f"{artifact_name}_summary",
                    summary_text,
                )
        # On fail: structural summary already stored by post_execute; nothing to do.
        pending = [k for k in pending if k != a.kind]
        ctx["_pending_posthooks"] = pending
        if not pending:
            await update_task(
                a.source_task_id, status="completed",
                context=_json.dumps(ctx),
            )
        else:
            await update_task(
                a.source_task_id, context=_json.dumps(ctx),
            )
        return

    logger.warning("posthook verdict: unknown kind", kind=a.kind)


async def _summary_kinds_for_source(source: dict, source_ctx: dict) -> list[str]:
    """Return summary:<name> kinds for large output artifacts on this source.

    Reads the stored artifact values from the blackboard; enqueues one
    summary kind per artifact whose stored text exceeds 3000 chars.
    """
    from src.workflows.engine.artifacts import ArtifactStore

    mission_id = source.get("mission_id")
    if mission_id is None:
        return []
    output_names = list(source_ctx.get("output_artifacts") or [])
    if not output_names:
        return []
    store = ArtifactStore()
    kinds: list[str] = []
    for name in output_names:
        val = await store.retrieve(mission_id, name)
        if val and isinstance(val, str) and len(val) > 3000:
            kinds.append(f"summary:{name}")
    return kinds
```

Note: `_summary_kinds_for_source` is async because `ArtifactStore.retrieve` is async; the caller (`_apply_posthook_verdict`) needs to `await` it.

Update the call site:

```python
        new_summary_kinds = await _summary_kinds_for_source(source, ctx)
```

- [ ] **Step 4: Run test**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_beckman_posthooks.py -v
rm -f worktree_test.db
```

Expected: all posthook tests pass so far.

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/apply.py tests/test_beckman_posthooks.py
git commit -m "feat(beckman): _apply_posthook_verdict handles grade pass + small output"
```

---

## Task 6: `_apply_posthook_verdict` — grade pass spawns summaries for large artifacts

**Files:**
- Test: `tests/test_beckman_posthooks.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
@pytest.mark.asyncio
async def test_grade_pass_spawns_summary_for_large_artifact(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, get_task, update_task, get_db
    await init_db()

    source_id = await add_task(
        title="source",
        description="",
        agent_type="writer",
        mission_id=1,
        context=json.dumps({
            "_pending_posthooks": ["grade"],
            "output_artifacts": ["big_out"],
        }),
    )
    await update_task(source_id, status="ungraded", result="..." * 2000)

    # Pre-seed the artifact store with a >3KB value.
    from src.workflows.engine.artifacts import ArtifactStore
    store = ArtifactStore()
    await store.store(1, "big_out", "x" * 5000)

    from general_beckman.result_router import PostHookVerdict
    from general_beckman.apply import _apply_one
    verdict = PostHookVerdict(
        source_task_id=source_id, kind="grade", passed=True, raw={},
    )
    grade_row = {"id": 999, "mission_id": 1, "agent_type": "grader"}
    await _apply_one(grade_row, verdict)

    # Source still ungraded (summary pending).
    refreshed = await get_task(source_id)
    assert refreshed["status"] == "ungraded"
    ctx = json.loads(refreshed["context"])
    assert ctx["_pending_posthooks"] == ["summary:big_out"]

    # Artifact summarizer task enqueued.
    db = await get_db()
    cursor = await db.execute(
        "SELECT id, context FROM tasks WHERE agent_type='artifact_summarizer'"
    )
    rows = list(await cursor.fetchall())
    assert len(rows) == 1
    sum_ctx = json.loads(rows[0]["context"])
    assert sum_ctx["source_task_id"] == source_id
    assert sum_ctx["artifact_name"] == "big_out"
```

- [ ] **Step 2: Run test**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_beckman_posthooks.py::test_grade_pass_spawns_summary_for_large_artifact -v
rm -f worktree_test.db
```

Expected: PASS (logic from Task 5 already supports this; this test locks the behaviour).

If it fails, debug. Likely causes: ArtifactStore not finding the seeded value (missing persistence layer), or path mismatch.

- [ ] **Step 3: Implementation (already done in Task 5)**

No new code. This task validates the previous implementation against the large-artifact case.

- [ ] **Step 4: Run full posthook suite**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_beckman_posthooks.py -v
rm -f worktree_test.db
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_beckman_posthooks.py
git commit -m "test(beckman): grade pass spawns per-artifact summary tasks"
```

---

## Task 7: `_apply_posthook_verdict` — grade fail retries source

**Files:**
- Test: `tests/test_beckman_posthooks.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
@pytest.mark.asyncio
async def test_grade_fail_retries_source_excludes_generating_model(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, get_task, update_task
    await init_db()
    source_id = await add_task(
        title="source",
        description="",
        agent_type="writer",
        mission_id=1,
        context=json.dumps({
            "_pending_posthooks": ["grade"],
            "generating_model": "qwen-7b",
        }),
    )
    await update_task(source_id, status="ungraded", worker_attempts=1)

    from general_beckman.result_router import PostHookVerdict
    from general_beckman.apply import _apply_one
    verdict = PostHookVerdict(
        source_task_id=source_id, kind="grade", passed=False,
        raw="reason: hallucinated claim",
    )
    await _apply_one({"id": 999, "mission_id": 1, "agent_type": "grader"}, verdict)

    refreshed = await get_task(source_id)
    assert refreshed["status"] == "pending"
    assert refreshed["worker_attempts"] == 2
    ctx = json.loads(refreshed["context"])
    assert "qwen-7b" in ctx["grade_excluded_models"]
    assert ctx["_pending_posthooks"] == []
```

- [ ] **Step 2: Run test**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_beckman_posthooks.py::test_grade_fail_retries_source_excludes_generating_model -v
rm -f worktree_test.db
```

Expected: PASS (logic from Task 5 supports this).

- [ ] **Step 3: Implementation (already in Task 5)**

Validation only.

- [ ] **Step 4: Run full posthook suite** (same command as Task 6).

- [ ] **Step 5: Commit**

```bash
git add tests/test_beckman_posthooks.py
git commit -m "test(beckman): grade fail retries source and excludes generating_model"
```

---

## Task 8: `_apply_posthook_verdict` — summary verdict handling

**Files:**
- Test: `tests/test_beckman_posthooks.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
@pytest.mark.asyncio
async def test_summary_pass_stores_artifact_and_completes_source(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, get_task, update_task
    await init_db()
    source_id = await add_task(
        title="source",
        description="",
        agent_type="writer",
        mission_id=1,
        context=json.dumps({"_pending_posthooks": ["summary:big_out"]}),
    )
    await update_task(source_id, status="ungraded")

    from general_beckman.result_router import PostHookVerdict
    from general_beckman.apply import _apply_one
    verdict = PostHookVerdict(
        source_task_id=source_id, kind="summary:big_out",
        passed=True, raw={"summary": "condensed output", "artifact_name": "big_out"},
    )
    await _apply_one({"id": 998, "mission_id": 1, "agent_type": "artifact_summarizer"}, verdict)

    refreshed = await get_task(source_id)
    assert refreshed["status"] == "completed"
    ctx = json.loads(refreshed["context"])
    assert ctx["_pending_posthooks"] == []

    from src.workflows.engine.artifacts import ArtifactStore
    stored = await ArtifactStore().retrieve(1, "big_out_summary")
    assert stored == "condensed output"


@pytest.mark.asyncio
async def test_summary_fail_keeps_structural_and_completes_source(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, get_task, update_task
    await init_db()
    source_id = await add_task(
        title="source",
        description="",
        agent_type="writer",
        mission_id=1,
        context=json.dumps({"_pending_posthooks": ["summary:big_out"]}),
    )
    await update_task(source_id, status="ungraded")

    from general_beckman.result_router import PostHookVerdict
    from general_beckman.apply import _apply_one
    verdict = PostHookVerdict(
        source_task_id=source_id, kind="summary:big_out",
        passed=False, raw={"artifact_name": "big_out"},
    )
    await _apply_one({"id": 997, "mission_id": 1, "agent_type": "artifact_summarizer"}, verdict)

    refreshed = await get_task(source_id)
    assert refreshed["status"] == "completed"
```

- [ ] **Step 2: Run tests**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_beckman_posthooks.py -v -k summary
rm -f worktree_test.db
```

Expected: both new tests pass (logic from Task 5 handles both).

- [ ] **Step 3: Implementation (already in Task 5)**

Validation only.

- [ ] **Step 4: Full posthook test run** (same as Task 6).

- [ ] **Step 5: Commit**

```bash
git add tests/test_beckman_posthooks.py
git commit -m "test(beckman): summary pass stores artifact; fail keeps structural"
```

---

## Task 9: Route result detects `posthook_verdict` signal from post-hook tasks

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/rewrite.py`
- Test: `tests/test_beckman_rewrite.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_beckman_rewrite.py`:

```python
from general_beckman.result_router import Complete, PostHookVerdict
from general_beckman.rewrite import rewrite_actions


def test_grader_task_complete_emits_posthook_verdict():
    task = {"id": 500, "mission_id": 1, "agent_type": "grader"}
    ctx = {}
    raw = {
        "status": "completed",
        "result": "grade json",
        "posthook_verdict": {
            "kind": "grade",
            "source_task_id": 100,
            "passed": True,
            "raw": {"score": 0.95},
        },
    }
    complete = Complete(task_id=500, result="grade json", iterations=1, metadata={}, raw=raw)
    out = rewrite_actions(task, ctx, [complete])
    kinds = [type(a).__name__ for a in out]
    assert "Complete" in kinds
    assert "PostHookVerdict" in kinds
    # Bookkeeping → no MissionAdvance.
    assert "MissionAdvance" not in kinds
    verdict = next(a for a in out if isinstance(a, PostHookVerdict))
    assert verdict.kind == "grade"
    assert verdict.source_task_id == 100
    assert verdict.passed is True
```

- [ ] **Step 2: Run test**

Expected: FAIL — rewrite doesn't translate `posthook_verdict` into a `PostHookVerdict` action.

- [ ] **Step 3: Extend `_rewrite_one`**

Modify `rewrite.py` — at the start of `_rewrite_one`, before the existing rule 1, add:

```python
    # Rule 0: post-hook task completion → translate into PostHookVerdict action.
    if isinstance(a, Complete) and task.get("agent_type") in (
        "grader", "artifact_summarizer",
    ):
        raw = a.raw or {}
        verdict_payload = raw.get("posthook_verdict") if isinstance(raw, dict) else None
        if isinstance(verdict_payload, dict):
            from general_beckman.result_router import PostHookVerdict
            return [
                a,
                PostHookVerdict(
                    source_task_id=verdict_payload["source_task_id"],
                    kind=verdict_payload["kind"],
                    passed=bool(verdict_payload.get("passed")),
                    raw=verdict_payload.get("raw") or {},
                ),
            ]
```

- [ ] **Step 4: Run tests**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_beckman_rewrite.py tests/test_beckman_posthooks.py -v
rm -f worktree_test.db
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/rewrite.py tests/test_beckman_rewrite.py
git commit -m "feat(beckman): translate posthook_verdict payload into PostHookVerdict action"
```

---

## Task 10: `GraderAgent`

**Files:**
- Create: `src/agents/grader.py`
- Modify: `src/agents/__init__.py`
- Test: `tests/test_grader_agent.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_grader_agent.py`:

```python
"""GraderAgent wraps grade_task and returns a posthook_verdict payload."""
import json
import pytest
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_grader_returns_posthook_verdict_shape(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task
    await init_db()
    source_id = await add_task(
        title="source", description="", agent_type="writer",
        context=json.dumps({"output_artifacts": ["out"]}),
    )

    from src.agents.grader import GraderAgent
    agent = GraderAgent()
    grader_task = {
        "id": 500,
        "agent_type": "grader",
        "context": json.dumps({
            "source_task_id": source_id,
            "generating_model": "qwen-7b",
        }),
    }

    fake_verdict = {
        "passed": True,
        "score": 0.9,
        "grader_model": "claude-sonnet",
        "cost": 0.002,
    }
    with patch("src.core.grading.grade_task", AsyncMock(return_value=fake_verdict)):
        result = await agent.execute(grader_task)

    assert result["status"] == "completed"
    assert "posthook_verdict" in result
    pv = result["posthook_verdict"]
    assert pv["kind"] == "grade"
    assert pv["source_task_id"] == source_id
    assert pv["passed"] is True
    assert pv["raw"] == fake_verdict


@pytest.mark.asyncio
async def test_grader_missing_source_returns_failed(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db
    await init_db()

    from src.agents.grader import GraderAgent
    agent = GraderAgent()
    task = {
        "id": 600,
        "agent_type": "grader",
        "context": json.dumps({"source_task_id": 99999}),
    }
    result = await agent.execute(task)
    assert result["status"] == "failed"
    assert "missing" in result["error"].lower() or "not found" in result["error"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_grader_agent.py -v
rm -f worktree_test.db
```

Expected: ImportError.

- [ ] **Step 3: Create `GraderAgent`**

Create `src/agents/grader.py`:

```python
"""GraderAgent — post-hook wrapper over src.core.grading.grade_task.

No ReAct loop. Reads `source_task_id` from the task context, fetches
the source task row, invokes `grade_task`, and returns a result dict
whose `posthook_verdict` field the Beckman rewrite layer translates
into a PostHookVerdict action.
"""
from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.infra.logging_config import get_logger

logger = get_logger("agents.grader")


class GraderAgent(BaseAgent):
    name = "grader"
    allowed_tools: list[str] = []

    async def execute(self, task: dict) -> dict:
        from src.core.grading import grade_task
        from src.infra.db import get_task

        ctx_raw = task.get("context") or "{}"
        try:
            ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
        except (json.JSONDecodeError, ValueError):
            ctx = {}

        source_task_id = ctx.get("source_task_id")
        if source_task_id is None:
            return {"status": "failed", "error": "grader: source_task_id missing"}

        source = await get_task(source_task_id)
        if source is None:
            return {
                "status": "failed",
                "error": f"grader: source task {source_task_id} missing",
            }

        verdict = await grade_task(source)
        passed = bool(verdict.get("passed", False))
        return {
            "status": "completed",
            "result": json.dumps(verdict, default=str),
            "model": verdict.get("grader_model", "unknown"),
            "cost": float(verdict.get("cost", 0.0)),
            "iterations": 1,
            "posthook_verdict": {
                "kind": "grade",
                "source_task_id": source_task_id,
                "passed": passed,
                "raw": verdict,
            },
        }
```

Register in `src/agents/__init__.py`:

```python
from .grader import GraderAgent
# ... existing imports ...

AGENT_REGISTRY = {
    # ... existing entries ...
    "grader": GraderAgent(),
}
```

- [ ] **Step 4: Run test**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_grader_agent.py -v
rm -f worktree_test.db
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agents/grader.py src/agents/__init__.py tests/test_grader_agent.py
git commit -m "feat(agents): GraderAgent wraps grade_task with posthook_verdict payload"
```

---

## Task 11: `ArtifactSummarizerAgent`

**Files:**
- Create: `src/agents/artifact_summarizer.py`
- Modify: `src/agents/__init__.py`
- Test: `tests/test_artifact_summarizer_agent.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_artifact_summarizer_agent.py`:

```python
"""ArtifactSummarizerAgent wraps _llm_summarize and returns a posthook_verdict."""
import json
import pytest
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_artifact_summarizer_success(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db
    await init_db()

    # Pre-seed artifact.
    from src.workflows.engine.artifacts import ArtifactStore
    await ArtifactStore().store(1, "big_out", "x" * 5000)

    from src.agents.artifact_summarizer import ArtifactSummarizerAgent
    agent = ArtifactSummarizerAgent()
    task = {
        "id": 700,
        "mission_id": 1,
        "agent_type": "artifact_summarizer",
        "context": json.dumps({
            "source_task_id": 100,
            "artifact_name": "big_out",
        }),
    }
    fake_summary = "short summary of content" * 5
    with patch(
        "src.workflows.engine.hooks._llm_summarize",
        AsyncMock(return_value=fake_summary),
    ):
        result = await agent.execute(task)

    assert result["status"] == "completed"
    pv = result["posthook_verdict"]
    assert pv["kind"] == "summary:big_out"
    assert pv["passed"] is True
    assert pv["raw"]["summary"] == fake_summary


@pytest.mark.asyncio
async def test_artifact_summarizer_degenerate_output_marked_failed(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db
    await init_db()
    from src.workflows.engine.artifacts import ArtifactStore
    await ArtifactStore().store(1, "big_out", "x" * 5000)

    from src.agents.artifact_summarizer import ArtifactSummarizerAgent
    agent = ArtifactSummarizerAgent()
    task = {
        "id": 701,
        "mission_id": 1,
        "agent_type": "artifact_summarizer",
        "context": json.dumps({
            "source_task_id": 100,
            "artifact_name": "big_out",
        }),
    }
    # _llm_summarize returning None indicates degenerate/empty output.
    with patch(
        "src.workflows.engine.hooks._llm_summarize",
        AsyncMock(return_value=None),
    ):
        result = await agent.execute(task)

    pv = result["posthook_verdict"]
    assert pv["passed"] is False
```

- [ ] **Step 2: Run test**

Expected: ImportError.

- [ ] **Step 3: Create `ArtifactSummarizerAgent`**

Create `src/agents/artifact_summarizer.py`:

```python
"""ArtifactSummarizerAgent — wraps _llm_summarize for the post-hook pipeline."""
from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.infra.logging_config import get_logger

logger = get_logger("agents.artifact_summarizer")


class ArtifactSummarizerAgent(BaseAgent):
    name = "artifact_summarizer"
    allowed_tools: list[str] = []

    async def execute(self, task: dict) -> dict:
        from src.workflows.engine.hooks import _llm_summarize
        from src.workflows.engine.artifacts import ArtifactStore

        ctx_raw = task.get("context") or "{}"
        try:
            ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
        except (json.JSONDecodeError, ValueError):
            ctx = {}

        source_task_id = ctx.get("source_task_id")
        artifact_name = ctx.get("artifact_name")
        if source_task_id is None or not artifact_name:
            return {
                "status": "failed",
                "error": "artifact_summarizer: source_task_id/artifact_name missing",
            }

        mission_id = task.get("mission_id")
        text = ""
        if mission_id is not None:
            val = await ArtifactStore().retrieve(mission_id, artifact_name)
            if isinstance(val, str):
                text = val
        if not text:
            return {
                "status": "failed",
                "error": f"artifact '{artifact_name}' empty or missing on blackboard",
            }

        summary = await _llm_summarize(text, artifact_name)
        passed = bool(summary) and isinstance(summary, str) and len(summary) >= 50

        return {
            "status": "completed",
            "result": summary or "",
            "model": "artifact_summarizer",
            "cost": 0.0,
            "iterations": 1,
            "posthook_verdict": {
                "kind": f"summary:{artifact_name}",
                "source_task_id": source_task_id,
                "passed": passed,
                "raw": {"summary": summary or "", "artifact_name": artifact_name},
            },
        }
```

Register in `src/agents/__init__.py`:

```python
from .artifact_summarizer import ArtifactSummarizerAgent
# ...

AGENT_REGISTRY = {
    # ...
    "artifact_summarizer": ArtifactSummarizerAgent(),
}
```

- [ ] **Step 4: Run test**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_artifact_summarizer_agent.py -v
rm -f worktree_test.db
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agents/artifact_summarizer.py src/agents/__init__.py tests/test_artifact_summarizer_agent.py
git commit -m "feat(agents): ArtifactSummarizerAgent wraps _llm_summarize"
```

---

## Task 12: Remove agent self-transition to `ungraded`

**Files:**
- Modify: `src/agents/base.py:2076-2097`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_grader_agent.py` (exercising the base path):

```python
@pytest.mark.asyncio
async def test_base_agent_does_not_self_transition_to_ungraded(tmp_path, monkeypatch):
    """BaseAgent's ReAct loop must return status='completed', never status='ungraded'."""
    import inspect
    from src.agents import base as _base_mod

    src = inspect.getsource(_base_mod)
    # No more transition_task(..., "ungraded", ...) call.
    assert 'transition_task(\n                            task_id, "ungraded"' not in src
    assert 'transition_task(task_id, "ungraded"' not in src
    # No more {"status": "ungraded"} return from the ReAct path.
    assert '"status": "ungraded"' not in src
```

(Source-level check keeps the test stable — don't need to reproduce the whole ReAct loop.)

- [ ] **Step 2: Run test**

Expected: FAIL (the strings are still present).

- [ ] **Step 3: Remove the ungraded self-transition**

Edit `src/agents/base.py` around line 2065-2098. Locate the block that looks like:

```python
                if grade_deferred:
                    # ... populates _ctx and ...
                    from src.core.state_machine import transition_task
                    await transition_task(
                        task_id, "ungraded",
                        context=_json.dumps(_ctx),
                    )
                    return {
                        "status": "ungraded",
                        "result": result,
                        "model": used_model,
                        "cost": total_cost,
                        "difficulty": reqs.difficulty,
                        "iterations": iteration + 1,
                        "tools_used_names": sorted(tools_used_names),
                    }
```

Delete the entire `if grade_deferred:` block. Leave the following `return {"status": "completed", ...}` block as the only terminal path. Remove any now-unused `grade_deferred` local / `_ctx` population that was only feeding the deleted block. Leave `generating_model` / `worker_completed_at` / `tools_used_names` in the completed-status return dict so Beckman can pick them up.

After deletion, the agent's terminal return dict must include:

```python
return {
    "status": "completed",
    "result": result,
    "model": used_model,
    "cost": total_cost,
    "difficulty": reqs.difficulty,
    "iterations": iteration + 1,
    "tools_used_names": sorted(tools_used_names),
    "generating_model": used_model,   # surfaced for post-hook context
}
```

- [ ] **Step 4: Run the focused test and broader agent suite**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 60 .venv/Scripts/python.exe -m pytest tests/test_grader_agent.py tests/test_artifact_summarizer_agent.py -v
rm -f worktree_test.db
```

Expected: new source-check test passes; previous agent tests unaffected.

- [ ] **Step 5: Commit**

```bash
git add src/agents/base.py tests/test_grader_agent.py
git commit -m "refactor(agents): drop self-transition to 'ungraded' status

Beckman now owns the ungraded transition via _apply_request_posthook
when a post-hook task is spawned. Agent simply returns status='completed'."
```

---

## Task 13: Remove dispatcher's `on_model_swap`

**Files:**
- Modify: `src/core/llm_dispatcher.py:379-395`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_llm_dispatcher.py` (or create if absent — if absent, create with standard imports):

```python
def test_dispatcher_has_no_on_model_swap():
    """Dispatcher is a pure pipe; swap-event handling lives in Beckman."""
    from src.core.llm_dispatcher import LLMDispatcher
    assert not hasattr(LLMDispatcher, "on_model_swap")
```

- [ ] **Step 2: Run test**

Expected: FAIL (method still exists).

- [ ] **Step 3: Delete the method**

Remove lines 379-395 from `src/core/llm_dispatcher.py` (the entire `async def on_model_swap(...)`). Keep `get_stats()` intact.

- [ ] **Step 4: Run test**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_llm_dispatcher.py::test_dispatcher_has_no_on_model_swap -v
rm -f worktree_test.db
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/llm_dispatcher.py tests/test_llm_dispatcher.py
git commit -m "refactor(dispatcher): remove on_model_swap (move to Beckman)"
```

---

## Task 14: Add Beckman's `on_model_swap`, redirect local_model_manager

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py`
- Modify: `src/models/local_model_manager.py:355`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_beckman_posthooks.py`:

```python
@pytest.mark.asyncio
async def test_beckman_on_model_swap_calls_accelerate_retries(monkeypatch):
    import general_beckman
    calls = {}
    async def fake_accel(tag):
        calls["tag"] = tag
        return 3
    monkeypatch.setattr("src.infra.db.accelerate_retries", fake_accel)
    await general_beckman.on_model_swap("old", "new")
    assert calls["tag"] == "model_swap"
```

- [ ] **Step 2: Run test**

Expected: FAIL (`AttributeError: general_beckman has no attribute 'on_model_swap'`).

- [ ] **Step 3: Add Beckman's `on_model_swap`**

Append to `packages/general_beckman/src/general_beckman/__init__.py`:

```python
async def on_model_swap(old_model: str | None, new_model: str | None) -> None:
    """Called when the local model manager swaps models.

    Wakes tasks whose retries were delayed waiting for *any* model to
    load. Grading is no longer triggered here — it's a regular task
    flowing through next_task().
    """
    try:
        from src.infra.db import accelerate_retries
        await accelerate_retries("model_swap")
    except Exception as e:
        from src.infra.logging_config import get_logger
        get_logger("beckman.on_model_swap").debug(
            f"accelerate_retries failed: {e}",
        )
```

Redirect the caller — `src/models/local_model_manager.py:355`:

Replace:

```python
                _swap_task = asyncio.ensure_future(
                    get_dispatcher().on_model_swap(old_litellm, new_litellm)
                )
```

with:

```python
                import general_beckman
                _swap_task = asyncio.ensure_future(
                    general_beckman.on_model_swap(old_litellm, new_litellm)
                )
```

Remove the now-unused `from src.core.llm_dispatcher import get_dispatcher` import if nothing else uses it.

- [ ] **Step 4: Run tests**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_beckman_posthooks.py tests/test_llm_dispatcher.py -v
rm -f worktree_test.db
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/__init__.py src/models/local_model_manager.py tests/test_beckman_posthooks.py
git commit -m "feat(beckman): on_model_swap owns accelerate_retries; local_model_manager redirects"
```

---

## Task 15: Remove Beckman's `ungraded` short-circuit; remove `queue_llm_summary` call site

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py`
- Modify: `src/workflows/engine/hooks.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_beckman_posthooks.py`:

```python
@pytest.mark.asyncio
async def test_on_task_finished_does_not_short_circuit_on_ungraded(tmp_path, monkeypatch):
    """Agent now returns completed; Beckman must route it fully."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, get_task
    await init_db()
    source_id = await add_task(
        title="s", description="", agent_type="writer", mission_id=1,
        context=json.dumps({"generating_model": "qwen-7b"}),
    )

    import general_beckman
    # Agent returned completed (post-refactor shape).
    result = {
        "status": "completed", "result": "out",
        "model": "qwen-7b", "cost": 0.001, "iterations": 1,
        "generating_model": "qwen-7b",
    }
    await general_beckman.on_task_finished(source_id, result)

    source = await get_task(source_id)
    assert source["status"] == "ungraded"
    ctx = json.loads(source["context"])
    assert ctx["_pending_posthooks"] == ["grade"]
```

- [ ] **Step 2: Run test**

Expected: FAIL — the existing short-circuit returns before routing.

- [ ] **Step 3: Remove the short-circuit**

In `packages/general_beckman/src/general_beckman/__init__.py` — delete the `if (result or {}).get("status") == "ungraded": return` block from `on_task_finished` (added as a stopgap earlier in this session). Leave the `post_execute_workflow_step` call and the normal route pipeline intact.

In `src/workflows/engine/hooks.py` — locate the `queue_llm_summary` call inside `post_execute_workflow_step` (around line 1125 pre-change):

```python
            # Queue LLM upgrade — orchestrator processes this outside task timeout
            await queue_llm_summary(mission_id, name, output_value)
```

Remove this line. The structural summary still gets stored above it; LLM upgrade will now be scheduled by Beckman via `RequestPostHook("summary:<name>")` when grade passes.

- [ ] **Step 4: Run the full Beckman suite**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 60 .venv/Scripts/python.exe -m pytest tests/test_beckman_posthooks.py tests/test_beckman_on_task_finished.py tests/test_beckman_rewrite.py tests/test_beckman_apply.py -v
rm -f worktree_test.db
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/__init__.py src/workflows/engine/hooks.py tests/test_beckman_posthooks.py
git commit -m "refactor(beckman/hooks): drop ungraded short-circuit and queue_llm_summary call"
```

---

## Task 16: Delete `drain_ungraded_tasks` and `drain_pending_summaries`

**Files:**
- Modify: `src/core/grading.py`
- Modify: `src/workflows/engine/hooks.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_beckman_posthooks.py`:

```python
def test_drain_functions_removed():
    from src.core import grading
    from src.workflows.engine import hooks
    assert not hasattr(grading, "drain_ungraded_tasks")
    assert not hasattr(hooks, "drain_pending_summaries")
    assert not hasattr(hooks, "queue_llm_summary")
```

- [ ] **Step 2: Run test**

Expected: FAIL (functions still exist).

- [ ] **Step 3: Delete the drain functions**

In `src/core/grading.py` — delete `async def drain_ungraded_tasks(...)` (lines 519-end-of-function). Keep `grade_task`, `apply_grade_result`, and all internal helpers.

In `src/workflows/engine/hooks.py` — delete `async def queue_llm_summary(...)` (lines 26-35) and `async def drain_pending_summaries(...)` (lines 38-122). Keep `_llm_summarize`.

Fix any imports left dangling in `src/core/llm_dispatcher.py` (the `drain_ungraded_tasks` import should already have been removed when `on_model_swap` was deleted in Task 13).

Run a grep to confirm no broken imports:

```bash
grep -rn "drain_ungraded_tasks\|drain_pending_summaries\|queue_llm_summary" src/ packages/ tests/ 2>/dev/null
```

If matches outside historical docstrings, fix or remove them.

- [ ] **Step 4: Run tests**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 60 .venv/Scripts/python.exe -m pytest tests/test_beckman_posthooks.py tests/test_beckman_on_task_finished.py tests/test_beckman_rewrite.py tests/test_beckman_apply.py tests/test_grader_agent.py tests/test_artifact_summarizer_agent.py tests/test_workflow_engine_advance.py tests/test_salako_workflow_advance.py tests/test_mechanical_context_shape.py -v
rm -f worktree_test.db
```

Expected: all pass. If `test_llm_dispatcher.py` tests reference `drain_ungraded_tasks`, delete or rewrite them here — note in commit message.

- [ ] **Step 5: Commit**

```bash
git add src/core/grading.py src/workflows/engine/hooks.py tests/test_beckman_posthooks.py tests/test_llm_dispatcher.py
git commit -m "refactor(core): delete drain_ungraded_tasks and drain_pending_summaries

Queue owns post-hook scheduling now; batch drain functions are redundant."
```

---

## Task 17: Boot migration — stale `ungraded` rows + drop `pending_llm_summaries` table

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py`
- Test: `tests/test_migration_ungraded_to_posthooks.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_migration_ungraded_to_posthooks.py`:

```python
"""Boot migration converts stale 'ungraded' rows into the post-hook shape."""
import json
import pytest


@pytest.mark.asyncio
async def test_stale_ungraded_row_gets_migrated(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, get_task, update_task, get_db
    await init_db()
    # Simulate a pre-refactor 'ungraded' row: no _pending_posthooks, has generating_model.
    source_id = await add_task(
        title="legacy", description="", agent_type="writer", mission_id=1,
        context=json.dumps({"generating_model": "qwen-7b"}),
    )
    await update_task(source_id, status="ungraded", result="legacy output")

    # Reset migration sentinel so the function actually runs.
    from general_beckman import posthook_migration
    posthook_migration._migrated = False

    await posthook_migration.run()

    refreshed = await get_task(source_id)
    ctx = json.loads(refreshed["context"])
    assert ctx["_pending_posthooks"] == ["grade"]

    db = await get_db()
    cursor = await db.execute(
        "SELECT id, context FROM tasks WHERE agent_type='grader'"
    )
    rows = list(await cursor.fetchall())
    assert len(rows) == 1
    grader_ctx = json.loads(rows[0]["context"])
    assert grader_ctx["source_task_id"] == source_id


@pytest.mark.asyncio
async def test_already_migrated_row_unchanged(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, get_task, update_task, get_db
    await init_db()
    source_id = await add_task(
        title="already", description="", agent_type="writer", mission_id=1,
        context=json.dumps({
            "generating_model": "qwen-7b",
            "_pending_posthooks": ["grade"],
        }),
    )
    await update_task(source_id, status="ungraded")

    from general_beckman import posthook_migration
    posthook_migration._migrated = False
    await posthook_migration.run()

    # No new grader row should spawn for an already-migrated source.
    db = await get_db()
    cursor = await db.execute(
        "SELECT id FROM tasks WHERE agent_type='grader'"
    )
    rows = list(await cursor.fetchall())
    assert len(rows) == 0
```

- [ ] **Step 2: Run test**

Expected: ImportError.

- [ ] **Step 3: Create migration module**

Create `packages/general_beckman/src/general_beckman/posthook_migration.py`:

```python
"""One-shot migration from legacy 'ungraded' rows to the post-hook shape.

Runs once per process via the `_migrated` sentinel. Safe to call
repeatedly; no-op after first successful run.
"""
from __future__ import annotations

import asyncio
import json

from src.infra.logging_config import get_logger

logger = get_logger("beckman.posthook_migration")

_migrated: bool = False
_lock: asyncio.Lock = asyncio.Lock()


async def run() -> None:
    """Migrate stale ungraded rows; drop the defunct pending_llm_summaries table."""
    global _migrated
    if _migrated:
        return
    async with _lock:
        if _migrated:
            return

        from src.infra.db import get_db, add_task

        db = await get_db()

        # Step 1: migrate ungraded rows without _pending_posthooks.
        cursor = await db.execute(
            "SELECT id, mission_id, context FROM tasks WHERE status='ungraded'"
        )
        rows = list(await cursor.fetchall())
        migrated_count = 0
        for row in rows:
            try:
                ctx = json.loads(row["context"] or "{}")
            except (json.JSONDecodeError, TypeError):
                ctx = {}
            if ctx.get("_pending_posthooks"):
                continue
            ctx["_pending_posthooks"] = ["grade"]
            await db.execute(
                "UPDATE tasks SET context = ? WHERE id = ?",
                (json.dumps(ctx), row["id"]),
            )
            # Spawn the grader post-hook.
            await add_task(
                title=f"Grade task #{row['id']}",
                description="",
                agent_type="grader",
                mission_id=row["mission_id"],
                depends_on=[],
                context=json.dumps({
                    "source_task_id": row["id"],
                    "generating_model": ctx.get("generating_model", ""),
                }),
            )
            migrated_count += 1

        # Step 2: drop orphaned llm_summary table.
        await db.execute("DROP TABLE IF EXISTS pending_llm_summaries")
        await db.commit()

        _migrated = True
        logger.info(
            "posthook_migration complete",
            ungraded_migrated=migrated_count,
        )
```

Wire it into `next_task()` in `packages/general_beckman/src/general_beckman/__init__.py`:

```python
async def next_task():
    from general_beckman.cron import fire_due
    from general_beckman.queue import pick_ready_task
    from general_beckman import posthook_migration

    await posthook_migration.run()  # one-shot; no-op after first success
    await fire_due()

    snap = _capacity_snapshot()
    return await pick_ready_task(_system_busy(snap))
```

- [ ] **Step 4: Run test**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_migration_ungraded_to_posthooks.py -v
rm -f worktree_test.db
```

Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/posthook_migration.py packages/general_beckman/src/general_beckman/__init__.py tests/test_migration_ungraded_to_posthooks.py
git commit -m "feat(beckman): one-shot migration of stale ungraded rows + drop pending_llm_summaries"
```

---

## Task 18: Progress-ping skip-list extension for post-hook tasks

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_beckman_posthooks.py`:

```python
@pytest.mark.asyncio
async def test_grader_task_does_not_emit_progress_ping(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra import db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db, add_task, update_task
    await init_db()
    grader_id = await add_task(
        title="Grade task #1", description="",
        agent_type="grader", mission_id=1,
        context=json.dumps({"source_task_id": 1}),
    )

    calls = []
    class FakeTG:
        async def send_notification(self, text):
            calls.append(text)

    monkeypatch.setattr(
        "src.app.telegram_bot.get_telegram",
        lambda: FakeTG(),
    )

    from general_beckman import _send_step_progress
    task = {
        "id": grader_id, "title": "Grade task #1",
        "mission_id": 1, "agent_type": "grader",
    }
    await _send_step_progress(task, "completed", {"status": "completed"})
    assert calls == []  # post-hook tasks are infrastructure, not progress.


@pytest.mark.asyncio
async def test_artifact_summarizer_task_does_not_emit_progress_ping(tmp_path, monkeypatch):
    class FakeTG:
        async def send_notification(self, text):
            raise AssertionError("ping should not fire")

    monkeypatch.setattr(
        "src.app.telegram_bot.get_telegram",
        lambda: FakeTG(),
    )
    from general_beckman import _send_step_progress
    task = {
        "id": 123, "title": "Summarize 'x' for #1",
        "mission_id": 1, "agent_type": "artifact_summarizer",
    }
    await _send_step_progress(task, "completed", {"status": "completed"})
```

- [ ] **Step 2: Run test**

Expected: first test fails (ping fires), second raises AssertionError.

- [ ] **Step 3: Extend the skip list**

In `packages/general_beckman/src/general_beckman/__init__.py`, update the `on_task_finished` progress-ping block AND `_send_step_progress`:

Change the gate in `on_task_finished` from:

```python
        if task.get("mission_id") and task.get("agent_type") != "mechanical":
```

to:

```python
        _bookkeeping = task.get("agent_type") in (
            "mechanical", "grader", "artifact_summarizer",
        )
        if task.get("mission_id") and not _bookkeeping:
```

And at the top of `_send_step_progress`, add an early return as a belt-and-suspenders guard:

```python
async def _send_step_progress(task: dict, status: str, result: dict) -> None:
    if task.get("agent_type") in ("mechanical", "grader", "artifact_summarizer"):
        return
    from src.app.telegram_bot import get_telegram
    # ... existing body ...
```

- [ ] **Step 4: Run tests**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 30 .venv/Scripts/python.exe -m pytest tests/test_beckman_posthooks.py -v -k progress_ping
rm -f worktree_test.db
```

Expected: both new tests pass.

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/__init__.py tests/test_beckman_posthooks.py
git commit -m "refactor(beckman): skip progress ping for grader/artifact_summarizer tasks"
```

---

## Task 19: Full targeted test suite green

**Files:**
- Test only.

- [ ] **Step 1: Run the full targeted suite**

```bash
DB_PATH="$PWD/worktree_test.db" timeout 120 .venv/Scripts/python.exe -m pytest \
    tests/test_beckman_posthooks.py \
    tests/test_beckman_cron_seed.py \
    tests/test_beckman_rewrite.py \
    tests/test_beckman_retry.py \
    tests/test_beckman_apply.py \
    tests/test_beckman_next_task.py \
    tests/test_beckman_on_task_finished.py \
    tests/test_salako_workflow_advance.py \
    tests/test_workflow_engine_advance.py \
    tests/test_mechanical_context_shape.py \
    tests/test_grader_agent.py \
    tests/test_artifact_summarizer_agent.py \
    tests/test_migration_ungraded_to_posthooks.py \
    tests/test_llm_dispatcher.py \
    -v
rm -f worktree_test.db
```

Expected: all pass. Count should be ≥ the original 33 plus the ~18 new tests.

- [ ] **Step 2: If any failures, debug and fix in place**

If a previously-green test now fails, read the diff to understand the interaction; fix code or test; re-run. Do NOT add green bypass commits.

- [ ] **Step 3: Write a passing-summary commit note**

Once fully green:

```bash
git commit --allow-empty -m "test: full targeted suite green after post-hook extraction

~51 tests covering beckman pipeline, post-hook actions, grader/summarizer
agents, migration, and rewrite layer."
```

- [ ] **Step 4: Smoke test plan (manual, user drives)**

Document in commit body (no automation yet — manual validation by user):

1. Restart KutAI.
2. Trigger an i2p mission. Observe:
   - First LLM task goes to `ungraded` (check via `/queue` — should see both the source row in ungraded AND a `grader` row pending).
   - On model swap or next cycle, grader task runs.
   - Grade pass → source → completed (or spawns summary task for large output).
   - Grade fail → source → pending with error, retries with different model.
3. Trigger `/shop` (mechanical pipeline). Observe: no grader or summarizer tasks spawned.
4. Check `/dlq` for any grade-DLQ entries.

- [ ] **Step 5: Nothing to commit this step**

Skip — step 3 already committed.

---

## Self-review notes

The following spec requirements map to tasks:

| Spec section | Tasks |
|---|---|
| §3.1 Agent cleanup | Task 12 |
| §3.1 Beckman rewrite | Task 3, Task 9 |
| §3.1 Beckman apply | Tasks 4-8 |
| §3.1 on_task_finished post-hook sync (pre-existing) | No change (kept from earlier commit) |
| §3.1 Dispatcher pure pipe | Task 13 |
| §3.1 local_model_manager redirect | Task 14 |
| §3.1 New agents | Tasks 10, 11 |
| §3.2 Post-hook kinds | Tasks 4, 5 |
| §3.3 Happy path | Tasks 3-8 |
| §3.4-3.6 Sequence diagrams | Covered by Tasks 5, 6, 7, 8 integration |
| §3.7 Action types | Task 1 |
| §3.8 determine_posthooks | Task 2 |
| §3.9 Skip-list | Task 3 (bookkeeping guard), Task 18 (progress ping) |
| §3.10 Dispatcher cleanup | Tasks 13, 14 |
| §3.11 Agents | Tasks 10, 11 |
| §3.12 Route-result detects verdict | Task 9 |
| §4 Failure cases | Tasks 5, 7, 8 |
| §5 Migration | Task 17 |
| §6 Deletions | Tasks 12, 13, 15, 16 |
| §7 Testing | Every task |

No gaps.
