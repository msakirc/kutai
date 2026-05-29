# CPS SP3 — In-Task Deadlock Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the three in-task `await_inline=True` post-hook call paths (grading, code_review, summarize) onto the durable continuation substrate so a cap-counted task no longer holds a lane slot while blocking on a child — closing the DLQ deadlock.

**Architecture:** Shape B (collapse the agent layer). The post-hook enqueues the raw_dispatch reviewer/summarizer child **directly** with `on_complete`/`on_error`/`cont_state` (via `enqueue`, never a direct `continuations` write); a named resume handler parses the child output, builds a `PostHookVerdict`, and calls the **existing, unchanged** `_apply_posthook_verdict`. The `grader`/`code_reviewer`/`artifact_summarizer` agent classes are deleted. Grading's 2-attempt retry becomes continuation-chaining in the resume handler.

**Tech Stack:** Python 3.10 async, aiosqlite, pytest-asyncio. Packages: `general_beckman` (substrate + apply + new handler module), `src.core` (grading, code_review), `src.workflows.engine` (hooks).

**Spec:** `docs/superpowers/specs/2026-05-29-cps-sp3-design.md` (rev2).

**Substrate contract (frozen — do NOT edit):** `docs/handoff/2026-05-28-sp3-kickoff.md` §"Substrate invariants SP3 MUST honor". The substrate (`continuations.py` fire logic, `db.py`, `enqueue`) is unchanged by SP3 except appending to `_HANDLER_MODULES`.

**Conventions (project rules — non-negotiable):**
- Run tests with a timeout prefix: `timeout 60 .venv/Scripts/python -m pytest tests/... -v`. Never run pytest without a timeout (zombie pytest holds SQLite write locks and crash-loops live KutAI).
- Prefix git with `rtk`.
- SQLite datetime is `strftime('%Y-%m-%d %H:%M:%S')` — never `isoformat()`.
- Lazy cross-module imports (inside functions) to avoid circular imports.
- `cont_state` must be JSON-serializable. Handler signature is `async (child_task_id: int, result: dict, state: dict) -> None`.
- **Parallel-safety:** SP3b runs concurrently and owns `src/core/llm_dispatcher.py` + the `request` callers. SP3 must NOT change `_task_result_to_request_response`'s signature/behavior. If a merge conflict appears in `_HANDLER_MODULES`, resolve take-both.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/core/grading.py` | Add pure `build_grading_spec(source, exclusions) -> dict`; keep `parse_grade_response`/`GradeResult`; remove `grade_task` (its enqueue+loop move to the resume handler). `apply_grade_result` stays (dead-but-tested). | Modify |
| `src/core/code_review.py` | Add pure `build_code_review_spec(source, exclusions) -> dict`; keep `parse_code_review_response`/`CodeReviewResult`; remove `code_review_task`. | Modify |
| `src/workflows/engine/hooks.py` | Add pure `build_summary_spec(text, artifact_name) -> dict`; remove `_llm_summarize`; remove the schema-validation producer-skip strings for deleted agents (`:1465`). | Modify |
| `packages/general_beckman/src/general_beckman/posthook_continuations.py` | **NEW.** Six handlers (`posthook.grade.resume`/`_err`, `posthook.code_review.resume`/`_err`, `posthook.summary.resume`/`_err`) + content extraction + grading chaining + `register_continuations()` (registers at import). | Create |
| `packages/general_beckman/src/general_beckman/continuations.py` | Append `"general_beckman.posthook_continuations"` to `_HANDLER_MODULES`. | Modify |
| `packages/general_beckman/src/general_beckman/apply.py` | New `_enqueue_posthook_llm_child(kind, source, source_ctx)` (enqueues child + continuation, mission_id in cont_state only). Wire `_apply_request_posthook` (grade/code_review/summary kinds) + the grade-pass summary loop (`:4217`) to it. Drop `_OVERHEAD_POSTHOOK_AGENTS`/`_posthook_kind` dead strings. Relocate grader-DLQ-cascade semantics into `posthook.grade.resume_err`. | Modify |
| `src/agents/grader.py`, `src/agents/code_reviewer.py`, `src/agents/artifact_summarizer.py` | DELETE. | Delete |
| `src/agents/__init__.py` | Remove `AGENT_REGISTRY` entries + imports for the three agents (`:21-23`, `:50-52`). | Modify |
| `src/core/task_classifier.py` | Remove router-prompt + keyword-table entries for the three agents (`:64-67`, `:81-82`, `:407-410`). | Modify |
| `packages/general_beckman/src/general_beckman/posthooks.py` | Drop dead agent strings from `_NO_POSTHOOKS_AGENT_TYPES` (`:74`). | Modify |
| `packages/general_beckman/src/general_beckman/rewrite.py` | Drop dead agent strings from Rule 0 (`:103-105`) + Rule 1 `is_bookkeeping` (`:255-259`). | Modify |
| `packages/general_beckman/src/general_beckman/__init__.py` | Drop dead agent strings from the three progress-ping sets (`:941-944`, `:983`, `:1026-1029`). | Modify |
| `src/workflows/engine/advance.py` (or `workflow_engine/advance.py`) | Drop dead agent strings from `_bookkeeping` (`:195`). | Modify |
| `tests/...` | Delete/update ~12 affected test files (see Task 13). Add new SP3 suites. | Create/Modify/Delete |

---

## Task 1: Extract `build_grading_spec` (pure spec-builder)

**Files:**
- Modify: `src/core/grading.py`
- Test: `tests/core/test_build_grading_spec.py`

`build_grading_spec` lifts the message + spec construction from today's `grade_task` (`grading.py:321-369`) into a pure function, parameterized by `exclusions` (the chaining handler passes the growing exclusion list). It performs the early auto-fail checks (`:307-314`: trivial/empty + degenerate) and returns **either** a ready `spec` dict **or** an immediate `GradeResult` (auto-fail) so the spawn helper can short-circuit without a child.

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_build_grading_spec.py`:

```python
"""SP3 Task 1 — build_grading_spec pure spec-builder."""
import json
import pytest


def test_build_grading_spec_returns_overhead_raw_dispatch_spec():
    from src.core.grading import build_grading_spec
    source = {"id": 7, "title": "T", "description": "D",
              "result": "x" * 200, "context": "{}"}
    out = build_grading_spec(source, exclusions=["bad-model"])
    assert isinstance(out, dict)
    spec = out  # ready spec when source is gradeable
    assert spec["agent_type"] == "reviewer"
    assert spec["kind"] == "overhead"
    llm = spec["context"]["llm_call"]
    assert llm["raw_dispatch"] is True
    assert llm["call_category"] == "overhead"
    assert "bad-model" in llm["exclude_models"]
    # response embedded, capped at 30000
    assert "x" * 100 in llm["messages"][1]["content"]


def test_build_grading_spec_auto_fails_trivial_output():
    from src.core.grading import build_grading_spec, GradeResult
    source = {"id": 7, "title": "T", "description": "D", "result": "  ", "context": "{}"}
    out = build_grading_spec(source, exclusions=[])
    assert isinstance(out, GradeResult)
    assert out.passed is False
    assert "auto-fail" in out.raw
```

- [ ] **Step 2: Run to verify it fails**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/core/test_build_grading_spec.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_grading_spec'`.

- [ ] **Step 3: Implement `build_grading_spec`**

In `src/core/grading.py`, add (lift the messages from `:321-336` and the spec from `:348-369` of the current `grade_task`; the trivial/degenerate checks from `:307-314`). The function takes the already-parsed `source` row and the current `exclusions` list:

```python
def build_grading_spec(source: dict, exclusions: list[str]):
    """Pure builder for the grading reviewer child.

    Returns a ready Beckman spec dict when the source is gradeable, OR a
    GradeResult (auto-fail) when the source is trivial/empty/degenerate
    (caller short-circuits to apply that verdict without enqueueing a child).
    """
    import time as _time
    import uuid as _uuid

    result_text = source.get("result", "")
    if not result_text or len(str(result_text).strip()) < 10:
        return GradeResult(passed=False, raw="auto-fail: trivial/empty output")

    from dogru_mu_samet import assess as cq_assess
    _cq = cq_assess(str(result_text))
    if _cq.is_degenerate:
        return GradeResult(passed=False, raw=f"auto-fail: {_cq.summary}")

    messages = [
        {"role": "system", "content": GRADING_SYSTEM},
        {"role": "user", "content": GRADING_PROMPT.format(
            title=str(source.get("title", ""))[:100],
            description=str(source.get("description", ""))[:500],
            response=str(result_text)[:30000],
        )},
    ]
    _suffix = f"{_time.monotonic_ns() % 1_000_000:06d}-{_uuid.uuid4().hex[:6]}"
    return {
        "title": f"grader:task#{source.get('id')}:{_suffix}",
        "description": "Grading review of task output",
        "agent_type": "reviewer",
        "kind": "overhead",
        "priority": 1,
        "context": {"llm_call": {
            "raw_dispatch": True,
            "call_category": "overhead",
            "task": "reviewer",
            "agent_type": "reviewer",
            "difficulty": 3,
            "messages": messages,
            "failures": [],
            "estimated_input_tokens": 800,
            "estimated_output_tokens": 600,
            "prefer_speed": True,
            "exclude_models": list(exclusions),
        }},
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/core/test_build_grading_spec.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
rtk git add src/core/grading.py tests/core/test_build_grading_spec.py
rtk git commit -m "feat(cps-sp3): extract build_grading_spec pure spec-builder"
```

---

## Task 2: Extract `build_code_review_spec`

**Files:**
- Modify: `src/core/code_review.py`
- Test: `tests/core/test_build_code_review_spec.py`

Lift from `code_review_task` (`code_review.py:140-175`, early checks `:120-131`). Same return contract: spec dict or a `CodeReviewResult` auto-fail.

- [ ] **Step 1: Write the failing test**

```python
"""SP3 Task 2 — build_code_review_spec."""
def test_build_code_review_spec_shape():
    from src.core.code_review import build_code_review_spec
    source = {"id": 9, "title": "T", "description": "D",
              "result": "def f(): pass\n" * 20,
              "context": '{"produces": ["a.py"]}'}
    spec = build_code_review_spec(source, exclusions=[])
    assert spec["agent_type"] == "reviewer"
    assert spec["kind"] == "overhead"
    assert spec["context"]["llm_call"]["raw_dispatch"] is True

def test_build_code_review_spec_auto_fails_trivial():
    from src.core.code_review import build_code_review_spec, CodeReviewResult
    out = build_code_review_spec({"id": 9, "result": "x", "context": "{}"}, exclusions=[])
    assert isinstance(out, CodeReviewResult)
    assert out.passed is False
```

- [ ] **Step 2: Run to verify fail**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/core/test_build_code_review_spec.py -v`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement `build_code_review_spec`**

In `src/core/code_review.py` add (lift verbatim from `code_review_task`'s body, parameterizing `exclusions`; parse the `produces` from `source.context`):

```python
def build_code_review_spec(source: dict, exclusions: list[str]):
    """Pure builder for the code-review reviewer child. Returns a spec dict,
    or a CodeReviewResult auto-fail for trivial/degenerate source."""
    import json as _json
    import time as _time
    import uuid as _uuid

    result_text = source.get("result", "")
    if not result_text or len(str(result_text).strip()) < 10:
        return CodeReviewResult(passed=False, raw="auto-fail: trivial/empty output")
    try:
        from dogru_mu_samet import assess as cq_assess
        _cq = cq_assess(str(result_text))
        if _cq.is_degenerate:
            return CodeReviewResult(passed=False, raw=f"auto-fail: {_cq.summary}")
    except Exception:
        pass

    ctx = source.get("context", "{}")
    if isinstance(ctx, str):
        try:
            ctx = _json.loads(ctx)
        except (ValueError, TypeError):
            ctx = {}
    produces = ctx.get("produces") or []

    messages = [
        {"role": "system", "content": CODE_REVIEW_SYSTEM},
        {"role": "user", "content": CODE_REVIEW_PROMPT.format(
            title=str(source.get("title", ""))[:100],
            description=str(source.get("description", ""))[:500],
            produces=_json.dumps(produces),
            response=str(result_text)[:30000],
        )},
    ]
    _suffix = f"{_time.monotonic_ns() % 1_000_000:06d}-{_uuid.uuid4().hex[:6]}"
    return {
        "title": f"code_reviewer:task#{source.get('id')}:{_suffix}",
        "description": "Code review of build-step output",
        "agent_type": "reviewer",
        "kind": "overhead",
        "priority": 1,
        "context": {"llm_call": {
            "raw_dispatch": True,
            "call_category": "overhead",
            "task": "reviewer",
            "agent_type": "reviewer",
            "difficulty": 4,
            "messages": messages,
            "failures": [],
            "estimated_input_tokens": 1500,
            "estimated_output_tokens": 800,
            "exclude_models": list(exclusions),
        }},
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/core/test_build_code_review_spec.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/core/code_review.py tests/core/test_build_code_review_spec.py
rtk git commit -m "feat(cps-sp3): extract build_code_review_spec pure spec-builder"
```

---

## Task 3: Extract `build_summary_spec`

**Files:**
- Modify: `src/workflows/engine/hooks.py`
- Test: `tests/workflows/engine/test_build_summary_spec.py`

Lift from `_llm_summarize` (`hooks.py:27-80`). **Critically: do NOT resolve or set `parent_id`/`mission_id`** — drop the `current_task_id` ContextVar block (`:49-57`). mission_id travels in `cont_state` (Task 8/9). Output cap 16000, `prefer_speed`/`prefer_local`.

- [ ] **Step 1: Write the failing test**

```python
"""SP3 Task 3 — build_summary_spec."""
def test_build_summary_spec_shape():
    from src.workflows.engine.hooks import build_summary_spec
    spec = build_summary_spec("long text " * 500, "user_stories")
    assert spec["agent_type"] == "summarizer"
    assert spec["kind"] == "overhead"
    llm = spec["context"]["llm_call"]
    assert llm["raw_dispatch"] is True
    assert llm["prefer_local"] is True
    assert "user_stories" in llm["messages"][1]["content"]
    # no mission_id / parent leakage in the spec
    assert "mission_id" not in spec
```

- [ ] **Step 2: Run to verify fail**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/workflows/engine/test_build_summary_spec.py -v`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement `build_summary_spec`**

```python
def build_summary_spec(text: str, artifact_name: str) -> dict:
    """Pure builder for the summarizer child. No mission_id/parent — those
    travel in cont_state (SP3 child-spec hygiene)."""
    import time as _time
    import uuid as _uuid

    truncated = text[:16000]
    messages = [
        {"role": "system", "content": (
            "You are a concise summarizer. Produce a summary that preserves "
            "ALL key facts, decisions, and data points. Target: under 400 "
            "words. No filler."
        )},
        {"role": "user", "content": (
            f"Summarize this '{artifact_name}' artifact. Keep every important "
            f"fact, number, name, and decision:\n\n{truncated}"
        )},
    ]
    _suffix = f"{_time.monotonic_ns() % 1_000_000:06d}-{_uuid.uuid4().hex[:6]}"
    return {
        "title": f"summarizer:{artifact_name}:{_suffix}",
        "description": f"LLM summarization of artifact '{artifact_name}'",
        "agent_type": "summarizer",
        "kind": "overhead",
        "context": {"llm_call": {
            "raw_dispatch": True,
            "call_category": "overhead",
            "task": "summarizer",
            "agent_type": "summarizer",
            "difficulty": 2,
            "messages": messages,
            "failures": [],
            "prefer_speed": True,
            "prefer_local": True,
            "estimated_input_tokens": min(len(text) // 4, 4000),
            "estimated_output_tokens": 500,
        }},
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/workflows/engine/test_build_summary_spec.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/workflows/engine/hooks.py tests/workflows/engine/test_build_summary_spec.py
rtk git commit -m "feat(cps-sp3): extract build_summary_spec pure spec-builder"
```

---

## Task 4: New `posthook_continuations.py` module + registration

**Files:**
- Create: `packages/general_beckman/src/general_beckman/posthook_continuations.py`
- Modify: `packages/general_beckman/src/general_beckman/continuations.py` (`_HANDLER_MODULES`)
- Test: `tests/beckman/test_posthook_continuations_register.py`

Skeleton + the shared content-extraction helper + registration. Handlers are filled in Tasks 5–7.

- [ ] **Step 1: Write the failing test**

```python
"""SP3 Task 4 — posthook continuation handlers register."""
def test_posthook_handlers_registered():
    from general_beckman import posthook_continuations as pc
    from general_beckman.continuations import _HANDLERS, register_startup_handlers
    pc.register_continuations()
    for name in (
        "posthook.grade.resume", "posthook.grade.resume_err",
        "posthook.code_review.resume", "posthook.code_review.resume_err",
        "posthook.summary.resume", "posthook.summary.resume_err",
    ):
        assert name in _HANDLERS, f"{name} not registered"

def test_module_in_handler_modules_static_list():
    from general_beckman.continuations import _HANDLER_MODULES
    assert "general_beckman.posthook_continuations" in _HANDLER_MODULES
```

- [ ] **Step 2: Run to verify fail**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/beckman/test_posthook_continuations_register.py -v`
Expected: FAIL — ModuleNotFoundError / missing list entry.

- [ ] **Step 3: Create the module skeleton + extraction helper + registration**

Create `packages/general_beckman/src/general_beckman/posthook_continuations.py`:

```python
"""CPS SP3 — post-hook continuation handlers (grading / code_review / summarize).

Shape B: the post-hook enqueues the raw_dispatch reviewer/summarizer child
directly with on_complete/on_error; these handlers parse the child output,
build a PostHookVerdict, and re-enter the EXISTING _apply_posthook_verdict.
The grader/code_reviewer/artifact_summarizer agent classes are deleted.
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("beckman.posthook_continuations")


def _extract_content(result: dict) -> str:
    """Dual-shape decode (matches src/app/interview.py:297-310).

    Normal terminal: result['result']['content']. Restart-reconcile:
    top-level result['content']. List blocks are joined.
    """
    result = result or {}
    inner = result.get("result")
    if isinstance(inner, dict):
        content = inner.get("content", "")
    elif inner is not None:
        content = inner
    else:
        content = result.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content or "")


# Handlers (filled in Tasks 5-7)
async def _grade_resume(child_task_id: int, result: dict, state: dict) -> None:
    raise NotImplementedError  # Task 5


async def _grade_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    raise NotImplementedError  # Task 5


async def _code_review_resume(child_task_id: int, result: dict, state: dict) -> None:
    raise NotImplementedError  # Task 6


async def _code_review_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    raise NotImplementedError  # Task 6


async def _summary_resume(child_task_id: int, result: dict, state: dict) -> None:
    raise NotImplementedError  # Task 7


async def _summary_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    raise NotImplementedError  # Task 7


def register_continuations() -> None:
    """Register SP3 post-hook CPS handlers. Idempotent."""
    try:
        from general_beckman.continuations import register_resume
        register_resume("posthook.grade.resume", _grade_resume)
        register_resume("posthook.grade.resume_err", _grade_resume_err)
        register_resume("posthook.code_review.resume", _code_review_resume)
        register_resume("posthook.code_review.resume_err", _code_review_resume_err)
        register_resume("posthook.summary.resume", _summary_resume)
        register_resume("posthook.summary.resume_err", _summary_resume_err)
    except Exception as exc:  # noqa: BLE001
        logger.debug("posthook continuation registration deferred", error=str(exc))


# Register at import so handlers are present for restart reconcile.
register_continuations()
```

- [ ] **Step 4: Append to `_HANDLER_MODULES`**

In `continuations.py:175-184`, add the new module to the list (take-both if SP3b also touched it):

```python
_HANDLER_MODULES: list[str] = [
    "mr_roboto.executors.analytics_digest",
    "mr_roboto.executors.classify_signals",
    # CPS SP2 — edge-group migrations:
    "src.app.telegram_bot",
    "src.app.interview",
    "src.app.meetings",
    "src.app.jobs.faq_regen",
    # CPS SP3 — in-task deadlock set:
    "general_beckman.posthook_continuations",
    # site #6 (investor_bullets) deferred to SP5+ — see SP2 spec §Site 6
]
```

- [ ] **Step 5: Run to verify pass**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/beckman/test_posthook_continuations_register.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add packages/general_beckman/src/general_beckman/posthook_continuations.py packages/general_beckman/src/general_beckman/continuations.py tests/beckman/test_posthook_continuations_register.py
rtk git commit -m "feat(cps-sp3): posthook_continuations module skeleton + _HANDLER_MODULES entry"
```

---

## Task 5: `posthook.grade.resume` + `.resume_err` (with chaining)

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/posthook_continuations.py`
- Test: `tests/beckman/test_posthook_grade_resume.py`

The resume parses the reviewer output. On parse-OK → build `PostHookVerdict(kind="grade", passed, raw)` and call `_apply_posthook_verdict`. On parse-fail when `state["attempt"]==0` → re-enqueue a 2nd reviewer child (failed grader added to exclusions, `attempt=1`). On parse-fail when `attempt==1` → auto-fail verdict. `resume_err` (child terminally failed) → auto-fail verdict (carries the relocated grader-DLQ-cascade semantics — `_apply_posthook_verdict`'s grade-fail branch already drives source retry/DLQ at cap).

> **`PostHookVerdict` import:** the dataclass lives in `general_beckman` (used by `apply.py`). Confirm the exact import path while implementing — grep `class PostHookVerdict` (likely `general_beckman.actions` or `rewrite.py`). Build it with `kind`, `source_task_id`, `passed`, `raw` to match `_apply_posthook_verdict`'s reads (`a.kind`, `a.source_task_id`, `a.passed`, `a.raw`).

- [ ] **Step 1: Write the failing tests**

```python
"""SP3 Task 5 — grade resume + chaining."""
import json
import pytest
from unittest.mock import AsyncMock, patch


def _grade_result(text):
    return {"result": {"content": text}, "status": "completed"}


@pytest.mark.asyncio
async def test_grade_resume_pass_applies_verdict():
    from general_beckman import posthook_continuations as pc
    raw = "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: PASS\nWELL_FORMED: PASS\nCOHERENT: PASS\n"
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._grade_resume(101, _grade_result(raw),
                               {"source_task_id": 7, "attempt": 0, "exclusions": []})
    assert ap.await_count == 1
    verdict = ap.call_args.args[1]
    assert verdict.kind == "grade" and verdict.source_task_id == 7
    assert verdict.passed is True


@pytest.mark.asyncio
async def test_grade_resume_parsefail_attempt0_chains_second_child():
    from general_beckman import posthook_continuations as pc
    with patch.object(pc, "_enqueue_grade_child", AsyncMock()) as enq, \
         patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._grade_resume(101, _grade_result("garbage no verdict"),
                               {"source_task_id": 7, "attempt": 0,
                                "exclusions": [], "grader_model": "qwen-thinking"})
    enq.assert_awaited_once()  # 2nd child enqueued
    # exclusions now include the failed grader, attempt bumped
    _, kwargs = enq.call_args
    assert kwargs["attempt"] == 1
    assert "qwen-thinking" in kwargs["exclusions"]
    ap.assert_not_awaited()  # no verdict applied yet


@pytest.mark.asyncio
async def test_grade_resume_parsefail_attempt1_autofails():
    from general_beckman import posthook_continuations as pc
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._grade_resume(102, _grade_result("still garbage"),
                               {"source_task_id": 7, "attempt": 1, "exclusions": ["m"]})
    verdict = ap.call_args.args[1]
    assert verdict.passed is False
    assert "grader_incapable" in str(verdict.raw)


@pytest.mark.asyncio
async def test_grade_resume_err_autofails():
    from general_beckman import posthook_continuations as pc
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._grade_resume_err(103, {"status": "failed", "error": "no candidates"},
                                   {"source_task_id": 7, "attempt": 0, "exclusions": []})
    verdict = ap.call_args.args[1]
    assert verdict.passed is False
    assert "grader call failed" in str(verdict.raw)
```

- [ ] **Step 2: Run to verify fail**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/beckman/test_posthook_grade_resume.py -v`
Expected: FAIL — `NotImplementedError` / missing `_enqueue_grade_child`.

- [ ] **Step 3: Implement the grade handlers + chaining helper**

In `posthook_continuations.py`, add module-level imports and replace the two `_grade_*` stubs. Note `_apply_posthook_verdict`, `_enqueue_grade_child`, and `PostHookVerdict` are referenced as module attributes so tests can patch them:

```python
async def _apply_posthook_verdict(child_task: dict, verdict) -> None:
    from general_beckman.apply import _apply_posthook_verdict as _impl
    await _impl(child_task, verdict)


def _make_grade_verdict(source_task_id: int, passed: bool, raw):
    from general_beckman.rewrite import PostHookVerdict  # confirm path on impl
    return PostHookVerdict(kind="grade", source_task_id=source_task_id,
                           passed=passed, raw=raw)


async def _enqueue_grade_child(source_task_id: int, exclusions: list, attempt: int,
                               mission_id=None) -> None:
    """Chain a 2nd grade child (attempt 1) via the apply-layer spawn helper."""
    from general_beckman.apply import _enqueue_posthook_llm_child
    from src.infra.db import get_task
    source = await get_task(source_task_id)
    if source is None:
        return
    await _enqueue_posthook_llm_child(
        "grade", source, _parse_ctx(source),
        exclusions=exclusions, attempt=attempt,
    )


def _parse_ctx(source: dict) -> dict:
    import json as _json
    ctx = source.get("context") or "{}"
    if isinstance(ctx, str):
        try:
            return _json.loads(ctx)
        except (ValueError, TypeError):
            return {}
    return ctx if isinstance(ctx, dict) else {}


async def _grade_resume(child_task_id: int, result: dict, state: dict) -> None:
    from src.core.grading import parse_grade_response
    source_task_id = state.get("source_task_id")
    attempt = int(state.get("attempt", 0))
    exclusions = list(state.get("exclusions") or [])
    raw = _extract_content(result)
    grader_model = (result.get("result") or {}).get("model", "") if isinstance(result.get("result"), dict) else result.get("model", "")

    try:
        verdict = parse_grade_response(raw)
        verdict.raw = raw
        await _apply_posthook_verdict(
            {"id": child_task_id},
            _make_grade_verdict(source_task_id, verdict.passed, _grade_raw_dict(verdict)),
        )
        return
    except ValueError:
        if attempt == 0:
            if grader_model and grader_model not in exclusions:
                exclusions.append(grader_model)
            await _enqueue_grade_child(
                source_task_id, exclusions=exclusions, attempt=1,
                mission_id=state.get("mission_id"),
            )
            return
        await _apply_posthook_verdict(
            {"id": child_task_id},
            _make_grade_verdict(
                source_task_id, False,
                f"auto-fail: grader_incapable after 2 attempts: {raw[:300]}",
            ),
        )


async def _grade_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    source_task_id = state.get("source_task_id")
    err = (result or {}).get("error", "unknown")
    await _apply_posthook_verdict(
        {"id": child_task_id},
        _make_grade_verdict(source_task_id, False,
                            f"auto-fail: grader call failed ({err})"),
    )
```

Add `_grade_raw_dict(verdict)` mirroring `grader.py:114-126` (the dataclass→dict normalization). `parse_grade_response` raising `ValueError` on unparseable output is the existing contract (`grading.py:404-409`).

- [ ] **Step 4: Run to verify pass**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/beckman/test_posthook_grade_resume.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
rtk git add packages/general_beckman/src/general_beckman/posthook_continuations.py tests/beckman/test_posthook_grade_resume.py
rtk git commit -m "feat(cps-sp3): grade resume handler with continuation-chaining retry"
```

---

## Task 6: `posthook.code_review.resume` + `.resume_err`

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/posthook_continuations.py`
- Test: `tests/beckman/test_posthook_code_review_resume.py`

Single-shot (no chaining). Parse → `PostHookVerdict(kind="code_review", passed, raw=issues)` → apply. `resume_err` → passed=False.

- [ ] **Step 1: Write the failing tests**

```python
"""SP3 Task 6 — code_review resume."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_code_review_resume_pass():
    from general_beckman import posthook_continuations as pc
    raw = "ISSUES:\n- NONE\nVERDICT: PASS\n"
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._code_review_resume(201, {"result": {"content": raw}},
                                     {"source_task_id": 9})
    v = ap.call_args.args[1]
    assert v.kind == "code_review" and v.passed is True


@pytest.mark.asyncio
async def test_code_review_resume_fail_carries_issues():
    from general_beckman import posthook_continuations as pc
    raw = "ISSUES:\n- missing auth check in a.py:12\nVERDICT: FAIL\n"
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._code_review_resume(202, {"result": {"content": raw}},
                                     {"source_task_id": 9})
    v = ap.call_args.args[1]
    assert v.passed is False


@pytest.mark.asyncio
async def test_code_review_resume_err():
    from general_beckman import posthook_continuations as pc
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._code_review_resume_err(203, {"status": "failed", "error": "x"},
                                         {"source_task_id": 9})
    assert ap.call_args.args[1].passed is False
```

- [ ] **Step 2: Run to verify fail**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/beckman/test_posthook_code_review_resume.py -v`
Expected: FAIL — NotImplementedError.

- [ ] **Step 3: Implement**

```python
def _make_cr_verdict(source_task_id: int, passed: bool, raw):
    from general_beckman.rewrite import PostHookVerdict
    return PostHookVerdict(kind="code_review", source_task_id=source_task_id,
                           passed=passed, raw=raw)


async def _code_review_resume(child_task_id: int, result: dict, state: dict) -> None:
    from src.core.code_review import parse_code_review_response
    source_task_id = state.get("source_task_id")
    raw = _extract_content(result)
    cr = parse_code_review_response(raw)
    await _apply_posthook_verdict(
        {"id": child_task_id},
        _make_cr_verdict(source_task_id, cr.passed,
                         {"passed": cr.passed, "issues": cr.issues, "raw": cr.raw}),
    )


async def _code_review_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    source_task_id = state.get("source_task_id")
    err = (result or {}).get("error", "unknown")
    await _apply_posthook_verdict(
        {"id": child_task_id},
        _make_cr_verdict(source_task_id, False,
                         {"passed": False, "raw": f"auto-fail: code-review call failed ({err})"}),
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/beckman/test_posthook_code_review_resume.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add packages/general_beckman/src/general_beckman/posthook_continuations.py tests/beckman/test_posthook_code_review_resume.py
rtk git commit -m "feat(cps-sp3): code_review resume handler"
```

---

## Task 7: `posthook.summary.resume` + `.resume_err`

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/posthook_continuations.py`
- Test: `tests/beckman/test_posthook_summary_resume.py`

The verdict `kind` is `f"summary:{artifact_name}"` (matches `apply.py:4275`). `passed = bool(summary) and len(summary) >= 50` and not degenerate; raw carries `{"summary", "artifact_name"}`. `resume_err` → passed=False (structural summary already stored by `post_execute`; just drain pending).

- [ ] **Step 1: Write the failing tests**

```python
"""SP3 Task 7 — summary resume."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_summary_resume_pass_builds_summary_kind_verdict():
    from general_beckman import posthook_continuations as pc
    summary = "This artifact describes the user stories. " * 4
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._summary_resume(301, {"result": {"content": summary}},
                                 {"source_task_id": 5, "artifact_name": "user_stories"})
    v = ap.call_args.args[1]
    assert v.kind == "summary:user_stories" and v.passed is True
    assert v.raw["summary"].startswith("This artifact")


@pytest.mark.asyncio
async def test_summary_resume_short_fails():
    from general_beckman import posthook_continuations as pc
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._summary_resume(302, {"result": {"content": "tiny"}},
                                 {"source_task_id": 5, "artifact_name": "x"})
    assert ap.call_args.args[1].passed is False


@pytest.mark.asyncio
async def test_summary_resume_err_fails_closed():
    from general_beckman import posthook_continuations as pc
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._summary_resume_err(303, {"status": "failed"},
                                     {"source_task_id": 5, "artifact_name": "x"})
    assert ap.call_args.args[1].passed is False
```

- [ ] **Step 2: Run to verify fail**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/beckman/test_posthook_summary_resume.py -v`
Expected: FAIL — NotImplementedError.

- [ ] **Step 3: Implement**

```python
def _make_summary_verdict(source_task_id: int, artifact_name: str, passed: bool, summary: str):
    from general_beckman.rewrite import PostHookVerdict
    return PostHookVerdict(
        kind=f"summary:{artifact_name}", source_task_id=source_task_id,
        passed=passed, raw={"summary": summary, "artifact_name": artifact_name},
    )


async def _summary_resume(child_task_id: int, result: dict, state: dict) -> None:
    source_task_id = state.get("source_task_id")
    artifact_name = state.get("artifact_name") or ""
    summary = _extract_content(result).strip()
    passed = bool(summary) and len(summary) >= 50
    if passed:
        try:
            from dogru_mu_samet import assess as cq_assess
            if cq_assess(summary).is_degenerate:
                passed = False
        except Exception:
            pass
    await _apply_posthook_verdict(
        {"id": child_task_id},
        _make_summary_verdict(source_task_id, artifact_name, passed, summary if passed else ""),
    )


async def _summary_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    source_task_id = state.get("source_task_id")
    artifact_name = state.get("artifact_name") or ""
    await _apply_posthook_verdict(
        {"id": child_task_id},
        _make_summary_verdict(source_task_id, artifact_name, False, ""),
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/beckman/test_posthook_summary_resume.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add packages/general_beckman/src/general_beckman/posthook_continuations.py tests/beckman/test_posthook_summary_resume.py
rtk git commit -m "feat(cps-sp3): summary resume handler"
```

---

## Task 8: Spawn helper `_enqueue_posthook_llm_child` + wire `_apply_request_posthook`

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/apply.py`
- Test: `tests/beckman/test_posthook_spawn_cps.py`

The helper builds the child spec (via `build_*_spec`), short-circuits to an immediate verdict on auto-fail, else `enqueue`s with `on_complete`/`on_error`/`cont_state` (mission_id in cont_state ONLY). Wire `_apply_request_posthook` so the three LLM kinds route through it instead of enqueueing an agent task. **mechanical kinds untouched.**

- [ ] **Step 1: Write the failing test**

```python
"""SP3 Task 8 — CPS spawn for grade/code_review/summary post-hooks."""
import json
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_grade_posthook_enqueues_child_with_continuation_no_agent_task():
    from general_beckman import apply as ap
    source = {"id": 7, "title": "T", "description": "D", "result": "x" * 200,
              "context": "{}", "mission_id": 3}
    with patch.object(ap, "enqueue", AsyncMock(return_value=999)) as enq, \
         patch("src.infra.db.get_task", AsyncMock(return_value=source)), \
         patch("src.infra.db.update_task", AsyncMock()):
        await ap._enqueue_posthook_llm_child("grade", source, {})
    enq.assert_awaited_once()
    _, kwargs = enq.call_args
    assert kwargs["on_complete"] == "posthook.grade.resume"
    assert kwargs["on_error"] == "posthook.grade.resume_err"
    assert kwargs["cont_state"]["source_task_id"] == 7
    # mission_id in cont_state, NOT on the child spec row
    assert kwargs["cont_state"].get("mission_id") == 3
    spec = enq.call_args.args[0]
    assert "mission_id" not in spec


@pytest.mark.asyncio
async def test_trivial_source_short_circuits_to_verdict_no_child():
    from general_beckman import apply as ap
    source = {"id": 7, "result": "  ", "context": "{}"}
    with patch.object(ap, "enqueue", AsyncMock()) as enq, \
         patch.object(ap, "_apply_posthook_verdict", AsyncMock()) as apv:
        await ap._enqueue_posthook_llm_child("grade", source, {})
    enq.assert_not_awaited()
    apv.assert_awaited_once()  # auto-fail applied directly
```

- [ ] **Step 2: Run to verify fail**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/beckman/test_posthook_spawn_cps.py -v`
Expected: FAIL — missing `_enqueue_posthook_llm_child`.

- [ ] **Step 3: Implement the helper**

In `apply.py`, add (near `_apply_request_posthook`). Import `enqueue` from the package:

```python
async def _enqueue_posthook_llm_child(kind: str, source: dict, source_ctx: dict,
                                      *, exclusions=None, attempt: int = 0) -> None:
    """Enqueue the raw_dispatch reviewer/summarizer child with a continuation.

    mission_id rides in cont_state ONLY (never on the child row — SP3 child-spec
    hygiene). On a trivial/degenerate source the builder returns a verdict, which
    we apply directly (no child)."""
    from general_beckman import enqueue
    source_id = source.get("id")
    mission_id = source.get("mission_id")
    excl = list(exclusions or source_ctx.get("grade_excluded_models")
                or source_ctx.get("review_excluded_models") or [])

    if kind == "grade":
        from src.core.grading import build_grading_spec, GradeResult
        built = build_grading_spec(source, excl)
        on_complete, on_error = "posthook.grade.resume", "posthook.grade.resume_err"
        if isinstance(built, GradeResult):
            await _apply_posthook_verdict({"id": source_id},
                _grade_autofail_verdict(source_id, built))
            return
    elif kind == "code_review":
        from src.core.code_review import build_code_review_spec, CodeReviewResult
        built = build_code_review_spec(source, excl)
        on_complete, on_error = "posthook.code_review.resume", "posthook.code_review.resume_err"
        if isinstance(built, CodeReviewResult):
            await _apply_posthook_verdict({"id": source_id},
                _cr_autofail_verdict(source_id, built))
            return
    elif kind.startswith("summary:"):
        artifact_name = kind.split(":", 1)[1]
        text = source_ctx.get("_summary_text") or ""  # see Task 9 for how text is resolved
        from src.workflows.engine.hooks import build_summary_spec
        built = build_summary_spec(text, artifact_name)
        on_complete, on_error = "posthook.summary.resume", "posthook.summary.resume_err"
    else:
        raise ValueError(f"_enqueue_posthook_llm_child: unsupported kind {kind!r}")

    cont_state = {
        "source_task_id": source_id, "kind": kind,
        "attempt": attempt, "exclusions": excl, "mission_id": mission_id,
    }
    if kind.startswith("summary:"):
        cont_state["artifact_name"] = kind.split(":", 1)[1]
    await enqueue(built, parent_id=source_id, on_complete=on_complete,
                  on_error=on_error, cont_state=cont_state, lane="overhead")
```

Add small `_grade_autofail_verdict`/`_cr_autofail_verdict` helpers that wrap the `GradeResult`/`CodeReviewResult` into a `PostHookVerdict(passed=False)`.

- [ ] **Step 4: Wire `_apply_request_posthook`**

In `_apply_request_posthook` (`apply.py:1153-1197`), after parking the source `ungraded`, branch the LLM kinds to the helper instead of `_posthook_agent_and_payload` + `add_task`:

```python
    if a.kind == "grade" or a.kind == "code_review" or a.kind.startswith("summary:"):
        await _enqueue_posthook_llm_child(a.kind, source, posthook_ctx)
        return
    # mechanical kinds keep the existing agent_type/payload + add_task path:
    agent_type, payload = _posthook_agent_and_payload(a, source, posthook_ctx)
    ...
```

(Leave `_posthook_agent_and_payload`'s mechanical branches; remove its now-dead grade/summary/code_review branches in Task 11.)

- [ ] **Step 5: Run to verify pass**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/beckman/test_posthook_spawn_cps.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add packages/general_beckman/src/general_beckman/apply.py tests/beckman/test_posthook_spawn_cps.py
rtk git commit -m "feat(cps-sp3): _enqueue_posthook_llm_child + wire _apply_request_posthook (CPS spawn)"
```

---

## Task 9: Wire the grade-pass summary loop to CPS

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/apply.py` (`:4211-4228`)
- Test: `tests/beckman/test_grade_pass_summary_cps.py`

The grade-pass branch spawns summary children (`:4217` `add_task(agent_type="artifact_summarizer")`). Route through `_enqueue_posthook_llm_child("summary:<name>", source, ctx_with_text)`. The summary text comes from the blackboard (`ArtifactStore().retrieve(mission_id, artifact_name)` — what `ArtifactSummarizerAgent.execute` did at `artifact_summarizer.py:60`). Resolve it here and pass via `source_ctx["_summary_text"]`.

- [ ] **Step 1: Write the failing test**

```python
"""SP3 Task 9 — grade-pass spawns summary children via CPS."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_grade_pass_spawns_summary_child_via_helper(monkeypatch):
    from general_beckman import apply as ap
    source = {"id": 5, "mission_id": 3, "status": "ungraded",
              "context": '{"_pending_posthooks": ["grade"]}'}
    with patch("src.infra.db.get_task", AsyncMock(return_value=source)), \
         patch("src.infra.db.update_task", AsyncMock()), \
         patch.object(ap, "_summary_kinds_for_source", AsyncMock(return_value=["summary:user_stories"])), \
         patch.object(ap, "_enqueue_posthook_llm_child", AsyncMock()) as enq, \
         patch.object(ap, "_record_and_resolve_confidence", AsyncMock()), \
         patch("src.workflows.engine.artifacts.ArtifactStore") as Store:
        Store.return_value.retrieve = AsyncMock(return_value="long artifact " * 300)
        from general_beckman.rewrite import PostHookVerdict
        v = PostHookVerdict(kind="grade", source_task_id=5, passed=True, raw={})
        await ap._apply_posthook_verdict({"id": 999}, v)
    enq.assert_awaited()
    assert enq.call_args.args[0].startswith("summary:")
```

- [ ] **Step 2: Run to verify fail**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/beckman/test_grade_pass_summary_cps.py -v`
Expected: FAIL (still uses `add_task(agent_type="artifact_summarizer")`).

- [ ] **Step 3: Implement**

Replace the `for kind in new_summary_kinds:` `add_task(...)` block (`apply.py:4215-4228`) with:

```python
        for kind in new_summary_kinds:
            pending.append(kind)
            artifact_name = kind.split(":", 1)[1]
            text = ""
            try:
                val = await ArtifactStore().retrieve(source.get("mission_id"), artifact_name)
                if isinstance(val, str):
                    text = val
            except Exception:
                pass
            ctx_with_text = dict(ctx)
            ctx_with_text["_summary_text"] = text
            await _enqueue_posthook_llm_child(kind, source, ctx_with_text)
```

- [ ] **Step 4: Run to verify pass**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/beckman/test_grade_pass_summary_cps.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add packages/general_beckman/src/general_beckman/apply.py tests/beckman/test_grade_pass_summary_cps.py
rtk git commit -m "feat(cps-sp3): grade-pass summary spawn routes through CPS helper"
```

---

## Task 10: Delete agent classes + registry/classifier refs

**Files:**
- Delete: `src/agents/grader.py`, `src/agents/code_reviewer.py`, `src/agents/artifact_summarizer.py`
- Modify: `src/agents/__init__.py`, `src/core/task_classifier.py`
- Test: `tests/agents/test_sp3_agents_deleted.py`

- [ ] **Step 1: Write the failing test**

```python
"""SP3 Task 10 — deleted post-hook agents are gone + unrouted."""
def test_agents_not_in_registry():
    from src.agents import AGENT_REGISTRY
    for name in ("grader", "code_reviewer", "artifact_summarizer"):
        assert name not in AGENT_REGISTRY

def test_classifier_does_not_route_to_deleted_agents():
    import src.core.task_classifier as tc
    src = open(tc.__file__, encoding="utf-8").read()
    for name in ("grader", "code_reviewer", "artifact_summarizer"):
        assert f'"{name}"' not in src, f"{name} still referenced in classifier"
```

- [ ] **Step 2: Run to verify fail**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/agents/test_sp3_agents_deleted.py -v`
Expected: FAIL — agents still registered/referenced.

- [ ] **Step 3: Delete + de-register**

```bash
rm src/agents/grader.py src/agents/code_reviewer.py src/agents/artifact_summarizer.py
```

In `src/agents/__init__.py`: remove the three imports (`:21-23`) and the three `AGENT_REGISTRY` entries (`:50-52`). In `src/core/task_classifier.py`: remove the router-prompt lines + keyword-table entries (`:64-67`, `:81-82`, `:407-410`). Grep to confirm zero remaining references: `rtk grep "grader\|code_reviewer\|artifact_summarizer" src/agents src/core/task_classifier.py`.

- [ ] **Step 4: Run to verify pass**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/agents/test_sp3_agents_deleted.py -v`
Expected: PASS. Also verify import health: `.venv/Scripts/python -c "import src.agents; import src.core.task_classifier"`.

- [ ] **Step 5: Commit**

```bash
rtk git add -A
rtk git commit -m "feat(cps-sp3): delete grader/code_reviewer/artifact_summarizer agents + classifier refs"
```

---

## Task 11: Drop dead agent strings from Beckman plumbing + relocate DLQ cascade

**Files:**
- Modify: `apply.py` (`_OVERHEAD_POSTHOOK_AGENTS:1145`, `_posthook_agent_and_payload` grade/summary/code_review branches `:1204-1261`, grader-DLQ-cascade `:832`), `posthooks.py:74`, `rewrite.py:103-105/255-259`, `general_beckman/__init__.py:941-944/983/1026-1029`, `advance.py:195`
- Test: `tests/beckman/test_sp3_dead_strings_removed.py`

The grader-DLQ-cascade (`apply.py:832`, in `_posthook_dlq_cascade`) made a grader DLQ permanently fail the source. Under Shape B the grade child failing terminally fires `posthook.grade.resume_err` → auto-fail verdict → `_apply_posthook_verdict` grade-fail branch already retries/DLQs the source at cap. So the cascade's grader branch is **subsumed** — remove the grader-specific cascade and rely on the resume_err path. (Verify no other caller depends on it.)

- [ ] **Step 1: Write the failing test**

```python
"""SP3 Task 11 — dead agent strings removed from plumbing."""
import re

PLUMBING = [
    "packages/general_beckman/src/general_beckman/apply.py",
    "packages/general_beckman/src/general_beckman/posthooks.py",
    "packages/general_beckman/src/general_beckman/rewrite.py",
    "packages/general_beckman/src/general_beckman/__init__.py",
]

def test_no_grader_artifact_summarizer_agent_strings_in_plumbing():
    for path in PLUMBING:
        src = open(path, encoding="utf-8").read()
        # grade/summary as POSTHOOK KINDS are fine; the AGENT TYPE strings are not.
        assert '"artifact_summarizer"' not in src, f"artifact_summarizer agent str in {path}"
        assert '"grader"' not in src, f"grader agent str in {path}"
```

> Note: `"code_review"` and `"summary:"` as post-hook *kinds* remain (they're verdict kinds, not agent types). Only the agent-type strings `"grader"`/`"artifact_summarizer"`/`"code_reviewer"` are removed. Adjust the assertion if a legitimate non-agent use of these literals exists.

- [ ] **Step 2: Run to verify fail**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/beckman/test_sp3_dead_strings_removed.py -v`
Expected: FAIL.

- [ ] **Step 3: Remove the strings**

- `apply.py`: delete `_OVERHEAD_POSTHOOK_AGENTS` + `_posthook_kind` (no longer needed — LLM children are enqueued `kind="overhead"` directly); delete the grade/summary/code_review branches of `_posthook_agent_and_payload` (keep mechanical branches); remove the grader branch of `_posthook_dlq_cascade` (`:832`).
- `posthooks.py:74` `_NO_POSTHOOKS_AGENT_TYPES`: drop `grader`/`artifact_summarizer`/`code_reviewer`.
- `rewrite.py:103-105` + `:255-259`: drop the three strings.
- `general_beckman/__init__.py:941-944`, `:983`, `:1026-1029`: drop the three strings from all three sets.
- `advance.py:195` `_bookkeeping`: drop the three strings.

- [ ] **Step 4: Run to verify pass + import health**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/beckman/test_sp3_dead_strings_removed.py -v`
Then: `timeout 120 .venv/Scripts/python -m pytest tests/beckman/ -q` (substrate + apply suites still green).
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add -A
rtk git commit -m "feat(cps-sp3): drop dead agent strings from beckman plumbing; relocate grader DLQ cascade to resume_err"
```

---

## Task 12: Remove `await_inline` from grading.py / code_review.py / hooks.py

**Files:**
- Modify: `src/core/grading.py` (remove `grade_task:276-427`), `src/core/code_review.py` (remove `code_review_task:98-206`), `src/workflows/engine/hooks.py` (remove `_llm_summarize:20-101`)
- Test: `tests/core/test_sp3_no_await_inline.py`

`grade_task`/`code_review_task`/`_llm_summarize` are dead after Tasks 8-9 (their callers — the deleted agents — are gone). Keep `parse_*`, `GradeResult`/`CodeReviewResult`, `apply_grade_result`, `build_*_spec`.

- [ ] **Step 1: Write the failing test**

```python
"""SP3 Task 12 — no await_inline in the migrated post-hook source files."""
FILES = ["src/core/grading.py", "src/core/code_review.py", "src/workflows/engine/hooks.py"]

def test_no_await_inline_true():
    for path in FILES:
        src = open(path, encoding="utf-8").read()
        assert "await_inline=True" not in src, f"await_inline=True still in {path}"

def test_parsers_and_builders_survive():
    from src.core.grading import parse_grade_response, build_grading_spec
    from src.core.code_review import parse_code_review_response, build_code_review_spec
    from src.workflows.engine.hooks import build_summary_spec
```

- [ ] **Step 2: Run to verify fail**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/core/test_sp3_no_await_inline.py -v`
Expected: FAIL — `await_inline=True` present.

- [ ] **Step 3: Remove the dead functions**

Delete `grade_task` (`grading.py:276-427`), `code_review_task` (`code_review.py:98-206`), `_llm_summarize` (`hooks.py:20-101`). Confirm nothing else imports them: `rtk grep "grade_task\|code_review_task\|_llm_summarize" src packages` (should hit only tests, handled in Task 13).

- [ ] **Step 4: Run to verify pass + import health**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/core/test_sp3_no_await_inline.py -v`
Then: `.venv/Scripts/python -c "import src.core.grading, src.core.code_review, src.workflows.engine.hooks"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add -A
rtk git commit -m "feat(cps-sp3): remove grade_task/code_review_task/_llm_summarize (await_inline gone)"
```

---

## Task 13: Update/delete affected existing tests

**Files:** (per spec rev2 §"Tests to rewrite / remove")
- Delete: `tests/test_grader_agent.py`, `tests/test_artifact_summarizer_agent.py`, `tests/test_code_review_posthook.py`, `tests/core/test_grading_enqueue.py`, `tests/workflows/engine/test_hooks_enqueue.py`
- Modify: `tests/test_lifecycle_fixes.py:144`, `tests/agents/test_prompt_quality.py:11-13`, `tests/test_posthook_kind.py:20-71`, `tests/test_reviewer_no_grade.py:28-37`, `tests/test_beckman_posthooks.py` (grader/summarizer assertions), `tests/test_beckman_rewrite.py:103`, `tests/test_beckman_on_task_finished.py:50-85`

- [ ] **Step 1: Delete the agent-path test files**

```bash
rm tests/test_grader_agent.py tests/test_artifact_summarizer_agent.py tests/test_code_review_posthook.py tests/core/test_grading_enqueue.py tests/workflows/engine/test_hooks_enqueue.py
```

- [ ] **Step 2: Update membership/path assertions**

For each modify-target: replace assertions that the agent exists / that `_posthook_kind("grader")=="overhead"` / that grade spawns `agent_type="grader"` / that `{"mechanical","grader"}` are siblings → with the CPS reality (grade post-hook enqueues a `reviewer` child carrying `on_complete="posthook.grade.resume"`; no grader agent task). For `test_prompt_quality.py:11-13`, remove the three agents from `LOW_TRAFFIC_AGENTS`. For `test_reviewer_no_grade.py` / `test_beckman_posthooks.py` membership tests, the `reviewer` agent type already returns `[]` from `determine_posthooks` — update the grader/summarizer cases to assert the agents no longer exist rather than their posthook behavior.

(Read each file and rewrite the specific assertions — do not blanket-delete; some files contain unrelated passing tests.)

- [ ] **Step 3: Run the touched suites**

Run: `timeout 120 .venv/Scripts/python -m pytest tests/test_posthook_kind.py tests/test_reviewer_no_grade.py tests/test_beckman_posthooks.py tests/test_beckman_rewrite.py tests/test_beckman_on_task_finished.py tests/test_lifecycle_fixes.py tests/agents/test_prompt_quality.py -q`
Expected: PASS (after rewrites).

- [ ] **Step 4: Commit**

```bash
rtk git add -A
rtk git commit -m "test(cps-sp3): update/remove tests that pinned the deleted agent-task path"
```

---

## Task 14: End-to-end integration — deadlock closure + C1 regression

**Files:**
- Test: `tests/beckman/test_cps_sp3_integration.py`

Drive the real `enqueue` → `add_task` → `on_task_finished` → resume → `_apply_posthook_verdict` path against a temp DB (mirror `tests/beckman/test_continuations_durable.py` fixture).

- [ ] **Step 1: Write the integration tests**

```python
"""SP3 Task 14 — end-to-end CPS integration: deadlock closure + C1 regression."""
import json
import pytest
# Reuse the _fresh_db / _close_db fixture pattern from
# tests/beckman/test_continuations_durable.py.


@pytest.mark.asyncio
async def test_grade_posthook_enqueues_child_and_returns_no_held_slot(tmp_path, monkeypatch):
    """Posthook spawns a reviewer child with a continuation row and returns —
    no grader agent task row, source parked 'ungraded'."""
    # ... set up source task 'ungraded' with a RequestPostHook(grade);
    # call _apply_request_posthook; assert:
    #   - a continuations row exists for the new child (resume_name='posthook.grade.resume')
    #   - the new child agent_type == 'reviewer' (NOT 'grader')
    #   - no task with agent_type='grader' was created
    #   - source still 'ungraded'


@pytest.mark.asyncio
async def test_failed_then_retried_completed_fires_grade_resume_once(tmp_path, monkeypatch):
    """C1 regression: reviewer child failed → re-pended → completed fires the
    resume exactly once on the final completed status (no silent drop)."""
    # Drive on_task_finished for the child with status='failed' (transient,
    # re-pends, continuation stays 'pending'); then status='completed';
    # assert the grade verdict was applied exactly once.


@pytest.mark.asyncio
async def test_double_terminal_fires_resume_once(tmp_path, monkeypatch):
    """CAS idempotency: two on_task_finished calls for the same completed child
    apply the verdict exactly once."""
```

- [ ] **Step 2: Run to verify fail, then implement the test bodies**

Fill the test bodies using the real DB helpers (`add_task` with `on_complete`, `on_task_finished`). Use a registered fake `reviewer`-output (a valid `VERDICT: PASS` string) to assert the source transitions to `completed`.

Run: `timeout 120 .venv/Scripts/python -m pytest tests/beckman/test_cps_sp3_integration.py -v`
Expected: PASS.

- [ ] **Step 3: Full regression gate**

Run:
```bash
timeout 300 .venv/Scripts/python -m pytest tests/beckman/ tests/core/test_grading.py tests/core/test_build_grading_spec.py tests/core/test_build_code_review_spec.py tests/workflows/engine/test_build_summary_spec.py tests/agents/ -q
```
Expected: PASS (note the two pre-existing flakes/fails the SP3 kickoff + SP2 handoff flagged: `test_enqueue_await_inline_blocks_until_resolved` cold-import flake; `test_reversibility_registry`; the 3 `test_dispatcher_*` fails owned by SP1.1/SP3b — confirm these are the SAME failures present on the branch point, not new).

- [ ] **Step 4: Commit**

```bash
rtk git add tests/beckman/test_cps_sp3_integration.py
rtk git commit -m "test(cps-sp3): e2e integration — deadlock closure + C1 regression + CAS idempotency"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** every spec §Changes item maps to a task — builders T1-T3; new module T4; handlers T5-T7 (chaining in T5); spawn refactor T8-T9; deletions T10-T11; await_inline removal T12; tests T13; substrate-invariant + deadlock-closure verification T14. ✓
- **Child-spec hygiene (HIGH):** enforced in T8 (mission_id in cont_state only; `assert "mission_id" not in spec`). ✓
- **Deletion completeness:** T10 (agents+registry+classifier) + T11 (plumbing: `_OVERHEAD_POSTHOOK_AGENTS`, `_NO_POSTHOOKS_AGENT_TYPES`, rewrite Rules 0/1, 3 progress-ping sets, advance.py, DLQ cascade). ✓
- **Type/name consistency:** handler names `posthook.<kind>.resume`/`_err` used identically in T4 registration, T5-T7 impls, T8-T9 spawn. `PostHookVerdict(kind, source_task_id, passed, raw)` consistent. Builders return `dict | GradeResult|CodeReviewResult` consistently. ✓
- **Open impl-time confirmations (flagged inline, not placeholders):** exact `PostHookVerdict` import path (T5 note); `advance.py` exact module path (`src/workflows/engine/` vs `workflow_engine/`); whether any non-agent literal use of the agent strings exists (T11 note). Each has a grep step.

---

## Execution

Per the AFK mandate: proceed with **subagent-driven-development** (fresh subagent per task, two-stage review between tasks) in an isolated worktree. Parallel SP3b session may be running — work in the worktree, never revert its commits, take-both on `_HANDLER_MODULES` merge conflicts.
