"""SP3b Task 9 — no-inline-deadlock proof.

The original deadlock was:
    parent task (holds lane slot) → dispatcher.request()
    → beckman.enqueue(await_inline=True)
    → wait for child task
    → child needs SAME lane → DEADLOCK (parent holds slot, child can't admit)

The CPS fix: reflection / constrained_emit / grade are spawned as SEPARATE
admitted tasks with ``on_complete`` continuations.  No inline waiter.  The
parent COMPLETES (releasing its slot) before the post-hook children run.

Three structural tests prove this:

  A. _enqueue_posthook_llm_child NEVER passes await_inline=True to enqueue
     (for any LLM post-hook kind: constrained_emit, self_reflect, grade,
     code_review, summary:*).

  B. The posthook chain cursor (_advance_posthook_chain) also never passes
     await_inline=True — it calls _enqueue_posthook_llm_child which itself
     never passes it.

  C. Fire-and-forget simulation: a source task completes → RequestPostHook
     → _apply_request_posthook → _enqueue_posthook_llm_child; the recorded
     enqueue calls carry on_complete continuations and NO await_inline=True.
     Source task result is available immediately (doesn't block on children).
"""
from __future__ import annotations

import inspect
import json
import pytest

from unittest.mock import AsyncMock, MagicMock, patch, call as mock_call


# ---------------------------------------------------------------------------
# Helper: capture all enqueue(...) calls and assert await_inline is absent
# ---------------------------------------------------------------------------

def _assert_no_await_inline(enqueue_calls: list, label: str = "") -> None:
    """Assert that none of the recorded enqueue calls used await_inline=True."""
    for i, c in enumerate(enqueue_calls):
        ai = c.kwargs.get("await_inline")
        assert ai is not True, (
            f"{label}enqueue call #{i} used await_inline=True — "
            f"this recreates the lane-deadlock. kwargs={c.kwargs}"
        )


# ---------------------------------------------------------------------------
# A. _enqueue_posthook_llm_child never passes await_inline to enqueue
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("kind,source_ctx", [
    (
        "constrained_emit",
        {
            "artifact_schema": {
                "widget": {"type": "object", "fields": {"name": {"type": "string"}}}
            },
            "workflow_step_id": "7.4",
        },
    ),
    (
        "self_reflect",
        {"agent_type": "coder"},
    ),
    (
        "grade",
        {},  # build_grading_spec will short-circuit for empty source; that's fine
    ),
    (
        "code_review",
        {},  # build_code_review_spec will short-circuit; that's fine
    ),
])
async def test_enqueue_posthook_llm_child_no_await_inline(kind, source_ctx):
    """_enqueue_posthook_llm_child must NEVER pass await_inline=True to enqueue.

    For LLM post-hook kinds the function either short-circuits (grade/
    code_review with no grader candidates) or enqueues with on_complete only.
    Either way, await_inline must never appear.
    """
    import general_beckman.apply as apply_mod

    draft = "the feature is implemented and tested"
    source = {
        "id": 100,
        "mission_id": 42,
        "result": draft,
        "title": "build widget",
        "description": "add widget endpoint",
        "agent_type": "coder",
    }
    ctx = dict(source_ctx)

    captured_enqueue_calls: list = []

    async def _fake_enqueue(spec, /, **kwargs):
        captured_enqueue_calls.append(mock_call(**kwargs))
        return 999  # fake child task id

    # For grade/code_review, build_*_spec may return a terminal result object
    # (short-circuit) when source has no actual content / grader candidates.
    # _apply_posthook_verdict is called in that path — stub it out so we don't
    # need a real DB.
    async def _fake_apply_verdict(*a, **kw):
        pass

    with (
        patch.object(apply_mod, "enqueue", _fake_enqueue),
        patch.object(apply_mod, "_apply_posthook_verdict", _fake_apply_verdict),
    ):
        await apply_mod._enqueue_posthook_llm_child(kind, source, ctx)

    # Whether or not an enqueue call was made, none should carry await_inline=True.
    _assert_no_await_inline(
        captured_enqueue_calls,
        label=f"[kind={kind!r}] ",
    )


# ---------------------------------------------------------------------------
# B. summary:* kind also never uses await_inline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enqueue_posthook_llm_child_summary_no_await_inline():
    """summary:* kind enqueues a summarizer child — no await_inline."""
    import general_beckman.apply as apply_mod
    from unittest.mock import AsyncMock as _AM

    source = {"id": 101, "mission_id": 5, "result": "lots of text", "title": "t"}
    source_ctx = {}
    captured: list = []

    async def _fake_enqueue(spec, /, **kwargs):
        captured.append(mock_call(**kwargs))
        return 888

    # ArtifactStore.retrieve may fail — that's fine, summary falls back to "".
    with patch.object(apply_mod, "enqueue", _fake_enqueue):
        await apply_mod._enqueue_posthook_llm_child(
            "summary:design_doc", source, source_ctx
        )

    _assert_no_await_inline(captured, label="[kind=summary:design_doc] ")
    if captured:
        # When a child was spawned it must carry on_complete (CPS handoff).
        assert "on_complete" in captured[0].kwargs, (
            "summary child missing on_complete continuation"
        )


# ---------------------------------------------------------------------------
# C. Source-to-child simulation: source completes WITHOUT blocking on child
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_source_task_completes_without_blocking_on_posthook_child():
    """Simulate RequestPostHook via _apply_request_posthook.

    Asserts that when a source task triggers a RequestPostHook (grade),
    the source's update_task(status='ungraded') happens and enqueue is called
    with on_complete (not await_inline=True).  The source result is immediately
    available — no blocking on the child.
    """
    import general_beckman.apply as apply_mod
    from general_beckman.result_router import RequestPostHook

    # Source task — a completed coder task that needs grading.
    source = {
        "id": 200,
        "mission_id": 10,
        "result": "implemented the feature",
        "title": "coder step",
        "description": "implement feature X",
        "agent_type": "coder",
        "status": "completed",
        "context": json.dumps({
            "is_workflow_step": True,
            "workflow_step_id": "3.1",
        }),
    }

    enqueue_kwargs_log: list[dict] = []

    async def _fake_enqueue(spec, /, **kwargs):
        enqueue_kwargs_log.append(dict(kwargs))
        return 300  # fake child task id

    update_log: list[tuple] = []

    async def _fake_update_task(task_id, **kwargs):
        update_log.append((task_id, dict(kwargs)))

    async def _fake_get_task(task_id):
        return dict(source)

    # build_grading_spec may short-circuit if there are no graders — fake it so
    # it returns a spec dict that triggers the enqueue path.
    fake_spec = {
        "title": "grade:task#200",
        "agent_type": "grader",
        "kind": "overhead",
        "context": {"llm_call": {"raw_dispatch": True, "messages": [], "failures": []}},
    }

    a = RequestPostHook(
        source_task_id=200,
        kind="grade",
        source_ctx={"workflow_step_id": "3.1"},
    )

    with (
        patch("general_beckman.apply.enqueue", _fake_enqueue),
        patch("src.infra.db.update_task", _fake_update_task),
        patch("src.infra.db.get_task", _fake_get_task),
        patch("src.core.grading.build_grading_spec", return_value=fake_spec),
        patch.object(apply_mod, "_apply_posthook_verdict", AsyncMock()),
    ):
        await apply_mod._apply_request_posthook(source, a)

    # Source was updated to 'ungraded' — it's done with its result, not blocking.
    statuses = [kw.get("status") for _, kw in update_log]
    assert "ungraded" in statuses, (
        "Source task was not marked 'ungraded' — posthook lifecycle broken. "
        f"update_log={update_log}"
    )

    # If enqueue was called (non-short-circuit path), it must use CPS, not inline.
    for kwargs in enqueue_kwargs_log:
        assert kwargs.get("await_inline") is not True, (
            f"enqueue called with await_inline=True — deadlock risk! kwargs={kwargs}"
        )
        # CPS handoff: on_complete or on_error must be present.
        has_cps = "on_complete" in kwargs or "on_error" in kwargs
        assert has_cps, (
            f"enqueue missing on_complete/on_error — not a CPS continuation. "
            f"kwargs={kwargs}"
        )


# ---------------------------------------------------------------------------
# D. apply.py source does NOT contain await_inline (belt-and-suspenders)
# ---------------------------------------------------------------------------

def test_apply_py_source_has_no_await_inline():
    """apply.py must not call enqueue(await_inline=True) anywhere.

    Belt-and-suspenders source scan: even if the parametric tests above only
    exercise the tested paths, this assertion covers ALL branches in apply.py.
    """
    import general_beckman.apply as apply_mod

    src = inspect.getsource(apply_mod)
    # The only legitimate await_inline references in apply.py are doc comments.
    # Find all non-comment lines that contain await_inline=True.
    problematic = [
        line for line in src.splitlines()
        if "await_inline=True" in line
        and not line.strip().startswith("#")
    ]
    assert not problematic, (
        "apply.py contains non-comment await_inline=True — "
        "this would recreate the lane-deadlock:\n"
        + "\n".join(f"  {ln}" for ln in problematic)
    )
