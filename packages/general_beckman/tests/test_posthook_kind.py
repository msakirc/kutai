"""Post-hook grade/summarize work must land on the OVERHEAD lane.

Bug (2026-05-26): the generic post-hook spawner (apply._apply_request_posthook)
called add_task() without kind=, so grade/summarize post-hooks defaulted to
kind='main_work' → runner='react'. As main_work they picked CLOUD models and
rode the 600s wall-clock cap; under network instability a cloud call hung the
full 600s → bare TimeoutError → DLQ at 6/6. These are single-call LLM
evaluation work and belong on the OVERHEAD lane (loaded local model, direct
runner).

SP3 reality: grade / code_review / summary post-hooks no longer flow through
add_task() at all. They are enqueued as raw_dispatch reviewer/summarizer
CHILDREN via general_beckman.enqueue(..., lane="overhead") with a durable
continuation. ``_posthook_kind`` therefore now returns ``"main_work"``
unconditionally — it only serves the mechanical post-hook branch that still
reaches add_task() (those route via agent_type='mechanical'). The OVERHEAD
lane assignment moved from the spawned task's ``kind`` to the explicit
``lane="overhead"`` argument on enqueue().
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


def test_posthook_kind_is_always_main_work():
    """SP3: _posthook_kind no longer routes grade/summary to 'overhead' — the
    LLM post-hooks bypass add_task() entirely. The only kind it returns now is
    'main_work' (mechanical post-hooks route via agent_type='mechanical')."""
    from general_beckman.apply import _posthook_kind
    assert _posthook_kind("reviewer") == "main_work"
    assert _posthook_kind("summarizer") == "main_work"
    assert _posthook_kind("mechanical") == "main_work"
    assert _posthook_kind("anything") == "main_work"


@pytest.mark.asyncio
async def test_grade_posthook_enqueues_reviewer_child_on_overhead_lane(monkeypatch):
    """End-to-end: a grade post-hook spawns a raw_dispatch reviewer CHILD via
    general_beckman.enqueue(lane="overhead") with the durable grade
    continuation — NOT an add_task(agent_type='grader', kind='overhead') row."""
    import general_beckman.apply as apply_mod

    add_task_calls: list = []

    async def fake_add_task(**kw):
        add_task_calls.append(kw)
        return 1

    async def fake_get_task(tid):
        # Non-trivial, non-degenerate result so build_grading_spec does not
        # short-circuit to an auto-fail verdict (which would skip the child).
        return {
            "id": tid, "mission_id": 5,
            "result": (
                "The service exposes a REST API backed by a normalized SQLite "
                "schema, with auth handled by signed session tokens."
            ),
            "context": "{}",
        }

    async def fake_update_task(*a, **k):
        return None

    monkeypatch.setattr("src.infra.db.add_task", fake_add_task)
    monkeypatch.setattr("src.infra.db.get_task", fake_get_task)
    monkeypatch.setattr("src.infra.db.update_task", fake_update_task)

    class _A:
        kind = "grade"
        source_task_id = 100
        source_ctx: dict = {}

    with patch.object(apply_mod, "enqueue", AsyncMock(return_value=999)) as enq:
        await apply_mod._apply_request_posthook({"id": 1}, _A())

    # The reviewer child was enqueued on the OVERHEAD lane via CPS.
    enq.assert_awaited_once()
    spec = enq.call_args.args[0]
    kwargs = enq.call_args.kwargs
    assert spec["agent_type"] == "reviewer"
    assert kwargs["lane"] == "overhead", \
        f"grade post-hook must enqueue on the overhead lane, got {kwargs.get('lane')!r}"
    assert kwargs["on_complete"] == "posthook.grade.resume"

    # No legacy grader agent-task row was created.
    assert not any(c.get("agent_type") == "grader" for c in add_task_calls)


# ---------------------------------------------------------------------------
# SP3b Task 3: self_reflect + constrained_emit registry entries
# ---------------------------------------------------------------------------

def test_reflect_and_emit_kinds_registered():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    assert "self_reflect" in POST_HOOK_REGISTRY
    assert "constrained_emit" in POST_HOOK_REGISTRY
    assert POST_HOOK_REGISTRY["constrained_emit"].default_severity == "blocker"
    assert POST_HOOK_REGISTRY["self_reflect"].default_severity == "warning"


def test_reflect_emit_child_types_are_recursion_guarded():
    from general_beckman.posthooks import _NO_POSTHOOKS_AGENT_TYPES
    assert "self_reflect" in _NO_POSTHOOKS_AGENT_TYPES
    assert "constrained_emit" in _NO_POSTHOOKS_AGENT_TYPES
