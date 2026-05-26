"""Post-hook grade/summarize tasks must spawn as OVERHEAD, not main_work.

Bug (2026-05-26): the generic post-hook spawner (apply._apply_request_posthook)
called add_task() without kind=, so grade/summarize post-hooks defaulted to
kind='main_work' → runner='react'. As main_work they picked CLOUD models and
rode the 600s wall-clock cap; under network instability a cloud call hung the
full 600s → bare TimeoutError → DLQ at 6/6. These are single-call LLM
evaluation work and belong on the OVERHEAD lane (loaded local model, direct
runner) — matching the inline grade (grading.py) and _llm_summarize (hooks.py)
which already use kind='overhead'. Mechanical post-hooks (verify_artifacts/
test_run/pattern_lint) route via agent_type='mechanical' and are untouched.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


def test_grader_posthook_is_overhead():
    from general_beckman.apply import _posthook_kind
    assert _posthook_kind("grader") == "overhead"


def test_artifact_summarizer_posthook_is_overhead():
    from general_beckman.apply import _posthook_kind
    assert _posthook_kind("artifact_summarizer") == "overhead"


def test_mechanical_posthook_is_not_overhead():
    from general_beckman.apply import _posthook_kind
    assert _posthook_kind("mechanical") == "main_work"


def test_reviewer_posthook_is_not_overhead():
    """Reviewers may need tools/ReAct — leave them main_work (out of scope)."""
    from general_beckman.apply import _posthook_kind
    assert _posthook_kind("code_reviewer") == "main_work"


@pytest.mark.asyncio
async def test_grade_posthook_spawns_with_kind_overhead(monkeypatch):
    """End-to-end: a grade post-hook must add_task(kind='overhead')."""
    import general_beckman.apply as apply_mod

    captured: dict = {}

    async def fake_add_task(**kw):
        captured.update(kw)
        return 1

    async def fake_get_task(tid):
        return {"id": tid, "mission_id": 5, "context": "{}"}

    async def fake_update_task(*a, **k):
        return None

    monkeypatch.setattr("src.infra.db.add_task", fake_add_task)
    monkeypatch.setattr("src.infra.db.get_task", fake_get_task)
    monkeypatch.setattr("src.infra.db.update_task", fake_update_task)

    class _A:
        kind = "grade"
        source_task_id = 100
        source_ctx: dict = {}

    await apply_mod._apply_request_posthook({"id": 1}, _A())

    assert captured.get("agent_type") == "grader"
    assert captured.get("kind") == "overhead", \
        f"grade post-hook must spawn overhead, got kind={captured.get('kind')!r}"
