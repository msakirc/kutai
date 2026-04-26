"""Test prompt-noise reduction on high-attempt retries (handoff item O).

Skill library and prior-steps narrative are dropped on attempt >=3.
Schema block + retry hint + deps stay — they're load-bearing for the
actual fix. Constrained-decoding Phase B handles structural shape, but
the draft phase still benefits from a leaner prompt on small models.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.base import BaseAgent


class _Agent(BaseAgent):
    name = "fake"


def _task_with_prior_steps(retry: int) -> dict:
    return {
        "id": 1,
        "title": "T",
        "description": "do thing",
        "worker_attempts": retry,
        "context": json.dumps({
            "is_workflow_step": True,
            "prior_steps": [
                {"title": "step A", "agent_type": "coder",
                 "status": "completed", "result": "narrative content here"},
                {"title": "step B", "agent_type": "writer",
                 "status": "completed", "result": "more narrative"},
            ],
        }),
    }


_PRIOR_HEADER = "## Results from Prior Steps (Inline)"


@pytest.mark.asyncio
async def test_prior_steps_kept_on_low_retry(monkeypatch):
    """Force the policy to include 'prior' so the gate is the only switch."""
    from src.memory import context_policy as cp
    monkeypatch.setattr(
        cp, "get_context_policy",
        lambda *_a, **_k: {"prior": True, "deps": True},
    )
    monkeypatch.setattr(cp, "apply_heuristics", lambda task, p: p)

    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value=None)
    with patch(
        "src.workflows.engine.hooks.get_artifact_store",
        return_value=fake_store,
    ):
        ctx = await _Agent()._build_context(_task_with_prior_steps(retry=1))
    assert _PRIOR_HEADER in ctx, (
        "prior-steps header expected on low-retry pass"
    )


@pytest.mark.asyncio
async def test_prior_steps_dropped_on_high_retry(monkeypatch):
    from src.memory import context_policy as cp
    monkeypatch.setattr(
        cp, "get_context_policy",
        lambda *_a, **_k: {"prior": True, "deps": True},
    )
    monkeypatch.setattr(cp, "apply_heuristics", lambda task, p: p)

    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value=None)
    with patch(
        "src.workflows.engine.hooks.get_artifact_store",
        return_value=fake_store,
    ):
        ctx = await _Agent()._build_context(_task_with_prior_steps(retry=4))
    assert _PRIOR_HEADER not in ctx, (
        "prior-steps header must be dropped on retry >= 3"
    )


@pytest.mark.asyncio
async def test_skills_skipped_on_high_retry(monkeypatch):
    """Skills block should not call find_relevant_skills on high-retry."""
    from src.memory import context_policy as cp
    monkeypatch.setattr(
        cp, "get_context_policy",
        lambda *_a, **_k: {"skills": True, "deps": True},
    )
    monkeypatch.setattr(cp, "apply_heuristics", lambda task, p: p)

    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value=None)
    fake_skills = AsyncMock(return_value=[])
    with patch(
        "src.workflows.engine.hooks.get_artifact_store",
        return_value=fake_store,
    ), patch(
        "src.memory.skills.find_relevant_skills", fake_skills,
    ):
        await _Agent()._build_context(_task_with_prior_steps(retry=5))
    # On high retry, skills block is gated out — no DB call.
    fake_skills.assert_not_called()


@pytest.mark.asyncio
async def test_skills_called_on_low_retry(monkeypatch):
    from src.memory import context_policy as cp
    monkeypatch.setattr(
        cp, "get_context_policy",
        lambda *_a, **_k: {"skills": True, "deps": True},
    )
    monkeypatch.setattr(cp, "apply_heuristics", lambda task, p: p)

    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value=None)
    fake_skills = AsyncMock(return_value=[])
    with patch(
        "src.workflows.engine.hooks.get_artifact_store",
        return_value=fake_store,
    ), patch(
        "src.memory.skills.find_relevant_skills", fake_skills,
    ):
        await _Agent()._build_context(_task_with_prior_steps(retry=1))
    # Low retry — skill lookup runs (returned [] but call happened)
    fake_skills.assert_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
