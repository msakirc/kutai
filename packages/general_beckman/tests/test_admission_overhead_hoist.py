"""Admission must hoist OVERHEAD selection hints out of context.llm_call.

Post-hook children (self_reflect / constrained_emit / grade ...) author their
overhead intent INSIDE context.llm_call: call_category="overhead", a low inner
difficulty, prefer_speed=True. The model is actually picked at Beckman
admission, which historically read only the top-level task row — so overhead
work was scored as medium-difficulty thinking main_work and cloud thinking
models (gemini) won over an idle local. These tests pin the hoist so admission
selects with the same overhead-aware args the husam worker uses on retry.
"""
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _mock_pick(provider="anthropic", model_name="claude-sonnet-4-6"):
    fake_model = MagicMock(is_local=False, provider=provider)
    fake_model.name = model_name
    return MagicMock(model=fake_model, composite=0.6)


def _breakdown(scalar: float):
    bd = MagicMock()
    bd.scalar = scalar
    return bd


def _overhead_task(tid=5, agent_type="self_reflect", inner_difficulty=6):
    return {
        "id": tid, "priority": 1, "difficulty": 5,  # top-level DEFAULT (wrong)
        "agent_type": agent_type, "kind": "overhead",
        "created_at": time.time(), "status": "ready",
        "context": {"llm_call": {
            "raw_dispatch": True,
            "call_category": "overhead",
            "task": "reviewer",
            "agent_type": agent_type,
            "difficulty": inner_difficulty,  # the value that should win
            "prefer_speed": True,
        }},
    }


async def _run_admission(task, captured):
    import general_beckman

    def _cap(*a, **k):
        captured.clear()
        captured.update(k)
        return _mock_pick()

    # Defeat the cross-tick admission fingerprint cache (module globals persist
    # between tests in one session and would short-circuit later candidates).
    general_beckman._last_admission_admitted = True
    general_beckman._last_admission_fp = None

    snap = MagicMock()
    snap.pressure_for = MagicMock(return_value=_breakdown(0.7))
    with patch("general_beckman.queue.pick_ready_top_k",
               new=AsyncMock(return_value=[task])), \
         patch("general_beckman._claim_task", new=AsyncMock(return_value=True)), \
         patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("fatih_hoca.select", side_effect=_cap), \
         patch("nerd_herd.refresh_snapshot",
               new=AsyncMock(return_value=snap), create=True):
        return await general_beckman.next_task()


@pytest.mark.asyncio
async def test_overhead_call_category_hoisted():
    captured = {}
    out = await _run_admission(_overhead_task(), captured)
    assert out is not None
    assert captured.get("call_category") == "overhead"


@pytest.mark.asyncio
async def test_overhead_inner_difficulty_overrides_top_level():
    captured = {}
    await _run_admission(_overhead_task(inner_difficulty=6), captured)
    # inner difficulty 6 must beat the top-level default 5
    assert captured.get("difficulty") == 6


@pytest.mark.asyncio
async def test_overhead_disables_thinking_and_prefers_speed():
    captured = {}
    await _run_admission(_overhead_task(), captured)
    assert captured.get("needs_thinking") is False
    assert captured.get("prefer_speed") is True


@pytest.mark.asyncio
async def test_main_work_unaffected_no_llm_call():
    """A plain agent row (no context.llm_call) keeps main_work defaults."""
    captured = {}
    task = {
        "id": 9, "priority": 5, "difficulty": 7,
        "agent_type": "coder", "created_at": time.time(), "status": "ready",
    }
    await _run_admission(task, captured)
    # call_category not forced to overhead; difficulty from top-level row
    assert captured.get("call_category") in (None, "main_work")
    assert captured.get("difficulty") == 7
