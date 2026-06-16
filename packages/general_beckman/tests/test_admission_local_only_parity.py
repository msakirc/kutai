"""Admission must select with the SAME local_only constraint the worker enforces.

Live regression (2026-06-16): admission built its ``fatih_hoca.select`` kwargs
by hand and never passed ``local_only``. A task the worker treats as
local_only (PII heuristic / classifier / context) was therefore admitted onto
a CLOUD model at admission, then categorically refused by the worker's
re-select (which DOES pass local_only) -> ``ModelCallFailed`` -> Beckman
re-pend -> admission re-picks cloud -> infinite admit-then-refuse loop.

The selector is meant to be the single selection mechanism; admission and the
worker must hand it identical constraints. This pins ``local_only`` parity.
"""
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _task(tid=1, priority=7, agent_type="analyst"):
    return {
        "id": tid, "priority": priority, "difficulty": 5,
        "agent_type": agent_type, "created_at": time.time(),
        "status": "ready",
        "title": "vault_summary",
        "description": "Summarize my password vault entries.",
    }


def _mock_local_pick():
    fake_model = MagicMock(is_local=True, provider="local")
    fake_model.name = "Qwen3.5-9B"
    return MagicMock(model=fake_model, composite=0.6)


@pytest.mark.asyncio
async def test_admission_passes_local_only_for_sensitive_task():
    import general_beckman

    captured = {}

    def _capture_select(*a, **k):
        captured.update(k)
        return _mock_local_pick()

    snap = MagicMock()
    snap.pressure_for = MagicMock(return_value=MagicMock(scalar=0.7))
    with patch("general_beckman.queue.pick_ready_top_k", new=AsyncMock(return_value=[_task()])), \
         patch("general_beckman._claim_task", new=AsyncMock(return_value=True)), \
         patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("fatih_hoca.select", side_effect=_capture_select), \
         patch("nerd_herd.refresh_snapshot", new=AsyncMock(return_value=snap), create=True):
        await general_beckman.next_task()

    assert captured, "fatih_hoca.select was never called"
    assert captured.get("local_only") is True, (
        "admission must pass local_only=True for a sensitive task so it does "
        "not admit onto a cloud model the worker will refuse"
    )
