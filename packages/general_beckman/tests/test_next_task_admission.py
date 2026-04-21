import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _task(tid=1, priority=5, difficulty=5, agent_type="coder"):
    return {
        "id": tid, "priority": priority, "difficulty": difficulty,
        "agent_type": agent_type, "created_at": time.time(),
        "status": "ready",
    }


def _mock_pick(provider="anthropic", model_name="claude-sonnet-4-6"):
    fake_model = MagicMock(is_local=False, provider=provider)
    fake_model.name = model_name
    return MagicMock(model=fake_model, composite=0.6)


@pytest.mark.asyncio
async def test_admits_when_pool_abundant():
    import general_beckman

    snap = MagicMock()
    snap.pressure_for = MagicMock(return_value=0.7)
    with patch("general_beckman.queue.pick_ready_top_k", new=AsyncMock(return_value=[_task(1, priority=7)])), \
         patch("general_beckman._currently_dispatched_count", new=AsyncMock(return_value=0)), \
         patch("general_beckman._claim_task", new=AsyncMock(return_value=True)), \
         patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("general_beckman.posthook_migration.run", new=AsyncMock(return_value=None)), \
         patch("fatih_hoca.select", return_value=_mock_pick()), \
         patch("nerd_herd.snapshot", return_value=snap, create=True):
        out = await general_beckman.next_task()
    assert out is not None
    assert out["id"] == 1
    assert out.get("preselected_pick") is not None


@pytest.mark.asyncio
async def test_holds_when_depleted_and_low_priority():
    import general_beckman

    snap = MagicMock()
    snap.pressure_for = MagicMock(return_value=-0.8)
    with patch("general_beckman.queue.pick_ready_top_k", new=AsyncMock(return_value=[_task(1, priority=3)])), \
         patch("general_beckman._currently_dispatched_count", new=AsyncMock(return_value=0)), \
         patch("general_beckman._claim_task", new=AsyncMock(return_value=True)), \
         patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("general_beckman.posthook_migration.run", new=AsyncMock(return_value=None)), \
         patch("fatih_hoca.select", return_value=_mock_pick()), \
         patch("nerd_herd.snapshot", return_value=snap, create=True):
        out = await general_beckman.next_task()
    assert out is None


@pytest.mark.asyncio
async def test_skips_candidate_when_hoca_none():
    import general_beckman

    snap = MagicMock()
    snap.pressure_for = MagicMock(return_value=0.7)
    tasks = [_task(1), _task(2)]
    picks = iter([None, _mock_pick()])
    with patch("general_beckman.queue.pick_ready_top_k", new=AsyncMock(return_value=tasks)), \
         patch("general_beckman._currently_dispatched_count", new=AsyncMock(return_value=0)), \
         patch("general_beckman._claim_task", new=AsyncMock(return_value=True)), \
         patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("general_beckman.posthook_migration.run", new=AsyncMock(return_value=None)), \
         patch("fatih_hoca.select", side_effect=lambda *a, **k: next(picks)), \
         patch("nerd_herd.snapshot", return_value=snap, create=True):
        out = await general_beckman.next_task()
    assert out is not None
    assert out["id"] == 2


@pytest.mark.asyncio
async def test_hard_cap_returns_none():
    import general_beckman
    with patch("general_beckman._currently_dispatched_count", new=AsyncMock(return_value=99)):
        out = await general_beckman.next_task()
    assert out is None
