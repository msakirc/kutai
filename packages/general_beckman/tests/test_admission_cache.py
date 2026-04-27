"""Admission-cache short-circuit: skip Hoca scan when state unchanged."""
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _task(tid=1, priority=5, agent_type="coder"):
    return {
        "id": tid, "priority": priority, "difficulty": 5,
        "agent_type": agent_type, "created_at": time.time(),
        "status": "ready", "worker_attempts": 0, "max_worker_attempts": 6,
    }


def _stable_snap():
    """Snap with deterministic, hashable in_flight + cloud state."""
    snap = MagicMock()
    snap.in_flight_calls = []
    snap.local = MagicMock()
    snap.local.model_name = "qwen-9b"
    snap.local.idle_seconds = 5.0
    snap.local.is_swapping = False
    snap.cloud = {}
    snap.pressure_for = MagicMock(return_value=-1.0)  # always reject
    return snap


@pytest.mark.asyncio
async def test_cache_skips_redundant_scan_when_state_unchanged():
    """After a REJECT tick, an immediate identical tick must skip Hoca."""
    import general_beckman

    # Reset module cache.
    general_beckman._last_admission_fp = None
    general_beckman._last_admission_admitted = True

    cands = [_task(1)]
    snap = _stable_snap()
    pick = MagicMock(composite=0.6)
    pick.model = MagicMock(is_local=True, provider="local")
    pick.model.name = "qwen-9b"

    select_spy = MagicMock(return_value=pick)
    with patch("general_beckman.queue.pick_ready_top_k", new=AsyncMock(return_value=cands)), \
         patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("fatih_hoca.select", new=select_spy), \
         patch("nerd_herd.refresh_snapshot", new=AsyncMock(return_value=snap), create=True):
        # Tick 1: full scan, REJECT.
        out1 = await general_beckman.next_task()
        assert out1 is None
        first_calls = select_spy.call_count
        assert first_calls >= 1

        # Tick 2: identical state. Must short-circuit (no new fatih_hoca calls).
        out2 = await general_beckman.next_task()
        assert out2 is None
        assert select_spy.call_count == first_calls, "cache failed to skip second tick"


@pytest.mark.asyncio
async def test_cache_invalidates_when_in_flight_changes():
    """Snapshot delta (e.g. a slot frees) must trigger re-evaluation."""
    import general_beckman

    general_beckman._last_admission_fp = None
    general_beckman._last_admission_admitted = True

    cands = [_task(1)]
    snap_a = _stable_snap()
    fake_call = MagicMock(task_id=999, model="qwen-9b", provider="", is_local=True)
    snap_a.in_flight_calls = [fake_call]
    snap_b = _stable_snap()
    snap_b.in_flight_calls = []  # slot freed

    pick = MagicMock(composite=0.6)
    pick.model = MagicMock(is_local=True, provider="local")
    pick.model.name = "qwen-9b"
    select_spy = MagicMock(return_value=pick)

    with patch("general_beckman.queue.pick_ready_top_k", new=AsyncMock(return_value=cands)), \
         patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("fatih_hoca.select", new=select_spy):
        with patch("nerd_herd.refresh_snapshot", new=AsyncMock(return_value=snap_a), create=True):
            await general_beckman.next_task()
        first_calls = select_spy.call_count
        with patch("nerd_herd.refresh_snapshot", new=AsyncMock(return_value=snap_b), create=True):
            await general_beckman.next_task()
        assert select_spy.call_count > first_calls, "cache failed to invalidate after in_flight delta"


@pytest.mark.asyncio
async def test_cache_bypassed_for_mechanical_candidates():
    """Mechanical tasks must always be considered, even on cache hit."""
    import general_beckman

    general_beckman._last_admission_fp = None
    general_beckman._last_admission_admitted = False  # priming a "skip" state
    snap = _stable_snap()
    cands = [_task(1, agent_type="mechanical")]

    # Pre-seed cache with matching fingerprint.
    general_beckman._last_admission_fp = general_beckman._admission_fingerprint(snap, cands)

    with patch("general_beckman.queue.pick_ready_top_k", new=AsyncMock(return_value=cands)), \
         patch("general_beckman._claim_task", new=AsyncMock(return_value=True)), \
         patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("nerd_herd.refresh_snapshot", new=AsyncMock(return_value=snap), create=True):
        out = await general_beckman.next_task()
    assert out is not None, "mechanical task starved by cache short-circuit"
    assert out["id"] == 1
