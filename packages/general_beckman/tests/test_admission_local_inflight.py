"""Regression tests for the local in-flight admission gate.

Bug history: 4 local researchers were admitted concurrently because a
legacy BECKMAN_HARD_CAP=4 short-circuit allowed them and the pressure
gate was dead. Fix wired admission to snap.pressure_for(pick.model);
for local, any entry with is_local=True in in_flight_calls yields a
-1.0 local pressure, which must fail even a max-urgency threshold.
"""
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _task(tid=1, priority=5, difficulty=5, agent_type="researcher"):
    return {
        "id": tid, "priority": priority, "difficulty": difficulty,
        "agent_type": agent_type, "created_at": time.time(),
        "status": "ready",
    }


def _mock_local_pick(model_name="qwen3-8b"):
    fake_model = MagicMock(is_local=True, provider="llama.cpp")
    fake_model.name = model_name
    return MagicMock(model=fake_model, composite=0.8)


def _breakdown(scalar: float):
    """Return a mock PressureBreakdown with the given scalar."""
    bd = MagicMock()
    bd.scalar = scalar
    return bd


def _mock_snapshot_local_busy():
    """Snapshot that mirrors real `_local_pressure`: -1.0 when any is_local."""
    snap = MagicMock()
    fake_inflight = MagicMock(is_local=True, provider="llama.cpp", model="qwen3-8b")
    snap.in_flight_calls = [fake_inflight]
    # Match real type semantics: local pick → -1.0 when any in_flight_calls is_local.
    def _pressure_for(model, **kwargs):
        if getattr(model, "is_local", False):
            scalar = -1.0 if any(c.is_local for c in snap.in_flight_calls) else 0.0
        else:
            scalar = 1.0
        return _breakdown(scalar)
    snap.pressure_for = _pressure_for
    return snap


@pytest.mark.asyncio
async def test_local_pick_rejected_when_local_already_in_flight():
    """Regression: 4 local researchers admitted concurrently (BECKMAN_HARD_CAP bug).

    When in_flight_calls contains a local entry, pressure_for(local_model)
    is -1.0. Threshold at priority=5 is 0.0. Admission must REJECT.
    """
    import general_beckman

    snap = _mock_snapshot_local_busy()
    with patch("general_beckman.queue.pick_ready_top_k",
               new=AsyncMock(return_value=[_task(1, priority=5, agent_type="researcher")])), \
         patch("general_beckman._claim_task", new=AsyncMock(return_value=True)), \
         patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("fatih_hoca.select", return_value=_mock_local_pick()), \
         patch("nerd_herd.refresh_snapshot", new=AsyncMock(return_value=snap), create=True):
        out = await general_beckman.next_task()
    assert out is None, "local task must be held when another local call is in-flight"


@pytest.mark.asyncio
async def test_local_pick_rejected_even_at_max_urgency():
    """Pressure -1.0 is below every threshold (min threshold = -0.5)."""
    import general_beckman

    snap = _mock_snapshot_local_busy()
    # priority=10 → urgency=1.0 → threshold = -0.5. -1.0 < -0.5 → still REJECT.
    with patch("general_beckman.queue.pick_ready_top_k",
               new=AsyncMock(return_value=[_task(1, priority=10, agent_type="researcher")])), \
         patch("general_beckman._claim_task", new=AsyncMock(return_value=True)), \
         patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("fatih_hoca.select", return_value=_mock_local_pick()), \
         patch("nerd_herd.refresh_snapshot", new=AsyncMock(return_value=snap), create=True):
        out = await general_beckman.next_task()
    assert out is None, "local in-flight must block even max-urgency local picks"


@pytest.mark.asyncio
async def test_sequential_local_admits_then_rejects_after_begin_call():
    """Regression: the slot persists across ReAct iterations.

    1. Empty in-flight → admit succeeds.
    2. Dispatcher _begin_call registers a task slot.
    3. Next Beckman tick with another local candidate must REJECT
       because _local_pressure sees the task slot's is_local=True entry.
    """
    import general_beckman
    import src.core.in_flight as in_flight_mod
    import src.core.llm_dispatcher as dispatcher_mod

    # Clear in-flight registries in case a prior test left residue.
    in_flight_mod._task_slots.clear()
    in_flight_mod._call_entries.clear()

    # Build a live snapshot whose in_flight_calls reflects dispatcher state.
    snap = MagicMock()
    def _current_in_flight():
        return [
            MagicMock(is_local=e.is_local, provider=e.provider, model=e.model)
            for e in list(in_flight_mod._task_slots.values())
            + list(in_flight_mod._call_entries.values())
        ]
    # `snap.in_flight_calls` is read inside _local_pressure in real code; here
    # we replicate that contract so pressure_for reads the dispatcher's state.
    type(snap).in_flight_calls = property(lambda self: _current_in_flight())
    def _pressure_for(model, **kwargs):
        if getattr(model, "is_local", False):
            scalar = -1.0 if any(c.is_local for c in snap.in_flight_calls) else 0.0
        else:
            scalar = 1.0
        return _breakdown(scalar)
    snap.pressure_for = _pressure_for

    # Phase 1: empty in-flight → admit.
    with patch("general_beckman.queue.pick_ready_top_k",
               new=AsyncMock(return_value=[_task(1, priority=5)])), \
         patch("general_beckman._claim_task", new=AsyncMock(return_value=True)), \
         patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("fatih_hoca.select", return_value=_mock_local_pick()), \
         patch("nerd_herd.refresh_snapshot", new=AsyncMock(return_value=snap), create=True):
        first = await general_beckman.next_task()
    assert first is not None and first["id"] == 1

    # Phase 2: dispatcher registers the slot (as it does in real request()).
    # Suppress the nerd_herd push since we test snapshot-reading contract only.
    with patch("src.core.in_flight._push", new=AsyncMock()):
        await dispatcher_mod._begin_call(
            category="main_work", model_name="qwen3-8b",
            provider="llama.cpp", is_local=True, task_id=1,
        )

    # Phase 3: a second candidate must be rejected.
    with patch("general_beckman.queue.pick_ready_top_k",
               new=AsyncMock(return_value=[_task(2, priority=5)])), \
         patch("general_beckman._claim_task", new=AsyncMock(return_value=True)), \
         patch("general_beckman.cron.fire_due", new=AsyncMock(return_value=None)), \
         patch("fatih_hoca.select", return_value=_mock_local_pick()), \
         patch("nerd_herd.refresh_snapshot", new=AsyncMock(return_value=snap), create=True):
        second = await general_beckman.next_task()
    assert second is None, "second local task must be held while slot is occupied"

    # Cleanup.
    with patch("src.core.in_flight._push", new=AsyncMock()):
        await dispatcher_mod.release_task(1)
