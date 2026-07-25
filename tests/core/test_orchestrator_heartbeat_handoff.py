"""TDD: startup-heartbeat handoff.

The boot fix (2026-07-25) stops cancelling the startup heartbeat before the
vector-store load. Instead the orchestrator cancels it *after* creating its own
heartbeat task, via _handoff_heartbeat — which must tolerate a missing or
already-finished predecessor so a fast/edge boot never raises.
"""
import asyncio

import pytest


async def test_handoff_cancels_live_predecessor():
    from src.core.orchestrator import _handoff_heartbeat

    pred = asyncio.create_task(asyncio.sleep(3600))
    await asyncio.sleep(0)  # let it start running

    _handoff_heartbeat(pred)

    with pytest.raises(asyncio.CancelledError):
        await pred
    assert pred.cancelled()


async def test_handoff_tolerates_none():
    from src.core.orchestrator import _handoff_heartbeat

    _handoff_heartbeat(None)  # no startup task to hand off — must not raise


async def test_handoff_tolerates_done_predecessor():
    from src.core.orchestrator import _handoff_heartbeat

    async def _quick():
        return 1

    pred = asyncio.create_task(_quick())
    await pred  # predecessor already finished
    _handoff_heartbeat(pred)  # must not re-cancel / raise on a done task
