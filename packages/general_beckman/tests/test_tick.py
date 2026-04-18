"""Tests for general_beckman.tick."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_tick_calls_watchdog_check_stuck_tasks():
    """tick() must invoke watchdog.check_stuck_tasks on every call."""
    import general_beckman
    with patch("general_beckman.watchdog.check_stuck_tasks", new=AsyncMock()) as w:
        await general_beckman.tick()
    w.assert_awaited()


@pytest.mark.asyncio
async def test_tick_swallows_subroutine_exceptions():
    """tick() must not raise if a subroutine raises."""
    import general_beckman

    async def boom(*a, **kw):
        raise RuntimeError("x")

    with patch("general_beckman.watchdog.check_stuck_tasks", side_effect=boom):
        await general_beckman.tick()  # must not raise


@pytest.mark.asyncio
async def test_tick_runs_scheduled_jobs_if_orchestrator_registered():
    """If an orchestrator with scheduled_jobs is registered, tick() delegates
    to its tick_* entry points."""
    import general_beckman
    from general_beckman import lifecycle
    sj = MagicMock()
    sj.tick_todos = AsyncMock()
    sj.tick_api_discovery = AsyncMock()
    sj.tick_digest = AsyncMock()
    sj.tick_price_watches = AsyncMock()
    sj.tick_benchmark_refresh = AsyncMock()
    sj.check_scheduled_tasks = AsyncMock()
    fake_orch = MagicMock(scheduled_jobs=sj)
    lifecycle.set_orchestrator(fake_orch)
    try:
        with patch("general_beckman.watchdog.check_stuck_tasks", new=AsyncMock()):
            await general_beckman.tick()
        # At minimum the cheap-to-call tick methods are invoked.
        sj.tick_todos.assert_awaited()
    finally:
        lifecycle.set_orchestrator(None)
