"""Tests for src.core.heartbeat — bump, stale_seconds, keepalive context."""
from __future__ import annotations

import asyncio
import pytest

from src.core import heartbeat as hb


def setup_function() -> None:
    """Reset heartbeat state between tests."""
    hb._HEARTBEATS.clear()


def test_bump_records_monotonic_timestamp():
    hb.bump(123)
    assert hb.stale_seconds(123) < 1.0


def test_stale_seconds_no_bump_returns_zero():
    """A task with no bump yet is treated as just-started, not stale."""
    assert hb.stale_seconds(999) == 0.0


def test_clear_drops_entry():
    hb.bump(7)
    hb.clear(7)
    assert hb.stale_seconds(7) == 0.0


@pytest.mark.asyncio
async def test_keepalive_pumps_during_long_await():
    """User feedback 2026-05-01: cloud non-streaming calls + local model
    swaps don't bump per-token; long ops starve the 300s watchdog. The
    keepalive context manager runs a background task that bumps the
    active heartbeat at a fixed interval until exit."""
    hb.current_task_id.set(42)
    hb.bump(42)
    initial_stale = hb.stale_seconds(42)

    # Use a tight interval so the test runs fast. With interval=0.05 and a
    # 0.4s body, the pump fires ~6-7 times. Each fire resets stale.
    async with hb.keepalive(interval=0.05):
        await asyncio.sleep(0.4)
        # Inside the body, the heartbeat must NOT have gone stale by the
        # body length — pump kept it fresh.
        assert hb.stale_seconds(42) <= 0.1, (
            f"keepalive failed to refresh; stale={hb.stale_seconds(42)}"
        )


@pytest.mark.asyncio
async def test_keepalive_cancels_on_exit():
    """Background pump must stop firing once the with-block exits.
    Without this, leaked tasks would keep bumping cleared heartbeats."""
    hb.current_task_id.set(99)
    hb.bump(99)

    async with hb.keepalive(interval=0.02):
        await asyncio.sleep(0.05)

    last_after_exit = hb.stale_seconds(99)
    await asyncio.sleep(0.15)  # would bump 7+ times if pump leaked
    new_stale = hb.stale_seconds(99)
    assert new_stale > last_after_exit + 0.1, (
        "keepalive pump leaked beyond context manager exit"
    )


@pytest.mark.asyncio
async def test_keepalive_propagates_body_exception():
    """The pump must shut down cleanly even when the body raises."""
    hb.current_task_id.set(55)
    hb.bump(55)

    with pytest.raises(RuntimeError, match="boom"):
        async with hb.keepalive(interval=0.02):
            await asyncio.sleep(0.05)
            raise RuntimeError("boom")
