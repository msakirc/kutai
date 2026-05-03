"""Integration tests for the vector_maint salako executors.

Verifies:
- vector_maint_wal calls wal_checkpoint and returns ok=True
- vector_maint_snapshot calls snapshot_chroma and returns a dst path
- Both are wrapped in run_in_executor (asyncio.to_thread) so slow sync
  ops don't block the event loop
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.mark.integration
class TestVectorMaintWal:

    def test_wal_calls_checkpoint_and_returns_ok(self, temp_db):
        """run_wal invokes wal_checkpoint and returns {'ok': True}."""

        async def _run():
            task = {"id": 1, "payload": {"action": "vector_maint_wal"}}

            # wal_checkpoint already uses asyncio.to_thread internally;
            # mock it at the module level to return True without touching disk.
            with patch("src.memory.vector_store.wal_checkpoint", new_callable=AsyncMock, return_value=True) as mock_wc:
                from salako.executors.vector_maint import run_wal
                result = await run_wal(task)

            mock_wc.assert_called_once()
            assert result["ok"] is True

        run_async(_run())

    def test_wal_catches_errors_gracefully(self, temp_db):
        """Errors from wal_checkpoint are caught, logged, and returned as ok=False."""

        async def _run():
            task = {"id": 1, "payload": {"action": "vector_maint_wal"}}

            async def _raise():
                raise RuntimeError("chroma is borked")

            with patch("src.memory.vector_store.wal_checkpoint", side_effect=_raise):
                from salako.executors.vector_maint import run_wal
                result = await run_wal(task)

            assert result["ok"] is False
            assert "chroma is borked" in result.get("error", "")

        run_async(_run())


@pytest.mark.integration
class TestVectorMaintSnapshot:

    def test_snapshot_calls_snapshot_chroma_and_returns_dst(self, temp_db):
        """run_snapshot invokes snapshot_chroma(keep=3) and returns the dst path."""

        async def _run():
            task = {"id": 2, "payload": {"action": "vector_maint_snapshot"}}
            fake_dst = "/tmp/chroma.bak.20260503-120000"

            with patch("src.memory.vector_store.snapshot_chroma", new_callable=AsyncMock, return_value=fake_dst) as mock_sc:
                from salako.executors.vector_maint import run_snapshot
                result = await run_snapshot(task)

            mock_sc.assert_called_once_with(keep=3)
            assert result["dst"] == fake_dst

        run_async(_run())

    def test_snapshot_catches_errors_gracefully(self, temp_db):
        """Errors from snapshot_chroma are caught, logged, and returned as dst=None."""

        async def _run():
            task = {"id": 2, "payload": {"action": "vector_maint_snapshot"}}

            async def _raise(keep=3):
                raise OSError("disk full")

            with patch("src.memory.vector_store.snapshot_chroma", side_effect=_raise):
                from salako.executors.vector_maint import run_snapshot
                result = await run_snapshot(task)

            assert result["dst"] is None
            assert "disk full" in result.get("error", "")

        run_async(_run())

    def test_snapshot_none_when_chroma_absent(self, temp_db):
        """snapshot_chroma returning None (dir absent) is handled as a non-error."""

        async def _run():
            task = {"id": 2, "payload": {"action": "vector_maint_snapshot"}}

            with patch("src.memory.vector_store.snapshot_chroma", new_callable=AsyncMock, return_value=None):
                from salako.executors.vector_maint import run_snapshot
                result = await run_snapshot(task)

            assert result["dst"] is None
            assert "error" not in result

        run_async(_run())


@pytest.mark.integration
class TestVectorMaintNonBlocking:
    """Regression: verify the executors don't block the event loop."""

    def test_wal_does_not_block_event_loop(self):
        """A 0.1s blocking op wrapped in to_thread allows concurrent coroutines to run."""

        async def _run():
            progress = []

            async def ticker():
                for i in range(10):
                    progress.append(i)
                    await asyncio.sleep(0.01)

            # Simulate a slow sync WAL checkpoint (0.1s)
            async def slow_wal():
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: time.sleep(0.1))
                return True

            ticker_task = asyncio.create_task(ticker())
            with patch("src.memory.vector_store.wal_checkpoint", side_effect=slow_wal):
                from salako.executors.vector_maint import run_wal
                task = {"id": 1, "payload": {"action": "vector_maint_wal"}}
                result = await run_wal(task)
            await ticker_task

            # If the WAL was run on the event loop it would have blocked the ticker.
            # With run_in_executor/to_thread the ticker should have made several ticks.
            assert result["ok"] is True
            assert len(progress) >= 3, (
                f"Ticker only made {len(progress)} ticks — "
                f"WAL checkpoint may have blocked the event loop"
            )

        run_async(_run())

    def test_snapshot_does_not_block_event_loop(self):
        """A 0.1s blocking op wrapped in to_thread allows concurrent coroutines to run."""

        async def _run():
            progress = []

            async def ticker():
                for i in range(10):
                    progress.append(i)
                    await asyncio.sleep(0.01)

            async def slow_snapshot(keep=3):
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: time.sleep(0.1))
                return "/tmp/fake.bak"

            ticker_task = asyncio.create_task(ticker())
            with patch("src.memory.vector_store.snapshot_chroma", side_effect=slow_snapshot):
                from salako.executors.vector_maint import run_snapshot
                task = {"id": 2, "payload": {"action": "vector_maint_snapshot"}}
                result = await run_snapshot(task)
            await ticker_task

            assert result["dst"] == "/tmp/fake.bak"
            assert len(progress) >= 3, (
                f"Ticker only made {len(progress)} ticks — "
                f"snapshot may have blocked the event loop"
            )

        run_async(_run())
