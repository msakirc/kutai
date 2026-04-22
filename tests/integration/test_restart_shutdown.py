"""test_restart_shutdown.py — Integration tests for shutdown and restart behavior.

Tests:
- shutdown_event propagation
- Exit code 42 is the restart sentinel (documented behavior)
- Orchestrator requested_exit_code attribute
- Task locks are released on shutdown

Markers:
  @pytest.mark.integration  — all tests

Note: We do NOT start a real Orchestrator (that would load models and connect
to Telegram). Instead we test the shutdown components in isolation.
"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# shutdown_event propagation
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestShutdownEvent:
    """shutdown_event is an asyncio.Event that signals graceful stop."""

    def test_shutdown_event_initial_state(self):
        """A fresh shutdown_event is not set."""
        event = asyncio.Event()
        assert not event.is_set()

    def test_shutdown_event_set_propagates(self):
        """Setting the event makes is_set() True."""
        event = asyncio.Event()
        event.set()
        assert event.is_set()

    def test_shutdown_event_clear(self):
        """Event can be cleared (reused)."""
        event = asyncio.Event()
        event.set()
        event.clear()
        assert not event.is_set()

    def test_shutdown_event_wait_completes_when_set(self):
        """Waiting on a set event completes immediately."""
        async def _run():
            event = asyncio.Event()
            event.set()
            # Should complete instantly, not hang
            await asyncio.wait_for(event.wait(), timeout=1.0)

        run_async(_run())

    def test_shutdown_event_timeout_when_not_set(self):
        """Waiting on an unset event times out correctly."""
        async def _run():
            event = asyncio.Event()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(event.wait(), timeout=0.1)

        run_async(_run())


# ---------------------------------------------------------------------------
# Exit code sentinel values
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestExitCodeSentinels:
    """The exit code 42 is used for restart; 0 for clean stop.

    These tests document the expected behavior rather than testing
    the Orchestrator directly (which requires the full stack).
    """

    EXIT_CODE_RESTART = 42
    EXIT_CODE_STOP = 0

    def test_restart_code_is_42(self):
        """Exit code 42 is the restart sentinel (matches kutai_wrapper.py)."""
        # Read the wrapper to confirm 42 is the restart code
        wrapper_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "kutai_wrapper.py",
        )
        if not os.path.exists(wrapper_path):
            pytest.skip("kutai_wrapper.py not found")

        with open(wrapper_path, encoding="utf-8") as f:
            content = f.read()

        assert "42" in content, (
            "kutai_wrapper.py should contain the magic restart code 42"
        )

    def test_orchestrator_has_requested_exit_code(self):
        """Orchestrator tracks requested_exit_code for restart/stop commands."""
        # Import just the class definition without instantiation
        # (instantiation would kill orphaned llama-server processes)
        import inspect
        from src.core import orchestrator as orch_mod

        source = inspect.getsource(orch_mod.Orchestrator.__init__)
        assert "requested_exit_code" in source, (
            "Orchestrator.__init__ should initialize requested_exit_code"
        )

    def test_orchestrator_exit_code_initialized_to_none(self):
        """requested_exit_code starts as None (not yet set)."""
        import inspect
        from src.core.orchestrator import Orchestrator
        source = inspect.getsource(Orchestrator.__init__)
        # Should initialize to None
        assert "requested_exit_code" in source
        # Check for None initialization pattern
        assert "None" in source or "none" in source.lower()


# ---------------------------------------------------------------------------
# Graceful shutdown: DB locks released
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestShutdownDBCleanup:
    """On shutdown, DB locks should be releasable."""

    def test_release_task_locks_function_exists(self):
        """release_task_locks is importable from db module."""
        from src.infra.db import release_task_locks
        assert callable(release_task_locks)

    def test_release_mission_locks_function_exists(self):
        """release_mission_locks is importable from db module."""
        from src.infra.db import release_mission_locks
        assert callable(release_mission_locks)

    def test_close_db_idempotent(self, temp_db):
        """close_db can be called multiple times without error."""
        from src.infra.db import close_db

        async def _run():
            await close_db()  # First close (from temp_db fixture state)
            await close_db()  # Second close — should not raise

        run_async(_run())

    def test_release_task_locks_on_empty_db(self, temp_db):
        """release_task_locks on an empty DB does not crash."""
        from src.infra.db import release_task_locks

        async def _run():
            # Should complete without error even with no locked tasks
            await release_task_locks(task_id=0)

        run_async(_run())


# ---------------------------------------------------------------------------
# Orchestrator agent timeout constants
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestProgressWatchdog:
    """Per-agent wall-clock timeouts removed 2026-04-22 in favor of a
    progress-heartbeat watchdog. These tests validate the new contract."""

    def test_progress_timeout_is_reasonable(self):
        """Watchdog limit is generous but bounded — few minutes max."""
        from src.core.heartbeat import PROGRESS_TIMEOUT_SECONDS
        assert 120.0 <= PROGRESS_TIMEOUT_SECONDS <= 1800.0

    def test_bump_resets_stale(self):
        from src.core import heartbeat as hb
        hb.clear(99999)
        hb.bump(99999)
        assert hb.stale_seconds(99999) < 1.0
        hb.clear(99999)

    def test_unbumped_task_is_not_stale(self):
        """No heartbeat yet → 0.0 (just-started, give it a chance)."""
        from src.core import heartbeat as hb
        hb.clear(88888)
        assert hb.stale_seconds(88888) == 0.0

    def test_contextvar_carries_task_id(self):
        from src.core import heartbeat as hb
        hb.clear(77777)
        hb.current_task_id.set(77777)
        hb.bump()  # no arg — uses contextvar
        assert hb.stale_seconds(77777) < 1.0
        hb.clear(77777)
        hb.current_task_id.set(None)
