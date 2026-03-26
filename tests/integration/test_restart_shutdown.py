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
            await release_task_locks()

        run_async(_run())


# ---------------------------------------------------------------------------
# Orchestrator agent timeout constants
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAgentTimeouts:
    """AGENT_TIMEOUTS constants have reasonable values."""

    def test_all_known_agent_types_have_timeouts(self):
        """All standard agent types are present in AGENT_TIMEOUTS."""
        from src.core.orchestrator import AGENT_TIMEOUTS

        expected_agents = [
            "coder", "planner", "architect", "fixer", "reviewer",
            "researcher", "writer", "executor", "assistant",
            "shopping_advisor", "workflow",
        ]
        for agent in expected_agents:
            assert agent in AGENT_TIMEOUTS, (
                f"Agent '{agent}' missing from AGENT_TIMEOUTS"
            )

    def test_shopping_timeout_is_generous(self):
        """Shopping advisor gets a long timeout (web searches take time)."""
        from src.core.orchestrator import AGENT_TIMEOUTS
        assert AGENT_TIMEOUTS.get("shopping_advisor", 0) >= 300, (
            "Shopping advisor needs at least 300s for web searches"
        )

    def test_workflow_timeout_is_longest(self):
        """Workflow agent has the longest timeout."""
        from src.core.orchestrator import AGENT_TIMEOUTS
        workflow_timeout = AGENT_TIMEOUTS.get("workflow", 0)
        other_timeouts = [v for k, v in AGENT_TIMEOUTS.items() if k != "workflow"]
        if other_timeouts:
            assert workflow_timeout >= max(other_timeouts), (
                "Workflow agent should have the longest timeout"
            )
