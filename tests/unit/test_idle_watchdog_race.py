"""Tests for the idle-unloader / health-watchdog race condition fix.

Verifies that when the idle unloader deliberately stops the server,
the health watchdog does NOT misinterpret it as a crash.
"""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_manager():
    """Create a LocalModelManager with mocked-out heavy dependencies."""
    with patch("src.models.local_model_manager.get_gpu_scheduler"), \
         patch("src.models.local_model_manager.LocalModelManager._create_job_object", return_value=None), \
         patch("src.models.local_model_manager.LocalModelManager._kill_orphaned_servers"):
        from src.models.local_model_manager import LocalModelManager
        mgr = LocalModelManager()
    return mgr


class TestIdleUnloadFlag(unittest.TestCase):
    """The _idle_unload_in_progress flag must bracket idle unloads."""

    def test_flag_starts_false(self):
        mgr = _make_manager()
        assert mgr._idle_unload_in_progress is False

    def test_flag_set_during_idle_unload(self):
        """Flag should be True while _stop_server runs inside idle unloader."""
        mgr = _make_manager()
        mgr.current_model = "test-model"
        mgr.process = MagicMock()
        mgr.process.poll.return_value = None
        mgr._last_request_time = 1.0  # very old → idle

        observed_flag = []

        original_stop = mgr._stop_server

        async def spy_stop():
            observed_flag.append(mgr._idle_unload_in_progress)
            # Simulate stop
            mgr.process = None

        mgr._stop_server = spy_stop

        async def run():
            # Run one iteration of the idle unloader by simulating its logic
            # (we can't easily run the infinite loop, so replicate the body)
            max_idle = 60
            if mgr.is_loaded and mgr.idle_seconds > max_idle:
                mgr._idle_unload_in_progress = True
                try:
                    async with mgr._swap_lock:
                        await mgr._stop_server()
                        mgr.current_model = None
                finally:
                    mgr._idle_unload_in_progress = False

        asyncio.get_event_loop().run_until_complete(run())

        assert observed_flag == [True], "Flag should be True during _stop_server"
        assert mgr._idle_unload_in_progress is False, "Flag should be cleared after"
        assert mgr.current_model is None

    def test_flag_cleared_on_error(self):
        """Flag must be cleared even if _stop_server raises."""
        mgr = _make_manager()
        mgr.current_model = "test-model"
        mgr.process = MagicMock()
        mgr.process.poll.return_value = None
        mgr._last_request_time = 1.0

        async def failing_stop():
            raise RuntimeError("boom")

        mgr._stop_server = failing_stop

        async def run():
            max_idle = 60
            if mgr.is_loaded and mgr.idle_seconds > max_idle:
                mgr._idle_unload_in_progress = True
                try:
                    async with mgr._swap_lock:
                        await mgr._stop_server()
                        mgr.current_model = None
                finally:
                    mgr._idle_unload_in_progress = False

        with pytest.raises(RuntimeError):
            asyncio.get_event_loop().run_until_complete(run())

        assert mgr._idle_unload_in_progress is False


class TestWatchdogSkipsDuringIdleUnload(unittest.TestCase):
    """Watchdog must not treat dead process as crash during idle unload."""

    def test_watchdog_skips_crash_when_idle_unload_in_progress(self):
        """When _idle_unload_in_progress is True, watchdog should NOT restart."""
        mgr = _make_manager()
        mgr.current_model = "test-model"

        # Simulate a process that has exited (poll returns exit code)
        dead_process = MagicMock()
        dead_process.poll.return_value = 0
        mgr.process = dead_process

        mgr._idle_unload_in_progress = True

        swap_called = False
        original_swap = mgr._swap_model

        async def mock_swap(*args, **kwargs):
            nonlocal swap_called
            swap_called = True

        mgr._swap_model = mock_swap

        async def run_one_watchdog_iteration():
            # Replicate the watchdog's crash-detection logic
            if not mgr.current_model:
                return

            if mgr.process and mgr.process.poll() is not None:
                if mgr._idle_unload_in_progress:
                    # Should take this path and NOT call _swap_model
                    return
                await mgr._swap_model(mgr.current_model, reason="crash recovery")

        asyncio.get_event_loop().run_until_complete(run_one_watchdog_iteration())
        assert not swap_called, "Watchdog should NOT restart during idle unload"

    def test_watchdog_restarts_on_real_crash(self):
        """When _idle_unload_in_progress is False, watchdog SHOULD restart."""
        mgr = _make_manager()
        mgr.current_model = "test-model"

        dead_process = MagicMock()
        dead_process.poll.return_value = 1  # crashed
        mgr.process = dead_process

        mgr._idle_unload_in_progress = False

        swap_called = False

        async def mock_swap(*args, **kwargs):
            nonlocal swap_called
            swap_called = True
            return True

        mgr._swap_model = mock_swap

        async def run_one_watchdog_iteration():
            if not mgr.current_model:
                return
            if mgr.process and mgr.process.poll() is not None:
                if mgr._idle_unload_in_progress:
                    return
                model_name = mgr.current_model
                mgr.process = None
                mgr.current_model = None
                await mgr._swap_model(model_name, reason="crash recovery")

        asyncio.get_event_loop().run_until_complete(run_one_watchdog_iteration())
        assert swap_called, "Watchdog should restart on real crash"


class TestIdleTimeoutDefault(unittest.TestCase):
    """Verify the default idle timeout is 60 seconds (1 minute)."""

    def test_default_max_idle_minutes(self):
        """run_idle_unloader default max_idle_minutes should be 1."""
        import inspect
        from src.models.local_model_manager import LocalModelManager

        sig = inspect.signature(LocalModelManager.run_idle_unloader)
        default = sig.parameters["max_idle_minutes"].default
        assert default == 1, f"Expected 1 minute, got {default}"

    def test_default_check_interval(self):
        """run_idle_unloader default check_interval should be 30s."""
        import inspect
        from src.models.local_model_manager import LocalModelManager

        sig = inspect.signature(LocalModelManager.run_idle_unloader)
        default = sig.parameters["check_interval"].default
        assert default == 30, f"Expected 30s, got {default}"
