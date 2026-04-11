"""Tests for yasar_usta.sidecar."""

import os
import sys
import tempfile
import asyncio
from pathlib import Path

from yasar_usta.sidecar import SidecarManager


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestSidecarManager:
    def test_pid_alive_no_file(self):
        mgr = SidecarManager(name="test", command=["echo"])
        assert mgr.pid_alive() is None

    def test_pid_alive_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "test.pid"
            pid_file.write_text("4000000")  # unlikely PID
            mgr = SidecarManager(
                name="test",
                command=["echo"],
                pid_file=str(pid_file),
            )
            assert mgr.pid_alive() is None
            assert not pid_file.exists()  # cleaned up

    def test_pid_alive_current_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "test.pid"
            pid_file.write_text(str(os.getpid()))
            mgr = SidecarManager(
                name="test",
                command=["echo"],
                pid_file=str(pid_file),
            )
            assert mgr.pid_alive() == os.getpid()

    def test_start_with_no_command(self):
        mgr = SidecarManager(name="test", command=[])
        run_async(mgr.start())
        # Should not crash, just log warning

    def test_is_alive_no_pid_no_url(self):
        mgr = SidecarManager(name="test", command=["echo"])
        result = run_async(mgr.is_alive())
        assert result is False

    def test_is_alive_with_current_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "test.pid"
            pid_file.write_text(str(os.getpid()))
            mgr = SidecarManager(
                name="test",
                command=["echo"],
                pid_file=str(pid_file),
            )
            result = run_async(mgr.is_alive())
            assert result is True

    def test_ensure_calls_start_when_dead(self):
        """ensure() should attempt start when not alive."""
        mgr = SidecarManager(name="test", command=[])
        # command is empty so start will log warning but not crash
        run_async(mgr.ensure())

    def test_stop_noop_when_no_pid(self):
        mgr = SidecarManager(name="test", command=["echo"])
        run_async(mgr.stop())
        # Should not crash

    def test_name_attribute(self):
        mgr = SidecarManager(name="my-sidecar", command=["python", "-m", "http.server"])
        assert mgr.name == "my-sidecar"

    def test_pid_file_path_conversion(self):
        mgr = SidecarManager(name="test", command=["echo"], pid_file="/tmp/test.pid")
        assert mgr.pid_file == Path("/tmp/test.pid")

    def test_no_pid_file_stays_none(self):
        mgr = SidecarManager(name="test", command=["echo"])
        assert mgr.pid_file is None
