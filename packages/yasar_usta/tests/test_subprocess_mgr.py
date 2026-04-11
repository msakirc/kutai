"""Tests for yasar_usta.subprocess_mgr."""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from yasar_usta.subprocess_mgr import SubprocessManager


class TestSubprocessManager:
    def setup_method(self):
        """Create a fresh event loop for each test."""
        self.loop = asyncio.new_event_loop()

    def teardown_method(self):
        """Close the event loop after each test."""
        self.loop.close()

    def run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_start_and_wait(self):
        """Start a trivial process, wait for clean exit."""
        python = sys.executable
        mgr = SubprocessManager(
            command=[python, "-c", "import sys; sys.exit(0)"],
            log_dir=tempfile.mkdtemp(),
        )
        self.run(mgr.start())
        assert mgr.running is True
        code = self.run(mgr.wait_for_exit())
        assert code == 0
        assert mgr.running is False

    def test_crash_exit_code(self):
        """Process that exits with code 1."""
        python = sys.executable
        mgr = SubprocessManager(
            command=[python, "-c", "import sys; sys.exit(1)"],
            log_dir=tempfile.mkdtemp(),
        )
        self.run(mgr.start())
        code = self.run(mgr.wait_for_exit())
        assert code == 1

    def test_stop_graceful(self):
        """Stop a long-running process gracefully."""
        python = sys.executable
        mgr = SubprocessManager(
            command=[python, "-c", "import time; time.sleep(60)"],
            log_dir=tempfile.mkdtemp(),
            stop_timeout=5,
        )
        self.run(mgr.start())
        assert mgr.running is True
        self.run(mgr.stop())
        assert mgr.running is False

    def test_stderr_capture(self):
        """Stderr lines are captured in the tail buffer."""
        python = sys.executable
        mgr = SubprocessManager(
            command=[python, "-c", "import sys; sys.stderr.write('error line\\n'); sys.exit(0)"],
            log_dir=tempfile.mkdtemp(),
        )
        self.run(mgr.start())
        self.run(mgr.wait_for_exit())
        # Give pipe reader a moment to finish
        self.run(asyncio.sleep(0.2))
        assert any("error line" in line for line in mgr.stderr_tail)

    def test_heartbeat_detection(self):
        """Heartbeat file check works."""
        with tempfile.TemporaryDirectory() as tmp:
            hb_file = Path(tmp) / "heartbeat"
            import time
            hb_file.write_text(str(time.time()))

            mgr = SubprocessManager(
                command=["echo", "noop"],
                log_dir=tmp,
                heartbeat_file=str(hb_file),
                heartbeat_stale_seconds=120,
            )
            assert mgr.is_heartbeat_stale() is False

            # Write old timestamp
            hb_file.write_text(str(time.time() - 200))
            assert mgr.is_heartbeat_stale() is True

    def test_no_heartbeat_file_not_stale(self):
        """Missing heartbeat file = not stale (still starting up)."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SubprocessManager(
                command=["echo", "noop"],
                log_dir=tmp,
                heartbeat_file=str(Path(tmp) / "nonexistent"),
            )
            assert mgr.is_heartbeat_stale() is False
