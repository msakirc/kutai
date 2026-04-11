"""Tests for yasar_usta.lock."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from yasar_usta.lock import is_pid_alive, acquire_lock, release_lock


class TestIsPidAlive:
    def test_current_process_is_alive(self):
        assert is_pid_alive(os.getpid()) is True

    def test_impossible_pid_is_dead(self):
        # PID 0 is kernel on unix, nonexistent on Windows
        # Use a very high PID unlikely to exist
        assert is_pid_alive(4_000_000) is False


class TestAcquireLock:
    def test_acquire_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            acquire_lock(tmp, name="test_guard")
            lock_file = Path(tmp) / "test_guard.lock"
            assert lock_file.exists()
            pid = int(lock_file.read_text().strip())
            assert pid == os.getpid()
            release_lock()

    def test_stale_lock_recovery(self):
        """If lock file has a dead PID, acquire should succeed."""
        with tempfile.TemporaryDirectory() as tmp:
            # Write a stale PID
            lock_file = Path(tmp) / "test_stale.lock"
            lock_file.write_text("0000000002")  # PID 2 is unlikely to be alive
            sentinel = Path(tmp) / "test_stale.lk"
            sentinel.write_text("L")
            # Should succeed (stale recovery)
            acquire_lock(tmp, name="test_stale")
            pid = int(lock_file.read_text().strip())
            assert pid == os.getpid()
            release_lock()
