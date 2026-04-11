"""Tests for yasar_usta.heartbeat."""

import asyncio
import os
import tempfile
import time
from pathlib import Path

from yasar_usta.heartbeat import (
    EXIT_RESTART,
    EXIT_STOP,
    HeartbeatWriter,
    write_heartbeat,
)


class TestConstants:
    def test_exit_restart(self):
        assert EXIT_RESTART == 42

    def test_exit_stop(self):
        assert EXIT_STOP == 0


class TestWriteHeartbeat:
    def test_writes_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "heartbeat")
            write_heartbeat(path)
            ts = float(Path(path).read_text())
            assert abs(ts - time.time()) < 2

    def test_writes_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "hb1")
            p2 = os.path.join(tmp, "hb2")
            write_heartbeat(p1, p2)
            assert Path(p1).exists()
            assert Path(p2).exists()

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "dir", "heartbeat")
            write_heartbeat(path)
            assert Path(path).exists()


class TestHeartbeatWriter:
    def test_writes_immediately_on_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "heartbeat")
            writer = HeartbeatWriter(path, interval=100)

            loop = asyncio.new_event_loop()
            try:
                task = loop.create_task(writer.run())
                loop.run_until_complete(asyncio.sleep(0.1))
                task.cancel()
                try:
                    loop.run_until_complete(task)
                except asyncio.CancelledError:
                    pass
            finally:
                loop.close()

            assert Path(path).exists()
            ts = float(Path(path).read_text())
            assert abs(ts - time.time()) < 2

    def test_cancel_stops_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "heartbeat")
            writer = HeartbeatWriter(path, interval=0.1)

            loop = asyncio.new_event_loop()
            try:
                task = loop.create_task(writer.run())
                loop.run_until_complete(asyncio.sleep(0.3))
                task.cancel()
                try:
                    loop.run_until_complete(task)
                except asyncio.CancelledError:
                    pass
            finally:
                loop.close()
            # Should not raise — writer exits cleanly on cancel
