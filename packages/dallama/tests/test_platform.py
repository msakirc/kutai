"""Tests for PlatformHelper — OS-specific process management."""
import subprocess
import sys
from unittest.mock import MagicMock, patch
import pytest
from dallama.platform import PlatformHelper

@pytest.fixture
def helper():
    return PlatformHelper()

def test_create_process_returns_popen(helper, tmp_path):
    stderr_path = str(tmp_path / "stderr.log")
    cmd = [sys.executable, "-c", "import time; time.sleep(10)"]
    proc = helper.create_process(cmd, stderr_path)
    assert isinstance(proc, subprocess.Popen)
    assert proc.poll() is None
    proc.kill()
    proc.wait()

def test_create_process_writes_stderr(helper, tmp_path):
    stderr_path = str(tmp_path / "stderr.log")
    cmd = [sys.executable, "-c", "import sys; sys.stderr.write('hello\\n')"]
    proc = helper.create_process(cmd, stderr_path)
    proc.wait(timeout=5)
    with open(stderr_path) as f:
        assert "hello" in f.read()

@pytest.mark.asyncio
async def test_graceful_stop_terminates(helper, tmp_path):
    stderr_path = str(tmp_path / "stderr.log")
    cmd = [sys.executable, "-c", "import time; time.sleep(60)"]
    proc = helper.create_process(cmd, stderr_path)
    assert proc.poll() is None
    await helper.graceful_stop(proc, timeout=5)
    assert proc.poll() is not None

@pytest.mark.asyncio
async def test_graceful_stop_force_kills_on_timeout(helper, tmp_path):
    stderr_path = str(tmp_path / "stderr.log")
    code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    cmd = [sys.executable, "-c", code]
    proc = helper.create_process(cmd, stderr_path)
    await helper.graceful_stop(proc, timeout=2)
    assert proc.poll() is not None

def test_kill_orphans_no_crash(helper):
    helper.kill_orphans("nonexistent-process-name-xyz")
