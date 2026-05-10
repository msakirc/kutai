"""Z10-T3B — per-mission container lifecycle.

Tests the three helpers introduced in Phase E:

* :func:`mission_container_name` — naming + legacy fallback warning.
* :func:`ensure_mission_container` — idempotent create-or-reuse.
* :func:`teardown_mission_container` — tolerates missing container.

Docker is mocked at the ``_run_quiet`` boundary so the suite runs on
hosts without a Docker daemon (CI, dev laptops without WSL).
"""
from __future__ import annotations

from typing import Iterator
from unittest.mock import MagicMock

import pytest

from src.tools import shell


class _DockerStub:
    """Records ``_run_quiet`` invocations + replays a scripted result list."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.script: list[tuple[int, str, str]] = []

    async def __call__(self, *args: str):
        self.calls.append(args)
        if self.script:
            return self.script.pop(0)
        return 1, "", "no such container"


@pytest.fixture
def stub_run_quiet(monkeypatch) -> Iterator[_DockerStub]:
    stub = _DockerStub()
    monkeypatch.setattr(shell, "_run_quiet", stub)
    yield stub


def test_mission_container_name_with_id() -> None:
    assert shell.mission_container_name(47) == "kutai-mission-47"
    assert shell.mission_container_name(1) == "kutai-mission-1"


def test_mission_container_name_legacy_fallback(monkeypatch, caplog) -> None:
    fake = MagicMock()
    monkeypatch.setattr(shell, "logger", fake)
    name = shell.mission_container_name(None)
    assert name == shell.CONTAINER_NAME
    # Warning emitted explaining the fallback.
    fake.warning.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_mission_container_idempotent_running(
    stub_run_quiet, monkeypatch
) -> None:
    """Second call should be a no-op when the container is already up."""
    # docker inspect → State.Running = true
    stub_run_quiet.script[:] = [(0, "true", ""), (0, "true", "")]

    ok1 = await shell.ensure_mission_container(47)
    ok2 = await shell.ensure_mission_container(47)
    assert ok1 is True
    assert ok2 is True
    # No ``docker run`` invocations — only inspects.
    run_calls = [c for c in stub_run_quiet.calls if c[:3] == ("docker", "run", "-d")]
    assert run_calls == []


@pytest.mark.asyncio
async def test_ensure_mission_container_starts_stopped(stub_run_quiet) -> None:
    """Exists-but-stopped path → ``docker start`` succeeds."""
    # inspect: rc=0, stdout="false" (exists but stopped)
    # start:   rc=0
    stub_run_quiet.script[:] = [(0, "false", ""), (0, "", "")]

    ok = await shell.ensure_mission_container(47)
    assert ok is True
    start_calls = [c for c in stub_run_quiet.calls if c[:2] == ("docker", "start")]
    assert len(start_calls) == 1
    assert start_calls[0][2] == "kutai-mission-47"


@pytest.mark.asyncio
async def test_ensure_mission_container_creates_when_missing(
    stub_run_quiet, monkeypatch
) -> None:
    """No container → run network-ls + network-create + run -d."""
    # Order of internal calls:
    # 1. docker inspect → rc=1 (not found)
    # 2. docker start  → rc=1
    # 3. docker network ls → rc=0 stdout="" (network missing)
    # 4. docker network create → rc=0
    # 5. docker run -d → rc=0
    stub_run_quiet.script[:] = [
        (1, "", "no such container"),
        (1, "", "no such container"),
        (0, "", ""),
        (0, "", ""),
        (0, "container-id-hash", ""),
    ]
    # Skip DB lookup for resource caps.
    async def _stub_caps(mission_id):
        return {"memory": "4g", "cpus": "2", "pids_limit": "512"}
    monkeypatch.setattr(shell, "_resolve_mission_resource_caps", _stub_caps)

    ok = await shell.ensure_mission_container(47)
    assert ok is True
    run_calls = [c for c in stub_run_quiet.calls if c[:3] == ("docker", "run", "-d")]
    assert len(run_calls) == 1
    flat = " ".join(run_calls[0])
    assert "kutai-mission-47" in flat
    assert "--memory" in flat and "4g" in flat
    assert "--cpus" in flat
    assert "--pids-limit" in flat


@pytest.mark.asyncio
async def test_teardown_tolerates_missing(stub_run_quiet) -> None:
    """Already-gone container → teardown still returns True."""
    # stop: rc=1, rm: rc=1 ("no such container"), network rm: rc=1
    stub_run_quiet.script[:] = [
        (1, "", "no such container"),
        (1, "", "no such container"),
        (1, "", "no such network"),
    ]
    ok = await shell.teardown_mission_container(47)
    assert ok is True
