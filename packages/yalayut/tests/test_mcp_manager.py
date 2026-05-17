"""yalayut.mcp_manager — start, health-probe, call, idle-shutdown."""
import sys
from pathlib import Path

import pytest

from yalayut.mcp_manager import McpManager

FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"


def _mcp_block(unhealthy=False):
    cmd = f"{sys.executable} {FAKE_SERVER}"
    if unhealthy:
        cmd += " --unhealthy"
    return {
        "transport": "stdio",
        "run_cmd": cmd,
        "env_required": [],
        "health_check": "list_tools",
        "idle_timeout_s": 300,
    }


@pytest.fixture
def manager(monkeypatch):
    mgr = McpManager()
    # Stub the DB persistence so tests run without a schema.
    procs = {}

    async def fake_record(artifact_id, **kw):
        procs.setdefault(artifact_id, {}).update(kw)

    async def fake_get(artifact_id):
        return procs.get(artifact_id)

    monkeypatch.setattr(mgr, "_persist_process", fake_record)
    monkeypatch.setattr(mgr, "_load_process", fake_get)
    return mgr


@pytest.mark.asyncio
async def test_start_and_health_probe_ok(manager):
    handle = await manager.ensure_running(artifact_id=31, mcp=_mcp_block())
    assert handle["health"] == "ready"
    assert handle["pid"] > 0
    await manager.shutdown(31)


@pytest.mark.asyncio
async def test_health_probe_failure_kills_process(manager):
    handle = await manager.ensure_running(artifact_id=32,
                                          mcp=_mcp_block(unhealthy=True))
    assert handle["health"] == "unhealthy"
    # Process must be killed; no leftover.
    assert manager.is_running(32) is False


@pytest.mark.asyncio
async def test_list_tools_returns_discovered(manager):
    await manager.ensure_running(artifact_id=33, mcp=_mcp_block())
    tools = await manager.list_tools(33)
    names = {t["name"] for t in tools}
    assert names == {"echo", "add"}
    await manager.shutdown(33)


@pytest.mark.asyncio
async def test_call_tool_echoes(manager):
    await manager.ensure_running(artifact_id=34, mcp=_mcp_block())
    res = await manager.call_tool(34, "echo", {"text": "hello"})
    assert res["ok"] is True
    assert "hello" in res["content"]
    await manager.shutdown(34)


@pytest.mark.asyncio
async def test_ensure_running_reuses_live_process(manager):
    h1 = await manager.ensure_running(artifact_id=35, mcp=_mcp_block())
    h2 = await manager.ensure_running(artifact_id=35, mcp=_mcp_block())
    assert h1["pid"] == h2["pid"]  # no second spawn
    await manager.shutdown(35)


@pytest.mark.asyncio
async def test_idle_shutdown_kills_stale(manager, monkeypatch):
    await manager.ensure_running(artifact_id=36, mcp=_mcp_block())
    # Force last_used into the distant past.
    manager._procs[36]["last_used"] = 0.0
    killed = await manager.sweep_idle(now=10_000_000.0)
    assert 36 in killed
    assert manager.is_running(36) is False
