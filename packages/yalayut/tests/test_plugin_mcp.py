"""yalayut.plugins.mcp — discovery, budget cap, namespacing, execution."""
import sys
from pathlib import Path

import pytest

from yalayut.plugins.mcp import McpPlugin, execute_mcp_tool, rank_tools_by_intent

FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"


def _mcp_row(env_status="ready"):
    return {
        "id": 41,
        "artifact_type": "mcp",
        "name": "mcp-cloudflare",
        "env_status": env_status,
        "manifest": {
            "name": "mcp-cloudflare",
            "mcp": {
                "transport": "stdio",
                "run_cmd": f"{sys.executable} {FAKE_SERVER}",
                "env_required": [],
                "tools_discover": True,
                "idle_timeout_s": 300,
            },
        },
    }


def test_rank_tools_caps_to_k():
    tools = [
        {"name": "a", "description": "deploy a worker to cloudflare"},
        {"name": "b", "description": "list dns records"},
        {"name": "c", "description": "manage kv namespace"},
        {"name": "d", "description": "purge cache"},
    ]
    ranked = rank_tools_by_intent(tools, "deploy a worker", k=3)
    assert len(ranked) == 3
    assert ranked[0]["name"] == "a"  # best match first


def test_to_application_skips_when_env_missing():
    plugin = McpPlugin()
    app = plugin.to_application(_mcp_row(env_status="missing_CLOUDFLARE_API_TOKEN"),
                                task_ctx={})
    assert app["payload"]["tools"] == []
    assert app["payload"]["skipped_reason"] == "missing_CLOUDFLARE_API_TOKEN"


@pytest.mark.asyncio
async def test_to_application_async_discovers_and_namespaces(monkeypatch):
    plugin = McpPlugin()
    app = await plugin.to_application_async(
        _mcp_row(), task_ctx={"step_intent": "echo some text", "_confidence": 0.8}
    )
    tool_names = {t["tool_name"] for t in app["payload"]["tools"]}
    # Namespaced: mcp_cloudflare__<tool>, double underscore.
    assert "mcp_cloudflare__echo" in tool_names
    # Budget cap: fake server has 2 tools, K_mcp=3, so both survive.
    assert len(app["payload"]["tools"]) <= 3


@pytest.mark.asyncio
async def test_per_step_total_budget(monkeypatch):
    plugin = McpPlugin()
    # 3 mcp rows, each yields 2 tools = 6; cap K_mcp_total=6 keeps all 6;
    # a 4th would be trimmed. Verify the trimming helper.
    from yalayut.plugins.mcp import enforce_step_budget
    apps = [
        {"payload": {"tools": [{"tool_name": f"m{i}__t{j}", "_score": 0.9 - j * 0.1}
                               for j in range(3)]}}
        for i in range(3)
    ]
    trimmed = enforce_step_budget(apps, k_total=6)
    total = sum(len(a["payload"]["tools"]) for a in trimmed)
    assert total == 6


@pytest.mark.asyncio
async def test_execute_mcp_tool_round_trip():
    tool_spec = {
        "tool_name": "mcp_cloudflare__echo",
        "artifact_id": 44,
        "mcp_tool_name": "echo",
        "mcp": {"transport": "stdio",
                "run_cmd": f"{sys.executable} {FAKE_SERVER}",
                "env_required": [], "idle_timeout_s": 300},
    }
    res = await execute_mcp_tool(tool_spec, {"text": "ping"})
    assert res["ok"] is True
    assert "ping" in res["response"]
    # Cleanup.
    from yalayut.mcp_manager import get_manager
    await get_manager().shutdown(44)
