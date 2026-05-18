"""yalayut.dispatch_tool — prefix routing for namespaced api/mcp tool-calls."""
import pytest

import yalayut


@pytest.mark.asyncio
async def test_dispatch_routes_api(monkeypatch):
    async def fake_api(tool_spec, params):
        return {"ok": True, "response": "api-ran", "error": None}

    monkeypatch.setattr("yalayut.plugins.api.execute_api_tool", fake_api)
    registry = {
        "api_coingecko__price": {"_yalayut_kind": "api",
                                 "tool_name": "api_coingecko__price"},
    }
    res = await yalayut.dispatch_tool("api_coingecko__price", {"ids": "btc"},
                                      registry)
    assert res["ok"] is True
    assert res["response"] == "api-ran"


@pytest.mark.asyncio
async def test_dispatch_routes_mcp(monkeypatch):
    async def fake_mcp(tool_spec, args):
        return {"ok": True, "response": "mcp-ran", "error": None}

    monkeypatch.setattr("yalayut.plugins.mcp.execute_mcp_tool", fake_mcp)
    registry = {
        "mcp_cloudflare__echo": {"_yalayut_kind": "mcp",
                                 "tool_name": "mcp_cloudflare__echo"},
    }
    res = await yalayut.dispatch_tool("mcp_cloudflare__echo", {"text": "x"},
                                      registry)
    assert res["ok"] is True
    assert res["response"] == "mcp-ran"


@pytest.mark.asyncio
async def test_dispatch_unknown_tool():
    res = await yalayut.dispatch_tool("not_a_yalayut_tool", {}, {})
    assert res["ok"] is False
    assert "unknown" in res["error"]
