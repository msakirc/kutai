"""yalayut.plugins.api — tool payload + execution."""
import pytest

from yalayut.plugins.api import ApiPlugin, execute_api_tool


def _api_row(env_status="ready"):
    return {
        "id": 21,
        "artifact_type": "api",
        "name": "api-coingecko",
        "name_original": "CoinGecko",
        "env_status": env_status,
        "manifest": {
            "name": "api-coingecko",
            "api": {
                "base_url": "https://api.coingecko.com/api/v3",
                "auth_type": "none",
                "auth_env": None,
                "verbs": [
                    {"verb": "price",
                     "endpoint": "/simple/price",
                     "params_schema": {"ids": "string", "vs_currencies": "string"}},
                ],
            },
        },
    }


def test_to_application_builds_tool_payload():
    plugin = ApiPlugin()
    app = plugin.to_application(_api_row(), task_ctx={})
    assert app["exposure_class"] == "tool"
    payload = app["payload"]
    assert payload["kind"] == "api"
    # Namespaced tool name: <artifact_slug>__<verb>, double underscore.
    assert payload["tools"][0]["tool_name"] == "api_coingecko__price"
    assert payload["tools"][0]["base_url"].endswith("/api/v3")
    assert payload["tools"][0]["endpoint"] == "/simple/price"


def test_to_application_synthesizes_get_verb_when_no_verbs():
    """public_apis_md manifests carry only a base_url — no verbs. The plugin
    must fall back to a single synthetic ``get`` verb so the API is callable."""
    row = _api_row()
    row["manifest"]["api"].pop("verbs")
    row["manifest"]["api"]["description"] = "Crypto prices"
    app = ApiPlugin().to_application(row, task_ctx={})
    tools = app["payload"]["tools"]
    assert len(tools) == 1
    assert tools[0]["tool_name"] == "api_coingecko__get"
    assert tools[0]["endpoint"] == ""
    assert tools[0]["description"] == "Crypto prices"


def test_to_application_empty_when_env_missing():
    plugin = ApiPlugin()
    app = plugin.to_application(_api_row(env_status="missing_X_API_KEY"),
                                task_ctx={})
    assert app["payload"]["tools"] == []
    assert app["payload"]["skipped_reason"] == "missing_X_API_KEY"


@pytest.mark.asyncio
async def test_execute_api_tool_calls_call_api(monkeypatch):
    captured = {}

    async def fake_call_api(api, endpoint=None, params=None):
        captured["endpoint"] = endpoint
        captured["params"] = params
        return '{"bitcoin": {"usd": 64000}}'

    monkeypatch.setattr("src.tools.free_apis.call_api", fake_call_api)
    tool_spec = {
        "tool_name": "api_coingecko__price",
        "base_url": "https://api.coingecko.com/api/v3",
        "endpoint": "/simple/price",
        "auth_type": "none",
        "auth_env": None,
    }
    res = await execute_api_tool(tool_spec, {"ids": "bitcoin", "vs_currencies": "usd"})
    assert res["ok"] is True
    assert "64000" in res["response"]
    assert captured["endpoint"] == "https://api.coingecko.com/api/v3/simple/price"
    assert captured["params"]["ids"] == "bitcoin"


@pytest.mark.asyncio
async def test_execute_api_tool_handles_error(monkeypatch):
    async def fake_call_api(api, endpoint=None, params=None):
        return "API error: HTTP 503"

    monkeypatch.setattr("src.tools.free_apis.call_api", fake_call_api)
    tool_spec = {"tool_name": "api_x__y", "base_url": "https://x",
                 "endpoint": "/y", "auth_type": "none", "auth_env": None}
    res = await execute_api_tool(tool_spec, {})
    assert res["ok"] is False
    assert "503" in res["error"]
