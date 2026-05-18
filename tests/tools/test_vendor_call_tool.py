"""Tests for the Z6 T3B LLM vendor_call tool."""
from __future__ import annotations

import json

import pytest

from src.tools import vendor_call as vc_mod
from src.tools.vendor_call import vendor_call_tool


class _FakeAdapter:
    def __init__(self, result=None, raises=None):
        self._result = result or {"status": "ok", "data": {"x": 1}}
        self._raises = raises
        self.calls: list = []

    async def execute(self, action, params):
        self.calls.append((action, params))
        if self._raises:
            raise self._raises
        return self._result


class _FakeRegistry:
    def __init__(self, mapping=None):
        self._m = mapping or {}

    def get(self, name):
        return self._m.get(name)

    def list_services(self):
        return sorted(self._m.keys())


def _install_registry(monkeypatch, mapping):
    import src.integrations.registry as reg_mod
    fake = _FakeRegistry(mapping)
    monkeypatch.setattr(reg_mod, "get_integration_registry", lambda: fake)


# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_service_refused():
    out = json.loads(await vendor_call_tool("", "list_things"))
    assert out["status"] == "refused"
    assert out["reason"] == "missing_service_or_action"


@pytest.mark.asyncio
async def test_researcher_denied_everything(monkeypatch):
    _install_registry(monkeypatch, {"stripe": _FakeAdapter()})
    out = json.loads(await vendor_call_tool(
        "stripe", "list_products", {}, agent="researcher",
    ))
    assert out["status"] == "refused"
    assert out["reason"] == "agent_not_allowed"


@pytest.mark.asyncio
async def test_unknown_agent_denied(monkeypatch):
    _install_registry(monkeypatch, {"stripe": _FakeAdapter()})
    out = json.loads(await vendor_call_tool(
        "stripe", "list_products", {}, agent="some_new_agent",
    ))
    assert out["status"] == "refused"


@pytest.mark.asyncio
async def test_executor_can_call_vercel(monkeypatch):
    adapter = _FakeAdapter(
        result={"status": "ok", "data": {"id": "dep_1"}, "status_code": 200},
    )
    _install_registry(monkeypatch, {"vercel": adapter})
    out = json.loads(await vendor_call_tool(
        "vercel", "list_deployments", {"limit": 5}, agent="executor",
    ))
    assert out["status"] == "ok"
    assert out["data"]["id"] == "dep_1"
    assert adapter.calls == [("list_deployments", {"limit": 5})]


@pytest.mark.asyncio
async def test_implementer_can_call_stripe_not_vercel(monkeypatch):
    _install_registry(monkeypatch, {
        "stripe": _FakeAdapter(),
        "vercel": _FakeAdapter(),
    })
    # stripe: allowed
    out_s = json.loads(await vendor_call_tool(
        "stripe", "list_products", {}, agent="implementer",
    ))
    assert out_s["status"] == "ok"
    # vercel: denied
    out_v = json.loads(await vendor_call_tool(
        "vercel", "list_deployments", {}, agent="implementer",
    ))
    assert out_v["status"] == "refused"


@pytest.mark.asyncio
async def test_cost_cap_refused(monkeypatch):
    monkeypatch.setenv("MAX_TOOL_CALL_COST_USD", "5.0")
    _install_registry(monkeypatch, {"stripe": _FakeAdapter()})
    out = json.loads(await vendor_call_tool(
        "stripe", "create_product", {}, agent="implementer",
        cost_estimate_usd=10.0,
    ))
    assert out["status"] == "refused"
    assert out["reason"] == "cost_cap_exceeded"
    assert out["cap_usd"] == 5.0


@pytest.mark.asyncio
async def test_params_as_json_string(monkeypatch):
    adapter = _FakeAdapter()
    _install_registry(monkeypatch, {"stripe": adapter})
    await vendor_call_tool(
        "stripe", "list_products",
        params='{"limit": 3, "active": true}',
        agent="implementer",
    )
    assert adapter.calls[0][1] == {"limit": 3, "active": True}


@pytest.mark.asyncio
async def test_adapter_not_registered(monkeypatch):
    _install_registry(monkeypatch, {})
    out = json.loads(await vendor_call_tool(
        "stripe", "list_products", {}, agent="implementer",
    ))
    assert out["status"] == "error"
    assert out["reason"] == "adapter_not_registered"


@pytest.mark.asyncio
async def test_adapter_raises_captured(monkeypatch):
    adapter = _FakeAdapter(raises=RuntimeError("kaboom"))
    _install_registry(monkeypatch, {"stripe": adapter})
    out = json.loads(await vendor_call_tool(
        "stripe", "list_products", {}, agent="implementer",
    ))
    assert out["status"] == "error"
    assert out["reason"] == "vendor_raised"
    assert "kaboom" in out["error"]


def test_tool_registered_in_registry():
    """The tool registers under TOOL_REGISTRY at import time."""
    from src.tools import TOOL_REGISTRY
    assert "vendor_call" in TOOL_REGISTRY
    entry = TOOL_REGISTRY["vendor_call"]
    assert callable(entry["function"])


def test_allowlist_defaults_to_deny():
    """Agents not in the allowlist get an empty allowed-services list."""
    from src.tools.vendor_call import AGENT_ALLOWLIST, _allowed_services
    assert _allowed_services("nonexistent_agent") == []
    # Pinned: researcher must remain empty (high blast radius if changed).
    assert AGENT_ALLOWLIST.get("researcher") == []
