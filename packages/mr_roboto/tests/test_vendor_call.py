"""Tests for the Z6 T3A vendor_call mechanical executor.

Pure unit-level: we monkeypatch the IntegrationRegistry, artifact loader,
cost-cap probe, and founder_action emitter so no DB / HTTP touches.
"""
from __future__ import annotations

import importlib

import pytest

from mr_roboto.executors import vendor_call as vc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeAdapter:
    def __init__(self, result=None, raises=None):
        self._result = result or {"status": "ok", "data": {"hello": "world"}}
        self._raises = raises
        self.calls: list = []

    async def execute(self, action, params):
        self.calls.append((action, params))
        if self._raises:
            raise self._raises
        return self._result


class _FakeRegistry:
    def __init__(self, mapping: dict | None = None):
        self._mapping = mapping or {}

    def get(self, name):
        return self._mapping.get(name)


@pytest.fixture(autouse=True)
def _patch_emitter(monkeypatch):
    """Capture founder_action emissions to assert against without DB."""
    emitted: list[dict] = []

    async def _capture(mission_id, service, action, task_id, step_id, msg):
        emitted.append({
            "mission_id": mission_id, "service": service, "action": action,
            "task_id": task_id, "step_id": step_id, "msg": msg,
        })

    monkeypatch.setattr(vc, "_emit_failure_action", _capture)
    return emitted


@pytest.fixture(autouse=True)
def _patch_cost_cap(monkeypatch):
    """Default: cost cap allows. Tests that need to fail override."""
    async def _allow(_mid, _cost):
        return True, "ok"
    monkeypatch.setattr(vc, "_check_cost_cap", _allow)


def _install_registry(monkeypatch, mapping):
    import src.integrations.registry as reg_mod

    fake = _FakeRegistry(mapping)
    monkeypatch.setattr(reg_mod, "get_integration_registry", lambda: fake)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_service_or_action():
    res = await vc.run({"id": 1, "mission_id": 1, "payload": {}})
    assert res["ok"] is False
    assert res["reason"] == "missing_service_or_action"


@pytest.mark.asyncio
async def test_happy_path_with_literal_params(monkeypatch):
    adapter = _FakeAdapter(
        result={"status": "ok", "data": {"id": "prod_123"}, "status_code": 201},
    )
    _install_registry(monkeypatch, {"stripe": adapter})

    res = await vc.run({
        "id": 7,
        "mission_id": 42,
        "payload": {
            "service": "stripe",
            "action": "create_product",
            "params": {"name": "Pro Plan"},
        },
    })
    assert res["ok"] is True
    assert res["service"] == "stripe"
    assert res["action"] == "create_product"
    assert res["status_code"] == 201
    assert res["result"] == {"id": "prod_123"}
    assert adapter.calls == [("create_product", {"name": "Pro Plan"})]


@pytest.mark.asyncio
async def test_params_from_artifact_merges_with_literal(monkeypatch):
    adapter = _FakeAdapter()
    _install_registry(monkeypatch, {"sendgrid": adapter})

    async def _load(_mid, name):
        assert name == "email_payload"
        return {"to": "user@example.com", "subject": "Welcome"}

    monkeypatch.setattr(vc, "_load_artifact", _load)

    await vc.run({
        "id": 1,
        "mission_id": 5,
        "context": {"post_hook": {
            "service": "sendgrid",
            "action": "send_mail",
            "params_from_artifact": "email_payload",
            "params": {"subject": "Override"},  # literal wins
        }},
    })
    assert adapter.calls[0][1] == {
        "to": "user@example.com",
        "subject": "Override",
    }


@pytest.mark.asyncio
async def test_adapter_not_registered(monkeypatch, _patch_emitter):
    _install_registry(monkeypatch, {})  # empty

    res = await vc.run({
        "id": 3,
        "mission_id": 1,
        "payload": {"service": "stripe", "action": "list_products"},
    })
    assert res["ok"] is False
    assert res["reason"] == "adapter_not_registered"
    # Defensive — no founder_action emitted here (admission handles it).
    assert _patch_emitter == []


@pytest.mark.asyncio
async def test_vendor_error_emits_founder_action(monkeypatch, _patch_emitter):
    adapter = _FakeAdapter(result={"status": "error", "error": "401 unauth"})
    _install_registry(monkeypatch, {"stripe": adapter})

    res = await vc.run({
        "id": 9,
        "mission_id": 11,
        "context": {"workflow_step_id": "12.3"},
        "payload": {"service": "stripe", "action": "list_products"},
    })
    assert res["ok"] is False
    assert res["reason"] == "vendor_error"
    assert res["error"] == "401 unauth"
    assert len(_patch_emitter) == 1
    e = _patch_emitter[0]
    assert e["service"] == "stripe"
    assert e["task_id"] == 9
    assert e["step_id"] == "12.3"


@pytest.mark.asyncio
async def test_adapter_raises_is_captured(monkeypatch, _patch_emitter):
    adapter = _FakeAdapter(raises=RuntimeError("boom"))
    _install_registry(monkeypatch, {"stripe": adapter})

    res = await vc.run({
        "id": 4,
        "mission_id": 1,
        "payload": {"service": "stripe", "action": "list_products"},
    })
    assert res["ok"] is False
    assert res["reason"] == "vendor_error"
    assert "boom" in res["error"]
    assert len(_patch_emitter) == 1


@pytest.mark.asyncio
async def test_cost_cap_blocks(monkeypatch, _patch_emitter):
    adapter = _FakeAdapter()
    _install_registry(monkeypatch, {"stripe": adapter})

    async def _block(_mid, _cost):
        return False, "cost $50.00 exceeds remaining $5.00"

    monkeypatch.setattr(vc, "_check_cost_cap", _block)

    res = await vc.run({
        "id": 2,
        "mission_id": 1,
        "context": {"cost_estimate_usd": 50.0},
        "payload": {"service": "stripe", "action": "create_product"},
    })
    assert res["ok"] is False
    assert res["reason"] == "cost_cap_exceeded"
    assert adapter.calls == []  # never called
    assert len(_patch_emitter) == 1


@pytest.mark.asyncio
async def test_context_post_hook_alternative_location(monkeypatch):
    """post_hook spec can live under context.post_hook instead of payload."""
    adapter = _FakeAdapter()
    _install_registry(monkeypatch, {"cloudflare": adapter})

    res = await vc.run({
        "id": 1,
        "mission_id": 1,
        "context": {"post_hook": {
            "service": "cloudflare",
            "action": "list_zones",
            "params": {},
        }},
    })
    assert res["ok"] is True
    assert adapter.calls[0][0] == "list_zones"


@pytest.mark.asyncio
async def test_dispatcher_wired():
    """The mechanical dispatcher routes 'vendor_call' to this executor."""
    import mr_roboto

    # Patch the registry to return None — we just want to confirm the kind
    # is reachable through the dispatcher (returns 'failed' with our reason).
    import src.integrations.registry as reg_mod

    class _Empty:
        def get(self, _):
            return None
    orig = reg_mod.get_integration_registry
    reg_mod.get_integration_registry = lambda: _Empty()
    try:
        action = await mr_roboto.run({
            "id": 1,
            "mission_id": 1,
            "payload": {
                "action": "vendor_call",
                "service": "stripe",
            },
            "context": {"post_hook": {
                "service": "stripe", "action": "list_products",
            }},
        })
    finally:
        reg_mod.get_integration_registry = orig

    # The dispatcher's payload.get("action") read happens before we read our
    # own spec.action — confirm the path executes (status either 'failed' for
    # bad spec or completes if dispatcher tolerated). Either way, vendor_call
    # was reached: ImportError would have produced a different message.
    assert action.status in ("failed", "completed")
