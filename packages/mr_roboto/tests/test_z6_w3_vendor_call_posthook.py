"""Z6 W3 — vendor_call post-hook wired on real verification steps.

Decision: chose Option-B-style — added two mechanical sibling
verification steps (13.2.verify, 13.3.verify) that fire vendor_call
against cloudflare.list_zones and sentry.list_projects respectively.
This exercises the T3A vendor_call executor on a path with legitimate
value (confirming the provider configured in the preceding step is
actually reachable with the credentials wired into the vault) rather
than synthesising a fictional production step.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_WF_JSON = _REPO / "src" / "workflows" / "i2p" / "i2p_v3.json"


def _load_steps() -> list[dict]:
    with _WF_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)["steps"]


def _by_id(sid: str) -> dict | None:
    for s in _load_steps():
        if s.get("id") == sid:
            return s
    return None


def test_w3_verify_steps_exist_with_vendor_call_payload():
    """Both verify steps must declare action=vendor_call and a post_hook
    context block so vendor_call.run picks up the spec."""
    for sid, svc, action in (
        ("13.2.verify", "cloudflare", "list_zones"),
        ("13.3.verify", "sentry", "list_projects"),
    ):
        s = _by_id(sid)
        assert s is not None, f"missing step {sid}"
        assert s["agent"] == "mechanical"
        assert s["executor"] == "mechanical"
        assert s["payload"]["action"] == "vendor_call"
        # context.post_hook is what vendor_call.run actually reads
        ph = s["context"]["post_hook"]
        assert ph["service"] == svc
        assert ph["action"] == action


def test_w3_expander_propagates_post_hook_to_task_context():
    """Expander hoists step.context.post_hook into task.context so
    vendor_call.run can find it at dispatch."""
    from src.workflows.engine.expander import expand_steps_to_tasks

    step = _by_id("13.2.verify")
    assert step is not None
    tasks = expand_steps_to_tasks([step], mission_id="1")
    assert tasks, "expander emitted no tasks for 13.2.verify"
    ctx = tasks[0]["context"]
    if isinstance(ctx, str):
        ctx = json.loads(ctx)
    assert ctx.get("post_hook", {}).get("service") == "cloudflare"
    assert ctx.get("post_hook", {}).get("action") == "list_zones"
    assert ctx.get("executor") == "mechanical"
    assert ctx.get("payload", {}).get("action") == "vendor_call"


@pytest.mark.asyncio
async def test_w3_vendor_call_executor_fires_on_step(monkeypatch):
    """vendor_call.run with the verify step's context shape calls
    registry.get(service).execute(action, params)."""
    from mr_roboto.executors.vendor_call import run as vendor_call_run

    class _Adapter:
        def __init__(self):
            self.calls: list[tuple[str, dict]] = []

        async def execute(self, action: str, params: dict) -> dict:
            self.calls.append((action, params))
            return {"status": "ok", "data": {"zones": []}, "status_code": 200}

    adapter = _Adapter()

    class _Registry:
        def get(self, name: str):
            return adapter if name == "cloudflare" else None

    import src.integrations.registry as reg_mod
    monkeypatch.setattr(
        reg_mod, "get_integration_registry", lambda: _Registry(),
    )

    task = {
        "mission_id": 1,
        "id": 100,
        "context": {
            "workflow_step_id": "13.2.verify",
            "post_hook": {
                "service": "cloudflare",
                "action": "list_zones",
                "params": {},
            },
        },
    }
    res = await vendor_call_run(task)
    assert res["ok"] is True
    assert res["service"] == "cloudflare"
    assert res["action"] == "list_zones"
    assert adapter.calls == [("list_zones", {})]


@pytest.mark.asyncio
async def test_w3_vendor_call_failure_emits_founder_action(monkeypatch, tmp_path):
    """When the vendor adapter raises, vendor_call.run must emit a
    founder_action with kind='generic' so the founder is alerted."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "z6_w3.db"))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "z6_w3.db"), raising=False)
    await db_mod.init_db()

    import src.founder_actions as fa
    mid = await db_mod.add_mission("w3 mission", "")

    class _BrokenAdapter:
        async def execute(self, action: str, params: dict) -> dict:
            raise RuntimeError("simulated cloudflare 500")

    class _Registry:
        def get(self, name: str):
            return _BrokenAdapter() if name == "cloudflare" else None

    import src.integrations.registry as reg_mod
    monkeypatch.setattr(
        reg_mod, "get_integration_registry", lambda: _Registry(),
    )

    from mr_roboto.executors.vendor_call import run as vendor_call_run
    res = await vendor_call_run({
        "mission_id": mid,
        "id": 999,
        "context": {
            "workflow_step_id": "13.2.verify",
            "post_hook": {
                "service": "cloudflare",
                "action": "list_zones",
                "params": {},
            },
        },
    })
    assert res["ok"] is False
    assert res["reason"] == "vendor_error"

    rows = await fa.list_by_mission(mid)
    assert rows, "expected a founder_action row"
    assert rows[0].kind == "generic"
    assert "cloudflare" in rows[0].title
