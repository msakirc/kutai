"""Z8 T5G — Twilio SMS + escalate_to_founder executors."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "esc.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


# ── sms_send ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sms_send_missing_to_fails():
    from packages.mr_roboto.src.mr_roboto.executors import sms_send

    res = await sms_send.run({"payload": {"body": "hi"}})
    assert res["ok"] is False
    assert "to" in res["reason"]


@pytest.mark.asyncio
async def test_sms_send_missing_from_fails(monkeypatch):
    from packages.mr_roboto.src.mr_roboto.executors import sms_send

    monkeypatch.delenv("TWILIO_FROM", raising=False)
    res = await sms_send.run({
        "payload": {"to": "+15551234567", "body": "hi"},
    })
    assert res["ok"] is False
    assert "from" in res["reason"]


@pytest.mark.asyncio
async def test_sms_send_missing_account_sid(monkeypatch):
    from packages.mr_roboto.src.mr_roboto.executors import sms_send

    monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
    monkeypatch.setenv("TWILIO_FROM", "+15557654321")
    res = await sms_send.run({
        "payload": {"to": "+15551234567", "body": "hi"},
    })
    assert res["ok"] is False
    assert "TWILIO_ACCOUNT_SID" in res["reason"]


@pytest.mark.asyncio
async def test_sms_send_calls_vendor_call(monkeypatch, tmp_path):
    """Monkey-patch the vendor_call hop and assert sms_send hands off correctly."""
    await _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_FROM", "+15557654321")

    from packages.mr_roboto.src.mr_roboto.executors import sms_send

    captured: dict = {}

    async def fake_vendor_call(task):
        captured["task"] = task
        return {"ok": True, "result": {"sid": "SM_FAKE_001"}}

    monkeypatch.setattr(sms_send, "vendor_call_run", fake_vendor_call, raising=False)
    # The executor does ``from .vendor_call import run as vendor_call_run``
    # *inside* the function — patch the module instead.
    from packages.mr_roboto.src.mr_roboto.executors import vendor_call as vc_mod
    monkeypatch.setattr(vc_mod, "run", fake_vendor_call)

    res = await sms_send.run({
        "mission_id": 1,
        "payload": {"to": "+15551234567", "body": "alert"},
    })
    assert res["ok"] is True
    assert res["sid"] == "SM_FAKE_001"
    spec = captured["task"]["context"]["post_hook"]
    assert spec["service"] == "twilio"
    assert spec["action"] == "send_sms"
    assert spec["params"]["To"] == "+15551234567"
    assert spec["params"]["From"] == "+15557654321"


@pytest.mark.asyncio
async def test_sms_send_daily_cap_enforced(monkeypatch, tmp_path):
    await _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_FROM", "+15557654321")
    monkeypatch.setenv("TWILIO_DAILY_CAP_USD", "0.001")  # cap below one SMS

    from packages.mr_roboto.src.mr_roboto.executors import sms_send

    res = await sms_send.run({
        "payload": {"to": "+15551234567", "body": "hi"},
    })
    assert res["ok"] is False
    assert res.get("capped") is True


# ── escalate_to_founder ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_escalate_low_severity_uses_telegram(tmp_path, monkeypatch):
    await _setup(tmp_path, monkeypatch)
    from packages.mr_roboto.src.mr_roboto.executors import escalate_to_founder

    res = await escalate_to_founder.run({
        "mission_id": 5,
        "payload": {
            "severity": "low",
            "title": "minor blip",
            "summary": "ignore",
        },
    })
    assert res["channel"] == "telegram"
    assert res["tier"] == 1
    assert res["founder_action_id"]  # truthy
    assert res["sms_sid"] is None


@pytest.mark.asyncio
async def test_escalate_critical_with_policy_sms(tmp_path, monkeypatch):
    await _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_FROM", "+15557654321")
    monkeypatch.setenv("ONCALL_PHONE", "+15551234567")

    # Default policy already routes tier3 → sms; explicitly seed policy.
    from src.ops.escalation_policy import Policy, set_policy
    await set_policy(Policy(mission_id=9, tier3_channel="sms"))

    from packages.mr_roboto.src.mr_roboto.executors import escalate_to_founder
    from packages.mr_roboto.src.mr_roboto.executors import vendor_call as vc_mod

    captured: dict = {}

    async def fake_vendor_call(task):
        captured["task"] = task
        return {"ok": True, "result": {"sid": "SM_CRITICAL_42"}}

    monkeypatch.setattr(vc_mod, "run", fake_vendor_call)

    res = await escalate_to_founder.run({
        "mission_id": 9,
        "payload": {
            "severity": "critical",
            "title": "DB disk full",
            "summary": "p99 free <2%",
        },
    })
    assert res["channel"] == "sms"
    assert res["tier"] == 3
    assert res["sms_sid"] == "SM_CRITICAL_42"
    assert res["founder_action_id"]
    # vendor_call was called once
    assert "task" in captured


# ── oncall_action wires escalate_to_founder ────────────────────────────────


@pytest.mark.asyncio
async def test_oncall_action_escalate_verb_real_wired(tmp_path, monkeypatch):
    await _setup(tmp_path, monkeypatch)
    from packages.mr_roboto.src.mr_roboto.executors import oncall_action

    res = await oncall_action.run({
        "mission_id": 1,
        "payload": {
            "verb": "escalate_to_founder",
            "params": {
                "severity": "high",
                "title": "5xx spike",
                "summary": "rate >5% for 10min",
            },
        },
    })
    # No longer a stub; real handler returns channel + founder_action_id.
    assert res.get("stub") is False
    assert "channel" in res
    assert res.get("founder_action_id")
