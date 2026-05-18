"""Z7 wiring-sweep #5 — cold-outreach inbound-reply path.

Before: outreach/handle_reply had no inbound caller (no ESP reply webhook),
and template_id='follow_up' was an inert free-form string nothing branched on.

Now: POST /webhook/outreach/reply/{product_id} dispatches outreach/handle_reply;
run_outreach_draft has a template registry that branches cold vs follow_up.
"""
from __future__ import annotations

import pytest


# ── template registry — template_id now drives the draft instruction ──────────

@pytest.mark.asyncio
async def test_outreach_draft_follow_up_template_branches(monkeypatch):
    from mr_roboto import outreach_draft

    captured = {}

    async def _fake_enqueue(spec, **kw):
        captured["spec"] = spec
        return {"task_id": 1}

    monkeypatch.setattr(outreach_draft, "enqueue", _fake_enqueue)

    await outreach_draft.run_outreach_draft(
        product_id="p1", mission_id=1,
        prospect_data={"email": "x@y.com", "reply_body": "interested!"},
        template_id="follow_up", list_id="l1",
    )
    desc = captured["spec"]["description"].lower()
    assert "follow-up" in desc and "reply_body" in desc, (
        "follow_up template_id did not branch the draft instruction"
    )


@pytest.mark.asyncio
async def test_outreach_draft_cold_template_is_distinct(monkeypatch):
    from mr_roboto import outreach_draft

    captured = {}

    async def _fake_enqueue(spec, **kw):
        captured["spec"] = spec
        return {"task_id": 1}

    monkeypatch.setattr(outreach_draft, "enqueue", _fake_enqueue)

    await outreach_draft.run_outreach_draft(
        product_id="p1", mission_id=1,
        prospect_data={"email": "x@y.com"},
        template_id="cold", list_id="l1",
    )
    desc = captured["spec"]["description"].lower()
    assert "cold" in desc and "first contact" in desc


# ── send_id extraction from mail reference headers ────────────────────────────

def test_extract_send_id_from_in_reply_to():
    pytest.importorskip("fastapi")  # helper lives in the fastapi webhook module
    from src.app.webhook_listener import _extract_send_id_from_refs
    assert _extract_send_id_from_refs(
        {"in_reply_to": "<outreach-4471@mail.acme.com>"}) == "4471"
    assert _extract_send_id_from_refs(
        {"References": "<x@y> <outreach-90@z>"}) == "90"
    assert _extract_send_id_from_refs({"in_reply_to": "<plain@y>"}) is None
    assert _extract_send_id_from_refs({}) is None


# ── the webhook route itself (skips when fastapi absent) ──────────────────────

@pytest.mark.asyncio
async def test_reply_webhook_enqueues_handle_reply(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    enqueued = []

    async def _fake_enqueue(spec, **kw):
        enqueued.append((spec, kw))
        return 1

    import general_beckman
    monkeypatch.setattr(general_beckman, "enqueue", _fake_enqueue)

    from src.app.webhook_listener import app
    client = TestClient(app)
    resp = client.post(
        "/webhook/outreach/reply/prod9",
        json={"send_id": 555, "text": "Yes, tell me more",
              "from": "Jane <jane@acme.com>"},
    )
    assert resp.status_code == 200
    assert resp.json()["send_id"] == 555
    assert len(enqueued) == 1
    spec, kw = enqueued[0]
    assert spec["payload"]["action"] == "outreach/handle_reply"
    assert spec["payload"]["send_id"] == 555
    assert spec["payload"]["reply_from"] == "jane@acme.com"
    assert spec["payload"]["product_id"] == "prod9"


@pytest.mark.asyncio
async def test_reply_webhook_rejects_missing_send_id(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from src.app.webhook_listener import app

    client = TestClient(app)
    resp = client.post("/webhook/outreach/reply/prod9",
                        json={"text": "hi", "from": "a@b.com"})
    assert resp.status_code == 400
