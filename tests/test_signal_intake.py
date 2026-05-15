"""Z9 T3A — signal intake + PII redaction tests.

Covers:
  * ``redact_user_pii`` — every PII class redacted, UUIDs survive, composable
    with ``redact_secrets``, recursive over dict/list.
  * Webhook intake for intercom / zendesk / posthog — a normalized
    ``raw_signal`` growth_events row is written with PII stripped.

Signature verification is stubbed (monkeypatch ``verify_signature``); the
real HMAC verifiers are exercised separately in ``test_webhook_signing.py``.
"""
from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from src.security.sensitivity import redact_secrets, redact_user_pii


# ---------------------------------------------------------------------------
# redact_user_pii
# ---------------------------------------------------------------------------


def test_redacts_email():
    out = redact_user_pii("reach me at jane.doe@example.com please")
    assert "example.com" not in out
    assert "jane.doe" not in out
    assert "[PII]" in out


def test_redacts_ipv4():
    out = redact_user_pii("client ip 203.0.113.45 connected")
    assert "203.0.113.45" not in out
    assert "[PII]" in out


def test_redacts_ipv6():
    for ip in ("2001:db8::ff00:42:8329", "fe80::1", "::1",
               "2001:0db8:85a3:0000:0000:8a2e:0370:7334"):
        out = redact_user_pii(f"server {ip} responded")
        assert ip not in out, f"{ip} not redacted: {out}"


def test_redacts_street_address():
    out = redact_user_pii("ship to 123 Main Street, apartment 4")
    assert "Main Street" not in out
    assert "[PII]" in out


def test_redacts_phone_number():
    for phone in ("+1 (555) 123-4567", "555-123-4567", "+44 20 7946 0958"):
        out = redact_user_pii(f"call {phone} now")
        assert phone not in out, f"{phone} not redacted: {out}"


def test_uuid_survives():
    u = str(uuid.uuid4())
    out = redact_user_pii(f"event id {u} recorded")
    assert u in out, f"UUID was redacted: {out}"


def test_plain_text_untouched():
    text = "the build passed, ship it at 3pm room 5"
    assert redact_user_pii(text) == text


def test_recursive_dict_and_list():
    payload = {
        "content": "mail a@b.com",
        "id": "11111111-2222-3333-4444-555555555555",
        "nested": {"ip": "10.0.0.1"},
        "tags": ["ok", "ping 192.168.0.1"],
    }
    out = redact_user_pii(payload)
    assert "a@b.com" not in out["content"]
    assert out["id"] == "11111111-2222-3333-4444-555555555555"
    assert "10.0.0.1" not in out["nested"]["ip"]
    assert "192.168.0.1" not in out["tags"][1]
    # Original is not mutated.
    assert payload["content"] == "mail a@b.com"


def test_composable_with_redact_secrets():
    text = "key sk-abcdefghijklmnopqrstuvwxyz from user joe@corp.com"
    out = redact_user_pii(redact_secrets(text))
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in out
    assert "joe@corp.com" not in out


def test_non_string_scalars_pass_through():
    assert redact_user_pii(None) is None
    assert redact_user_pii(42) == 42
    assert redact_user_pii(True) is True


# ---------------------------------------------------------------------------
# Webhook signal intake
# ---------------------------------------------------------------------------


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "signal.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


@pytest.fixture
def client():
    from src.app.webhook_listener import app
    return TestClient(app)


def _stub_signature_ok(monkeypatch):
    async def _ok(*a, **kw):
        return True
    monkeypatch.setattr("src.app.webhook_listener.verify_signature", _ok)


async def _fetch_raw_signals(db_mod):
    conn = await db_mod.get_db()
    async with conn.execute(
        "SELECT mission_id, kind, properties_json, segment "
        "FROM growth_events WHERE kind='raw_signal'"
    ) as cur:
        rows = await cur.fetchall()
    return [
        {
            "mission_id": r[0],
            "kind": r[1],
            "properties": json.loads(r[2]),
            "segment": r[3],
        }
        for r in rows
    ]


@pytest.mark.asyncio
async def test_intercom_webhook_stores_raw_signal(tmp_path, monkeypatch, client):
    db_mod = await _setup(tmp_path, monkeypatch)
    _stub_signature_ok(monkeypatch)

    payload = {
        "id": "intercom-evt-1",
        "topic": "conversation.user.created",
        "data": {
            "item": {
                "id": "conv-99",
                "state": "open",
                "created_at": 1715000000,
                "source": {"body": "App crashes, email me at user@example.com"},
            }
        },
    }
    r = client.post("/webhook/intercom", json=payload)
    assert r.status_code == 200
    assert r.json()["kind"] == "raw_signal"

    signals = await _fetch_raw_signals(db_mod)
    assert len(signals) == 1
    props = signals[0]["properties"]
    assert props["provider"] == "intercom"
    assert props["signal_type"] == "support_ticket"
    assert props["external_id"] == "intercom-evt-1"
    assert "user@example.com" not in props["content"]
    assert "[PII]" in props["content"]
    assert signals[0]["segment"] is None


@pytest.mark.asyncio
async def test_zendesk_webhook_stores_raw_signal(tmp_path, monkeypatch, client):
    db_mod = await _setup(tmp_path, monkeypatch)
    _stub_signature_ok(monkeypatch)

    payload = {
        "ticket": {
            "id": 4242,
            "subject": "Billing problem",
            "description": "Charged twice, contact me at 555-867-5309",
            "status": "open",
            "priority": "high",
            "created_at": "2026-05-15T10:00:00Z",
        }
    }
    r = client.post("/webhook/zendesk", json=payload)
    assert r.status_code == 200
    assert r.json()["kind"] == "raw_signal"

    signals = await _fetch_raw_signals(db_mod)
    assert len(signals) == 1
    props = signals[0]["properties"]
    assert props["provider"] == "zendesk"
    assert props["signal_type"] == "support_ticket"
    assert "555-867-5309" not in props["content"]
    assert props["raw_meta"]["ticket_id"] == 4242


@pytest.mark.asyncio
async def test_posthog_webhook_stores_raw_signal(tmp_path, monkeypatch, client):
    db_mod = await _setup(tmp_path, monkeypatch)
    _stub_signature_ok(monkeypatch)

    payload = {
        "uuid": "ph-event-7",
        "event": "checkout_completed",
        "distinct_id": "anon-1",
        "timestamp": "2026-05-15T12:00:00Z",
        "properties": {"note": "from 192.168.1.10", "amount": 29},
    }
    r = client.post("/webhook/posthog", json=payload)
    assert r.status_code == 200
    assert r.json()["kind"] == "raw_signal"

    signals = await _fetch_raw_signals(db_mod)
    assert len(signals) == 1
    props = signals[0]["properties"]
    assert props["provider"] == "posthog"
    assert props["signal_type"] == "analytics_event"
    assert props["content"] == "checkout_completed"
    assert props["external_id"] == "ph-event-7"
    # PII inside nested raw_meta.properties is redacted too.
    assert "192.168.1.10" not in json.dumps(props["raw_meta"])


@pytest.mark.asyncio
async def test_signal_provider_does_not_enqueue_alert_triage(
    tmp_path, monkeypatch, client
):
    db_mod = await _setup(tmp_path, monkeypatch)
    _stub_signature_ok(monkeypatch)

    client.post("/webhook/intercom", json={"id": "no-task-1", "data": {"item": {}}})

    conn = await db_mod.get_db()
    async with conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE agent_type='mechanical'"
    ) as cur:
        (n,) = await cur.fetchone()
    assert n == 0  # signal providers store, never triage


@pytest.mark.asyncio
async def test_signal_duplicate_event_not_restored(tmp_path, monkeypatch, client):
    db_mod = await _setup(tmp_path, monkeypatch)
    _stub_signature_ok(monkeypatch)

    payload = {"id": "dup-1", "data": {"item": {"source": {"body": "hi"}}}}
    r1 = client.post("/webhook/zendesk", json={"id": "dup-1", "ticket": {"id": 1}})
    r2 = client.post("/webhook/zendesk", json={"id": "dup-1", "ticket": {"id": 1}})
    assert r1.json()["kind"] == "raw_signal"
    assert r2.json()["status"] == "duplicate"

    signals = await _fetch_raw_signals(db_mod)
    assert len(signals) == 1


@pytest.mark.asyncio
async def test_signal_bad_signature_rejected(tmp_path, monkeypatch, client):
    await _setup(tmp_path, monkeypatch)

    async def _bad(*a, **kw):
        return False
    monkeypatch.setattr("src.app.webhook_listener.verify_signature", _bad)

    r = client.post("/webhook/posthog", json={"uuid": "x", "event": "y"})
    assert r.status_code == 401
