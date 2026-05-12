"""Z8 T3A — webhook listener tests (dedup + signature gating).

Signature verification is stubbed at this tier (T3B owns the real verifier);
tests monkeypatch ``src.app.webhook_signing.verify_signature`` so each test
can choose pass/fail independently.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "webhook.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    # Reset cached singleton so init_db opens against the new path.
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


@pytest.fixture
def client():
    from src.app.webhook_listener import app
    return TestClient(app)


@pytest.mark.asyncio
async def test_health_endpoint(tmp_path, monkeypatch, client):
    await _setup(tmp_path, monkeypatch)
    r = client.get("/webhook/__health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_bad_signature_returns_401(tmp_path, monkeypatch, client):
    await _setup(tmp_path, monkeypatch)

    async def _bad(*a, **kw):
        return False

    monkeypatch.setattr("src.app.webhook_listener.verify_signature", _bad)

    r = client.post("/webhook/sentry", json={"event_id": "x"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_missing_event_id_returns_400(tmp_path, monkeypatch, client):
    await _setup(tmp_path, monkeypatch)

    async def _ok(*a, **kw):
        return True

    monkeypatch.setattr("src.app.webhook_listener.verify_signature", _ok)
    r = client.post("/webhook/sentry", json={"data": {}})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_duplicate_event_returns_200_not_reprocessed(
    tmp_path, monkeypatch, client
):
    db_mod = await _setup(tmp_path, monkeypatch)

    async def _ok(*a, **kw):
        return True

    monkeypatch.setattr("src.app.webhook_listener.verify_signature", _ok)

    payload = {"event_id": "abc", "type": "issue_alert", "data": {}}

    r1 = client.post("/webhook/sentry", json=payload)
    r2 = client.post("/webhook/sentry", json=payload)

    assert r1.status_code == 200
    assert r1.json()["status"] == "accepted"
    assert r2.status_code == 200
    assert r2.json()["status"] == "duplicate"

    conn = await db_mod.get_db()
    async with conn.execute(
        "SELECT COUNT(*) FROM webhook_events"
    ) as cur:
        (n,) = await cur.fetchone()
    assert n == 1

    async with conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE agent_type='mechanical'"
    ) as cur:
        (m,) = await cur.fetchone()
    assert m == 1


@pytest.mark.asyncio
async def test_distinct_events_both_enqueue(tmp_path, monkeypatch, client):
    db_mod = await _setup(tmp_path, monkeypatch)

    async def _ok(*a, **kw):
        return True

    monkeypatch.setattr("src.app.webhook_listener.verify_signature", _ok)

    client.post("/webhook/sentry", json={"event_id": "a", "type": "issue_alert"})
    client.post("/webhook/sentry", json={"event_id": "b", "type": "issue_alert"})

    conn = await db_mod.get_db()
    async with conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE agent_type='mechanical'"
    ) as cur:
        (m,) = await cur.fetchone()
    assert m == 2


@pytest.mark.asyncio
async def test_alert_triage_lands_on_ongoing_lane(tmp_path, monkeypatch, client):
    db_mod = await _setup(tmp_path, monkeypatch)

    async def _ok(*a, **kw):
        return True

    monkeypatch.setattr("src.app.webhook_listener.verify_signature", _ok)

    client.post("/webhook/sentry", json={"event_id": "lane", "type": "x"})

    conn = await db_mod.get_db()
    async with conn.execute(
        "SELECT lane FROM tasks WHERE agent_type='mechanical'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    assert row[0] == "ongoing"
