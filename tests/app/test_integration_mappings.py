"""Z8 T3E — webhook routing through integration_mappings."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "mappings.db"
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


async def _bypass_sig(monkeypatch):
    async def _ok(*a, **kw):
        return True

    monkeypatch.setattr("src.app.webhook_listener.verify_signature", _ok)


@pytest.mark.asyncio
async def test_integration_mappings_table_exists(tmp_path, monkeypatch):
    db_mod = await _setup(tmp_path, monkeypatch)
    conn = await db_mod.get_db()
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='integration_mappings'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_webhook_routes_to_mapped_mission(
    tmp_path, monkeypatch, client
):
    db_mod = await _setup(tmp_path, monkeypatch)
    await _bypass_sig(monkeypatch)

    mid = await db_mod.add_mission(title="watch", description="ongoing watcher")
    conn = await db_mod.get_db()
    await conn.execute(
        "INSERT INTO integration_mappings (integration_id, product_id, mission_id) "
        "VALUES (?, NULL, ?)",
        ("sentry", mid),
    )
    await conn.commit()

    r = client.post(
        "/webhook/sentry",
        json={"event_id": "route-1", "type": "issue_alert"},
    )
    assert r.status_code == 200

    async with conn.execute(
        "SELECT mission_id FROM tasks WHERE agent_type='mechanical' "
        "AND title LIKE 'alert_triage:%'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    assert row[0] == mid


@pytest.mark.asyncio
async def test_product_id_disambiguates_between_missions(
    tmp_path, monkeypatch, client
):
    db_mod = await _setup(tmp_path, monkeypatch)
    await _bypass_sig(monkeypatch)

    m1 = await db_mod.add_mission(title="prod-A", description="A")
    m2 = await db_mod.add_mission(title="prod-B", description="B")
    conn = await db_mod.get_db()
    await conn.execute(
        "INSERT INTO integration_mappings (integration_id, product_id, mission_id) "
        "VALUES ('sentry', 'A', ?)",
        (m1,),
    )
    await conn.execute(
        "INSERT INTO integration_mappings (integration_id, product_id, mission_id) "
        "VALUES ('sentry', 'B', ?)",
        (m2,),
    )
    await conn.commit()

    r = client.post(
        "/webhook/sentry",
        json={"event_id": "prod-event", "type": "x", "product_id": "B"},
    )
    assert r.status_code == 200

    async with conn.execute(
        "SELECT mission_id FROM tasks WHERE agent_type='mechanical' "
        "AND title LIKE 'alert_triage:%'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    assert row[0] == m2


@pytest.mark.asyncio
async def test_specific_product_wins_over_null_catchall(
    tmp_path, monkeypatch, client
):
    db_mod = await _setup(tmp_path, monkeypatch)
    await _bypass_sig(monkeypatch)

    m_catchall = await db_mod.add_mission(title="all", description="*")
    m_specific = await db_mod.add_mission(title="X-only", description="X")
    conn = await db_mod.get_db()
    await conn.execute(
        "INSERT INTO integration_mappings (integration_id, product_id, mission_id) "
        "VALUES ('sentry', NULL, ?)",
        (m_catchall,),
    )
    await conn.execute(
        "INSERT INTO integration_mappings (integration_id, product_id, mission_id) "
        "VALUES ('sentry', 'X', ?)",
        (m_specific,),
    )
    await conn.commit()

    r = client.post(
        "/webhook/sentry",
        json={"event_id": "spec-1", "type": "x", "product_id": "X"},
    )
    assert r.status_code == 200

    async with conn.execute(
        "SELECT mission_id FROM tasks WHERE agent_type='mechanical' "
        "AND title LIKE 'alert_triage:%'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    assert row[0] == m_specific


@pytest.mark.asyncio
async def test_no_mapping_yields_null_mission(tmp_path, monkeypatch, client):
    db_mod = await _setup(tmp_path, monkeypatch)
    await _bypass_sig(monkeypatch)

    r = client.post(
        "/webhook/sentry",
        json={"event_id": "orphan-1", "type": "x"},
    )
    assert r.status_code == 200

    conn = await db_mod.get_db()
    async with conn.execute(
        "SELECT mission_id FROM tasks WHERE agent_type='mechanical' "
        "AND title LIKE 'alert_triage:%'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    assert row[0] is None
