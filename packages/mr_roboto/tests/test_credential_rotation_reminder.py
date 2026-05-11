"""Z6 T7A — credential rotation reminder tests."""
from __future__ import annotations

import datetime

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "rotate.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    import src.founder_actions as fa
    fa._reset_lifecycle_cache()
    return db_mod, fa


async def _insert_credential(
    db_mod, service: str,
    *,
    created_at: str | None = None,
    rotated_at: str | None = None,
    expires_at: str | None = None,
    encrypted_data: str = "x",
):
    db = await db_mod.get_db()
    now = datetime.datetime.utcnow().isoformat()
    created = created_at or now
    await db.execute(
        "INSERT INTO credentials "
        "(service_name, encrypted_data, created_at, updated_at, "
        " rotated_at, expires_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (service, encrypted_data, created, now, rotated_at, expires_at),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_fresh_credential_does_not_fire(tmp_path, monkeypatch):
    db_mod, fa = await _setup(tmp_path, monkeypatch)
    now = datetime.datetime.utcnow()
    # Created 10 days ago, expires in 365 days, never rotated — fresh.
    await _insert_credential(
        db_mod, "github",
        created_at=(now - datetime.timedelta(days=10)).isoformat(),
        expires_at=(now + datetime.timedelta(days=365)).isoformat(),
    )

    from mr_roboto.executors.credential_rotation_reminder import (
        credential_rotation_reminder,
    )
    res = await credential_rotation_reminder()
    assert res["ok"]
    assert res["scanned"] == 1
    assert res["due"] == 0
    assert res["emitted"] == []

    actions = await fa.list_pending()
    assert actions == []


@pytest.mark.asyncio
async def test_expiring_soon_fires(tmp_path, monkeypatch):
    db_mod, fa = await _setup(tmp_path, monkeypatch)
    now = datetime.datetime.utcnow()
    await _insert_credential(
        db_mod, "stripe",
        created_at=(now - datetime.timedelta(days=5)).isoformat(),
        expires_at=(now + datetime.timedelta(days=7)).isoformat(),
    )

    from mr_roboto.executors.credential_rotation_reminder import (
        credential_rotation_reminder,
    )
    res = await credential_rotation_reminder()
    assert res["ok"]
    assert res["due"] == 1
    assert len(res["emitted"]) == 1

    actions = await fa.list_pending()
    assert len(actions) == 1
    a = actions[0]
    assert a.kind == "credential_paste"
    assert "stripe" in a.title
    assert a.expected_output_kind == "credential"


@pytest.mark.asyncio
async def test_never_rotated_and_old_fires(tmp_path, monkeypatch):
    db_mod, fa = await _setup(tmp_path, monkeypatch)
    now = datetime.datetime.utcnow()
    # Created 120 days ago, no rotated_at, no expires_at — overdue.
    await _insert_credential(
        db_mod, "sendgrid",
        created_at=(now - datetime.timedelta(days=120)).isoformat(),
    )

    from mr_roboto.executors.credential_rotation_reminder import (
        credential_rotation_reminder,
    )
    res = await credential_rotation_reminder()
    assert res["due"] == 1
    actions = await fa.list_pending()
    assert len(actions) == 1
    assert actions[0].kind == "credential_paste"
    assert "sendgrid" in actions[0].title


@pytest.mark.asyncio
async def test_recently_rotated_does_not_fire(tmp_path, monkeypatch):
    db_mod, fa = await _setup(tmp_path, monkeypatch)
    now = datetime.datetime.utcnow()
    # Old created_at, but rotated_at recent — should NOT fire on the
    # 'never_rotated' arm. No upcoming expiry either.
    await _insert_credential(
        db_mod, "vercel",
        created_at=(now - datetime.timedelta(days=400)).isoformat(),
        rotated_at=(now - datetime.timedelta(days=2)).isoformat(),
    )

    from mr_roboto.executors.credential_rotation_reminder import (
        credential_rotation_reminder,
    )
    res = await credential_rotation_reminder()
    assert res["due"] == 0
    actions = await fa.list_pending()
    assert actions == []


@pytest.mark.asyncio
async def test_duplicate_emit_is_guarded(tmp_path, monkeypatch):
    db_mod, fa = await _setup(tmp_path, monkeypatch)
    now = datetime.datetime.utcnow()
    await _insert_credential(
        db_mod, "cloudflare",
        created_at=(now - datetime.timedelta(days=200)).isoformat(),
    )

    from mr_roboto.executors.credential_rotation_reminder import (
        credential_rotation_reminder,
    )
    res1 = await credential_rotation_reminder()
    assert len(res1["emitted"]) == 1
    res2 = await credential_rotation_reminder()
    assert res2["emitted"] == []
    assert res2["skipped_duplicate"] == 1

    actions = await fa.list_pending()
    assert len(actions) == 1  # still just the one


@pytest.mark.asyncio
async def test_no_credentials_is_clean(tmp_path, monkeypatch):
    db_mod, fa = await _setup(tmp_path, monkeypatch)
    from mr_roboto.executors.credential_rotation_reminder import (
        credential_rotation_reminder,
    )
    res = await credential_rotation_reminder()
    assert res["ok"]
    assert res["scanned"] == 0
    assert res["emitted"] == []


def test_cron_seed_registers_weekly_cadence():
    from general_beckman.cron_seed import INTERNAL_CADENCES
    matches = [
        c for c in INTERNAL_CADENCES
        if c.get("title") == "credential_rotation_reminder"
    ]
    assert len(matches) == 1
    assert matches[0].get("interval_seconds") == 604800  # 7 days
    assert matches[0].get("payload", {}).get("_executor") == (
        "credential_rotation_reminder"
    )
