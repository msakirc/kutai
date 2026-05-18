"""Z6 polish P1 — urgent flag + DM bypass for founder_actions."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "fa_urgent.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    import src.founder_actions as fa
    fa._reset_lifecycle_cache()
    return db_mod, fa


@pytest.mark.asyncio
async def test_urgent_column_exists(tmp_path, monkeypatch):
    db_mod, _ = await _setup(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    cur = await db.execute("PRAGMA table_info(founder_actions)")
    cols = {row[1] for row in await cur.fetchall()}
    assert "urgent" in cols


@pytest.mark.asyncio
async def test_create_persists_urgent_true(tmp_path, monkeypatch):
    db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    action = await fa.create(
        mid, "legal_counsel", "Stripe dispute", "why",
        ["review"], urgent=True, notify_telegram=False,
    )
    assert action.urgent is True
    # round-trip
    fetched = await fa.get(action.id)
    assert fetched is not None
    assert fetched.urgent is True
    assert fetched.to_dict()["urgent"] is True


@pytest.mark.asyncio
async def test_create_defaults_urgent_false(tmp_path, monkeypatch):
    db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("m", "")
    action = await fa.create(
        mid, "credential_paste", "Rotate gh", "why",
        ["gen"], notify_telegram=False,
    )
    assert action.urgent is False
    fetched = await fa.get(action.id)
    assert fetched is not None
    assert fetched.urgent is False


@pytest.mark.asyncio
async def test_notifier_dms_admin_on_urgent(monkeypatch):
    """urgent=True must DM TELEGRAM_ADMIN_CHAT_ID, not call thread poster."""
    from src.app import telegram_bot as tb_mod
    monkeypatch.setattr(tb_mod, "TELEGRAM_ADMIN_CHAT_ID", "12345", raising=False)
    iface = tb_mod.TelegramInterface.__new__(tb_mod.TelegramInterface)
    iface.app = MagicMock()
    iface.app.bot = MagicMock()
    iface.app.bot.send_message = AsyncMock()

    # Stub thread poster so we can prove it was NOT called.
    import src.app.telegram_topics as topics_mod
    thread_poster = AsyncMock()
    monkeypatch.setattr(topics_mod, "post_to_mission_thread", thread_poster)

    action = MagicMock()
    action.urgent = True
    action.mission_id = 42
    action.to_dict.return_value = {
        "id": 1, "mission_id": 42, "kind": "legal_counsel",
        "title": "Stripe dispute", "why": "test", "instructions": [],
        "status": "pending", "urgent": True,
    }
    await iface._notify_founder_action(action)
    iface.app.bot.send_message.assert_awaited_once()
    call_kwargs = iface.app.bot.send_message.await_args.kwargs
    assert call_kwargs["chat_id"] == 12345
    assert "URGENT" in call_kwargs["text"]
    # The thread poster must not have been used.
    thread_poster.assert_not_awaited()


@pytest.mark.asyncio
async def test_notifier_uses_thread_on_regular(monkeypatch):
    """urgent=False (default) must post via mission thread, not DM."""
    from src.app import telegram_bot as tb_mod
    iface = tb_mod.TelegramInterface.__new__(tb_mod.TelegramInterface)
    iface.app = MagicMock()
    iface.app.bot = MagicMock()
    iface.app.bot.send_message = AsyncMock()

    import src.app.telegram_topics as topics_mod
    thread_poster = AsyncMock()
    monkeypatch.setattr(topics_mod, "post_to_mission_thread", thread_poster)

    action = MagicMock()
    action.urgent = False
    action.mission_id = 7
    action.to_dict.return_value = {
        "id": 1, "mission_id": 7, "kind": "credential_paste",
        "title": "paste key", "why": "test", "instructions": [],
        "status": "pending", "urgent": False,
    }
    await iface._notify_founder_action(action)
    thread_poster.assert_awaited_once()
    # The direct DM bypass must not have been used.
    iface.app.bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_stripe_dispute_check_emits_urgent(tmp_path, monkeypatch):
    """T5D stripe_dispute_check must mark new disputes as urgent."""
    db_mod, fa = await _setup(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("stripe", "")

    # Stub vendor_call to return one fake dispute.
    from mr_roboto.executors import stripe_dispute_check as sdc

    async def fake_vc(task, service, action, params):
        return {
            "ok": True,
            "result": {"data": [{
                "id": "dp_test_1", "amount": 5000, "currency": "usd",
                "reason": "fraudulent",
            }]},
        }

    monkeypatch.setattr(sdc, "_vc", fake_vc)
    monkeypatch.setattr(sdc, "_workspace_root", lambda: str(tmp_path))

    res = await sdc.run({"mission_id": mid, "id": 1})
    assert res["ok"]
    assert res["new_disputes"] == 1
    actions = await fa.list_by_mission(mid)
    assert len(actions) == 1
    assert actions[0].urgent is True
    assert actions[0].kind == "legal_counsel"


@pytest.mark.asyncio
async def test_credential_rotation_urgent_only_when_expired(
    tmp_path, monkeypatch,
):
    """Expired creds (days_to_expiry < 0) emit urgent; expiring-soon don't."""
    db_mod, fa = await _setup(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    # Two credentials: one expired yesterday, one expiring in 7 days.
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    expired_at = (now - timedelta(days=1)).isoformat()
    soon_at = (now + timedelta(days=7)).isoformat()
    created_at = (now - timedelta(days=120)).isoformat()
    await db.execute(
        "INSERT INTO credentials "
        "(service_name, encrypted_data, expires_at, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("vendor_expired", "enc", expired_at, created_at, created_at),
    )
    await db.execute(
        "INSERT INTO credentials "
        "(service_name, encrypted_data, expires_at, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("vendor_soon", "enc", soon_at, created_at, created_at),
    )
    await db.commit()

    from mr_roboto.executors.credential_rotation_reminder import (
        credential_rotation_reminder,
    )
    res = await credential_rotation_reminder()
    assert res["ok"]
    actions = await fa.list_pending()
    by_title = {a.title: a for a in actions}
    assert "Rotate vendor_expired credential" in by_title
    assert "Rotate vendor_soon credential" in by_title
    assert by_title["Rotate vendor_expired credential"].urgent is True
    assert by_title["Rotate vendor_soon credential"].urgent is False
