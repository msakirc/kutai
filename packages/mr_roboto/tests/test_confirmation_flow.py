"""Z10 T1C — mr_roboto confirmation gate skeleton."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock

import pytest

import mr_roboto


async def _init_db(tmp_path, monkeypatch):
    db_path = tmp_path / "confirm_flow.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


@pytest.mark.asyncio
async def test_default_flow_no_confirmation_row(tmp_path, monkeypatch):
    db_mod = await _init_db(tmp_path, monkeypatch)

    # notify_user is `irreversible` per the registry, but without
    # require_confirmation the dispatcher must not open a confirmation row.
    mock_notify = AsyncMock(return_value={"sent": True})
    import mr_roboto.notify_user as _nu  # noqa: F401
    monkeypatch.setattr(
        sys.modules["mr_roboto.notify_user"], "notify_user", mock_notify
    )

    task = {
        "id": 1,
        "mission_id": 10,
        "payload": {"action": "notify_user", "message": "hi"},
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.reversibility == "irreversible"

    import aiosqlite
    async with aiosqlite.connect(db_mod.DB_PATH) as db:
        cur = await db.execute("SELECT COUNT(*) FROM action_confirmations")
        n = (await cur.fetchone())[0]
    assert n == 0


@pytest.mark.asyncio
async def test_require_confirmation_approved_proceeds(
    tmp_path, monkeypatch
):
    db_mod = await _init_db(tmp_path, monkeypatch)

    mock_notify = AsyncMock(return_value={"sent": True})
    import mr_roboto.notify_user as _nu  # noqa: F401
    monkeypatch.setattr(
        sys.modules["mr_roboto.notify_user"], "notify_user", mock_notify
    )

    task = {
        "id": 2,
        "mission_id": 11,
        "payload": {
            "action": "notify_user",
            "message": "hi",
            "require_confirmation": True,
        },
    }

    async def _approver():
        # Wait a beat for the dispatcher to open the confirmation row,
        # then approve it.
        import aiosqlite
        for _ in range(100):
            await asyncio.sleep(0.05)
            async with aiosqlite.connect(db_mod.DB_PATH) as db:
                cur = await db.execute(
                    "SELECT id FROM action_confirmations "
                    "WHERE task_id = 2 AND verdict = 'pending' "
                    "ORDER BY id DESC LIMIT 1"
                )
                row = await cur.fetchone()
            if row:
                await db_mod.resolve_confirmation(row[0], "approved")
                return
        raise AssertionError("dispatcher never opened a confirmation row")

    approver = asyncio.create_task(_approver())
    action = await mr_roboto.run(task)
    await approver

    assert action.status == "completed"
    assert action.reversibility == "irreversible"
    # The notify executor must have actually run.
    assert mock_notify.await_count == 1


@pytest.mark.asyncio
async def test_require_confirmation_rejected_blocks(
    tmp_path, monkeypatch
):
    db_mod = await _init_db(tmp_path, monkeypatch)

    mock_notify = AsyncMock(return_value={"sent": True})
    import mr_roboto.notify_user as _nu  # noqa: F401
    monkeypatch.setattr(
        sys.modules["mr_roboto.notify_user"], "notify_user", mock_notify
    )

    task = {
        "id": 3,
        "mission_id": 12,
        "payload": {
            "action": "notify_user",
            "message": "bye",
            "require_confirmation": True,
        },
    }

    async def _rejecter():
        import aiosqlite
        for _ in range(100):
            await asyncio.sleep(0.05)
            async with aiosqlite.connect(db_mod.DB_PATH) as db:
                cur = await db.execute(
                    "SELECT id FROM action_confirmations "
                    "WHERE task_id = 3 AND verdict = 'pending' "
                    "ORDER BY id DESC LIMIT 1"
                )
                row = await cur.fetchone()
            if row:
                await db_mod.resolve_confirmation(row[0], "rejected")
                return
        raise AssertionError("dispatcher never opened a confirmation row")

    rejecter = asyncio.create_task(_rejecter())
    action = await mr_roboto.run(task)
    await rejecter

    assert action.status == "rejected"
    # Executor must NOT have run.
    assert mock_notify.await_count == 0
