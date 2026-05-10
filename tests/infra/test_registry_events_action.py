"""Z10 T1C — registry_events action-scope audit row tests."""
from __future__ import annotations

import json
import aiosqlite
import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "registry_events.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_path, db_mod


@pytest.mark.asyncio
async def test_record_action_event_inserts(tmp_path, monkeypatch):
    db_path, db_mod = await _setup(tmp_path, monkeypatch)
    rid = await db_mod.record_action_event(
        verb="notify_user",
        reversibility="irreversible",
        mission_id=7,
        task_id=88,
        payload={"message": "hi"},
        status="completed",
    )
    assert rid > 0

    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT scope, target, event, verb, reversibility, "
            "mission_id, task_id, payload_json FROM registry_events "
            "WHERE id = ?",
            (rid,),
        )
        row = await cur.fetchone()
    assert row is not None
    scope, target, event, verb, rev, mid, tid, payload_json = row
    assert scope == "action"
    assert target == "notify_user"
    assert event == "notify_user"
    assert verb == "notify_user"
    assert rev == "irreversible"
    assert mid == 7
    assert tid == 88
    parsed = json.loads(payload_json)
    assert parsed["status"] == "completed"
    assert parsed["payload"]["message"] == "hi"
