import asyncio
import json

import pytest

from src.infra.db import init_db, get_db
import yalayut


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def test_capture_hint_inserts_internal_hint(loop):
    async def _run():
        await init_db()
        task = {
            "id": 999,
            "title": "Wire JWT auth into FastAPI",
            "description": "Add JWT bearer auth middleware to the API.",
            "agent_type": "coder",
        }
        outcome = {"status": "completed", "iterations": 3,
                   "result": json.dumps({"ok": True})}
        await yalayut.capture_hint(task, outcome)
        db = await get_db()
        cur = await db.execute(
            "SELECT artifact_type, kind, exposure_class, vet_tier, enabled "
            "FROM yalayut_index WHERE kind = 'internal_hint'")
        row = await cur.fetchone()
        await cur.close()
        assert row is not None
        assert row[0] == "skill"
        assert row[1] == "internal_hint"
        assert row[2] == "inject"
        assert row[3] == 0  # T0
        assert row[4] == 1  # enabled
    loop.run_until_complete(_run())


def test_capture_hint_skips_single_iteration(loop):
    async def _run():
        await init_db()
        before_db = await get_db()
        cur = await before_db.execute(
            "SELECT COUNT(*) FROM yalayut_index WHERE kind = 'internal_hint'")
        before = (await cur.fetchone())[0]
        await cur.close()
        # 1 iteration → no capture (nothing learned).
        await yalayut.capture_hint(
            {"id": 1, "title": "trivial", "description": "x",
             "agent_type": "coder"},
            {"status": "completed", "iterations": 1})
        cur = await before_db.execute(
            "SELECT COUNT(*) FROM yalayut_index WHERE kind = 'internal_hint'")
        after = (await cur.fetchone())[0]
        await cur.close()
        assert after == before
    loop.run_until_complete(_run())


def test_capture_hint_skips_failed_task(loop):
    async def _run():
        await init_db()
        await yalayut.capture_hint(
            {"id": 2, "title": "failed task", "description": "x",
             "agent_type": "coder"},
            {"status": "failed", "iterations": 4})
        db = await get_db()
        cur = await db.execute(
            "SELECT COUNT(*) FROM yalayut_index "
            "WHERE kind = 'internal_hint' AND name LIKE '%failed-task%'")
        assert (await cur.fetchone())[0] == 0
        await cur.close()
    loop.run_until_complete(_run())
