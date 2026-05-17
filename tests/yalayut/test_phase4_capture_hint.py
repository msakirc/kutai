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


def test_capture_hint_registered_in_post_hook_registry(loop):
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    assert "capture_hint" in POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["capture_hint"]
    assert spec.verb == "capture_hint"
    # advisory — capture failure must not DLQ the source task.
    assert spec.default_severity == "warning"


def test_capture_hint_routes_to_mechanical(loop):
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.apply import RequestPostHook  # dataclass

    hook = RequestPostHook(source_task_id=55, kind="capture_hint", source_ctx={})
    source = {"id": 55, "title": "the task", "agent_type": "coder"}
    source_ctx = {"title": "the task",
                  "description": "do the thing"}
    agent_type, payload = _posthook_agent_and_payload(hook, source, source_ctx)
    assert agent_type == "mechanical"
    assert payload["payload"]["action"] == "capture_hint"
    assert payload["payload"]["source_task"]["id"] == 55
