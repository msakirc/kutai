"""Beckman pushes a QueueProfile to nerd_herd after queue-change events."""
import json

import pytest
from unittest.mock import patch

from nerd_herd.types import QueueProfile


@pytest.mark.asyncio
async def test_build_and_push_sends_profile(tmp_path):
    from general_beckman.queue_profile_push import build_and_push
    import aiosqlite

    db = tmp_path / "t.db"
    async with aiosqlite.connect(db) as conn:
        # Minimal schema: tasks with status + context JSON (difficulty lives
        # in context.classification.difficulty or context.difficulty).
        await conn.execute(
            "CREATE TABLE tasks ("
            " id INTEGER PRIMARY KEY, status TEXT, context TEXT,"
            " next_retry_at TEXT, depends_on TEXT, completed_at TIMESTAMP,"
            " agent_type TEXT)"
        )
        # (status, context_json, next_retry_at) — depends_on='[]' so all are unblocked
        rows = [
            ("pending", json.dumps({"classification": {"difficulty": 3}}), None),
            ("pending", json.dumps({"classification": {"difficulty": 8}}), None),
            ("pending", json.dumps({"difficulty": 9}), None),
            ("done",    json.dumps({"classification": {"difficulty": 8}}), None),
            ("pending", json.dumps({"difficulty": 1}), None),
            # Retry-gated task: not yet dispatchable, should be excluded.
            ("pending", json.dumps({"difficulty": 8}), "2099-01-01 00:00:00"),
        ]
        await conn.executemany(
            "INSERT INTO tasks(status, context, next_retry_at, depends_on)"
            " VALUES(?, ?, ?, '[]')", rows,
        )
        await conn.commit()

    from general_beckman.queue_profile_push import _reset_cache_for_tests
    _reset_cache_for_tests()

    pushed = []
    with patch("nerd_herd.push_queue_profile",
               side_effect=lambda p: pushed.append(p)):
        await build_and_push(str(db))

    assert len(pushed) == 1
    assert isinstance(pushed[0], QueueProfile)
    assert pushed[0].total_ready_count == 4
    assert pushed[0].hard_tasks_count == 2


@pytest.mark.asyncio
async def test_build_and_push_missing_db_noop(tmp_path):
    from general_beckman.queue_profile_push import build_and_push
    # Unreadable/missing DB must not raise — fire-and-forget.
    await build_and_push(str(tmp_path / "missing.db"))


@pytest.mark.asyncio
async def test_projected_tokens_uses_learned_btable(tmp_path):
    from general_beckman.queue_profile_push import build_profile, _reset_cache_for_tests
    from general_beckman.btable_cache import set_btable
    import aiosqlite
    db = tmp_path / "bt.db"
    async with aiosqlite.connect(db) as conn:
        await conn.execute(
            "CREATE TABLE tasks (id INTEGER PRIMARY KEY, status TEXT, context TEXT,"
            " next_retry_at TEXT, depends_on TEXT, completed_at TIMESTAMP, agent_type TEXT)")
        ctx = json.dumps({"workflow_step_id": "1.0a", "workflow_phase": "research"})
        await conn.execute(
            "INSERT INTO tasks(status, context, next_retry_at, depends_on, agent_type)"
            " VALUES('pending', ?, NULL, '[]', 'researcher')", (ctx,))
        await conn.commit()
    try:
        _reset_cache_for_tests()
        set_btable({("researcher", "1.0a", "research"):
                    {"samples_n": 10, "in_p90": 1000, "out_p90": 500, "iters_p90": 2}})
        learned = await build_profile(str(db))
        _reset_cache_for_tests()
        set_btable({})
        cold = await build_profile(str(db))
        assert learned.projected_tokens == 3000, learned.projected_tokens  # (1000+500)*2
        assert cold.projected_tokens > learned.projected_tokens  # worst-case static default
    finally:
        set_btable({})
        _reset_cache_for_tests()


@pytest.mark.asyncio
async def test_cold_btable_projects_static_default(tmp_path):
    from general_beckman.queue_profile_push import build_profile, _reset_cache_for_tests
    from general_beckman.btable_cache import set_btable
    import aiosqlite
    db = tmp_path / "cold.db"
    async with aiosqlite.connect(db) as conn:
        await conn.execute(
            "CREATE TABLE tasks (id INTEGER PRIMARY KEY, status TEXT, context TEXT,"
            " next_retry_at TEXT, depends_on TEXT, completed_at TIMESTAMP, agent_type TEXT)")
        ctx = json.dumps({"workflow_step_id": "1.0a", "workflow_phase": "research"})
        await conn.execute(
            "INSERT INTO tasks(status, context, next_retry_at, depends_on, agent_type)"
            " VALUES('pending', ?, NULL, '[]', 'researcher')", (ctx,))
        await conn.commit()
    try:
        set_btable({}); _reset_cache_for_tests()
        a = await build_profile(str(db))
        _reset_cache_for_tests()
        b = await build_profile(str(db))
        assert a.projected_tokens == b.projected_tokens > 3000
    finally:
        set_btable({}); _reset_cache_for_tests()
