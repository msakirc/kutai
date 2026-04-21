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
            " next_retry_at TEXT)"
        )
        # (status, context_json, next_retry_at)
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
            "INSERT INTO tasks(status, context, next_retry_at)"
            " VALUES(?, ?, ?)", rows,
        )
        await conn.commit()

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
