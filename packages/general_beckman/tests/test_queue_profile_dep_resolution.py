import json
import os
import tempfile
from pathlib import Path

import pytest
import aiosqlite


def _setup_tasks_table(conn):
    return conn.execute(
        """CREATE TABLE tasks (
            id INTEGER PRIMARY KEY, status TEXT, agent_type TEXT,
            next_retry_at TIMESTAMP, depends_on TEXT, completed_at TIMESTAMP,
            context TEXT
        )"""
    )


@pytest.mark.asyncio
async def test_queue_profile_excludes_blocked_tasks(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        async with aiosqlite.connect(str(db_path)) as conn:
            await _setup_tasks_table(conn)
            # t1 unblocked; t2 depends on t1 (still pending → blocked)
            await conn.execute(
                "INSERT INTO tasks (id, status, agent_type, depends_on, context) "
                "VALUES (1, 'pending', 'analyst', '[]', "
                "'{\"workflow_step_id\":\"s1\",\"workflow_phase\":\"p1\",\"difficulty\":5}')"
            )
            await conn.execute(
                "INSERT INTO tasks (id, status, agent_type, depends_on, context) "
                "VALUES (2, 'pending', 'analyst', '[1]', '{\"difficulty\":7}')"
            )
            await conn.commit()
        # Reset the in-process completed_ids cache so this test doesn't see leakage
        from general_beckman.queue_profile_push import build_profile, _reset_cache_for_tests
        _reset_cache_for_tests()
        profile = await build_profile(str(db_path))
        # only t1 unblocked
        assert profile.total_ready_count == 1


@pytest.mark.asyncio
async def test_queue_profile_includes_dep_resolved(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        async with aiosqlite.connect(str(db_path)) as conn:
            await _setup_tasks_table(conn)
            await conn.execute(
                "INSERT INTO tasks VALUES (1, 'completed', 'analyst', NULL, '[]', "
                "datetime('now'), '{}')"
            )
            await conn.execute(
                "INSERT INTO tasks VALUES (2, 'pending', 'analyst', NULL, '[1]', NULL, "
                "'{\"workflow_step_id\":\"s2\",\"workflow_phase\":\"p2\",\"difficulty\":7}')"
            )
            await conn.commit()
        from general_beckman.queue_profile_push import build_profile, _reset_cache_for_tests
        _reset_cache_for_tests()
        profile = await build_profile(str(db_path))
        assert profile.total_ready_count == 1
        assert profile.hard_tasks_count == 1
        assert profile.by_difficulty.get(7) == 1


@pytest.mark.asyncio
async def test_queue_profile_projects_tokens_and_calls(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        async with aiosqlite.connect(str(db_path)) as conn:
            await _setup_tasks_table(conn)
            await conn.execute(
                "INSERT INTO tasks VALUES (1, 'pending', 'analyst', NULL, '[]', NULL, "
                "'{\"workflow_step_id\":\"4.5b\",\"workflow_phase\":\"phase_4\","
                "\"difficulty\":7}')"
            )
            await conn.commit()
        from general_beckman.queue_profile_push import build_profile, _reset_cache_for_tests
        _reset_cache_for_tests()
        profile = await build_profile(str(db_path))
        # 4.5b is in STEP_TOKEN_OVERRIDES with iters=12, in=10k, out=100k
        # projected_tokens = (10k+100k)*12 = 1.32M
        assert profile.projected_tokens >= 1_000_000
        assert profile.projected_calls >= 12
