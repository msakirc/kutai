import os
import tempfile
from pathlib import Path

import pytest
import aiosqlite


@pytest.mark.asyncio
async def test_model_call_tokens_table_created(monkeypatch):
    """init_db must create model_call_tokens with expected schema."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        # Reset module-level singleton and redirect DB_PATH before init
        import src.infra.db as db_mod
        monkeypatch.setattr(db_mod, "DB_PATH", str(db_path))
        db_mod._db_connection = None
        await db_mod.init_db()
        # Read schema before closing
        async with aiosqlite.connect(str(db_path)) as conn:
            cur = await conn.execute("PRAGMA table_info(model_call_tokens)")
            cols = {row[1]: row[2] for row in await cur.fetchall()}
        # Close the module singleton so Windows can delete the tempdir
        await db_mod.close_db(checkpoint=False)
        db_mod._db_connection = None
        expected = {
            "id": "INTEGER",
            "timestamp": "TIMESTAMP",
            "task_id": "INTEGER",
            "agent_type": "TEXT",
            "workflow_step_id": "TEXT",
            "workflow_phase": "TEXT",
            "call_category": "TEXT",
            "model": "TEXT",
            "provider": "TEXT",
            "is_streaming": "INTEGER",
            "prompt_tokens": "INTEGER",
            "completion_tokens": "INTEGER",
            "reasoning_tokens": "INTEGER",
            "total_tokens": "INTEGER",
            "duration_ms": "INTEGER",
            "iteration_n": "INTEGER",
            "success": "INTEGER",
        }
        for k, v in expected.items():
            assert k in cols, f"missing column {k}"
            assert cols[k] == v, f"column {k}: expected {v}, got {cols[k]}"


@pytest.mark.asyncio
async def test_step_token_stats_table_created(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        import src.infra.db as db_mod
        monkeypatch.setattr(db_mod, "DB_PATH", str(db_path))
        db_mod._db_connection = None
        await db_mod.init_db()
        async with aiosqlite.connect(str(db_path)) as conn:
            cur = await conn.execute("PRAGMA table_info(step_token_stats)")
            cols = {row[1]: row[2] for row in await cur.fetchall()}
        expected = {"agent_type", "workflow_step_id", "workflow_phase",
                    "samples_n", "in_p50", "in_p90", "in_p99",
                    "out_p50", "out_p90", "out_p99",
                    "iters_p50", "iters_p90", "iters_p99", "updated_at"}
        assert expected.issubset(set(cols.keys()))
        # Cleanup before tempdir teardown (Windows file locks)
        await db_mod.close_db()
        db_mod._db_connection = None


@pytest.mark.asyncio
async def test_model_pick_log_has_outcome_column(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        import src.infra.db as db_mod
        monkeypatch.setattr(db_mod, "DB_PATH", str(db_path))
        db_mod._db_connection = None
        await db_mod.init_db()
        async with aiosqlite.connect(str(db_path)) as conn:
            cur = await conn.execute("PRAGMA table_info(model_pick_log)")
            cols = {row[1] for row in await cur.fetchall()}
        assert "outcome" in cols
        await db_mod.close_db()
        db_mod._db_connection = None
