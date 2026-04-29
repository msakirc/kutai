import os
import tempfile
from pathlib import Path

import pytest
import aiosqlite


@pytest.mark.asyncio
async def test_rollup_writes_step_token_stats(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        import src.infra.db as db_mod
        monkeypatch.setattr(db_mod, "DB_PATH", str(db_path))
        db_mod._db_connection = None
        await db_mod.init_db()
        # Seed model_call_tokens with 10 rows for one (agent, step, phase) key
        async with aiosqlite.connect(str(db_path)) as conn:
            for i in range(10):
                await conn.execute(
                    """INSERT INTO model_call_tokens (
                        agent_type, workflow_step_id, workflow_phase, call_category,
                        model, provider, is_streaming, prompt_tokens, completion_tokens,
                        total_tokens, duration_ms, iteration_n, success
                    ) VALUES ('analyst','3.5','phase_3','main_work','gpt','openai',0,
                              ?, ?, ?, 1000, ?, 1)""",
                    (1000 + i*100, 2000 + i*100, 3000 + i*200, i + 1),
                )
            await conn.commit()
        from general_beckman.btable_rollup import run_rollup
        rows_written = await run_rollup(str(db_path))
        assert rows_written >= 1
        async with aiosqlite.connect(str(db_path)) as conn:
            async with conn.execute(
                "SELECT samples_n, in_p90 FROM step_token_stats "
                "WHERE agent_type='analyst' AND workflow_step_id='3.5'"
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
        assert row[0] == 10
        assert row[1] > 0
        await db_mod.close_db()
        db_mod._db_connection = None


@pytest.mark.asyncio
async def test_rollup_refreshes_btable_cache(monkeypatch):
    """After rollup, btable_cache.get_btable() returns the rolled-up rows."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        import src.infra.db as db_mod
        monkeypatch.setattr(db_mod, "DB_PATH", str(db_path))
        db_mod._db_connection = None
        await db_mod.init_db()
        async with aiosqlite.connect(str(db_path)) as conn:
            for i in range(8):
                await conn.execute(
                    """INSERT INTO model_call_tokens (
                        agent_type, workflow_step_id, workflow_phase, call_category,
                        model, provider, is_streaming, prompt_tokens, completion_tokens,
                        total_tokens, duration_ms, iteration_n, success
                    ) VALUES ('writer','4.15a1','phase_4','main_work','gpt','openai',0,
                              ?, ?, ?, 1000, ?, 1)""",
                    (5000 + i*500, 4000 + i*400, 9000 + i*900, i + 1),
                )
            await conn.commit()
        from general_beckman.btable_rollup import run_rollup
        from general_beckman.btable_cache import get_btable
        await run_rollup(str(db_path))
        btable = get_btable()
        key = ("writer", "4.15a1", "phase_4")
        assert key in btable
        assert btable[key]["samples_n"] == 8
        assert btable[key]["out_p90"] > 0
        await db_mod.close_db()
        db_mod._db_connection = None
