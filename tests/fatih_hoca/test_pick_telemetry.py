"""Pick telemetry: model_pick_log table must exist after init_db()."""
from __future__ import annotations

import pytest
import aiosqlite


@pytest.mark.asyncio
async def test_model_pick_log_table_exists(tmp_path, monkeypatch):
    """After init_db(), model_pick_log must exist with expected columns."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))

    from src.infra.db import init_db
    await init_db()

    async with aiosqlite.connect(tmp_path / "test.db") as db:
        cur = await db.execute("PRAGMA table_info(model_pick_log)")
        cols = {row[1] for row in await cur.fetchall()}

    expected = {
        "id", "timestamp", "task_name", "agent_type", "difficulty",
        "call_category", "picked_model", "picked_score", "picked_reasons",
        "candidates_json", "failures_json", "snapshot_summary",
    }
    assert expected.issubset(cols), f"missing columns: {expected - cols}"


@pytest.mark.asyncio
async def test_model_pick_log_indexes_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db, get_db
    await init_db()

    # Use the same db connection that init_db() populated
    db = await get_db()
    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='model_pick_log'"
    )
    idx_names = {row[0] for row in await cur.fetchall()}

    assert "idx_pick_log_task" in idx_names, f"missing task index, got: {idx_names}"
    assert "idx_pick_log_model" in idx_names, f"missing model index, got: {idx_names}"
