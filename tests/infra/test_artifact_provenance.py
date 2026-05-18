"""Z10 T1C — artifact_provenance API tests."""
from __future__ import annotations

import asyncio
import aiosqlite
import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "provenance.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_path, db_mod


@pytest.mark.asyncio
async def test_record_and_fetch_single_row(tmp_path, monkeypatch):
    db_path, db_mod = await _setup(tmp_path, monkeypatch)
    rid = await db_mod.record_artifact_write(
        path="src/foo.py",
        task_id=42,
        step_id="5.1.coder",
        model_id="claude-haiku",
        retry_n=0,
        reviewer_verdict_id=None,
        mission_id=100,
    )
    assert rid > 0
    rows = await db_mod.get_artifact_provenance("src/foo.py")
    assert len(rows) == 1
    r = rows[0]
    assert r["task_id"] == 42
    assert r["step_id"] == "5.1.coder"
    assert r["model_id"] == "claude-haiku"
    assert r["mission_id"] == 100


@pytest.mark.asyncio
async def test_multiple_writes_chronological(tmp_path, monkeypatch):
    db_path, db_mod = await _setup(tmp_path, monkeypatch)
    await db_mod.record_artifact_write(
        path="src/bar.py", task_id=1, model_id="m1",
    )
    # Tiny delay so written_at differs (or fall back to id-DESC ordering).
    await asyncio.sleep(0.01)
    await db_mod.record_artifact_write(
        path="src/bar.py", task_id=2, model_id="m2",
    )
    await asyncio.sleep(0.01)
    await db_mod.record_artifact_write(
        path="src/bar.py", task_id=3, model_id="m3",
    )

    rows = await db_mod.get_artifact_provenance("src/bar.py")
    assert len(rows) == 3
    # Most-recent first.
    assert [r["task_id"] for r in rows] == [3, 2, 1]


@pytest.mark.asyncio
async def test_join_with_tasks_and_tokens(tmp_path, monkeypatch):
    db_path, db_mod = await _setup(tmp_path, monkeypatch)

    # Insert a real task row so the LEFT JOIN populates agent_type.
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO tasks (id, title, agent_type, status) "
            "VALUES (?, ?, ?, ?)",
            (501, "do thing", "coder", "completed"),
        )
        # Two token rows for this task — totals should be summed.
        await db.execute(
            "INSERT INTO model_call_tokens "
            "(task_id, model, provider, total_tokens, success, "
            " prompt_tokens, completion_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (501, "m", "p", 100, 1, 60, 40),
        )
        await db.execute(
            "INSERT INTO model_call_tokens "
            "(task_id, model, provider, total_tokens, success, "
            " prompt_tokens, completion_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (501, "m", "p", 50, 1, 30, 20),
        )
        await db.commit()

    await db_mod.record_artifact_write(
        path="src/joined.py", task_id=501, model_id="m",
    )

    rows = await db_mod.get_artifact_provenance("src/joined.py")
    assert len(rows) == 1
    r = rows[0]
    assert r["agent_type"] == "coder"
    assert r["prompt_tokens"] == 90
    assert r["completion_tokens"] == 60
