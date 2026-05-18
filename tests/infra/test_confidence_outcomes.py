"""Z10 T4B — confidence_outcomes table + record/resolve API."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "confidence.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_path, db_mod


async def _seed_task_with_confidence(db_mod, *, conf_cat="high", conf_num=0.9):
    db = await db_mod.get_db()
    await db.execute(
        "INSERT INTO tasks "
        "(title, description, agent_type, status, confidence_categorical,"
        " confidence_numeric, mission_id) "
        "VALUES (?, ?, ?, 'pending', ?, ?, NULL)",
        ("calibration test task", "", "coder", conf_cat, conf_num),
    )
    cur = await db.execute("SELECT id FROM tasks WHERE title = ?",
                           ("calibration test task",))
    row = await cur.fetchone()
    await cur.close()
    task_id = row[0]
    # Seed pick log for this task title
    await db.execute(
        "INSERT INTO model_pick_log "
        "(task_name, agent_type, picked_model, picked_score,"
        " candidates_json, provider) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("calibration test task", "coder", "gpt-oss-20b",
         0.5, "[]", "local"),
    )
    await db.commit()
    return task_id


@pytest.mark.asyncio
async def test_record_claim_inserts_null_outcome(tmp_path, monkeypatch):
    _, db_mod = await _setup(tmp_path, monkeypatch)
    task_id = await _seed_task_with_confidence(db_mod)
    claim_id = await db_mod.record_confidence_claim(task_id)
    assert claim_id and claim_id > 0
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT task_id, model_id, task_kind, confidence_categorical, "
        " outcome_correct FROM confidence_outcomes WHERE id = ?",
        (claim_id,),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == task_id
    assert row[1] == "gpt-oss-20b"
    assert row[2] == "coder"
    assert row[3] == "high"
    assert row[4] is None  # outcome unresolved


@pytest.mark.asyncio
async def test_record_skips_when_no_confidence(tmp_path, monkeypatch):
    _, db_mod = await _setup(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    await db.execute(
        "INSERT INTO tasks (title, description, agent_type, status) "
        "VALUES ('plain', '', 'coder', 'pending')"
    )
    cur = await db.execute("SELECT id FROM tasks WHERE title='plain'")
    row = await cur.fetchone()
    await cur.close()
    await db.commit()
    claim_id = await db_mod.record_confidence_claim(row[0])
    assert claim_id is None


@pytest.mark.asyncio
async def test_resolve_updates_columns(tmp_path, monkeypatch):
    _, db_mod = await _setup(tmp_path, monkeypatch)
    task_id = await _seed_task_with_confidence(db_mod)
    claim_id = await db_mod.record_confidence_claim(task_id)
    ok = await db_mod.resolve_confidence_outcome(
        claim_id, correct=True, source="reviewer_verdict",
        reviewer_verdict_id=99, notes="approved",
    )
    assert ok is True
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT outcome_correct, resolution_source, reviewer_verdict_id,"
        " notes FROM confidence_outcomes WHERE id = ?", (claim_id,),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == 1
    assert row[1] == "reviewer_verdict"
    assert row[2] == 99
    assert row[3] == "approved"


@pytest.mark.asyncio
async def test_resolve_is_idempotent(tmp_path, monkeypatch):
    _, db_mod = await _setup(tmp_path, monkeypatch)
    task_id = await _seed_task_with_confidence(db_mod)
    claim_id = await db_mod.record_confidence_claim(task_id)
    assert await db_mod.resolve_confidence_outcome(
        claim_id, correct=True, source="reviewer_verdict") is True
    # Second call must noop, not flip
    assert await db_mod.resolve_confidence_outcome(
        claim_id, correct=False, source="regression") is False
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT outcome_correct, resolution_source FROM confidence_outcomes"
        " WHERE id = ?", (claim_id,),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == 1  # still original verdict
    assert row[1] == "reviewer_verdict"


@pytest.mark.asyncio
async def test_outstanding_filters_by_age(tmp_path, monkeypatch):
    _, db_mod = await _setup(tmp_path, monkeypatch)
    task_id = await _seed_task_with_confidence(db_mod)
    claim_id = await db_mod.record_confidence_claim(task_id)
    # Force picked_at into the past
    db = await db_mod.get_db()
    await db.execute(
        "UPDATE confidence_outcomes SET picked_at = "
        "datetime('now', '-5 hours') WHERE id = ?",
        (claim_id,),
    )
    await db.commit()
    older = await db_mod.outstanding_confidence_claims(older_than_hours=2)
    ids = [r["id"] for r in older]
    assert claim_id in ids
    # Tighten window — claim no longer qualifies
    fresh = await db_mod.outstanding_confidence_claims(older_than_hours=10)
    assert claim_id not in [r["id"] for r in fresh]
