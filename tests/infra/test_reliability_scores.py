"""Z10 T4B — confidence_reliability_scores rollup."""
from __future__ import annotations

import pytest


async def _setup(tmp_path, monkeypatch):
    db_path = tmp_path / "reliability.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


async def _seed_outcomes(
    db_mod, *, model_id="gpt-oss-20b", task_kind="coder",
    bucket="high", correct=70, incorrect=30,
):
    db = await db_mod.get_db()
    for i in range(correct):
        await db.execute(
            "INSERT INTO confidence_outcomes "
            "(task_id, model_id, picked_at, task_kind, "
            " confidence_categorical, outcome_correct, resolution_source) "
            "VALUES (?, ?, datetime('now'), ?, ?, 1, 'reviewer_verdict')",
            (i + 1, model_id, task_kind, bucket),
        )
    for i in range(incorrect):
        await db.execute(
            "INSERT INTO confidence_outcomes "
            "(task_id, model_id, picked_at, task_kind, "
            " confidence_categorical, outcome_correct, resolution_source) "
            "VALUES (?, ?, datetime('now'), ?, ?, 0, 'reviewer_verdict')",
            (i + 1000, model_id, task_kind, bucket),
        )
    await db.commit()


@pytest.mark.asyncio
async def test_recompute_writes_reliability(tmp_path, monkeypatch):
    db_mod = await _setup(tmp_path, monkeypatch)
    await _seed_outcomes(db_mod, correct=70, incorrect=30)
    n = await db_mod.recompute_reliability_scores()
    assert n == 1
    rel = await db_mod.get_reliability("gpt-oss-20b", "coder", "high")
    assert rel is not None
    assert rel["sample_n"] == 100
    assert rel["correct_n"] == 70
    assert abs(rel["reliability"] - 0.70) < 1e-6


@pytest.mark.asyncio
async def test_repeated_recompute_upserts(tmp_path, monkeypatch):
    db_mod = await _setup(tmp_path, monkeypatch)
    await _seed_outcomes(db_mod, correct=60, incorrect=40)
    await db_mod.recompute_reliability_scores()
    # Add more outcomes — second call should upsert, not duplicate
    await _seed_outcomes(db_mod, correct=20, incorrect=0)
    await db_mod.recompute_reliability_scores()
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT COUNT(*) FROM confidence_reliability_scores "
        "WHERE model_id='gpt-oss-20b' AND task_kind='coder' "
        "  AND confidence_bucket='high'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == 1
    rel = await db_mod.get_reliability("gpt-oss-20b", "coder", "high")
    assert rel["sample_n"] == 120
    assert rel["correct_n"] == 80


@pytest.mark.asyncio
async def test_calibration_matrix_dumps_rows(tmp_path, monkeypatch):
    db_mod = await _setup(tmp_path, monkeypatch)
    await _seed_outcomes(db_mod, model_id="A", task_kind="coder")
    await _seed_outcomes(db_mod, model_id="B", task_kind="researcher",
                         bucket="med", correct=10, incorrect=2)
    await db_mod.recompute_reliability_scores()
    rows = await db_mod.calibration_matrix()
    assert len(rows) == 2
    keys = {(r["model_id"], r["task_kind"], r["confidence_bucket"])
            for r in rows}
    assert ("A", "coder", "high") in keys
    assert ("B", "researcher", "med") in keys


@pytest.mark.asyncio
async def test_recompute_skips_unresolved(tmp_path, monkeypatch):
    db_mod = await _setup(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    # Unresolved row should not enter the rollup
    await db.execute(
        "INSERT INTO confidence_outcomes "
        "(task_id, model_id, picked_at, task_kind, "
        " confidence_categorical, outcome_correct) "
        "VALUES (1, 'm', datetime('now'), 'k', 'high', NULL)"
    )
    await db.commit()
    n = await db_mod.recompute_reliability_scores()
    assert n == 0
