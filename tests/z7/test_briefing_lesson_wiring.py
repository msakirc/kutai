"""Z7 wiring-sweep #8 — briefing_compose recovered-lessons round-trip.

The briefing's `## Recovered Failures` section reads mission_lessons rows via
`json_extract(source_ref, '$.mission_id')`. That only works if the production
mission_lessons writers actually put `mission_id` in `source_ref` — three of
them did not. Plus the `briefing_compose` posthook must be declared on a real
i2p step or it never fires.

Host-path coverage:
  1. i2p_v3.json step 15.14 declares the briefing_compose posthook.
  2. emit_lessons_from_dlq_patterns() writes source_ref with a queryable
     mission_id (dlq_pattern writer).
"""
from __future__ import annotations

import json
import os

import pytest


def test_i2p_15_14_declares_briefing_compose():
    """The posthook is opt-in (auto_wire_triggers=[]); a real step must
    declare it or it never runs."""
    path = os.path.join("src", "workflows", "i2p", "i2p_v3.json")
    with open(path, encoding="utf-8") as f:
        wf = json.load(f)
    step = next((s for s in wf["steps"] if s.get("id") == "15.14"), None)
    assert step is not None, "step 15.14 missing"
    assert "briefing_compose" in (step.get("post_hooks") or []), (
        "15.14 must declare the briefing_compose posthook"
    )


async def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "z7_lesson_wiring.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


@pytest.mark.asyncio
async def test_dlq_pattern_lesson_carries_queryable_mission_id(tmp_path, monkeypatch):
    """emit_lessons_from_dlq_patterns must write source_ref.mission_id so the
    briefing's json_extract query finds the row."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    from src.infra.dead_letter import quarantine_task
    from src.infra.mission_lessons import emit_lessons_from_dlq_patterns

    cur = await db.execute(
        "INSERT INTO missions (title, status, created_at) "
        "VALUES ('M', 'failed', datetime('now'))")
    await db.commit()
    mission_id = cur.lastrowid

    # 3+ DLQ rows in one (mission, error_category) group → a lesson is emitted.
    for i in range(3):
        tc = await db.execute(
            "INSERT INTO tasks (mission_id, title, status, agent_type, created_at) "
            "VALUES (?, ?, 'failed', 'coder', datetime('now'))",
            (mission_id, f"task {i}"))
        await db.commit()
        await quarantine_task(
            task_id=tc.lastrowid,
            mission_id=mission_id,
            error="ImportError: no module named widget",
            error_category="exhausted",
            original_agent="coder",
            attempts_snapshot=5,
        )

    emitted = await emit_lessons_from_dlq_patterns()
    assert emitted >= 1

    # The briefing's exact query — json_extract on source_ref.
    q = await db.execute(
        "SELECT COUNT(*) FROM mission_lessons "
        "WHERE json_extract(source_ref, '$.mission_id') = ?",
        (mission_id,),
    )
    assert (await q.fetchone())[0] >= 1, (
        "dlq_pattern lesson not found by the briefing's mission_id query"
    )
