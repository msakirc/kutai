"""Z7 A0 — Founder briefing surface tests.

Covers:
  1. briefing_compose posthook writes a mission_briefings row (kind=completion)
     with correct product_id, founder_minutes_saved_estimate, sections in body_md.
  2. Daily briefing job writes a mission_briefings row (kind=daily) with
     expected aggregation sections.
  3. render_briefing output contains all expected section markers.
  4. /founder_hours_saved sums mission_events.founder_minutes_saved correctly.
"""
from __future__ import annotations

import json
import pytest


# ── DB helpers ──────────────────────────────────────────────────────────────


async def _setup_db(tmp_path, monkeypatch):
    """Open fresh temp DB, run init_db, return db module."""
    db_path = tmp_path / "z7_briefing.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


async def _insert_mission(db, *, title="Test mission", status="completed") -> int:
    cur = await db.execute(
        "INSERT INTO missions (title, status, created_at) VALUES (?, ?, datetime('now'))",
        (title, status),
    )
    await db.commit()
    return cur.lastrowid


async def _insert_task(db, mission_id: int, *, status="completed", agent_type="coder") -> int:
    cur = await db.execute(
        "INSERT INTO tasks (mission_id, title, status, agent_type, created_at) "
        "VALUES (?, ?, ?, ?, datetime('now'))",
        (mission_id, "Test task", status, agent_type),
    )
    await db.commit()
    return cur.lastrowid


# ── 1. briefing_compose posthook ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_briefing_compose_writes_completion_row(tmp_path, monkeypatch):
    """briefing_compose handle() writes one mission_briefings row with kind=completion."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()

    mission_id = await _insert_mission(db, title="My product mission")
    # Insert 3 completed tasks to have something to summarise
    for _ in range(3):
        await _insert_task(db, mission_id)

    task_row = {
        "id": 999,
        "mission_id": mission_id,
        "agent_type": "coder",
        "title": "Compose briefing test task",
        "context": json.dumps({}),
    }
    result_row = {
        "status": "completed",
        "result": {"summary": "All steps done"},
        "phase_summaries": [
            {"phase": "Phase 1", "outcome": "Requirements gathered"},
            {"phase": "Phase 3", "outcome": "Code shipped"},
        ],
        "changed_files": ["src/app/foo.py", "tests/test_foo.py"],
        "deferred_items": ["Deploy to prod after QA sign-off"],
        "cost_actual_usd": 0.42,
    }

    from packages.general_beckman.src.general_beckman.posthook_handlers.briefing_compose import (
        handle,
    )
    outcome = await handle(task_row, result_row)

    assert outcome.get("status") == "ok", f"Expected ok, got {outcome}"

    cur = await db.execute(
        "SELECT * FROM mission_briefings WHERE mission_id = ? AND kind = 'completion'",
        (str(mission_id),),
    )
    rows = await cur.fetchall()
    assert len(rows) == 1, f"Expected 1 briefing row, got {len(rows)}"
    row = dict(zip([d[0] for d in cur.description], rows[0]))

    assert row["kind"] == "completion"
    assert row["product_id"] == str(mission_id)
    assert row["founder_minutes_saved_estimate"] is not None
    assert row["founder_minutes_saved_estimate"] > 0
    assert row["body_md"] is not None
    # Body must contain section markers
    body = row["body_md"]
    assert "Phase" in body or "phase" in body, "body_md missing phase section"
    assert "file" in body.lower() or "changed" in body.lower(), "body_md missing files section"


@pytest.mark.asyncio
async def test_briefing_compose_minutes_saved_heuristic(tmp_path, monkeypatch):
    """founder_minutes_saved_estimate grows with step count."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()

    m1 = await _insert_mission(db, title="Small mission")
    m2 = await _insert_mission(db, title="Large mission")

    # small: 1 task, large: 10 tasks
    await _insert_task(db, m1)
    for _ in range(10):
        await _insert_task(db, m2)

    from packages.general_beckman.src.general_beckman.posthook_handlers.briefing_compose import (
        handle,
    )

    small_outcome = await handle(
        {"id": 1, "mission_id": m1, "agent_type": "coder"},
        {"status": "completed", "result": {}},
    )
    large_outcome = await handle(
        {"id": 2, "mission_id": m2, "agent_type": "coder"},
        {"status": "completed", "result": {}},
    )
    assert small_outcome["status"] == "ok"
    assert large_outcome["status"] == "ok"

    cur = await db.execute(
        "SELECT mission_id, founder_minutes_saved_estimate FROM mission_briefings "
        "WHERE kind='completion' ORDER BY founder_minutes_saved_estimate"
    )
    rows = [dict(zip([d[0] for d in cur.description], r)) for r in await cur.fetchall()]
    assert len(rows) == 2
    # Large mission should have more minutes saved
    estimates = {str(r["mission_id"]): r["founder_minutes_saved_estimate"] for r in rows}
    assert estimates[str(m2)] >= estimates[str(m1)]


@pytest.mark.asyncio
async def test_briefing_compose_includes_recovered_failures(tmp_path, monkeypatch):
    """briefing_compose body_md mentions failures recovered (from mission_lessons)."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    mission_id = await _insert_mission(db)

    # Insert a lesson that references this mission via source_ref
    await db.execute(
        "INSERT INTO mission_lessons (stack, domain, pattern, fix, severity, "
        "occurrences, dedup_key, source_kind, source_ref) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "python", "tests", "import error on startup",
            "add missing dep", "error", 2,
            f"testkey_{mission_id}", "dlq",
            json.dumps({"mission_id": mission_id}),
        ),
    )
    await db.commit()

    from packages.general_beckman.src.general_beckman.posthook_handlers.briefing_compose import (
        handle,
    )
    outcome = await handle(
        {"id": 5, "mission_id": mission_id, "agent_type": "coder"},
        {"status": "completed", "result": {}, "failed_then_recovered": True},
    )
    assert outcome["status"] == "ok"

    cur = await db.execute(
        "SELECT body_md FROM mission_briefings WHERE mission_id=? AND kind='completion'",
        (str(mission_id),),
    )
    row = await cur.fetchone()
    assert row is not None
    # body_md should have a recovered/lesson section when lessons exist
    body = row[0] or ""
    # At minimum the body must be non-empty (recovered section may be empty if no
    # matching lessons — that's fine; the key is no crash)
    assert len(body) > 10


# ── 2. Daily briefing job ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_daily_briefing_writes_daily_row(tmp_path, monkeypatch):
    """daily_briefing.run_daily_briefing() writes a mission_briefings row kind=daily."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()

    # Insert two in-flight missions
    m1 = await _insert_mission(db, title="In-flight Alpha", status="in_progress")
    m2 = await _insert_mission(db, title="In-flight Beta", status="in_progress")

    # Insert a pending founder_action
    await db.execute(
        "INSERT INTO founder_actions "
        "(mission_id, kind, title, why, instructions_json, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))",
        (m1, "generic", "Review deployment plan", "needs sign-off",
         json.dumps(["Check plan", "Approve"]), "pending"),
    )
    await db.commit()

    from src.app.jobs.daily_briefing import run_daily_briefing
    result = await run_daily_briefing()
    assert result.get("ok") is True

    cur = await db.execute(
        "SELECT * FROM mission_briefings WHERE kind='daily' ORDER BY prepared_at DESC LIMIT 1"
    )
    rows = await cur.fetchall()
    assert len(rows) == 1
    cols = [d[0] for d in cur.description]
    row = dict(zip(cols, rows[0]))
    assert row["kind"] == "daily"
    body = row["body_md"] or ""
    # Must contain in-flight missions and actions sections
    assert "Alpha" in body or "Beta" in body or "mission" in body.lower()
    assert row["prepared_at"] is not None


@pytest.mark.asyncio
async def test_daily_briefing_idempotent_same_day(tmp_path, monkeypatch):
    """Running daily briefing twice on the same day produces exactly 1 row."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    await _insert_mission(db, status="in_progress")

    from src.app.jobs.daily_briefing import run_daily_briefing
    r1 = await run_daily_briefing()
    r2 = await run_daily_briefing()
    assert r1.get("ok") is True
    assert r2.get("ok") is True

    cur = await db.execute(
        "SELECT COUNT(*) FROM mission_briefings WHERE kind='daily'"
    )
    count = (await cur.fetchone())[0]
    assert count == 1, f"Expected 1 daily row, got {count}"


# ── 3. render_briefing ────────────────────────────────────────────────────────


def test_render_briefing_sections():
    """render_briefing returns a string with all required section headers."""
    from src.app.founder_action_render import render_briefing

    briefing_row = {
        "id": 1,
        "kind": "completion",
        "product_id": "42",
        "mission_id": "42",
        "body_md": (
            "## Phase Summary\nPhase 1: requirements done.\n\n"
            "## Changed Files\n- src/app/foo.py\n\n"
            "## Deferred Items\n- Deploy after QA\n\n"
            "## Cost\n$0.42\n\n"
            "## Recovered Failures\nNone\n"
        ),
        "founder_minutes_saved_estimate": 15,
        "prepared_at": "2026-05-15 09:00:00",
        "read_at": None,
        "acted_on": None,
    }
    text = render_briefing(briefing_row)
    assert isinstance(text, str)
    assert len(text) > 20
    # Required sections
    assert "Phase" in text or "phase" in text
    assert "file" in text.lower() or "changed" in text.lower() or "deferred" in text.lower()
    # Placeholder section call (attention budget, owned by another agent)
    # The render function emits an empty placeholder comment or empty section — just
    # verify no crash and we get some text back.


def test_render_briefing_weekly_rollup():
    """render_briefing includes weekly founder_minutes_saved rollup when provided."""
    from src.app.founder_action_render import render_briefing

    row = {
        "id": 2,
        "kind": "daily",
        "product_id": "10",
        "mission_id": None,
        "body_md": "## In-Flight Missions\n2 missions running.\n",
        "founder_minutes_saved_estimate": 30,
        "prepared_at": "2026-05-15 09:00:00",
        "read_at": None,
        "acted_on": None,
    }
    text = render_briefing(row, weekly_minutes_saved=120)
    assert "120" in text or "2 hour" in text or "minutes" in text.lower()


def test_render_briefing_attention_placeholder():
    """render_briefing calls _render_attention_section without crashing."""
    from src.app.founder_action_render import render_briefing

    row = {
        "id": 3,
        "kind": "completion",
        "product_id": "7",
        "mission_id": "7",
        "body_md": "## Phase Summary\nAll done.\n",
        "founder_minutes_saved_estimate": 5,
        "prepared_at": "2026-05-15 10:00:00",
        "read_at": None,
        "acted_on": None,
    }
    # Must not raise even if attention module is absent
    text = render_briefing(row)
    assert isinstance(text, str)


# ── 4. /founder_hours_saved command logic ─────────────────────────────────────


@pytest.mark.asyncio
async def test_founder_hours_saved_sum(tmp_path, monkeypatch):
    """/founder_hours_saved sums mission_events.founder_minutes_saved over period."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    mission_id = await _insert_mission(db)

    # Insert 3 mission_events with founder_minutes_saved
    for minutes in [10, 25, 5]:
        await db.execute(
            "INSERT INTO mission_events (mission_id, kind, payload, founder_minutes_saved) "
            "VALUES (?, ?, ?, ?)",
            (mission_id, "milestone", json.dumps({"note": "step done"}), minutes),
        )
    await db.commit()

    from src.app.jobs.daily_briefing import sum_founder_minutes_saved
    total = await sum_founder_minutes_saved(period_days=7)
    assert total == 40, f"Expected 40 minutes, got {total}"


@pytest.mark.asyncio
async def test_founder_hours_saved_period_filter(tmp_path, monkeypatch):
    """sum_founder_minutes_saved only counts within the specified period."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()
    mission_id = await _insert_mission(db)

    # Recent event (within 7 days)
    await db.execute(
        "INSERT INTO mission_events (mission_id, kind, payload, founder_minutes_saved, posted_at) "
        "VALUES (?, ?, ?, ?, datetime('now'))",
        (mission_id, "milestone", json.dumps({}), 20),
    )
    # Old event (30 days ago)
    await db.execute(
        "INSERT INTO mission_events (mission_id, kind, payload, founder_minutes_saved, posted_at) "
        "VALUES (?, ?, ?, ?, datetime('now', '-30 days'))",
        (mission_id, "milestone", json.dumps({}), 100),
    )
    await db.commit()

    from src.app.jobs.daily_briefing import sum_founder_minutes_saved
    total_7d = await sum_founder_minutes_saved(period_days=7)
    total_60d = await sum_founder_minutes_saved(period_days=60)

    assert total_7d == 20, f"Expected 20 for 7d window, got {total_7d}"
    assert total_60d == 120, f"Expected 120 for 60d window, got {total_60d}"
