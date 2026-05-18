"""Z9 Growth T3D — DLQ feedback hook tests.

Covers ``src.infra.dlq_feedback.mine_dlq_patterns``:

  * a recurring failure cluster (>= MIN_OCCURRENCES) writes one
    ``dlq_pattern`` growth_events row with the right occurrence_count;
  * re-running the miner is idempotent — no duplicate rows;
  * a sub-threshold group produces no row;
  * the dlq_signal_review cron + mine_dlq_patterns executor are wired.
"""
from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest

import src.infra.db as db_mod
from src.infra.dlq_feedback import MIN_OCCURRENCES, mine_dlq_patterns


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """A real KutAI DB (full schema via init_db) at a temp path."""
    db_path = str(tmp_path / "dlq_feedback_test.db")
    # Reset cached connection + point DB_PATH at the temp file.
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    _run(db_mod.init_db())
    yield db_path
    # Close so the next test gets a clean connection.
    conn = getattr(db_mod, "_db_connection", None)
    if conn is not None:
        _run(conn.close())
    db_mod._db_connection = None
    db_mod._db_connection_path = None


async def _seed_dlq(rows: list[dict]) -> None:
    """Insert dead_letter_tasks rows directly."""
    from src.infra.dead_letter import _ensure_dlq_table

    await _ensure_dlq_table()
    db = await db_mod.get_db()
    for r in rows:
        await db.execute(
            """INSERT OR REPLACE INTO dead_letter_tasks
               (task_id, mission_id, error, error_category, original_agent,
                attempts_snapshot, quarantined_at, resolved_at, resolution)
               VALUES (?, ?, ?, ?, ?, 0, datetime('now'), NULL, NULL)""",
            (
                r["task_id"],
                r.get("mission_id"),
                r.get("error", "boom"),
                r.get("error_category", "timeout"),
                r.get("original_agent", "coder"),
            ),
        )
    await db.commit()


def test_recurring_pattern_emits_growth_event(temp_db):
    """3 failures with the same (category, agent) → one dlq_pattern row."""
    _run(_seed_dlq([
        {"task_id": 101, "error_category": "timeout", "original_agent": "coder"},
        {"task_id": 102, "error_category": "timeout", "original_agent": "coder"},
        {"task_id": 103, "error_category": "timeout", "original_agent": "coder"},
    ]))

    written = _run(mine_dlq_patterns())
    assert written == 1

    events = _run(db_mod.get_growth_events(kind="dlq_pattern"))
    assert len(events) == 1
    props = events[0]["properties_json"]
    assert props["error_category"] == "timeout"
    assert props["agent"] == "coder"
    assert props["occurrence_count"] == 3
    assert sorted(props["sample_task_ids"]) == [101, 102, 103]
    assert props["pattern_key"]
    assert props["first_seen"] and props["last_seen"]
    # global signal — not bound to a mission
    assert events[0]["mission_id"] is None


def test_idempotent_rerun_no_duplicates(temp_db):
    """Running the miner twice must not duplicate the dlq_pattern row."""
    _run(_seed_dlq([
        {"task_id": 201, "error_category": "parse_error", "original_agent": "fixer"},
        {"task_id": 202, "error_category": "parse_error", "original_agent": "fixer"},
        {"task_id": 203, "error_category": "parse_error", "original_agent": "fixer"},
    ]))

    first = _run(mine_dlq_patterns())
    second = _run(mine_dlq_patterns())
    assert first == 1
    assert second == 0  # already emitted — deduped

    events = _run(db_mod.get_growth_events(kind="dlq_pattern"))
    assert len(events) == 1


def test_sub_threshold_group_no_row(temp_db):
    """A group below MIN_OCCURRENCES must not produce a dlq_pattern row."""
    assert MIN_OCCURRENCES == 3
    _run(_seed_dlq([
        {"task_id": 301, "error_category": "auth_failure", "original_agent": "researcher"},
        {"task_id": 302, "error_category": "auth_failure", "original_agent": "researcher"},
    ]))

    written = _run(mine_dlq_patterns())
    assert written == 0

    events = _run(db_mod.get_growth_events(kind="dlq_pattern"))
    assert events == []


def test_distinct_groups_emit_separately(temp_db):
    """Two distinct clusters above threshold → two dlq_pattern rows."""
    _run(_seed_dlq([
        {"task_id": 401, "error_category": "timeout", "original_agent": "coder"},
        {"task_id": 402, "error_category": "timeout", "original_agent": "coder"},
        {"task_id": 403, "error_category": "timeout", "original_agent": "coder"},
        {"task_id": 411, "error_category": "rate_limit", "original_agent": "planner"},
        {"task_id": 412, "error_category": "rate_limit", "original_agent": "planner"},
        {"task_id": 413, "error_category": "rate_limit", "original_agent": "planner"},
        # sub-threshold — ignored
        {"task_id": 421, "error_category": "network_error", "original_agent": "coder"},
    ]))

    written = _run(mine_dlq_patterns())
    assert written == 2

    events = _run(db_mod.get_growth_events(kind="dlq_pattern"))
    keys = {e["properties_json"]["pattern_key"] for e in events}
    assert len(keys) == 2


def test_resolved_dlq_rows_excluded(temp_db):
    """Resolved DLQ rows are not mined."""
    _run(_seed_dlq([
        {"task_id": 501, "error_category": "timeout", "original_agent": "coder"},
        {"task_id": 502, "error_category": "timeout", "original_agent": "coder"},
        {"task_id": 503, "error_category": "timeout", "original_agent": "coder"},
    ]))
    # Resolve one — now only 2 remain unresolved, below threshold.
    from src.infra.dead_letter import resolve_dlq_task
    _run(resolve_dlq_task(503, resolution="manual"))

    written = _run(mine_dlq_patterns())
    assert written == 0


# ---------------------------------------------------------------------------
# Cron + executor wiring
# ---------------------------------------------------------------------------

def test_dlq_signal_review_cron_seeded():
    from general_beckman.cron_seed import INTERNAL_CADENCES

    match = [c for c in INTERNAL_CADENCES if c["title"] == "dlq_signal_review"]
    assert len(match) == 1, "dlq_signal_review cadence not seeded"
    cadence = match[0]
    assert cadence["interval_seconds"] == 604800  # weekly
    assert cadence["payload"]["_executor"] == "mine_dlq_patterns"


def test_mine_dlq_patterns_executor_dispatched():
    """mr_roboto must route the mine_dlq_patterns action."""
    init_path = (
        Path(__file__).resolve().parents[1]
        / "packages" / "mr_roboto" / "src" / "mr_roboto" / "__init__.py"
    )
    text = init_path.read_text(encoding="utf-8")
    assert 'action == "mine_dlq_patterns"' in text
