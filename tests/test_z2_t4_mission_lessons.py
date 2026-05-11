"""Z2 T4A + T4B — mission_lessons schema, dedup upsert, and populators.

Tests:
  T4A — schema CRUD + dedup
  T4B — posthook-fail populator via _maybe_emit_lesson_from_posthook_fail
  T4B — DLQ emitter via emit_lessons_from_dlq_patterns
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures — in-memory SQLite via a patched get_db
# ---------------------------------------------------------------------------

def _make_db_path(tmp_path: Path) -> str:
    """Create the mission_lessons table (and minimal tables) in a temp DB."""
    db_path = str(tmp_path / "test.db")
    con = sqlite3.connect(db_path)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS mission_lessons (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            stack        TEXT    NOT NULL,
            domain       TEXT    NOT NULL,
            pattern      TEXT    NOT NULL,
            fix          TEXT    NOT NULL DEFAULT '',
            severity     TEXT    NOT NULL DEFAULT 'warning',
            occurrences  INTEGER NOT NULL DEFAULT 1,
            dedup_key    TEXT    NOT NULL UNIQUE,
            source_kind  TEXT    NOT NULL,
            source_ref   TEXT    NOT NULL DEFAULT '{}',
            suppressed   INTEGER NOT NULL DEFAULT 0,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_mission_lessons_stack_domain
            ON mission_lessons(stack, domain);
        CREATE INDEX IF NOT EXISTS idx_mission_lessons_occurrences
            ON mission_lessons(occurrences DESC, last_seen_at DESC);

        -- Minimal tables for populator tests
        CREATE TABLE IF NOT EXISTS dead_letter_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            mission_id INTEGER,
            error TEXT,
            error_category TEXT DEFAULT 'unknown',
            original_agent TEXT,
            attempts_snapshot INTEGER DEFAULT 0,
            quarantined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            resolution TEXT,
            UNIQUE(task_id)
        );
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            feedback TEXT,
            worker_attempts INTEGER DEFAULT 0,
            max_worker_attempts INTEGER DEFAULT 15,
            mission_id INTEGER,
            agent_type TEXT,
            status TEXT DEFAULT 'pending',
            context TEXT DEFAULT '{}',
            error TEXT,
            error_category TEXT,
            result TEXT,
            task_state TEXT
        );
        CREATE TABLE IF NOT EXISTS missions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            context TEXT DEFAULT '{}'
        );
    """)
    con.commit()
    con.close()
    return db_path


class _AsyncCon:
    """Thin async wrapper around a synchronous sqlite3 connection."""

    def __init__(self, db_path: str):
        self._path = db_path
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row

    async def execute(self, sql: str, params=()):
        cur = self._con.execute(sql, params)
        return _AsyncCursor(cur, self._con)

    async def commit(self):
        self._con.commit()

    async def close(self):
        self._con.close()


class _AsyncCursor:
    def __init__(self, cur: sqlite3.Cursor, con: sqlite3.Connection):
        self._cur = cur
        self._con = con
        self.lastrowid = cur.lastrowid
        self.description = cur.description

    async def fetchone(self):
        row = self._cur.fetchone()
        if row is None:
            return None
        return tuple(row)

    async def fetchall(self):
        return [tuple(r) for r in self._cur.fetchall()]


# ---------------------------------------------------------------------------
# T4A — Basic CRUD + dedup
# ---------------------------------------------------------------------------

class TestMissionLessonsCRUD:
    """Unit tests for upsert_mission_lesson, top_mission_lessons, suppress."""

    def _run(self, coro):
        return asyncio.new_event_loop().run_until_complete(coro)

    def setup_method(self):
        self._tmp = tempfile.mkdtemp()
        self._db_path = _make_db_path(Path(self._tmp))
        self._con = _AsyncCon(self._db_path)

    def teardown_method(self):
        self._run(self._con.close())

    def _patch(self):
        return patch(
            "src.infra.db.get_db",
            AsyncMock(return_value=self._con),
        )

    # ── upsert basics ─────────────────────────────────────────────────────

    def test_upsert_new_row_occurrences_1(self):
        from src.infra.mission_lessons import upsert_mission_lesson
        with self._patch():
            row_id = self._run(upsert_mission_lesson(
                "fastapi", "imports", "module not found",
                source_kind="posthook_fail",
            ))
        assert isinstance(row_id, int) and row_id > 0

        con = sqlite3.connect(self._db_path)
        row = con.execute("SELECT occurrences FROM mission_lessons WHERE id = ?", (row_id,)).fetchone()
        con.close()
        assert row[0] == 1

    def test_upsert_same_dedup_key_bumps_occurrences(self):
        """Same (stack, domain, pattern) → occurrences goes from 1 to 2."""
        from src.infra.mission_lessons import upsert_mission_lesson
        with self._patch():
            id1 = self._run(upsert_mission_lesson(
                "fastapi", "imports", "Module Not Found",
                source_kind="posthook_fail",
            ))
            id2 = self._run(upsert_mission_lesson(
                "fastapi", "imports", "module  not  found.",  # case+whitespace variant
                source_kind="posthook_fail",
            ))
        assert id1 == id2, "Same dedup_key must resolve to same row"

        con = sqlite3.connect(self._db_path)
        occ = con.execute("SELECT occurrences FROM mission_lessons WHERE id = ?", (id1,)).fetchone()[0]
        con.close()
        assert occ == 2

    def test_different_domain_separate_row(self):
        from src.infra.mission_lessons import upsert_mission_lesson
        with self._patch():
            id_a = self._run(upsert_mission_lesson(
                "fastapi", "imports", "missing dep",
                source_kind="dlq_pattern",
            ))
            id_b = self._run(upsert_mission_lesson(
                "fastapi", "quality", "missing dep",
                source_kind="dlq_pattern",
            ))
        assert id_a != id_b

    def test_fix_not_overwritten_by_empty(self):
        """Empty fix on conflict must NOT overwrite existing non-empty fix."""
        from src.infra.mission_lessons import upsert_mission_lesson
        with self._patch():
            self._run(upsert_mission_lesson(
                "fastapi", "imports", "bad import",
                fix="run pip install X",
                source_kind="posthook_fail",
            ))
            row_id = self._run(upsert_mission_lesson(
                "fastapi", "imports", "bad import",
                fix="",  # empty — should NOT clobber
                source_kind="posthook_fail",
            ))

        con = sqlite3.connect(self._db_path)
        fix = con.execute("SELECT fix FROM mission_lessons WHERE id = ?", (row_id,)).fetchone()[0]
        con.close()
        assert fix == "run pip install X"

    def test_fix_updated_when_new_nonempty(self):
        """Non-empty fix on conflict DOES update the fix column."""
        from src.infra.mission_lessons import upsert_mission_lesson
        with self._patch():
            self._run(upsert_mission_lesson(
                "fastapi", "imports", "bad import v2",
                fix="old fix",
                source_kind="posthook_fail",
            ))
            row_id = self._run(upsert_mission_lesson(
                "fastapi", "imports", "bad import v2",
                fix="new better fix",
                source_kind="posthook_fail",
            ))

        con = sqlite3.connect(self._db_path)
        fix = con.execute("SELECT fix FROM mission_lessons WHERE id = ?", (row_id,)).fetchone()[0]
        con.close()
        assert fix == "new better fix"

    # ── top_mission_lessons ───────────────────────────────────────────────

    def test_top_n_ordering_by_occurrences(self):
        """Row with occurrences=5 ranks above occurrences=2."""
        from src.infra.mission_lessons import upsert_mission_lesson, top_mission_lessons

        with self._patch():
            # Insert two rows, bump first 5 times total (5 inserts with same key).
            for _ in range(5):
                self._run(upsert_mission_lesson(
                    "nextjs", "schema", "type error A",
                    source_kind="dlq_pattern",
                ))
            for _ in range(2):
                self._run(upsert_mission_lesson(
                    "nextjs", "schema", "type error B",
                    source_kind="dlq_pattern",
                ))
            lessons = self._run(top_mission_lessons("nextjs", "schema", limit=10))

        assert len(lessons) == 2
        assert lessons[0]["occurrences"] == 5
        assert lessons[1]["occurrences"] == 2

    def test_suppressed_row_excluded(self):
        """suppressed=1 rows are excluded from top_mission_lessons."""
        from src.infra.mission_lessons import (
            upsert_mission_lesson, top_mission_lessons, suppress_mission_lesson,
        )
        with self._patch():
            row_id = self._run(upsert_mission_lesson(
                "django", "quality", "bad output",
                source_kind="posthook_fail",
            ))
            self._run(suppress_mission_lesson(row_id))
            lessons = self._run(top_mission_lessons("django", "quality"))

        assert all(r["id"] != row_id for r in lessons)

    def test_domain_none_matches_any(self):
        """domain=None in top_mission_lessons returns rows from any domain."""
        from src.infra.mission_lessons import upsert_mission_lesson, top_mission_lessons
        with self._patch():
            self._run(upsert_mission_lesson("rails", "imports", "p1", source_kind="dlq_pattern"))
            self._run(upsert_mission_lesson("rails", "quality", "p2", source_kind="dlq_pattern"))
            lessons = self._run(top_mission_lessons("rails", None, limit=10))

        assert len(lessons) == 2


# ---------------------------------------------------------------------------
# T4B — DLQ emitter
# ---------------------------------------------------------------------------

class TestDLQEmitter:
    """emit_lessons_from_dlq_patterns: 3+ DLQ rows sharing category → lesson."""

    def _run(self, coro):
        return asyncio.new_event_loop().run_until_complete(coro)

    def setup_method(self):
        self._tmp = tempfile.mkdtemp()
        self._db_path = _make_db_path(Path(self._tmp))
        self._con = _AsyncCon(self._db_path)

    def teardown_method(self):
        self._run(self._con.close())

    def _patch(self):
        return patch(
            "src.infra.db.get_db",
            AsyncMock(return_value=self._con),
        )

    def test_three_dlq_rows_same_category_emits_lesson(self):
        """3 DLQ rows with same error_category → 1 lesson upserted."""
        # Seed 3 tasks + 3 DLQ rows.
        con = sqlite3.connect(self._db_path)
        for i in range(1, 4):
            con.execute(
                "INSERT INTO tasks (id, title, mission_id) VALUES (?, ?, 1)",
                (i, f"task {i}"),
            )
            con.execute(
                "INSERT INTO dead_letter_tasks "
                "(task_id, mission_id, error, error_category) VALUES (?, 1, ?, 'quality')",
                (i, f"schema fail #{i}"),
            )
        con.commit()
        con.close()

        with self._patch():
            n = self._run(
                __import__(
                    "src.infra.mission_lessons", fromlist=["emit_lessons_from_dlq_patterns"]
                ).emit_lessons_from_dlq_patterns()
            )

        assert n >= 1

        con = sqlite3.connect(self._db_path)
        count = con.execute("SELECT COUNT(*) FROM mission_lessons WHERE domain = 'quality'").fetchone()[0]
        con.close()
        assert count >= 1

    def test_two_dlq_rows_below_threshold_no_lesson(self):
        """2 DLQ rows (< 3 threshold) → no lesson emitted."""
        con = sqlite3.connect(self._db_path)
        for i in range(10, 12):
            con.execute("INSERT INTO tasks (id, title, mission_id) VALUES (?, ?, 2)", (i, f"task {i}"))
            con.execute(
                "INSERT INTO dead_letter_tasks "
                "(task_id, mission_id, error, error_category) VALUES (?, 2, 'bad', 'timeout')",
                (i,),
            )
        con.commit()
        con.close()

        with self._patch():
            n = self._run(
                __import__(
                    "src.infra.mission_lessons", fromlist=["emit_lessons_from_dlq_patterns"]
                ).emit_lessons_from_dlq_patterns()
            )

        assert n == 0


# ---------------------------------------------------------------------------
# T4B — posthook-fail populator integration
# ---------------------------------------------------------------------------

class TestPosthookFailPopulator:
    """Drive _maybe_emit_lesson_from_posthook_fail directly."""

    def _run(self, coro):
        return asyncio.new_event_loop().run_until_complete(coro)

    def setup_method(self):
        self._tmp = tempfile.mkdtemp()
        self._db_path = _make_db_path(Path(self._tmp))
        self._con = _AsyncCon(self._db_path)

    def teardown_method(self):
        self._run(self._con.close())

    def test_posthook_fail_upserts_lesson(self):
        from general_beckman.apply import _maybe_emit_lesson_from_posthook_fail

        source = {"id": 42, "mission_id": None}

        with patch("src.infra.db.get_db", AsyncMock(return_value=self._con)):
            self._run(_maybe_emit_lesson_from_posthook_fail(
                source=source,
                kind="test_run",
                error_str="pytest failed: 3 errors",
                feedback="Fix the broken test assertions",
                attempts=15,
            ))

        con = sqlite3.connect(self._db_path)
        row = con.execute(
            "SELECT domain, severity, source_kind, occurrences FROM mission_lessons WHERE domain = 'test_run'"
        ).fetchone()
        con.close()

        assert row is not None
        assert row[0] == "test_run"
        assert row[1] == "blocker"
        assert row[2] == "posthook_fail"
        assert row[3] == 1

    def test_posthook_fail_idempotent(self):
        """Same posthook fail emitted twice → occurrences bumps to 2, no duplicate."""
        from general_beckman.apply import _maybe_emit_lesson_from_posthook_fail

        source = {"id": 43, "mission_id": None}
        kwargs = dict(
            source=source, kind="imports_check",
            error_str="missing: requests", feedback="pip install requests",
            attempts=15,
        )
        with patch("src.infra.db.get_db", AsyncMock(return_value=self._con)):
            self._run(_maybe_emit_lesson_from_posthook_fail(**kwargs))
            self._run(_maybe_emit_lesson_from_posthook_fail(**kwargs))

        con = sqlite3.connect(self._db_path)
        rows = con.execute("SELECT occurrences FROM mission_lessons WHERE domain = 'imports_check'").fetchall()
        con.close()
        assert len(rows) == 1
        assert rows[0][0] == 2

    def test_posthook_fail_exception_does_not_propagate(self):
        """Exception inside emit must be swallowed (try/except)."""
        from general_beckman.apply import _maybe_emit_lesson_from_posthook_fail

        # get_db raises — should not propagate
        with patch("src.infra.db.get_db", AsyncMock(side_effect=RuntimeError("db down"))):
            # Must not raise
            self._run(_maybe_emit_lesson_from_posthook_fail(
                source={"id": 99, "mission_id": None},
                kind="grounding",
                error_str="boom",
                feedback="",
                attempts=5,
            ))
