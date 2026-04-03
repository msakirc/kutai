# tests/test_todo.py
"""
Tests for Todo Items feature:
  - DB CRUD: add_todo, get_todos, toggle_todo, delete_todo
  - NL classification: "remind me to buy milk" -> todo
  - Keyword extraction: strip "remind me to" prefix
  - Product idea still classified as mission (not todo)
"""
import asyncio
import os
import sys
import tempfile
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    """Run an async coroutine synchronously for tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─── DB CRUD Tests ─────────────────────────────────────────────────────────

class TestTodoDB(unittest.TestCase):
    """Test todo CRUD operations against a real temp database."""

    @classmethod
    def setUpClass(cls):
        """Create a temp database and initialise schema."""
        cls._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls._tmp.close()
        cls.db_path = cls._tmp.name

        # Patch DB_PATH before importing db module
        import src.app.config as config
        cls._orig_db_path = config.DB_PATH
        config.DB_PATH = cls.db_path

        import src.infra.db as db_mod
        db_mod.DB_PATH = cls.db_path
        # Reset the singleton connection so it uses the new path
        db_mod._db_connection = None
        cls.db_mod = db_mod

        run_async(db_mod.init_db())

    @classmethod
    def tearDownClass(cls):
        """Close DB and clean up temp file."""
        run_async(cls.db_mod.close_db())
        import src.app.config as config
        config.DB_PATH = cls._orig_db_path
        try:
            os.unlink(cls.db_path)
        except OSError:
            pass

    def test_add_todo(self):
        todo_id = run_async(self.db_mod.add_todo("Buy milk"))
        self.assertIsInstance(todo_id, int)
        self.assertGreater(todo_id, 0)

    def test_get_todos(self):
        run_async(self.db_mod.add_todo("Item A"))
        run_async(self.db_mod.add_todo("Item B"))
        todos = run_async(self.db_mod.get_todos())
        self.assertGreaterEqual(len(todos), 2)
        titles = [t["title"] for t in todos]
        self.assertIn("Item A", titles)
        self.assertIn("Item B", titles)

    def test_get_todos_filter_status(self):
        tid = run_async(self.db_mod.add_todo("Pending item"))
        pending = run_async(self.db_mod.get_todos(status="pending"))
        ids = [t["id"] for t in pending]
        self.assertIn(tid, ids)

        done = run_async(self.db_mod.get_todos(status="done"))
        done_ids = [t["id"] for t in done]
        self.assertNotIn(tid, done_ids)

    def test_get_todo(self):
        tid = run_async(self.db_mod.add_todo("Specific item"))
        todo = run_async(self.db_mod.get_todo(tid))
        self.assertIsNotNone(todo)
        self.assertEqual(todo["title"], "Specific item")
        self.assertEqual(todo["status"], "pending")

    def test_get_todo_nonexistent(self):
        todo = run_async(self.db_mod.get_todo(99999))
        self.assertIsNone(todo)

    def test_toggle_todo(self):
        tid = run_async(self.db_mod.add_todo("Toggle me"))
        # First toggle: pending -> done
        new_status = run_async(self.db_mod.toggle_todo(tid))
        self.assertEqual(new_status, "done")

        todo = run_async(self.db_mod.get_todo(tid))
        self.assertEqual(todo["status"], "done")
        self.assertIsNotNone(todo["completed_at"])

        # Second toggle: done -> pending
        new_status = run_async(self.db_mod.toggle_todo(tid))
        self.assertEqual(new_status, "pending")

        todo = run_async(self.db_mod.get_todo(tid))
        self.assertEqual(todo["status"], "pending")
        self.assertIsNone(todo["completed_at"])

    def test_delete_todo(self):
        tid = run_async(self.db_mod.add_todo("Delete me"))
        run_async(self.db_mod.delete_todo(tid))
        todo = run_async(self.db_mod.get_todo(tid))
        self.assertIsNone(todo)

    def test_add_todo_with_priority_and_source(self):
        tid = run_async(self.db_mod.add_todo(
            "High priority item", priority="high", source="implicit"
        ))
        todo = run_async(self.db_mod.get_todo(tid))
        self.assertEqual(todo["priority"], "high")
        self.assertEqual(todo["source"], "implicit")

    def test_update_todo(self):
        tid = run_async(self.db_mod.add_todo("Update me"))
        run_async(self.db_mod.update_todo(tid, title="Updated title", priority="high"))
        todo = run_async(self.db_mod.get_todo(tid))
        self.assertEqual(todo["title"], "Updated title")
        self.assertEqual(todo["priority"], "high")

    def test_todo_reminder_seeded(self):
        """The init_db should seed the todo reminder scheduled task."""
        async def _check():
            db = await self.db_mod.get_db()
            cursor = await db.execute(
                "SELECT * FROM scheduled_tasks WHERE id = 9999"
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

        sched = run_async(_check())
        self.assertIsNotNone(sched)
        self.assertEqual(sched["title"], "Todo Reminder")
        self.assertIn("todo_reminder", sched["context"])

    def test_suggestion_columns_exist(self):
        """Todo items should have suggestion and suggestion_at columns."""
        todo_id = run_async(self.db_mod.add_todo("Test suggestion"))
        todo = run_async(self.db_mod.get_todo(todo_id))
        self.assertIsNone(todo["suggestion"])
        self.assertIsNone(todo["suggestion_at"])

    def test_update_todo_suggestion(self):
        """update_todo should accept suggestion and suggestion_at."""
        todo_id = run_async(self.db_mod.add_todo("Suggest me"))
        run_async(self.db_mod.update_todo(
            todo_id,
            suggestion="Compare prices online",
            suggestion_agent="shopping_advisor",
            suggestion_at="2026-04-03 10:00:00",
        ))
        todo = run_async(self.db_mod.get_todo(todo_id))
        self.assertEqual(todo["suggestion"], "Compare prices online")
        self.assertEqual(todo["suggestion_agent"], "shopping_advisor")
        self.assertEqual(todo["suggestion_at"], "2026-04-03 10:00:00")

    def test_clear_todo_suggestion(self):
        """Clearing suggestion resets both fields to NULL."""
        todo_id = run_async(self.db_mod.add_todo("Clear me"))
        run_async(self.db_mod.update_todo(
            todo_id,
            suggestion="Old suggestion",
            suggestion_agent="researcher",
            suggestion_at="2026-04-03 10:00:00",
        ))
        run_async(self.db_mod.update_todo(
            todo_id, suggestion=None, suggestion_agent=None, suggestion_at=None,
        ))
        todo = run_async(self.db_mod.get_todo(todo_id))
        self.assertIsNone(todo["suggestion"])
        self.assertIsNone(todo["suggestion_at"])


# ─── NL Classification Tests ──────────────────────────────────────────────

class TestTodoClassification(unittest.TestCase):
    """Test keyword-based classification detects todo patterns.

    We replicate the keyword classifier inline to avoid importing
    telegram_bot (which requires python-telegram-bot at import time).
    """

    @staticmethod
    def classify(text: str) -> dict:
        """Replicate _classify_message_by_keywords from telegram_bot.py."""
        lower = text.lower()

        # Todo items
        if any(w in lower for w in [
            "remind me", "don't forget", "dont forget", "todo",
            "add to list", "add to my list", "need to buy", "need to get",
            "remember to", "note to self", "hatirla", "unutma", "listeye ekle",
        ]):
            return {"type": "todo"}
        # Bug reports
        if any(w in lower for w in [
            "bug", "error", "broken", "crash", "doesn't work", "not working",
            "failed", "exception", "traceback", "issue with",
        ]):
            return {"type": "bug_report"}
        # Feature requests
        if any(w in lower for w in [
            "feature", "could you add", "would be nice", "suggestion",
            "it would help if", "can we have", "please add", "wish list",
        ]):
            return {"type": "feature_request"}
        # Mission (long or project-like)
        if len(text) > 200 or any(w in lower for w in [
            "research", "create a", "build", "analyze", "develop", "plan",
            "design a", "implement a", "set up", "write a report", "strategy",
        ]):
            result = {"type": "mission"}
            # Inline product-idea detection
            _PRODUCT_KEYWORDS = [
                "build ", "create ", "make ", "develop ", "design ",
                " app ", " app.", " application", " platform", " website", " webapp",
                " web app", " tool ", " saas", " service ", " system ",
                " bot ", " dashboard", " portal", " marketplace",
                " that allows", " that lets", " that enables", " where users",
                " for users to", " for people to",
            ]
            desc_lower = f" {text.lower()} "
            has_verb = any(kw in desc_lower for kw in _PRODUCT_KEYWORDS[:5])
            has_noun = any(kw in desc_lower for kw in _PRODUCT_KEYWORDS[5:])
            has_phrase = any(kw in desc_lower for kw in _PRODUCT_KEYWORDS[11:])
            if (has_verb and has_noun) or has_phrase:
                result["workflow"] = "i2p"
            return result
        return {"type": "task"}

    def test_remind_me_to(self):
        result = self.classify("Remind me to buy milk")
        self.assertEqual(result["type"], "todo")

    def test_dont_forget(self):
        result = self.classify("Don't forget the meeting at 3pm")
        self.assertEqual(result["type"], "todo")

    def test_note_to_self(self):
        result = self.classify("Note to self: update the invoice")
        self.assertEqual(result["type"], "todo")

    def test_need_to_buy(self):
        result = self.classify("I need to buy eggs")
        self.assertEqual(result["type"], "todo")

    def test_turkish_hatirla(self):
        result = self.classify("hatirla: market listesi")
        self.assertEqual(result["type"], "todo")

    def test_todo_keyword(self):
        result = self.classify("todo: finish the report")
        self.assertEqual(result["type"], "todo")

    def test_product_idea_not_todo(self):
        """A product/app idea should be classified as mission, not todo."""
        result = self.classify(
            "Build a web app that lets users manage their inventory"
        )
        self.assertEqual(result["type"], "mission")
        self.assertEqual(result.get("workflow"), "i2p")

    def test_bug_report_not_todo(self):
        result = self.classify("The login page is broken and shows an error")
        self.assertEqual(result["type"], "bug_report")

    def test_simple_task_not_todo(self):
        result = self.classify("Deploy the latest version")
        self.assertEqual(result["type"], "task")


# ─── Keyword Extraction Tests ─────────────────────────────────────────────

class TestTodoTitleExtraction(unittest.TestCase):
    """Test that the NL prefix stripping extracts clean todo titles."""

    @staticmethod
    def _extract_title(text: str) -> str:
        """Replicate the extraction logic from _handle_todo_from_message."""
        import re
        title = text
        for prefix in [
            r"remind me to\s+",
            r"don'?t forget to\s+",
            r"dont forget to\s+",
            r"remember to\s+",
            r"note to self[:\s]+",
            r"add to (?:my )?list[:\s]+",
            r"need to (?:buy|get)\s+",
            r"todo[:\s]+",
            r"hatirla[:\s]+",
            r"unutma[:\s]+",
            r"listeye ekle[:\s]+",
        ]:
            title = re.sub(f"^{prefix}", "", title, flags=re.IGNORECASE).strip()
        return title if title else text

    def test_remind_me_to(self):
        self.assertEqual(self._extract_title("Remind me to buy milk"), "buy milk")

    def test_dont_forget_to(self):
        self.assertEqual(
            self._extract_title("Don't forget to call the dentist"),
            "call the dentist",
        )

    def test_todo_colon(self):
        self.assertEqual(
            self._extract_title("todo: finish the report"),
            "finish the report",
        )

    def test_note_to_self(self):
        self.assertEqual(
            self._extract_title("Note to self: update CV"),
            "update CV",
        )

    def test_need_to_buy(self):
        self.assertEqual(self._extract_title("need to buy eggs"), "eggs")

    def test_plain_text_unchanged(self):
        self.assertEqual(
            self._extract_title("some random text"),
            "some random text",
        )


from datetime import datetime, timedelta


def test_format_age():
    from src.app.reminders import _format_age
    now = datetime(2026, 3, 26, 12, 0, 0)
    assert _format_age(now - timedelta(minutes=30), now) == "30m ago"
    assert _format_age(now - timedelta(hours=3), now) == "3h ago"
    assert _format_age(now - timedelta(days=2), now) == "2d ago"
    assert _format_age(now - timedelta(weeks=3), now) == "3w ago"
    assert _format_age(now - timedelta(minutes=2), now) == "2m ago"
    # Edge: just created
    assert _format_age(now, now) == "now"


class TestSchedulerNextRun(unittest.TestCase):
    """Verify the scheduler stores next_run in SQLite-compatible format."""

    @classmethod
    def setUpClass(cls):
        """Create a temp database and initialise schema."""
        cls._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls._tmp.close()
        cls.db_path = cls._tmp.name

        import src.app.config as config
        cls._orig_db_path = config.DB_PATH
        config.DB_PATH = cls.db_path

        import src.infra.db as db_mod
        db_mod.DB_PATH = cls.db_path
        db_mod._db_connection = None
        cls.db_mod = db_mod

        async def _setup():
            await db_mod.init_db()
            # Insert a test scheduled task with a known id so UPDATE/SELECT work.
            db = await db_mod.get_db()
            await db.execute(
                """INSERT OR IGNORE INTO scheduled_tasks
                   (id, title, cron_expression, agent_type, tier, enabled)
                   VALUES (9999, '_test_sched', '0 * * * *', 'executor', 'cheap', 1)"""
            )
            await db.commit()
        run_async(_setup())

    @classmethod
    def tearDownClass(cls):
        run_async(cls.db_mod.close_db())
        import src.app.config as config
        config.DB_PATH = cls._orig_db_path
        try:
            os.unlink(cls.db_path)
        except OSError:
            pass

    def test_next_run_format_matches_sqlite(self):
        """next_run must be stored as 'YYYY-MM-DD HH:MM:SS' (no T, no tz offset)
        so that 'next_run <= datetime(\"now\")' works in SQLite."""
        from datetime import timezone

        async def _check_next_run_format():
            db = await self.db_mod.get_db()

            _DB_DT_FMT = "%Y-%m-%d %H:%M:%S"
            now = datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
            next_run = datetime(2026, 3, 27, 12, 0, 0, tzinfo=timezone.utc)

            await self.db_mod.update_scheduled_task(
                9999,
                last_run=now.strftime(_DB_DT_FMT),
                next_run=next_run.strftime(_DB_DT_FMT),
            )

            cursor = await db.execute(
                "SELECT next_run FROM scheduled_tasks WHERE id = 9999"
            )
            row = await cursor.fetchone()
            stored = row[0]
            self.assertEqual(stored, "2026-03-27 12:00:00")
            self.assertNotIn("T", stored)
            self.assertNotIn("+", stored)

        run_async(_check_next_run_format())

    def test_due_query_finds_past_next_run(self):
        """Scheduled task with next_run in the past should be returned by get_due_scheduled_tasks."""
        async def _check():
            await self.db_mod.update_scheduled_task(
                9999, next_run="2020-01-01 00:00:00"
            )
            due = await self.db_mod.get_due_scheduled_tasks()
            ids = [s["id"] for s in due]
            self.assertIn(9999, ids)

        run_async(_check())

    def test_due_query_skips_future_next_run(self):
        """Scheduled task with next_run in the future should NOT be returned."""
        async def _check():
            await self.db_mod.update_scheduled_task(
                9999, next_run="2099-12-31 23:59:59"
            )
            due = await self.db_mod.get_due_scheduled_tasks()
            ids = [s["id"] for s in due]
            self.assertNotIn(9999, ids)

        run_async(_check())


class TestComputeNextRun(unittest.TestCase):
    """Test the cron next-run calculator (extracted from Orchestrator)."""

    @staticmethod
    def _compute_next_run(cron_expr, after):
        """Replicate Orchestrator._compute_next_run to avoid importing telegram deps."""
        from datetime import timedelta
        try:
            parts = cron_expr.strip().split()
            if len(parts) != 5:
                return None
            minute, hour, day, month, weekday = parts
            if minute != "*" and hour == "*":
                m = int(minute)
                candidate = after.replace(minute=m, second=0, microsecond=0)
                if candidate <= after:
                    candidate += timedelta(hours=1)
                return candidate
            if minute != "*" and hour != "*":
                m = int(minute)
                if "," in hour:
                    hours = sorted(int(h) for h in hour.split(","))
                    for h in hours:
                        candidate = after.replace(hour=h, minute=m, second=0, microsecond=0)
                        if candidate > after:
                            return candidate
                    candidate = after.replace(hour=hours[0], minute=m, second=0, microsecond=0)
                    return candidate + timedelta(days=1)
                h = int(hour)
                candidate = after.replace(hour=h, minute=m, second=0, microsecond=0)
                if candidate <= after:
                    candidate += timedelta(days=1)
                return candidate
            return after + timedelta(hours=1)
        except Exception:
            return None

    def test_comma_separated_hours(self):
        from datetime import timezone
        after = datetime(2026, 3, 27, 7, 30, 0, tzinfo=timezone.utc)
        result = self._compute_next_run("0 6,8,10,12,14,16,18 * * *", after)
        self.assertIsNotNone(result)
        self.assertEqual(result.hour, 8)
        self.assertEqual(result.minute, 0)

    def test_all_hours_passed_wraps_to_next_day(self):
        from datetime import timezone
        after = datetime(2026, 3, 27, 18, 30, 0, tzinfo=timezone.utc)
        result = self._compute_next_run("0 6,8,10,12,14,16,18 * * *", after)
        self.assertIsNotNone(result)
        self.assertEqual(result.day, 28)
        self.assertEqual(result.hour, 6)
        self.assertEqual(result.minute, 0)

    def test_hourly_at_minute(self):
        from datetime import timezone
        after = datetime(2026, 3, 27, 10, 15, 0, tzinfo=timezone.utc)
        result = self._compute_next_run("30 * * * *", after)
        self.assertIsNotNone(result)
        self.assertEqual(result.hour, 10)
        self.assertEqual(result.minute, 30)


if __name__ == "__main__":
    unittest.main()
