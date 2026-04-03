"""Tests for todo UI message builders."""
import asyncio
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestTodoListMessage(unittest.TestCase):
    """Test the todo list (overview) message builder."""

    @classmethod
    def setUpClass(cls):
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
        run_async(db_mod.init_db())

    @classmethod
    def tearDownClass(cls):
        run_async(cls.db_mod.close_db())
        import src.app.config as config
        config.DB_PATH = cls._orig_db_path
        try:
            os.unlink(cls.db_path)
        except OSError:
            pass

    def test_empty_list(self):
        from src.app.reminders import build_todo_list_message
        text, markup = run_async(build_todo_list_message())
        self.assertIsNone(text)

    def test_list_shows_numbered_todos(self):
        from src.app.reminders import build_todo_list_message
        run_async(self.db_mod.add_todo("Item A"))
        run_async(self.db_mod.add_todo("Item B"))
        text, markup = run_async(build_todo_list_message())
        self.assertIn("1.", text)
        self.assertIn("2.", text)
        self.assertIn("Item A", text)
        self.assertIn("Item B", text)

    def test_suggestion_hint_shown(self):
        from src.app.reminders import build_todo_list_message
        tid = run_async(self.db_mod.add_todo("Item C"))
        run_async(self.db_mod.update_todo(
            tid, suggestion="Compare prices online for best deals",
            suggestion_agent="shopping_advisor",
            suggestion_at="2026-04-03 10:00:00",
        ))
        text, markup = run_async(build_todo_list_message())
        # Should show truncated suggestion hint
        self.assertIn("Compare prices", text)
        self.assertIn("💡", text)

    def test_numbered_buttons(self):
        """Buttons should be numbered, not action-specific."""
        from src.app.reminders import build_todo_list_message
        text, markup = run_async(build_todo_list_message())
        self.assertIsNotNone(markup)
        # Check inline keyboard has numbered buttons
        buttons = []
        for row in markup.inline_keyboard:
            for btn in row:
                buttons.append(btn.text)
        # Should have at least "1", "2", "3" style buttons
        numbered = [b for b in buttons if b.strip().isdigit()]
        self.assertGreater(len(numbered), 0)


class TestTodoDetailMessage(unittest.TestCase):
    """Test the todo detail view builder."""

    @classmethod
    def setUpClass(cls):
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
        run_async(db_mod.init_db())

    @classmethod
    def tearDownClass(cls):
        run_async(cls.db_mod.close_db())
        import src.app.config as config
        config.DB_PATH = cls._orig_db_path
        try:
            os.unlink(cls.db_path)
        except OSError:
            pass

    def test_detail_with_suggestion(self):
        from src.app.reminders import build_todo_detail_message
        tid = run_async(self.db_mod.add_todo("Buy milk"))
        run_async(self.db_mod.update_todo(
            tid, suggestion="Compare milk prices online",
            suggestion_agent="shopping_advisor",
            suggestion_at="2026-04-03 10:00:00",
        ))
        todo = run_async(self.db_mod.get_todo(tid))
        text, markup = build_todo_detail_message(todo)
        self.assertIn("Buy milk", text)
        self.assertIn("Compare milk prices", text)
        self.assertIn("💡", text)
        # Should have Help button when suggestion exists
        btn_texts = [b.text for row in markup.inline_keyboard for b in row]
        self.assertIn("🤖 Help", btn_texts)
        self.assertIn("✅ Done", btn_texts)
        self.assertIn("✏️ Edit", btn_texts)

    def test_detail_without_suggestion(self):
        from src.app.reminders import build_todo_detail_message
        tid = run_async(self.db_mod.add_todo("Call dentist"))
        todo = run_async(self.db_mod.get_todo(tid))
        text, markup = build_todo_detail_message(todo)
        self.assertIn("Call dentist", text)
        btn_texts = [b.text for row in markup.inline_keyboard for b in row]
        self.assertNotIn("🤖 Help", btn_texts)
        self.assertIn("✅ Done", btn_texts)


if __name__ == "__main__":
    unittest.main()
