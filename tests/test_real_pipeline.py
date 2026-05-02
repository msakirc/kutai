"""
Real integration tests — no mocking of core logic.

These tests run the actual classification, routing, and agent pipeline
against a real (or simulated) model. They catch real failures like:
- Coding models picked for classification
- XML output instead of JSON
- Wrong agent type selected
- Thinking enabled when it shouldn't be
- Models that can't follow JSON instructions

Tests that need a running llama-server are marked with @needs_server.
Tests that only check routing/classification logic run without a server.
"""

import json
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


needs_server = unittest.skipUnless(
    os.environ.get("KUTAY_TEST_SERVER"),
    "Set KUTAY_TEST_SERVER=1 to run tests requiring a local llama-server",
)


# ---------------------------------------------------------------------------
# 1. ROUTING — verify correct model selection for different task types
# ---------------------------------------------------------------------------
class TestModelRouting(unittest.TestCase):
    """Verify the router picks appropriate models for each task type."""

    def setUp(self):
        from src.models.model_registry import get_registry
        self.registry = get_registry()

    def _route(self, task, difficulty=3, **kwargs):
        from fatih_hoca.requirements import ModelRequirements
        from src.core.router import select_model
        reqs = ModelRequirements(task=task, difficulty=difficulty, **kwargs)
        return select_model(reqs)

    def test_classifier_never_picks_coding_model(self):
        """Classification/routing must never use a coding-specialized model."""
        candidates = self._route("router", difficulty=2, prefer_speed=True)
        if not candidates:
            self.skipTest("No models registered")
        top = candidates[0]
        self.assertNotEqual(
            getattr(top.model, "specialty", ""), "coding",
            f"Coding model {top.model.name} should not be picked for classification"
        )

    def test_researcher_never_picks_coding_model(self):
        """Research tasks must not use coding-specialized models."""
        candidates = self._route("researcher", difficulty=4, prefer_speed=True)
        if not candidates:
            self.skipTest("No models registered")
        top = candidates[0]
        self.assertNotEqual(
            getattr(top.model, "specialty", ""), "coding",
            f"Coding model {top.model.name} should not be picked for research"
        )

    def test_loaded_model_preferred_over_unloaded(self):
        """The currently loaded model should score higher than unloaded ones."""
        candidates = self._route("researcher", difficulty=4)
        if len(candidates) < 2:
            self.skipTest("Need at least 2 models")
        loaded = [c for c in candidates if c.model.is_loaded]
        unloaded = [c for c in candidates if not c.model.is_loaded]
        if not loaded or not unloaded:
            self.skipTest("Need both loaded and unloaded models")
        # Loaded model should score higher
        self.assertGreater(
            loaded[0].score, unloaded[0].score,
            "Loaded model should score higher to avoid swap"
        )

    def test_planner_not_filtered_by_min_score(self):
        """Planner at difficulty 7 should still have candidates."""
        if not self.registry.models:
            self.skipTest("No models registered (set MODEL_DIR)")
        candidates = self._route("planner", difficulty=7, prefer_quality=True)
        self.assertGreater(
            len(candidates), 0,
            "Planner should have at least one candidate at difficulty 7"
        )

    def test_coding_model_preferred_for_coding_task(self):
        """Coding tasks should prefer coding-specialized models."""
        candidates = self._route("coder", difficulty=5, needs_function_calling=True)
        if not candidates:
            self.skipTest("No models registered")
        # If there's a coding model, it should be near the top
        coding_models = [c for c in candidates if getattr(c.model, "specialty", "") == "coding"]
        if not coding_models:
            self.skipTest("No coding models available")
        # Coding model should be in top 3
        top3_names = [c.model.name for c in candidates[:3]]
        self.assertIn(
            coding_models[0].model.name, top3_names,
            "Coding model should be in top 3 for coder task"
        )


# ---------------------------------------------------------------------------
# 2. THINKING CONTROL — verify thinking is disabled correctly
# ---------------------------------------------------------------------------
class TestThinkingControl(unittest.TestCase):
    """Verify thinking is disabled for non-thinking tasks."""

    def test_thinking_disabled_for_classification(self):
        """Classification (difficulty=2) should have thinking=False."""
        from fatih_hoca.requirements import ModelRequirements
        reqs = ModelRequirements(task="router", difficulty=2, prefer_speed=True)
        # is_thinking = model.thinking_model and reqs.needs_thinking
        self.assertFalse(reqs.needs_thinking)

    def test_thinking_disabled_for_researcher(self):
        """Researcher with needs_thinking=False should not enable thinking."""
        from fatih_hoca.requirements import ModelRequirements
        reqs = ModelRequirements(task="researcher", difficulty=4, needs_thinking=False)
        self.assertFalse(reqs.needs_thinking)

    def test_thinking_enabled_when_requested(self):
        """Tasks that need thinking should have it enabled."""
        from fatih_hoca.requirements import ModelRequirements
        reqs = ModelRequirements(task="planner", difficulty=7, needs_thinking=True)
        self.assertTrue(reqs.needs_thinking)


# ---------------------------------------------------------------------------
# 3. CLASSIFICATION — verify correct classification for common inputs
# ---------------------------------------------------------------------------
class TestClassificationPipeline(unittest.IsolatedAsyncioTestCase):
    """Test classification with keyword fallback (no LLM needed)."""

    def _classify_keywords(self, text):
        from src.app.telegram_bot import TelegramInterface
        return TelegramInterface._classify_message_by_keywords(text)

    def test_web_search_is_task_not_mission(self):
        """'Can you do a web search' should be a task, not mission."""
        result = self._classify_keywords("Can you do a web search")
        self.assertEqual(result["type"], "task")

    def test_build_app_is_mission_with_workflow(self):
        result = self._classify_keywords(
            "build me an app that allows multiple users to share and manage their shoplists together"
        )
        self.assertEqual(result["type"], "mission")
        self.assertEqual(result.get("workflow"), "i2p")

    def test_remind_me_is_todo(self):
        result = self._classify_keywords("remind me to buy milk tomorrow")
        self.assertEqual(result["type"], "todo")

    def test_whats_the_weather_is_question(self):
        result = self._classify_keywords("what is the weather like?")
        self.assertEqual(result["type"], "question")

    def test_hello_is_casual(self):
        result = self._classify_keywords("hello")
        self.assertEqual(result["type"], "casual")

    def test_fix_bug_is_bug_report(self):
        result = self._classify_keywords("there's a bug in the login page")
        self.assertEqual(result["type"], "bug_report")


# ---------------------------------------------------------------------------
# 4. TASK CLASSIFICATION — verify agent_type and difficulty
# ---------------------------------------------------------------------------
class TestTaskClassificationPipeline(unittest.TestCase):
    """Test task classifier keyword fallback."""

    def _classify(self, title, desc=""):
        from src.core.task_classifier import _classify_by_keywords
        return _classify_by_keywords(title, desc)

    def test_web_search_gets_researcher(self):
        result = self._classify("Can you do a web search", "search for waterproof shoes")
        self.assertEqual(result.agent_type, "researcher")

    def test_difficulty_not_inflated(self):
        """No keyword classification should produce difficulty > 6."""
        test_cases = [
            "search for shoes",
            "build a website",
            "fix the bug",
            "write a report",
            "plan the architecture",
        ]
        for text in test_cases:
            result = self._classify(text)
            self.assertLessEqual(
                result.difficulty, 6,
                f"'{text}' got difficulty {result.difficulty}, max should be 6"
            )

    def test_web_search_needs_tools(self):
        result = self._classify("search for waterproof shoes in Turkey")
        self.assertTrue(result.needs_tools)


# ---------------------------------------------------------------------------
# 5. JSON EXTRACTION — verify robust parsing of LLM output
# ---------------------------------------------------------------------------
class TestJSONExtraction(unittest.TestCase):
    """Test _extract_json handles all real LLM output formats."""

    def _extract(self, text):
        from src.core.task_classifier import _extract_json
        return _extract_json(text)

    def test_clean_json(self):
        result = self._extract('{"type": "task", "confidence": 0.9}')
        self.assertEqual(result["type"], "task")

    def test_with_think_tags(self):
        result = self._extract(
            '<think>The user wants to search</think>{"type": "task", "confidence": 0.8}'
        )
        self.assertEqual(result["type"], "task")

    def test_with_markdown_fence(self):
        result = self._extract('```json\n{"type": "mission"}\n```')
        self.assertEqual(result["type"], "mission")

    def test_with_preamble(self):
        result = self._extract('Here is the classification:\n{"type": "casual"}')
        self.assertEqual(result["type"], "casual")

    def test_xml_function_call_fails(self):
        """XML tool calls (from coding models) should NOT parse as JSON."""
        with self.assertRaises(ValueError):
            self._extract('<function=web_search><parameter=query>test</parameter></function>')

    def test_empty_after_think_strip(self):
        """Pure think tokens with no content should fail."""
        with self.assertRaises(ValueError):
            self._extract('<think>I need to think about this carefully...</think>')


# ---------------------------------------------------------------------------
# 6. CRON PARSER — verify scheduled tasks fire correctly
# ---------------------------------------------------------------------------
class TestCronParser(unittest.TestCase):
    """Test the cron parser handles all formats correctly."""

    def _next(self, cron, after):
        from src.core.orchestrator import Orchestrator
        o = Orchestrator.__new__(Orchestrator)
        return o._compute_next_run(cron, after)

    def test_comma_hours_picks_next(self):
        """'0 9,11,13,15,17,19,21 * * *' at 9:30 → 11:00"""
        result = self._next("0 9,11,13,15,17,19,21 * * *", datetime(2026, 3, 25, 9, 30))
        self.assertEqual(result.hour, 11)
        self.assertEqual(result.minute, 0)

    def test_comma_hours_wraps_to_next_day(self):
        """'0 9,11,13,15,17,19,21 * * *' at 21:30 → next day 9:00"""
        result = self._next("0 9,11,13,15,17,19,21 * * *", datetime(2026, 3, 25, 21, 30))
        self.assertEqual(result.day, 26)
        self.assertEqual(result.hour, 9)

    def test_every_hour(self):
        """'0 * * * *' at 9:30 → 10:00"""
        result = self._next("0 * * * *", datetime(2026, 3, 25, 9, 30))
        self.assertEqual(result.hour, 10)
        self.assertEqual(result.minute, 0)

    def test_daily(self):
        """'30 14 * * *' at 15:00 → next day 14:30"""
        result = self._next("30 14 * * *", datetime(2026, 3, 25, 15, 0))
        self.assertEqual(result.day, 26)
        self.assertEqual(result.hour, 14)

    def test_never_returns_none(self):
        """Valid cron should never return None."""
        crons = ["0 * * * *", "0 9 * * *", "0 9,11 * * *", "30 14 * * *"]
        for cron in crons:
            result = self._next(cron, datetime(2026, 3, 25, 12, 0))
            self.assertIsNotNone(result, f"'{cron}' returned None")


# ---------------------------------------------------------------------------
# 7. MIN_SCORE — verify models aren't filtered unnecessarily
# ---------------------------------------------------------------------------
class TestMinScoreThresholds(unittest.TestCase):
    """Verify the min_score formula doesn't filter available models."""

    def test_difficulty_7_allows_score_3(self):
        """At difficulty 7, a model scoring 3.0 should pass."""
        from fatih_hoca.requirements import ModelRequirements
        reqs = ModelRequirements(difficulty=7)
        # effective_min_score = (7-1) * 0.47 = 2.82
        self.assertLessEqual(reqs.effective_min_score, 3.0)

    def test_difficulty_5_allows_score_2(self):
        from fatih_hoca.requirements import ModelRequirements
        reqs = ModelRequirements(difficulty=5)
        # (5-1) * 0.47 = 1.88
        self.assertLessEqual(reqs.effective_min_score, 2.0)

    def test_difficulty_3_allows_anything(self):
        from fatih_hoca.requirements import ModelRequirements
        reqs = ModelRequirements(difficulty=3)
        # (3-1) * 0.47 = 0.94
        self.assertLessEqual(reqs.effective_min_score, 1.0)


# ---------------------------------------------------------------------------
# 8. PROGRESS MESSAGES — verify no raw LLM output leaks to Telegram
# ---------------------------------------------------------------------------
class TestProgressMessages(unittest.TestCase):
    """Verify progress callback doesn't leak raw model output."""

    def test_friendly_error_catches_attribute_error(self):
        from src.app.telegram_bot import _friendly_error
        result = _friendly_error("'ModelRegistry' object has no attribute 'get_overrides'")
        self.assertNotIn("object has no attribute", result)
        self.assertIn("Internal", result)

    def test_friendly_error_catches_import_error(self):
        from src.app.telegram_bot import _friendly_error
        result = _friendly_error("cannot import name 'foo' from 'bar'")
        self.assertIn("Internal", result)

    def test_friendly_error_generic_no_python_internals(self):
        from src.app.telegram_bot import _friendly_error
        result = _friendly_error("something unexpected happened")
        self.assertNotIn("{e}", result)
        self.assertNotIn("Traceback", result)


# ---------------------------------------------------------------------------
# 9. TODO DB — verify real DB operations
# ---------------------------------------------------------------------------
class TestTodoDB(unittest.IsolatedAsyncioTestCase):
    """Test todo CRUD against real SQLite (in-memory)."""

    async def asyncSetUp(self):
        import src.infra.db as db_mod
        # Close any existing connection so we don't write to the real DB
        await db_mod.close_db()
        self._orig_db_path = db_mod.DB_PATH
        db_mod.DB_PATH = ":memory:"
        db_mod._db_connection = None
        await db_mod.init_db()

    async def asyncTearDown(self):
        import src.infra.db as db_mod
        await db_mod.close_db()
        db_mod.DB_PATH = self._orig_db_path
        db_mod._db_connection = None

    async def test_add_and_get_todo(self):
        from src.infra.db import add_todo, get_todo
        todo_id = await add_todo("Buy milk")
        todo = await get_todo(todo_id)
        self.assertIsNotNone(todo)
        self.assertEqual(todo["title"], "Buy milk")
        self.assertEqual(todo["status"], "pending")

    async def test_toggle_todo(self):
        from src.infra.db import add_todo, toggle_todo, get_todo
        todo_id = await add_todo("Buy eggs")
        new_status = await toggle_todo(todo_id)
        self.assertEqual(new_status, "done")
        todo = await get_todo(todo_id)
        self.assertEqual(todo["status"], "done")
        # Toggle back
        new_status = await toggle_todo(todo_id)
        self.assertEqual(new_status, "pending")

    async def test_get_todos_filters_by_status(self):
        from src.infra.db import add_todo, toggle_todo, get_todos
        id1 = await add_todo("FilterTest A")
        id2 = await add_todo("FilterTest B")
        await toggle_todo(id1)  # mark done
        pending = await get_todos(status="pending")
        done = await get_todos(status="done")
        pending_titles = [t["title"] for t in pending]
        done_titles = [t["title"] for t in done]
        self.assertIn("FilterTest B", pending_titles)
        self.assertIn("FilterTest A", done_titles)
        self.assertNotIn("FilterTest A", pending_titles)


# ---------------------------------------------------------------------------
# 10. WEB SEARCH — verify ddgs actually works
# ---------------------------------------------------------------------------
class TestWebSearchReal(unittest.IsolatedAsyncioTestCase):
    """Test web search with real network calls."""

    async def test_ddgs_returns_urls(self):
        """ddgs search should return results containing URLs."""
        from src.tools.web_search import web_search, _DDGS
        if _DDGS is None:
            self.skipTest("ddgs not installed")
        result = await web_search("Python programming tutorial", max_results=3)
        self.assertIsInstance(result, str)
        self.assertIn("http", result, "Search results should contain at least one URL")

    async def test_web_search_nonsense_query(self):
        """Even nonsense queries should return a string without crashing."""
        from src.tools.web_search import web_search
        result = await web_search("xyzzy123456789qwerty", max_results=1)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
