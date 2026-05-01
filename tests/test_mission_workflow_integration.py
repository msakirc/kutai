"""Integration tests for the mission workflow — from message classification
through task classification to workflow creation.

Tests cover:
- Product-idea detection heuristics
- Message classification (keyword fallback + LLM path)
- Task classification (LLM + keyword fallback)
- Workflow definition loading and WorkflowRunner.start

The telegram library is not installed in the test environment, so we mock
the entire module hierarchy before importing TelegramInterface.
"""

import json
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Mock the telegram package so we can import telegram_bot without installing
# python-telegram-bot.
# ---------------------------------------------------------------------------
_telegram_mock = MagicMock()
_telegram_ext_mock = MagicMock()

_telegram_mock.Update = MagicMock
_telegram_mock.InlineKeyboardButton = MagicMock
_telegram_mock.InlineKeyboardMarkup = MagicMock
_telegram_ext_mock.Application = MagicMock()
_telegram_ext_mock.CommandHandler = MagicMock
_telegram_ext_mock.MessageHandler = MagicMock
_telegram_ext_mock.CallbackQueryHandler = MagicMock
_telegram_ext_mock.filters = MagicMock()
_telegram_ext_mock.ContextTypes = MagicMock()

sys.modules.setdefault("telegram", _telegram_mock)
sys.modules.setdefault("telegram.ext", _telegram_ext_mock)
sys.modules.setdefault("litellm", MagicMock())

from src.app.telegram_bot import TelegramInterface, _looks_like_product_idea
from src.core.task_classifier import (
    TaskClassification,
    _classify_by_keywords,
    _extract_json,
    classify_task,
)
from src.workflows.engine.loader import load_workflow
from src.workflows.engine.runner import WorkflowRunner


# ── 1. Product Idea Detection ─────────────────────────────────────────────


class TestProductIdeaDetection(unittest.TestCase):
    """Verify _looks_like_product_idea catches product/app descriptions."""

    def test_looks_like_product_idea_positive(self):
        text = "build me an app that allows multiple users to share and manage their shoplists together"
        self.assertTrue(_looks_like_product_idea(text))

    def test_looks_like_product_idea_with_platform(self):
        text = "create a website for tracking expenses"
        self.assertTrue(_looks_like_product_idea(text))

    def test_looks_like_product_idea_negative_research(self):
        text = "research best Python frameworks"
        self.assertFalse(_looks_like_product_idea(text))

    def test_looks_like_product_idea_negative_question(self):
        text = "what is asyncio?"
        self.assertFalse(_looks_like_product_idea(text))


# ── 2. Message Classification (keyword fallback) ──────────────────────────


class TestMessageClassification(unittest.TestCase):
    """Verify _classify_message_by_keywords returns correct types."""

    def test_keyword_fallback_mission_with_workflow(self):
        text = "build me an app that allows users to share shoplists"
        result = TelegramInterface._classify_message_by_keywords(text)
        self.assertEqual(result["type"], "mission")
        self.assertEqual(result.get("workflow"), "i2p")

    def test_keyword_fallback_mission_without_workflow(self):
        text = "research best frameworks for web development"
        result = TelegramInterface._classify_message_by_keywords(text)
        self.assertEqual(result["type"], "mission")
        self.assertNotIn("workflow", result)

    def test_keyword_fallback_task(self):
        text = "fix the bug in login page"
        result = TelegramInterface._classify_message_by_keywords(text)
        # "bug" keyword triggers bug_report, not mission
        self.assertNotEqual(result["type"], "mission")

    def test_keyword_fallback_question(self):
        text = "what is Docker?"
        result = TelegramInterface._classify_message_by_keywords(text)
        self.assertEqual(result["type"], "question")

    def test_keyword_fallback_casual(self):
        text = "hello"
        result = TelegramInterface._classify_message_by_keywords(text)
        self.assertEqual(result["type"], "casual")


# ── 3. LLM-Based Message Classification ──────────────────────────────────


class TestLLMClassification(unittest.IsolatedAsyncioTestCase):
    """Verify _classify_user_message LLM path and fallback."""

    def _make_interface(self):
        """Create a minimal TelegramInterface without calling __init__."""
        obj = object.__new__(TelegramInterface)
        obj._pending_clarifications = {}
        return obj

    @patch("src.core.router.call_model", new_callable=AsyncMock)
    async def test_llm_returns_mission_workflow(self, mock_call):
        mock_call.return_value = {
            "content": '{"type": "mission", "confidence": 0.95, "workflow": "i2p"}'
        }
        iface = self._make_interface()
        result = await iface._classify_user_message(
            "build me an app that allows users to share shoplists"
        )
        self.assertEqual(result["type"], "mission")
        self.assertEqual(result.get("workflow"), "i2p")

    @patch("src.core.router.call_model", new_callable=AsyncMock)
    async def test_llm_returns_task(self, mock_call):
        mock_call.return_value = {
            "content": '{"type": "task", "confidence": 0.85}'
        }
        iface = self._make_interface()
        result = await iface._classify_user_message("fix the login page CSS")
        self.assertEqual(result["type"], "task")
        self.assertNotIn("workflow", result)

    @patch("src.core.router.call_model", new_callable=AsyncMock)
    async def test_llm_failure_falls_to_keywords(self, mock_call):
        mock_call.side_effect = RuntimeError("model unavailable")
        iface = self._make_interface()
        result = await iface._classify_user_message("what is Docker?")
        # Keyword fallback should still classify correctly
        self.assertEqual(result["type"], "question")

    def test_extract_json_with_think_tags(self):
        raw = '<think>reasoning about the task</think>{"type":"task"}'
        result = _extract_json(raw)
        self.assertEqual(result["type"], "task")


# ── 4. Task Classification ────────────────────────────────────────────────


class TestTaskClassification(unittest.IsolatedAsyncioTestCase):
    """Verify classify_task with mocked LLM and keyword fallback."""

    async def test_classify_shoplist_app(self):
        mock_dispatcher = MagicMock()
        mock_dispatcher.request = AsyncMock(return_value={
            "content": json.dumps({
                "agent_type": "planner",
                "difficulty": 6,
                "needs_tools": False,
                "needs_vision": False,
                "needs_thinking": True,
                "local_only": False,
                "priority": "normal",
            })
        })
        with patch("src.core.llm_dispatcher.get_dispatcher", return_value=mock_dispatcher):
            result = await classify_task(
                "Build shoplist app",
                "Build an app that allows multiple users to share and manage shoplists",
            )
        self.assertIsInstance(result, TaskClassification)
        self.assertEqual(result.agent_type, "planner")
        self.assertEqual(result.difficulty, 6)
        self.assertTrue(result.needs_thinking)
        self.assertEqual(result.method, "llm")

    async def test_classify_web_search(self):
        mock_dispatcher = MagicMock()
        mock_dispatcher.request = AsyncMock(return_value={
            "content": json.dumps({
                "agent_type": "researcher",
                "difficulty": 4,
                "needs_tools": True,
                "needs_vision": False,
                "needs_thinking": False,
                "local_only": False,
                "priority": "normal",
            })
        })
        with patch("src.core.llm_dispatcher.get_dispatcher", return_value=mock_dispatcher):
            result = await classify_task(
                "Research Python frameworks",
                "Search the web for the best Python frameworks for web development",
            )
        self.assertIsInstance(result, TaskClassification)
        self.assertEqual(result.agent_type, "researcher")
        self.assertEqual(result.difficulty, 4)
        self.assertTrue(result.needs_tools)
        self.assertEqual(result.method, "llm")

    def test_keyword_difficulty_not_inflated(self):
        """Keyword fallback should not assign inflated difficulty."""
        result = _classify_by_keywords("fix the login bug", "The login page crashes")
        self.assertLessEqual(result.difficulty, 6)
        result2 = _classify_by_keywords("write a test", "Add unit tests for the parser")
        self.assertLessEqual(result2.difficulty, 6)
        result3 = _classify_by_keywords("deploy the app", "Run deploy script")
        self.assertLessEqual(result3.difficulty, 6)


# ── 5. Workflow Creation ──────────────────────────────────────────────────


class TestWorkflowCreation(unittest.TestCase):
    """Verify workflow definition loading and structure."""

    def test_workflow_definition_exists(self):
        wf = load_workflow("i2p_v3")
        self.assertEqual(wf.plan_id, "i2p_v3")
        self.assertEqual(wf.version, "2.0")

    def test_workflow_has_phases(self):
        wf = load_workflow("i2p_v3")
        self.assertGreater(len(wf.phases), 0)
        phase_ids = [p["id"] for p in wf.phases]
        self.assertIn("phase_0", phase_ids)
        self.assertIn("phase_1", phase_ids)


class TestWorkflowRunnerStart(unittest.IsolatedAsyncioTestCase):
    """Verify WorkflowRunner.start creates a mission with correct context."""

    @patch("src.infra.db.add_scheduled_task", new_callable=AsyncMock, return_value=1)
    @patch("src.infra.db.add_task", new_callable=AsyncMock)
    @patch("src.infra.db.add_mission", new_callable=AsyncMock)
    async def test_workflow_runner_start(self, mock_add_mission, mock_add_task, mock_add_sched):
        mock_add_mission.return_value = 42
        # Return incrementing task IDs
        task_counter = iter(range(1, 500))
        mock_add_task.side_effect = lambda **kwargs: next(task_counter)

        runner = WorkflowRunner()
        # Mock artifact store to avoid DB calls
        runner.artifact_store = MagicMock()
        runner.artifact_store.store = AsyncMock()

        mission_id = await runner.start(
            "i2p_v3",
            initial_input={"raw_idea": "Build a shoplist sharing app"},
        )

        self.assertEqual(mission_id, 42)

        # Verify add_mission was called with workflow_name in context
        mock_add_mission.assert_called_once()
        call_kwargs = mock_add_mission.call_args[1]
        context = call_kwargs["context"]
        self.assertEqual(context["workflow_name"], "i2p_v3")
        self.assertEqual(context["initial_input"]["raw_idea"], "Build a shoplist sharing app")

        # Verify tasks were created
        self.assertGreater(mock_add_task.call_count, 0)


if __name__ == "__main__":
    unittest.main()
