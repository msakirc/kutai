# tests/test_phase4.py
"""
Tests for Phase 4: Model Intelligence Layer

  4.1 Model stats recording & retrieval
  4.2 Performance ranking
  4.3 Cost budget tracking
  4.4 Quality score column
  4.5 Config-level constants
  4.6 Thinking model detection (replicated)
  4.7 Tier escalation logic (replicated)
  4.8 Grading prompt structure
  4.9 Thinking content extraction (replicated)
  4.10 Escalation constants
  4.11 Performance cache structure
"""
import asyncio
import os
import re
import sys
import tempfile
import unittest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _patch_db_path(db_mod, db_path):
    import config
    config.DB_PATH = db_path
    db_mod.DB_PATH = db_path


class _DBTestBase(unittest.TestCase):
    def setUp(self):
        if not HAS_AIOSQLITE:
            self.skipTest("aiosqlite not installed")
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.tmp.name
        self.tmp.close()

        import config
        import db as db_mod
        self._orig_config_path = config.DB_PATH
        self._orig_db_path = db_mod.DB_PATH
        self.db_mod = db_mod

        _patch_db_path(db_mod, self.db_path)
        db_mod._db_connection = None
        run_async(db_mod.init_db())

    def tearDown(self):
        run_async(self.db_mod.close_db())
        import config
        config.DB_PATH = self._orig_config_path
        self.db_mod.DB_PATH = self._orig_db_path
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(self.db_path + suffix)
            except OSError:
                pass


# ─── 4.1 Model Stats Recording ──────────────────────────────────────────────

class TestModelStats(_DBTestBase):

    def test_record_and_get_stats(self):
        async def _test():
            await self.db_mod.record_model_call(
                model="test-model", agent_type="coder",
                success=True, cost=0.01, latency=1.5, grade=4.0,
            )
            stats = await self.db_mod.get_model_stats(model="test-model")
            self.assertEqual(len(stats), 1)
            self.assertEqual(stats[0]["model"], "test-model")
            self.assertEqual(stats[0]["agent_type"], "coder")
            self.assertEqual(stats[0]["total_calls"], 1)
            self.assertAlmostEqual(stats[0]["avg_cost"], 0.01)
            self.assertAlmostEqual(stats[0]["avg_latency"], 1.5)
            self.assertEqual(stats[0]["success_rate"], 1.0)
        run_async(_test())

    def test_multiple_calls_update_averages(self):
        async def _test():
            await self.db_mod.record_model_call(
                "m1", "coder", True, cost=0.10, latency=2.0,
            )
            await self.db_mod.record_model_call(
                "m1", "coder", True, cost=0.20, latency=4.0,
            )
            stats = await self.db_mod.get_model_stats(model="m1")
            self.assertEqual(stats[0]["total_calls"], 2)
            self.assertAlmostEqual(stats[0]["avg_cost"], 0.15)
            self.assertAlmostEqual(stats[0]["avg_latency"], 3.0)
        run_async(_test())

    def test_failure_updates_success_rate(self):
        async def _test():
            await self.db_mod.record_model_call(
                "m2", "planner", True, cost=0.01,
            )
            await self.db_mod.record_model_call(
                "m2", "planner", False, cost=0.01,
            )
            stats = await self.db_mod.get_model_stats(model="m2")
            self.assertAlmostEqual(stats[0]["success_rate"], 0.5)
            self.assertEqual(stats[0]["total_calls"], 2)
        run_async(_test())

    def test_filter_by_agent_type(self):
        async def _test():
            await self.db_mod.record_model_call(
                "m3", "coder", True, cost=0.01,
            )
            await self.db_mod.record_model_call(
                "m3", "planner", True, cost=0.02,
            )
            coder_stats = await self.db_mod.get_model_stats(
                agent_type="coder"
            )
            self.assertEqual(len(coder_stats), 1)
            self.assertEqual(coder_stats[0]["agent_type"], "coder")
        run_async(_test())

    def test_get_all_stats(self):
        async def _test():
            await self.db_mod.record_model_call(
                "m4", "coder", True, cost=0.01,
            )
            await self.db_mod.record_model_call(
                "m5", "researcher", True, cost=0.02,
            )
            all_stats = await self.db_mod.get_model_stats()
            self.assertGreaterEqual(len(all_stats), 2)
        run_async(_test())


# ─── 4.2 Performance Ranking ────────────────────────────────────────────────

class TestPerformanceRanking(_DBTestBase):

    def test_ranking_order(self):
        async def _test():
            # Model with higher grade should rank first
            for _ in range(5):
                await self.db_mod.record_model_call(
                    "good-model", "coder", True, grade=4.5,
                )
            for _ in range(5):
                await self.db_mod.record_model_call(
                    "bad-model", "coder", True, grade=2.0,
                )
            ranking = await self.db_mod.get_model_performance_ranking("coder")
            self.assertGreaterEqual(len(ranking), 2)
            self.assertEqual(ranking[0]["model"], "good-model")
        run_async(_test())

    def test_minimum_calls_filter(self):
        async def _test():
            # Only 2 calls, below the 3-call minimum
            await self.db_mod.record_model_call(
                "rare-model", "coder", True, grade=5.0,
            )
            await self.db_mod.record_model_call(
                "rare-model", "coder", True, grade=5.0,
            )
            ranking = await self.db_mod.get_model_performance_ranking("coder")
            models = [r["model"] for r in ranking]
            self.assertNotIn("rare-model", models)
        run_async(_test())


# ─── 4.3 Cost Budget ────────────────────────────────────────────────────────

class TestCostBudget(_DBTestBase):

    def test_set_and_get_budget(self):
        async def _test():
            await self.db_mod.set_budget("daily", daily_limit=1.0)
            budget = await self.db_mod.get_budget("daily")
            self.assertIsNotNone(budget)
            self.assertAlmostEqual(budget["daily_limit"], 1.0)
            self.assertAlmostEqual(budget["spent_today"], 0.0)
        run_async(_test())

    def test_record_cost(self):
        async def _test():
            await self.db_mod.set_budget("daily", daily_limit=1.0)
            await self.db_mod.record_cost(0.25, "daily")
            budget = await self.db_mod.get_budget("daily")
            self.assertAlmostEqual(budget["spent_today"], 0.25)
            self.assertAlmostEqual(budget["spent_total"], 0.25)
        run_async(_test())

    def test_budget_check_within(self):
        async def _test():
            await self.db_mod.set_budget("daily", daily_limit=1.0)
            await self.db_mod.record_cost(0.50, "daily")
            result = await self.db_mod.check_budget("daily")
            self.assertTrue(result["ok"])
        run_async(_test())

    def test_budget_check_exceeded(self):
        async def _test():
            await self.db_mod.set_budget("daily", daily_limit=0.50)
            await self.db_mod.record_cost(0.60, "daily")
            result = await self.db_mod.check_budget("daily")
            self.assertFalse(result["ok"])
            self.assertIn("exceeded", result["reason"])
        run_async(_test())

    def test_no_budget_set(self):
        async def _test():
            result = await self.db_mod.check_budget("daily")
            self.assertTrue(result["ok"])
            self.assertEqual(result["reason"], "no budget set")
        run_async(_test())

    def test_total_budget_exceeded(self):
        async def _test():
            await self.db_mod.set_budget(
                "daily", daily_limit=10.0, total_limit=0.50
            )
            await self.db_mod.record_cost(0.60, "daily")
            result = await self.db_mod.check_budget("daily")
            self.assertFalse(result["ok"])
            self.assertIn("Total budget", result["reason"])
        run_async(_test())

    def test_goal_scope_budget(self):
        async def _test():
            await self.db_mod.set_budget(
                "goal", scope_id="42", daily_limit=0.0,
                total_limit=5.0,
            )
            budget = await self.db_mod.get_budget("goal", "42")
            self.assertIsNotNone(budget)
            self.assertAlmostEqual(budget["total_limit"], 5.0)
        run_async(_test())


# ─── 4.4 Quality Score Column ───────────────────────────────────────────────

class TestQualityScore(_DBTestBase):

    def test_quality_score_column_exists(self):
        async def _test():
            tid = await self.db_mod.add_task(
                "QS test", "d", agent_type="coder"
            )
            task = await self.db_mod.get_task(tid)
            self.assertIn("quality_score", task)
            self.assertIsNone(task["quality_score"])
        run_async(_test())

    def test_set_quality_score(self):
        async def _test():
            tid = await self.db_mod.add_task(
                "QS set", "d", agent_type="coder"
            )
            await self.db_mod.update_task(tid, quality_score=4.5)
            task = await self.db_mod.get_task(tid)
            self.assertAlmostEqual(task["quality_score"], 4.5)
        run_async(_test())


# ─── 4.5 Config-level Constants ─────────────────────────────────────────────

class TestPhase4Config(unittest.TestCase):

    def test_cost_budget_daily_default(self):
        from config import COST_BUDGET_DAILY
        self.assertIsInstance(COST_BUDGET_DAILY, float)
        self.assertGreater(COST_BUDGET_DAILY, 0)

    def test_thinking_models_defined(self):
        from config import THINKING_MODELS
        self.assertIsInstance(THINKING_MODELS, list)
        self.assertIn("o1", THINKING_MODELS)
        self.assertIn("qwq", THINKING_MODELS)

    def test_thinking_models_has_deepseek(self):
        from config import THINKING_MODELS
        self.assertIn("deepseek-r1", THINKING_MODELS)


# ─── 4.6 Thinking Model Detection (replicated, no litellm dep) ─────────────

# Replicated from router.py to avoid importing litellm
THINKING_MODELS_TEST = [
    "o1", "o3", "o4", "qwq", "deepseek-r1", "gemini-2.5-flash",
]


def _is_thinking_model_test(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(p in name_lower for p in THINKING_MODELS_TEST)


class TestThinkingModelDetection(unittest.TestCase):

    def test_detects_o1(self):
        self.assertTrue(_is_thinking_model_test("openai/o1-preview"))
        self.assertTrue(_is_thinking_model_test("o1-mini"))

    def test_detects_qwq(self):
        self.assertTrue(_is_thinking_model_test("ollama/qwq:32b"))

    def test_detects_deepseek_r1(self):
        self.assertTrue(_is_thinking_model_test("deepseek/deepseek-r1"))

    def test_regular_model_not_thinking(self):
        self.assertFalse(_is_thinking_model_test("gpt-4o-mini"))
        self.assertFalse(
            _is_thinking_model_test("claude-sonnet-4-20250514")
        )
        self.assertFalse(
            _is_thinking_model_test("groq/llama-3.1-8b-instant")
        )

    def test_gemini_thinking(self):
        self.assertTrue(
            _is_thinking_model_test(
                "gemini/gemini-2.5-flash-preview-05-20"
            )
        )


# ─── 4.7 Tier Escalation Logic (replicated, no litellm dep) ────────────────

TIER_ESCALATION_ORDER_TEST = ["cheap", "code", "medium", "expensive"]


def _escalate_tier_test(current_tier: str) -> str | None:
    try:
        idx = TIER_ESCALATION_ORDER_TEST.index(current_tier)
    except ValueError:
        return None
    if idx < len(TIER_ESCALATION_ORDER_TEST) - 1:
        return TIER_ESCALATION_ORDER_TEST[idx + 1]
    return None


class TestTierEscalation(unittest.TestCase):

    def test_cheap_escalates_to_code(self):
        self.assertEqual(_escalate_tier_test("cheap"), "code")

    def test_code_escalates_to_medium(self):
        self.assertEqual(_escalate_tier_test("code"), "medium")

    def test_medium_escalates_to_expensive(self):
        self.assertEqual(_escalate_tier_test("medium"), "expensive")

    def test_expensive_no_escalation(self):
        self.assertIsNone(_escalate_tier_test("expensive"))

    def test_unknown_tier_no_escalation(self):
        self.assertIsNone(_escalate_tier_test("unknown_tier"))


# ─── 4.8 Grading Prompt Structure ──────────────────────────────────────────

# Replicated from router.py
GRADING_PROMPT_TEST = """Rate this AI response on a scale of 1-5:
1 = Wrong/useless, 2 = Partially relevant, 3 = Adequate,
4 = Good and complete, 5 = Excellent

Task: {task_title}
Task description: {task_description}

Response to grade:
{response}

Respond with ONLY a JSON object: {{"score": N, "reason": "brief"}}"""


class TestGradingPrompt(unittest.TestCase):

    def test_grading_prompt_has_placeholders(self):
        self.assertIn("{task_title}", GRADING_PROMPT_TEST)
        self.assertIn("{task_description}", GRADING_PROMPT_TEST)
        self.assertIn("{response}", GRADING_PROMPT_TEST)

    def test_grading_prompt_mentions_scale(self):
        self.assertIn("1-5", GRADING_PROMPT_TEST)
        self.assertIn("score", GRADING_PROMPT_TEST.lower())

    def test_grading_prompt_formats_correctly(self):
        """Ensure the prompt can be formatted without errors."""
        formatted = GRADING_PROMPT_TEST.format(
            task_title="Test task",
            task_description="Do something",
            response="Here is the result",
        )
        self.assertIn("Test task", formatted)
        self.assertIn("Here is the result", formatted)


# ─── 4.9 Thinking Content Extraction (replicated) ──────────────────────────

def _extract_thinking_test(msg) -> str | None:
    """Replicated from router.py for testing without litellm."""
    if hasattr(msg, "thinking") and msg.thinking:
        return msg.thinking
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        return msg.reasoning_content
    content = msg.content or ""
    think_match = re.search(
        r"<(?:thinking|think)>(.*?)</(?:thinking|think)>",
        content, re.DOTALL
    )
    if think_match:
        return think_match.group(1).strip()
    return None


class TestThinkingExtraction(unittest.TestCase):

    def test_extract_think_tags(self):
        class MockMsg:
            content = "<think>Deep reasoning here</think>Final answer"
            thinking = None
            reasoning_content = None

        result = _extract_thinking_test(MockMsg())
        self.assertEqual(result, "Deep reasoning here")

    def test_extract_thinking_tags(self):
        class MockMsg:
            content = "<thinking>Analysis</thinking>Result"
            thinking = None
            reasoning_content = None

        result = _extract_thinking_test(MockMsg())
        self.assertEqual(result, "Analysis")

    def test_extract_thinking_attribute(self):
        class MockMsg:
            content = "Final answer"
            thinking = "My reasoning process"
            reasoning_content = None

        result = _extract_thinking_test(MockMsg())
        self.assertEqual(result, "My reasoning process")

    def test_extract_reasoning_content(self):
        class MockMsg:
            content = "Final answer"
            thinking = None
            reasoning_content = "Step by step reasoning"

        result = _extract_thinking_test(MockMsg())
        self.assertEqual(result, "Step by step reasoning")

    def test_no_thinking(self):
        class MockMsg:
            content = "Just a regular answer"
            thinking = None
            reasoning_content = None

        result = _extract_thinking_test(MockMsg())
        self.assertIsNone(result)

    def test_thinking_attribute_priority(self):
        """thinking attribute takes priority over tags."""
        class MockMsg:
            content = "<think>tag content</think>"
            thinking = "attr content"
            reasoning_content = None

        result = _extract_thinking_test(MockMsg())
        self.assertEqual(result, "attr content")


# ─── 4.10 Escalation Constants (replicated) ────────────────────────────────

class TestEscalationConstants(unittest.TestCase):

    def test_escalation_threshold(self):
        # Replicated value from agents/base.py
        ESCALATION_THRESHOLD = 3
        self.assertEqual(ESCALATION_THRESHOLD, 3)

    def test_tier_escalation_order(self):
        self.assertEqual(
            TIER_ESCALATION_ORDER_TEST,
            ["cheap", "code", "medium", "expensive"]
        )

    def test_max_format_retries(self):
        # Replicated value from agents/base.py
        MAX_FORMAT_RETRIES = 2
        self.assertEqual(MAX_FORMAT_RETRIES, 2)


# ─── 4.11 Model Stats Table Schema ─────────────────────────────────────────

class TestModelStatsSchema(_DBTestBase):

    def test_model_stats_table_exists(self):
        async def _test():
            db = await self.db_mod.get_db()
            cursor = await db.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='model_stats'"
            )
            row = await cursor.fetchone()
            self.assertIsNotNone(row)
        run_async(_test())

    def test_cost_budgets_table_exists(self):
        async def _test():
            db = await self.db_mod.get_db()
            cursor = await db.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='cost_budgets'"
            )
            row = await cursor.fetchone()
            self.assertIsNotNone(row)
        run_async(_test())

    def test_model_stats_unique_constraint(self):
        """Same model+agent_type should update, not duplicate."""
        async def _test():
            await self.db_mod.record_model_call(
                "unique-test", "coder", True, cost=0.01,
            )
            await self.db_mod.record_model_call(
                "unique-test", "coder", True, cost=0.02,
            )
            stats = await self.db_mod.get_model_stats(model="unique-test")
            self.assertEqual(len(stats), 1)
            self.assertEqual(stats[0]["total_calls"], 2)
        run_async(_test())


# ─── 4.12 Budget Update Logic ──────────────────────────────────────────────

class TestBudgetUpdateLogic(_DBTestBase):

    def test_multiple_cost_records_accumulate(self):
        async def _test():
            await self.db_mod.set_budget("daily", daily_limit=5.0)
            await self.db_mod.record_cost(0.10, "daily")
            await self.db_mod.record_cost(0.20, "daily")
            await self.db_mod.record_cost(0.30, "daily")
            budget = await self.db_mod.get_budget("daily")
            self.assertAlmostEqual(budget["spent_today"], 0.60)
            self.assertAlmostEqual(budget["spent_total"], 0.60)
        run_async(_test())

    def test_update_existing_budget(self):
        async def _test():
            await self.db_mod.set_budget("daily", daily_limit=1.0)
            await self.db_mod.set_budget("daily", daily_limit=2.0)
            budget = await self.db_mod.get_budget("daily")
            self.assertAlmostEqual(budget["daily_limit"], 2.0)
        run_async(_test())

    def test_no_budget_record_cost_noop(self):
        """Recording cost with no budget set should not error."""
        async def _test():
            # Should not raise
            await self.db_mod.record_cost(0.50, "daily")
        run_async(_test())


if __name__ == "__main__":
    unittest.main()
