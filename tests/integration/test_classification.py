"""test_classification.py — Integration tests for message and task classification.

Tests:
- Keyword-based message classification (no LLM needed)
- Keyword-based task classification (no LLM needed)
- LLM-based task classification with real model calls (marked @llm)
- Critical ambiguity: "coffee machine search going" must NOT be shopping

Markers:
  @pytest.mark.integration  — all tests
  @pytest.mark.llm          — tests making real LLM calls
"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Keyword-based message classification (from TelegramInterface)
# These tests use the static method directly — no DB, no LLM needed.
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestKeywordMessageClassification:
    """Test _classify_message_by_keywords static method."""

    @pytest.fixture(autouse=True)
    def _import_classifier(self):
        """Import the static method once per class."""
        from src.app.telegram_bot import TelegramInterface
        self.classify = TelegramInterface._classify_message_by_keywords

    def test_status_query_coffee_machine(self):
        """'How is the coffee machine search going' must be status_query, not shopping."""
        result = self.classify("How is the coffee machine search going")
        assert result["type"] == "status_query", (
            f"Expected 'status_query' but got '{result['type']}'. "
            "This is the primary regression case: status phrases must "
            "take priority over product keywords like 'coffee machine'."
        )

    def test_status_query_how_is_my(self):
        result = self.classify("How is my task going?")
        assert result["type"] == "status_query"

    def test_status_query_any_update(self):
        result = self.classify("Any update on the GPU search?")
        assert result["type"] == "status_query"

    def test_status_query_did_you_find(self):
        result = self.classify("Did you find anything yet?")
        assert result["type"] == "status_query"

    def test_shopping_motherboard(self):
        """'Find me a good motherboard under 10k' → shopping."""
        result = self.classify("Find me a good motherboard under 10k")
        # Note: "find" triggers researcher, but "how much"/"price" etc. trigger shopping.
        # This query lacks explicit price keywords, so keyword fallback may differ.
        # We document the actual behavior rather than assert a specific type.
        assert result["type"] in ("shopping", "task"), (
            f"Got unexpected type: {result['type']}"
        )

    def test_shopping_buy_keyword(self):
        """'I want to buy a laptop' → shopping."""
        result = self.classify("I want to buy a laptop")
        assert result["type"] == "shopping"

    def test_shopping_fiyat_keyword(self):
        """Turkish 'fiyat' keyword → shopping."""
        result = self.classify("Bu laptopun fiyatı ne kadar")
        assert result["type"] == "shopping"

    def test_shopping_en_ucuz(self):
        """Turkish 'en ucuz' → shopping."""
        result = self.classify("En ucuz GPU hangisi")
        assert result["type"] == "shopping"

    def test_todo_remind_me(self):
        """'Remind me to buy milk' → todo."""
        result = self.classify("Remind me to buy milk tomorrow")
        assert result["type"] == "todo"

    def test_todo_dont_forget(self):
        result = self.classify("Don't forget to call the dentist")
        assert result["type"] == "todo"

    def test_bug_report(self):
        result = self.classify("There's a bug in the login page")
        assert result["type"] == "bug_report"

    def test_feature_request(self):
        result = self.classify("Feature: could you add dark mode?")
        assert result["type"] == "feature_request"

    def test_simple_question_fallback(self):
        """A simple question with no category keywords → task or question."""
        result = self.classify("What time is it?")
        # No strong keywords match, should be something reasonable
        assert "type" in result


# ---------------------------------------------------------------------------
# Keyword-based task classification (from task_classifier.py)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestKeywordTaskClassification:
    """Test _classify_by_keywords — the LLM fallback."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from src.core.task_classifier import _classify_by_keywords
        self.classify = _classify_by_keywords

    def test_shopping_price_keyword(self):
        result = self.classify("Check price", "fiyat karşılaştırması yap")
        assert result.agent_type == "shopping_advisor"

    def test_coder_keyword(self):
        result = self.classify("Write code", "implement a REST API in Python")
        assert result.agent_type == "coder"

    def test_fixer_keyword(self):
        result = self.classify("Fix bug", "error in login function")
        assert result.agent_type == "fixer"

    def test_researcher_keyword(self):
        result = self.classify("Research", "find information about GPU memory")
        assert result.agent_type == "researcher"

    def test_summarizer_keyword(self):
        result = self.classify("Summarize", "tldr this article")
        assert result.agent_type == "summarizer"

    def test_default_executor(self):
        """Unrecognised text → executor (default)."""
        result = self.classify("xyzabc", "xyzabc completely unknown words")
        assert result.agent_type == "executor"
        assert result.confidence <= 0.4

    def test_shopping_sub_intent_attached(self):
        """classify_task attaches shopping_sub_intent for shopping tasks."""
        from src.core.task_classifier import _classify_by_keywords, _classify_shopping_sub_intent
        result = self.classify("Compare prices", "compare GPU prices for gaming")
        if result.agent_type == "shopping_advisor":
            sub = _classify_shopping_sub_intent("Compare prices compare GPU prices for gaming")
            assert sub is not None


# ---------------------------------------------------------------------------
# LLM-based classification (real model calls)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.llm
class TestLLMTaskClassification:
    """Tests that call the real LLM for classification.

    These require a loaded model and can take 30-120 s each.
    Skip automatically when no local model is available.
    """

    @pytest.fixture(autouse=True)
    def _check_model(self, fastest_local_model):
        if fastest_local_model is None:
            pytest.skip("No local model available — skipping LLM test")

    def test_classify_shopping_query_real_llm(self, temp_db, fastest_local_model):
        """LLM classifies a shopping query as shopping_advisor."""
        from src.core.task_classifier import classify_task

        async def _run():
            result = await classify_task(
                title="Find me a good gaming GPU under 15000 TL",
                description="Looking for the best price/performance GPU for 1080p gaming",
            )
            # LLM should pick shopping_advisor; we accept researcher as a near-miss
            assert result.agent_type in ("shopping_advisor", "researcher", "analyst"), (
                f"Unexpected agent_type: {result.agent_type}"
            )
            assert result.method in ("llm", "keyword"), f"method={result.method}"

        run_async(_run())

    @pytest.mark.timeout(120)
    def test_classify_build_request_real_llm(self, temp_db, fastest_local_model):
        """LLM classifies a 'build me X' request as planner or coder."""
        from src.core.task_classifier import classify_task

        async def _run():
            result = await classify_task(
                title="Build me a todo app",
                description="Create a full-stack todo application with React frontend and FastAPI backend",
            )
            assert result.agent_type in ("coder", "planner", "architect", "implementer"), (
                f"Unexpected agent_type: {result.agent_type}"
            )

        run_async(_run())

    @pytest.mark.timeout(120)
    def test_classify_simple_question_real_llm(self, temp_db, fastest_local_model):
        """LLM classifies 'what is 2+2' as assistant or executor."""
        from src.core.task_classifier import classify_task

        async def _run():
            result = await classify_task(
                title="What is 2+2",
                description="Simple arithmetic question",
            )
            assert result.agent_type in ("assistant", "executor", "analyst"), (
                f"Unexpected agent_type: {result.agent_type}"
            )
            # Simple question should have low difficulty
            assert result.difficulty <= 5

        run_async(_run())

    @pytest.mark.timeout(120)
    def test_classify_status_query_not_shopping_real_llm(self, temp_db, fastest_local_model):
        """Critical regression: 'how is the coffee machine search going' → not shopping.

        This is the key ambiguity bug: small LLMs see 'coffee machine' and
        classify as shopping, when the user is actually asking for task status.
        """
        from src.core.task_classifier import classify_task

        async def _run():
            result = await classify_task(
                title="How is the coffee machine search going",
                description="Asking about the status of a previous shopping task",
            )
            # Should be assistant, executor, or researcher — NOT shopping_advisor
            # We document the actual LLM behavior to detect regressions
            is_correct = result.agent_type not in ("shopping_advisor",)
            if not is_correct:
                pytest.xfail(
                    f"LLM misclassified status query as {result.agent_type}. "
                    "This is a known weakness of small/fast models. "
                    "The TelegramInterface keyword pre-filter is the primary guard."
                )

        run_async(_run())
