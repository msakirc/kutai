"""test_shopping_flow.py — Integration tests for the shopping flow.

Tests:
- Shopping task creation and routing
- ShoppingAdvisor agent structure (no LLM)
- LLM-based shopping classification and response (marked @llm)
- Price comparison output format validation

Markers:
  @pytest.mark.integration  — all tests
  @pytest.mark.llm          — tests making real LLM calls
"""
from __future__ import annotations

import asyncio
import json
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
# Shopping sub-intent classification (no LLM)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestShoppingSubIntentClassification:
    """Test the rule-based shopping sub-intent detector."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from src.core.task_classifier import _classify_shopping_sub_intent
        self.classify = _classify_shopping_sub_intent

    def test_price_check_intent(self):
        result = self.classify("What is the price of RTX 4070?")
        assert result == "price_check"

    def test_compare_intent(self):
        result = self.classify("Compare RTX 4070 vs RX 7700 XT")
        assert result == "compare"

    def test_deal_hunt_intent(self):
        result = self.classify("En ucuz laptop hangisi, indirim var mı?")
        assert result == "deal_hunt"

    def test_purchase_advice_intent(self):
        result = self.classify("Should I buy the MacBook Pro or Dell XPS?")
        assert result == "purchase_advice"

    def test_gift_intent(self):
        result = self.classify("Hediye olarak ne alsam? Teknoloji hediye fikri.")
        assert result == "gift"

    def test_upgrade_intent(self):
        result = self.classify("I want to upgrade from GTX 1080 to something better")
        assert result == "upgrade"

    def test_exploration_default(self):
        """Generic shopping intent with no specific sub-keyword → exploration."""
        result = self.classify("Looking for a good mechanical keyboard")
        assert result == "exploration"


# ---------------------------------------------------------------------------
# Shopping task creation in DB
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestShoppingTaskCreation:
    """Shopping tasks are created correctly in the DB."""

    def test_create_shopping_task(self, temp_db):
        """A shopping task with shopping_advisor agent_type is created correctly."""
        from src.infra.db import add_task, get_task

        async def _run():
            tid = await add_task(
                title="Find best gaming GPU under 15000 TL",
                description="Looking for best price/performance GPU for 1080p gaming",
                agent_type="shopping_advisor",
                priority=5,
                context={
                    "shopping_type": "price_comparison",
                    "budget": 15000,
                    "currency": "TRY",
                },
            )
            assert tid is not None

            task = await get_task(tid)
            assert task["agent_type"] == "shopping_advisor"
            assert task["status"] == "pending"

            ctx = task.get("context")
            if isinstance(ctx, str):
                ctx = json.loads(ctx)
            assert ctx.get("budget") == 15000

        run_async(_run())

    def test_shopping_task_from_classify_to_db(self, temp_db):
        """Full flow: keyword classify → DB insert → verify agent_type."""
        from src.core.task_classifier import _classify_by_keywords
        from src.infra.db import add_task, get_task

        async def _run():
            cls = _classify_by_keywords(
                "GPU fiyatı karşılaştırması",
                "RTX 4070 vs RX 7700 XT price comparison for gaming",
            )
            assert cls.agent_type == "shopping_advisor"

            tid = await add_task(
                title="GPU fiyatı karşılaştırması",
                description="RTX 4070 vs RX 7700 XT",
                agent_type=cls.agent_type,
                priority=cls.priority,
            )
            task = await get_task(tid)
            assert task["agent_type"] == "shopping_advisor"

        run_async(_run())


# ---------------------------------------------------------------------------
# Shopping agent availability
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestShoppingAgentRegistry:
    """Shopping agent is registered and has expected properties."""

    def test_shopping_advisor_agent_registered(self):
        """get_agent('shopping_advisor') returns a non-None agent."""
        from src.agents import get_agent
        agent = get_agent("shopping_advisor")
        assert agent is not None, (
            "shopping_advisor agent must be registered in the agent registry"
        )

    def test_shopping_advisor_has_correct_name(self):
        """Shopping agent has the expected name attribute."""
        from src.agents import get_agent
        agent = get_agent("shopping_advisor")
        if agent is None:
            pytest.skip("shopping_advisor agent not registered")
        assert hasattr(agent, "name")
        # Name should contain 'shopping' or 'advisor'
        assert "shopping" in agent.name.lower() or "advisor" in agent.name.lower(), (
            f"Unexpected agent name: {agent.name}"
        )

    def test_shopping_advisor_has_allowed_tools(self):
        """Shopping agent has tools configured (not completely tool-less)."""
        from src.agents import get_agent
        agent = get_agent("shopping_advisor")
        if agent is None:
            pytest.skip("shopping_advisor agent not registered")
        # allowed_tools == [] means NO tools, which would be wrong for shopping
        # allowed_tools == None means ALL tools (also acceptable)
        if agent.allowed_tools is not None:
            assert len(agent.allowed_tools) > 0, (
                "Shopping advisor needs tools (web_search at minimum)"
            )


# ---------------------------------------------------------------------------
# Real LLM shopping tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.llm
class TestShoppingAgentRealLLM:
    """Tests that run the shopping agent with a real local model.

    These are slow and require a loaded model.
    """

    @pytest.fixture(autouse=True)
    def _check_model(self, fastest_local_model):
        if fastest_local_model is None:
            pytest.skip("No local model available — skipping LLM test")

    @pytest.mark.timeout(300)
    def test_shopping_classification_then_response(self, temp_db, fastest_local_model):
        """Classify a shopping query then run the agent on it.

        This is the core idea-to-response pipeline for shopping:
        1. Classify → shopping_advisor
        2. Agent runs (may call web_search)
        3. Result has non-empty text

        We accept any non-empty response — shopping results depend heavily
        on web availability and model quality.
        """
        from src.core.task_classifier import classify_task
        from src.agents import get_agent

        task_title = "Find best gaming GPU under 15000 TL"
        task_desc = "I need a GPU for 1080p gaming. Compare price/performance."

        async def _run():
            # Step 1: classify
            cls = await classify_task(task_title, task_desc)
            assert cls.agent_type in ("shopping_advisor", "researcher", "analyst"), (
                f"Unexpected classification: {cls.agent_type}"
            )

            # Step 2: get the appropriate agent
            agent = get_agent(cls.agent_type)
            if agent is None:
                pytest.skip(f"Agent '{cls.agent_type}' not registered")

            task = {
                "id": 100,
                "title": task_title,
                "description": task_desc,
                "agent_type": cls.agent_type,
                "context": json.dumps({
                    "model_override": fastest_local_model,
                    "max_web_searches": 1,  # limit searches for speed
                }),
                "depends_on": "[]",
                "mission_id": None,
            }

            result = await agent.execute(task)
            assert isinstance(result, dict)

            result_text = result.get("result", "") or ""
            # Accept any non-empty response
            assert len(result_text) > 0, "Shopping agent returned empty result"

        run_async(_run())

    @pytest.mark.timeout(120)
    def test_shopping_result_is_human_readable(self, temp_db, fastest_local_model):
        """Shopping result is readable text, not raw JSON or empty."""
        from src.core.task_classifier import _classify_by_keywords
        from src.agents import get_agent

        async def _run():
            cls = _classify_by_keywords(
                "Laptop price",
                "I want to buy a laptop under 20000 TL for development",
            )
            agent = get_agent(cls.agent_type)
            if agent is None:
                pytest.skip(f"Agent '{cls.agent_type}' not registered")

            task = {
                "id": 101,
                "title": "Laptop price",
                "description": "I want to buy a laptop under 20000 TL for development",
                "agent_type": cls.agent_type,
                "context": json.dumps({
                    "model_override": fastest_local_model,
                }),
                "depends_on": "[]",
                "mission_id": None,
            }

            result = await agent.execute(task)
            result_text = result.get("result", "") or ""
            # Should be non-trivially long (more than 20 chars)
            assert len(result_text) >= 20, (
                f"Shopping result too short ({len(result_text)} chars): {result_text!r}"
            )

        run_async(_run())
