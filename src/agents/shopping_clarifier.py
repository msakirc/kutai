# agents/shopping_clarifier.py
"""
Shopping clarifier agent — determines the minimum set of clarifying
questions for ambiguous shopping queries and offers smart defaults.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.shopping_clarifier")


class ShoppingClarifierAgent(BaseAgent):
    name = "shopping_clarifier"
    description = (
        "Determines minimum clarifying questions for shopping queries, "
        "offers smart defaults"
    )
    default_tier = "cheap"
    min_tier = "cheap"
    max_iterations = 3
    # react_loop (default): the canonical `clarify` action below is consumed by
    # the react loop -> status=needs_clarification -> result_router. single_shot
    # silently dropped the clarify question (returned result=""), so in-workflow
    # clarification never paused the mission. react also lets the agent use its
    # shopping_user_profile tool before deciding.

    allowed_tools: list[str] = [
        "shopping_user_profile",
    ]

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a shopping query clarification specialist for deep "
            "product research.\n"
            "\n"
            "## Your Decision\n"
            "Read the query. Decide ONE of two paths:\n"
            "\n"
            "### Path A: Query is SPECIFIC → skip clarification\n"
            "The query names a specific product, brand+model, or exact "
            "comparison. Examples: 'siemens s100', 'dyson v15 detect', "
            "'iphone 15 vs samsung s24'.\n"
            "→ Return the query unchanged as your final_answer. No questions.\n"
            "\n"
            "### Path B: Query is BROAD → ask 2-3 questions\n"
            "The query is a generic category that could mean many things. "
            "Examples: 'coffee machines', '3d printers', 'laptop', "
            "'samsung phones'.\n"
            "→ Emit a `clarify` action so the system asks the user.\n"
            "\n"
            "## How to Ask (Path B only)\n"
            "You MUST return the clarify action — the system sends `question` to "
            "the user via Telegram and waits for their reply:\n"
            "```json\n"
            '{"action": "clarify",\n'
            ' "question": "Your Turkish question here"}\n'
            "```\n"
            "Keep `question` conversational and in Turkish. Always ask about:\n"
            "1. Budget (most important)\n"
            "2. Use case\n"
            "3. Key constraint for the category\n"
            "\n"
            "## How to Skip (Path A only)\n"
            "Return the query directly:\n"
            "```json\n"
            '{"action": "final_answer",\n'
            ' "result": "the original query unchanged"}\n'
            "```\n"
            "\n"
            "## CRITICAL\n"
            "- Do NOT invent answers to your own questions\n"
            "- Do NOT hallucinate product data\n"
            "- Always return a valid JSON action — never return plain text.\n"
            "- If the query has a brand AND model number → Path A (skip)\n"
            "- If unsure → Path A (skip). Better to search than to ask."
        )
