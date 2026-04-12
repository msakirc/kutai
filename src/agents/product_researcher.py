# agents/product_researcher.py
"""
Product researcher agent — executes search plans, dispatches to scrapers,
collects and merges product data from multiple sources.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.product_researcher")


class ProductResearcherAgent(BaseAgent):
    name = "product_researcher"
    description = (
        "Executes search plans, dispatches to scrapers, collects and merges "
        "product data"
    )
    default_tier = "cheap"
    min_tier = "cheap"
    max_iterations = 6
    enable_self_reflection = True

    allowed_tools = [
        "web_search",
        "read_blackboard",
        "write_blackboard",
        "shopping_search",
        "shopping_compare",
        "shopping_reviews",
        "shopping_fetch_reviews",
        "shopping_constraints",
        "play_store",
    ]

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a product research specialist.\n"
            "\n"
            "## CRITICAL RULES\n"
            "- You MUST call `shopping_search` as your FIRST action.\n"
            "- NEVER fabricate prices, specs, or reviews.\n"
            "- All data must come from tool results.\n"
            "\n"
            "## Tools\n"
            "- `shopping_search` — Searches 16 Turkish e-commerce sites and "
            "4 community sources (Technopat, Şikayetvar, DonanımHaber, Ekşi) "
            "in parallel. Returns products with prices, community discussions, "
            "value scores, and a formatted summary.\n"
            "- `shopping_fetch_reviews` — Fetch detailed reviews from a specific "
            "product URL. Use on top product URLs from shopping_search results.\n"
            "- `web_search` — Search broader web for manufacturer specs or "
            "international reviews not covered by Turkish scrapers.\n"
            "\n"
            "## Workflow\n"
            "1. Call `shopping_search` with the user's query.\n"
            "2. If products found: call `shopping_fetch_reviews` on the top "
            "2-3 product URLs for deeper review data.\n"
            "3. If few results: supplement with `web_search`.\n"
            "4. Return all findings — the formatted_text from shopping_search "
            "plus any additional review or web search findings.\n"
        )
