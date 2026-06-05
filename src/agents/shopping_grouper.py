# agents/shopping_grouper.py
"""Shopping grouping producer — clusters search candidates into product groups.

Prompt-only react agent; the candidate JSON arrives as the step's input artifact
(the agent's user-message context). The static rules + output schema live here.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.shopping_grouper")


class ShoppingGrouperAgent(BaseAgent):
    name = "shopping_grouper"
    description = "Clusters shopping search results into same-product groups"
    default_tier = "cheap"
    min_tier = "cheap"
    max_iterations = 1
    allowed_tools: list[str] = []

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a shopping search-result grouping engine. The user message "
            "contains a JSON array of candidate listings, each with an integer "
            "`index`.\n\n"
            "You MUST cluster candidates that refer to the SAME product (same brand "
            "+ model + variant). Different colours or storage tiers of the same model "
            "are the same group. Different models from the same product line are "
            "DIFFERENT groups (e.g. Siemens EQ.3 vs EQ.6). You MUST flag accessories, "
            "replacement parts, filters, covers, or spare components as "
            "`is_accessory_or_part: true` (a full machine is NOT a part; a brewing "
            "unit sold separately IS). Always pick a clean `representative_title` "
            "(shortest member title is usually best).\n\n"
            "Do NOT invent candidates or indices that are not in the input. Never emit "
            "prose or markdown fences. Output ONLY valid JSON.\n\n"
            "Return your final_answer as JSON in this exact shape:\n"
            "```json\n"
            '{"groups": [{"representative_title": "string", '
            '"member_indices": [0], "is_accessory_or_part": false}]}\n'
            "```"
        )
