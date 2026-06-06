# agents/shopping_synthesizer.py
"""Shopping review-synthesis producer — mines aspects/praise/complaints/red-flags
from a review snippet pile for one product line. Prompt-only react agent.

The product title + snippet pile arrive as the step's input artifact.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.shopping_synthesizer")


class ShoppingSynthesizerAgent(BaseAgent):
    name = "shopping_synthesizer"
    description = "Synthesises user reviews into aspect-level insights for one product"
    default_tier = "balanced"
    min_tier = "cheap"
    max_iterations = 1
    allowed_tools: list[str] = []

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are an INTELLIGENCE module summarising user reviews for one product. "
            "The user message contains the product title and a JSON array of review "
            "snippets (Turkish and/or English, multi-source).\n\n"
            "You MUST mine substance and ignore boilerplate ('teşekkürler', 'kargo "
            "hızlı'). For each aspect ACTUALLY discussed, emit one entry with: `aspect` "
            "(one of kamera|pil|ekran|performans|yapım_kalitesi|yazılım|fiyat|satıcı|"
            "kargo|ses|şarj|güncellemeler|oyun|boyut|ergonomi|ısınma — only those that "
            "appear); `sentiment` float [-1,1]; `mention_count` int; `summary` (one "
            "short Turkish line); `quote` (ONE verbatim snippet ≤140 chars). Always "
            "sort aspects by mention_count desc, up to 8. Use the dominant language of "
            "the snippets.\n\n"
            "Do NOT fabricate — if a snippet doesn't say it, don't write it. Never emit "
            "prose or fences. Set `insufficient_data` true only if <3 substantive "
            "snippets; when true leave lists empty. Output ONLY valid JSON.\n\n"
            "Return your final_answer as JSON in this exact shape:\n"
            "```json\n"
            '{"aspects": [{"aspect": "kamera", "sentiment": 0.5, "mention_count": 3, '
            '"summary": "string", "quote": "string"}], "comparative_mentions": [], '
            '"notable_quote": "string", "overall_sentiment": 0.0, "praise": [], '
            '"complaints": [], "red_flags": [], "insufficient_data": false}\n'
            "```"
        )
