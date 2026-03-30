# agents/researcher.py
"""
Researcher agent — searches for information, evaluates sources,
and synthesizes findings into clear, actionable summaries.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.researcher")


class ResearcherAgent(BaseAgent):
    name = "researcher"
    description = "Researches topics, gathers information, synthesizes findings"
    default_tier = "medium"
    min_tier = "cheap"
    # 4 iterations max. Perplexica results include a hint to finalize
    # immediately, so most tasks complete in 2 iterations.
    max_iterations = 4
    enable_self_reflection = True

    allowed_tools = [
        "web_search",
        "read_file",
        "write_file",
        "file_tree",
        "api_lookup",
        "api_call",
        "play_store",
        "github",
        "pharmacy",
    ]

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a research specialist. You find accurate, useful "
            "information and present it clearly.\n"
            "\n"
            "## Your Workflow\n"
            "1. **Search** — Use `web_search` once. The search tool already "
            "queries multiple sources and synthesizes results.\n"
            "2. **Evaluate** — If the result is comprehensive, go straight to "
            "final_answer. Only search again if the first result is clearly "
            "incomplete or off-topic.\n"
            "3. **Finalize** — Present findings with `final_answer`.\n"
            "\n"
            "## IMPORTANT: Be efficient\n"
            "- ONE search is usually enough. Do NOT search multiple times "
            "unless the first result genuinely lacks the answer.\n"
            "- After getting search results, respond with `final_answer` "
            "immediately. Do not search again with rephrased queries.\n"
            "\n"
            "## Rules\n"
            "- Cite sources when possible (include URLs).\n"
            "- If you can't find reliable information, say so honestly.\n"
            "- Keep summaries focused and actionable — no filler.\n"
            "\n"
            "## final_answer format\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": "## Research: [Topic]\\n\\n### Key Findings\\n...",\n'
            '  "memories": {\n'
            '    "research_topic_key_finding": "concise finding worth remembering"\n'
            "  }\n"
            "}\n"
            "```\n"
            "\n"
            "Use `memories` to store key facts that later agents might need "
            "(e.g., recommended library, API URL, important version constraint)."
        )
