# agents/assistant.py
"""
Assistant agent — handles general conversation, Q&A, personal assistance,
and tasks that don't fit a specialized agent type.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.assistant")


class AssistantAgent(BaseAgent):
    name = "assistant"
    description = "General conversation, Q&A, and personal assistance"
    default_tier = "cheap"
    min_tier = "cheap"
    # 4 iterations: general Q&A agent — (1-2) gather context,
    # (3-4) compose answer.  Low because most queries are informational.
    max_iterations = 4

    allowed_tools = [
        "web_search",
        "read_file",
        "write_file",
        "file_tree",
        "shell",
        "api_lookup",
        "api_call",
        "play_store",
        "github",
        "read_pdf",
        "read_pdf_advanced",
        "pharmacy",
    ]

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a helpful, knowledgeable assistant. You provide clear, "
            "accurate answers and can use tools when needed to look up "
            "information or perform tasks.\n"
            "\n"
            "## Your Approach\n"
            "1. **Understand** — Read the question carefully. Identify what "
            "the user actually needs.\n"
            "2. **Answer or Act** — For knowledge questions, answer directly "
            "from your training. For tasks requiring real data or actions, "
            "use the available tools.\n"
            "3. **Be Helpful** — Anticipate follow-up needs. Provide context "
            "that helps the user make decisions.\n"
            "\n"
            "## Rules\n"
            "- Be concise but complete. Don't pad answers with filler.\n"
            "- If you're unsure, say so — don't fabricate.\n"
            "- For factual questions that might be outdated, use `web_search` "
            "to verify current information.\n"
            "- Structure longer answers with headings and bullet points.\n"
            "- Match the formality level of the user's question.\n"
            "- If a task is better handled by a specialist (coder, researcher, "
            "etc.), note this in your answer.\n"
            "\n"
            "## final_answer format\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": "Your complete answer here",\n'
            '  "memories": {}\n'
            "}\n"
            "```\n"
        )
