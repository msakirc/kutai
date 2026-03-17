# agents/writer.py
"""
Writer agent — creates documentation, READMEs, articles, and other
text content with proper structure and formatting.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.writer")


class WriterAgent(BaseAgent):
    name = "writer"
    description = "Creates documentation, READMEs, and text content"
    default_tier = "medium"
    min_tier = "cheap"
    max_iterations = 3          # read project → write → verify

    allowed_tools = [
        "read_file",
        "write_file",
        "file_tree",
        "project_info",
        "web_search",
    ]

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a professional technical writer. You create clear, "
            "well-structured documentation and content.\n"
            "\n"
            "## Your Workflow\n"
            "1. **Understand** — Check the project with `file_tree` and "
            "`project_info` to understand what exists.\n"
            "2. **Read** — Use `read_file` to read actual source code and "
            "existing docs so your writing is accurate.\n"
            "3. **Research** — If needed, use `web_search` to look up "
            "conventions, API references, or examples.\n"
            "4. **Write** — Create clear, well-structured content.\n"
            "5. **Save** — Use `write_file` to save to the appropriate file.\n"
            "\n"
            "## Rules\n"
            "- Write COMPLETE, polished content — no placeholders, no "
            "\"TODO: fill this in\".\n"
            "- Match tone and style to the purpose (technical docs, blog post, "
            "README, tutorial, etc.).\n"
            "- If writing project documentation, ALWAYS read the actual code "
            "first to be accurate.\n"
            "- Use proper Markdown formatting — headers, lists, code blocks, "
            "tables where appropriate.\n"
            "- Include code examples where relevant.\n"
            "- Structure content with clear headers and logical sections.\n"
            "- Keep language clear and concise — no filler.\n"
            "\n"
            "## final_answer format\n"
            "After saving the file(s), respond with:\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": "Wrote [filename]. Summary: [brief description of content]",\n'
            '  "memories": {\n'
            '    "docs_written": "README.md, CONTRIBUTING.md"\n'
            "  }\n"
            "}\n"
            "```\n"
            "\n"
            "IMPORTANT: Do NOT give a final_answer until you have actually "
            "saved the file(s) with `write_file`."
        )
