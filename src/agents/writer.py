# agents/writer.py
"""
Writer agent — creates documentation, READMEs, articles, and other
text content with proper structure and formatting.

The system prompt is schema-aware: when the workflow step declares an
artifact_schema with ``type == "markdown"``, the agent switches from the
"write file + return summary" pattern to "emit markdown content
directly in result". The old pattern reliably failed required-section
validation because the result field carried only a summary blurb while
the actual content lived in a file the schema validator never read
(observed on i2p_v3 steps 7.15, 11.4, 12.1 before the switch).
"""
import json

from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.writer")


def _detect_markdown_schema(task: dict) -> bool:
    """Return True if the task's artifact_schema declares a markdown output."""
    ctx = task.get("context") or {}
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
            if isinstance(ctx, str):
                ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError, ValueError):
            return False
    if not isinstance(ctx, dict):
        return False
    schema = ctx.get("artifact_schema")
    if not isinstance(schema, dict):
        return False
    for v in schema.values():
        if isinstance(v, dict) and v.get("type") == "markdown":
            return True
    return False


_FILE_WRITE_PROMPT = (
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


_INLINE_MARKDOWN_PROMPT = (
    "You are a professional technical writer producing a structured "
    "markdown artifact for a workflow step. The artifact_schema for "
    "this step requires the FULL markdown content in the `result` "
    "field — not a summary, not a filename.\n"
    "\n"
    "## Your Workflow\n"
    "1. **Understand** — Read the task instruction and any provided "
    "input artifacts in the prompt.\n"
    "2. **Research** — If `read_file` / `web_search` are available and "
    "the task needs verification, use them. Otherwise skip straight to "
    "writing.\n"
    "3. **Write** — Produce the complete markdown directly. Every "
    "section the schema requires must appear with substantive content "
    "beneath the heading.\n"
    "\n"
    "## Rules\n"
    "- DO NOT call `write_file`. The workflow engine persists the "
    "artifact itself once you return — calling write_file wastes an "
    "iteration and leaves the schema validator looking at a summary "
    "blurb in `result`.\n"
    "- Write COMPLETE content — no placeholders, no \"TODO: fill in\".\n"
    "- Use proper markdown — `## Heading`, lists, code blocks, tables.\n"
    "- Include every section the instruction or schema names. Do not "
    "skip or rename headings — required-section validation matches by "
    "exact heading text.\n"
    "- Length should match the workflow step's expected output token "
    "budget. Aim for substantive content per section, not one-liners.\n"
    "\n"
    "## final_answer format\n"
    "Return the full markdown as the value of `result`:\n"
    "```json\n"
    "{\n"
    '  "action": "final_answer",\n'
    '  "result": "# Title\\n\\n## Section One\\n\\nFull content "\n'
    '             "here...\\n\\n## Section Two\\n\\n..."\n'
    "}\n"
    "```\n"
    "Inner quotes inside the markdown must be escaped exactly once "
    "(`\\\"`). Do not double-escape — the workflow engine canonicalizes "
    "your output and an extra escape layer just makes retries harder."
)


class WriterAgent(BaseAgent):
    name = "writer"
    description = "Creates documentation, READMEs, and text content"
    default_tier = "medium"
    min_tier = "cheap"
    # 5 iterations: (1) read project context, (2-3) write docs/prose,
    # (4) verify output, (5) fix issues.  Workflow steps with large context
    # need more room than simple doc generation.
    max_iterations = 5
    enable_self_reflection = True

    allowed_tools = [
        "read_file",
        "write_file",
        "file_tree",
        "project_info",
        "web_search",
    ]

    def get_system_prompt(self, task: dict) -> str:
        if _detect_markdown_schema(task):
            return _INLINE_MARKDOWN_PROMPT
        return _FILE_WRITE_PROMPT
