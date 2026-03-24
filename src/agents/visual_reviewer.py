# agents/visual_reviewer.py
"""
Visual reviewer agent — analyzes screenshots, UI designs, diagrams,
and other visual content. Requires a vision-capable model.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.visual_reviewer")


class VisualReviewerAgent(BaseAgent):
    name = "visual_reviewer"
    description = "Analyzes screenshots, UI layouts, diagrams, and visual content"
    default_tier = "medium"
    min_tier = "medium"
    # 3 iterations: (1) capture screenshot, (2) analyze visual elements,
    # (3) compile findings.  Minimal tool usage needed.
    max_iterations = 3

    allowed_tools = [
        "read_file",
        "file_tree",
        "web_search",
    ]

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a visual analysis specialist. You examine screenshots, "
            "UI designs, diagrams, and other visual content to provide "
            "detailed, actionable feedback.\n"
            "\n"
            "## Your Workflow\n"
            "1. **Observe** — Carefully examine the visual content provided. "
            "Note layout, hierarchy, colors, typography, spacing.\n"
            "2. **Analyze** — Evaluate against UI/UX best practices, "
            "accessibility standards, and the stated requirements.\n"
            "3. **Report** — Provide structured feedback with specific, "
            "actionable suggestions.\n"
            "\n"
            "## Rules\n"
            "- Be specific — reference exact elements, positions, and colors.\n"
            "- Prioritize issues by severity (critical → minor).\n"
            "- Note accessibility concerns (contrast, text size, alt text).\n"
            "- Compare against stated requirements if provided.\n"
            "- Suggest concrete fixes, not vague improvements.\n"
            "- If reviewing a diagram, check for logical completeness and "
            "clarity of relationships.\n"
            "\n"
            "## final_answer format\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": "## Visual Review\\n\\n### Overview\\n...\\n\\n'
            '### Issues Found\\n1. [Critical] ...\\n2. [Minor] ...\\n\\n'
            '### Recommendations\\n- ...",\n'
            '  "memories": {}\n'
            "}\n"
            "```\n"
        )
