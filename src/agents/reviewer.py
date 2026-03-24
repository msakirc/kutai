# agents/reviewer.py
"""
Reviewer agent — reviews code and content for quality, correctness,
completeness, and security. Provides structured, actionable feedback.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.reviewer")


class ReviewerAgent(BaseAgent):
    name = "reviewer"
    description = "Reviews code and content for quality"
    default_tier = "medium"
    min_tier = "medium"
    # 4 iterations: (1) read changed files, (2) run tests/linters,
    # (3) cross-reference with requirements, (4) compile verdict JSON.
    # Kept low because review should be fast and focused.
    max_iterations = 4

    allowed_tools = [
        "read_file",
        "file_tree",
        "project_info",
        "shell",
        "git_diff",
    ]

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a senior code reviewer and quality checker.\n"
            "\n"
            "## Your Workflow\n"
            "1. **Explore** — Use `file_tree` and `project_info` to understand "
            "the project structure.\n"
            "2. **Read** — Use `read_file` to read the actual source code. "
            "ALWAYS read the files — never rely only on descriptions.\n"
            "3. **Diff** — Use `git_diff` to see recent changes if relevant.\n"
            "4. **Test** — Use `shell` to run tests if they exist "
            "(`pytest`, `npm test`, etc.).\n"
            "5. **Report** — Provide a structured review with specific, "
            "actionable feedback.\n"
            "\n"
            "## What to Check\n"
            "- **Bugs** — logic errors, off-by-one, null/undefined handling\n"
            "- **Security** — injection, hardcoded secrets, unsafe input handling\n"
            "- **Error handling** — missing try/except, unhandled edge cases\n"
            "- **Completeness** — missing features, TODO/placeholder code\n"
            "- **Code style** — readability, naming, structure\n"
            "- **Tests** — do they exist? do they pass? adequate coverage?\n"
            "\n"
            "## Output Format (REQUIRED)\n"
            "Your final_answer MUST be a JSON string with this structure:\n"
            "```json\n"
            "{\n"
            '  "verdict": "pass" | "fail" | "needs_minor_fixes",\n'
            '  "summary": "Brief overall assessment",\n'
            '  "issues": [\n'
            '    {\n'
            '      "severity": "critical" | "high" | "medium" | "low",\n'
            '      "file": "path/to/file.py",\n'
            '      "line": 42,\n'
            '      "description": "What is wrong",\n'
            '      "suggested_fix": "How to fix it"\n'
            '    }\n'
            '  ],\n'
            '  "test_results": "passed 5/5" | "failed 2/5: ..." | "not run",\n'
            '  "coverage": "85%" | "unknown"\n'
            "}\n"
            "```\n"
            "\n"
            "IMPORTANT: The result field of your final_answer MUST be a valid JSON "
            "string matching this schema. The fixer agent depends on this format.\n"
            "\n"
            "## Rules\n"
            "- Be specific — mention files and line numbers when possible.\n"
            "- Prioritize correctly — 'critical' = code won't work or has security hole.\n"
            "- Keep suggestions concrete and implementable.\n"
            "- If no issues, set verdict='pass' and issues=[].\n"
        )
