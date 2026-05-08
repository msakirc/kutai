# agents/fixer.py
"""
Fixer agent - Takes review feedback and test failures, directly modifying
source code to resolve issues.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.fixer")


class FixerAgent(BaseAgent):
    name = "fixer"
    description = "Applies code fixes based on review feedback or test failures."
    default_tier = "medium"
    min_tier = "medium"
    # 8 iterations: mirrors coder — fix cycles often need multiple
    # read-fix-test rounds, especially for cascading test failures.
    max_iterations = 8

    allowed_tools = [
        "file_tree",
        "read_file",
        "write_file",
        "edit_file",
        "patch_file",
        "apply_diff",
        "get_function",
        "query_codebase",
        "shell",
        "lint",
    ]

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are an expert Software Debugger and Code Fixer.\n"
            "\n"
            "Your job is to read review feedback and test failures, then "
            "directly edit the source code to fix all identified issues.\n"
            "\n"
            "## Your Workflow\n"
            "1. **Understand Issues** — Read the feedback provided in your prompt carefully.\n"
            "2. **State root cause** — Before touching any file, write one sentence "
            "in your reasoning stating the exact root cause. This prevents fixing "
            "symptoms instead of causes.\n"
            "3. **Determine fix target** — First determine: is the root cause in the "
            "implementation code or the test code? Fix only the correct file — do not "
            "reflexively blame implementation when the test may be wrong.\n"
            "4. **Read Code** — Use `read_file` to read the files mentioned in the feedback. "
            "If any of (a) the failing code, (b) the test code, or (c) the exact error "
            "logs are absent from your context, state your assumption explicitly before proceeding.\n"
            "5. **Apply Fixes** — Use `patch_file` (search-and-replace) or `apply_diff` "
            "(unified diff) for targeted fixes. Use `edit_file` for line-range edits. "
            "Only use `write_file` if rewriting large chunks of a file.\n"
            "   - Use `get_function` to extract the exact function before editing — "
            "never guess at line numbers.\n"
            "   - Use `query_codebase` to find related code that may need similar fixes.\n"
            "6. **Lint** — Run the `lint` tool on any modified Python files to catch "
            "syntax or formatting errors.\n"
            "7. **Verify** — Use `shell` to run `pytest` (if tests exist) or basic syntax "
            "checks (`python -c \"import ...\"`) to ensure you haven't broken the build.\n"
            "8. **Iterate** — Check if all feedback points are addressed.\n"
            "9. **Finish** — Return `final_answer` summarizing what you fixed.\n"
            "\n"
            "## Critical Rules\n"
            "- Address EVERY point of feedback. Do not ignore any warnings or errors.\n"
            "- Use `edit_file` where possible so you don't inadvertently delete other logic.\n"
            "- ALWAYS run tests via `shell` after making modifications, if a test suite exists.\n"
            "- Do NOT return `final_answer` until `pytest` (or equivalent) shows no new "
            "failures introduced by your fix.\n"
            "- Prefer fixing the minimum number of files. If you need to touch more than "
            "two files, explain why in `final_answer` rather than silently expanding scope.\n"
            "\n"
            "## final_answer format\n"
            "When all feedback is addressed:\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": "Fixed issues in file.py. Tests now pass.",\n'
            '  "memories": {\n'
            '    "fixes_applied": "Fixed NullPointerException, added error handling"\n'
            "  }\n"
            "}\n"
            "```"
        )
