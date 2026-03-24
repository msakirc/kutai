# agents/error_recovery.py
"""
Error Recovery agent — receives failed tasks with error logs,
diagnoses root cause, either fixes and retries or escalates
with a clear diagnosis.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.error_recovery")


class ErrorRecoveryAgent(BaseAgent):
    name = "error_recovery"
    description = "Diagnoses failed tasks, fixes or escalates with clear analysis"
    default_tier = "medium"
    min_tier = "cheap"
    # 4 iterations: (1) read error context, (2) diagnose root cause,
    # (3) attempt fix, (4) verify.  Kept tight to avoid runaway recovery loops.
    max_iterations = 4
    enable_self_reflection = True

    allowed_tools = [
        "shell",
        "read_file",
        "write_file",
        "file_tree",
        "run_code",
    ]

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are an error recovery specialist. A task has failed and "
            "you need to diagnose why and attempt to fix it.\n"
            "\n"
            "## Your Process\n"
            "1. **Read the error** carefully — understand the root cause.\n"
            "2. **Investigate** — check files, logs, or run diagnostic commands.\n"
            "3. **Diagnose** — identify the root cause category:\n"
            "   - Bad prompt (LLM misunderstood the task)\n"
            "   - Missing tool (needed capability not available)\n"
            "   - Model too weak (task needs a stronger model)\n"
            "   - Missing dependency (package, file, or config not present)\n"
            "   - Environment issue (Docker, permissions, network)\n"
            "   - Logic error (incorrect approach or code)\n"
            "4. **Fix** — if fixable, apply the fix and verify it works.\n"
            "5. **Report** — provide a clear diagnosis and what you did.\n"
            "\n"
            "## Response Format\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": "## Error Recovery Report\\n\\n'
            '**Root Cause:** ...\\n**Category:** ...\\n'
            '**Fix Applied:** ...\\n**Verified:** yes/no",\n'
            '  "memories": {\n'
            '    "error_pattern_X": "description for future avoidance"\n'
            "  }\n"
            "}\n"
            "```"
        )
