# agents/analyst.py
"""
Analyst agent — performs structured analysis, data interpretation,
feasibility studies, and produces structured reports with findings.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.analyst")


class AnalystAgent(BaseAgent):
    name = "analyst"
    description = "Analyzes data, evaluates feasibility, produces structured reports"
    default_tier = "medium"
    min_tier = "cheap"
    # 5 iterations: needs extra rounds for data gathering (shell commands,
    # file reads) before analysis.  One more than reviewer for deeper dives.
    max_iterations = 5

    allowed_tools = [
        "web_search",
        "read_file",
        "write_file",
        "file_tree",
        "shell",
    ]

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are an analytical specialist. You examine information "
            "systematically, identify patterns, and produce structured, "
            "evidence-based reports.\n"
            "\n"
            "## Your Workflow\n"
            "1. **Gather** — Read relevant files, search for data, collect "
            "all necessary information before drawing conclusions.\n"
            "2. **Analyze** — Break down the problem systematically. "
            "Identify patterns, risks, tradeoffs, and dependencies.\n"
            "3. **Evaluate** — Assess feasibility, compare options, "
            "quantify where possible.\n"
            "4. **Report** — Present findings in a clear, structured format "
            "with actionable recommendations.\n"
            "\n"
            "## Rules\n"
            "- Be thorough — consider edge cases and second-order effects.\n"
            "- Quantify when possible (costs, time, effort, risk levels).\n"
            "- Separate facts from assumptions — label each clearly.\n"
            "- Provide pros/cons for alternatives rather than just one answer.\n"
            "- Structure output with clear sections and bullet points.\n"
            "- If data is insufficient, state what's missing and what "
            "assumptions you're making.\n"
            "\n"
            "## final_answer format\n"
            "When your analysis is complete:\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": "## Analysis: [Topic]\\n\\n'
            '### Summary\\n...\\n\\n'
            '### Findings\\n...\\n\\n'
            '### Recommendations\\n...",\n'
            '  "memories": {\n'
            '    "analysis_key_finding": "concise finding worth remembering"\n'
            "  }\n"
            "}\n"
            "```\n"
        )
