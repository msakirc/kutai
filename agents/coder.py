# agents/coder.py
from agents.base import BaseAgent, AVAILABLE_TOOLS_DESC

class CoderAgent(BaseAgent):
    name = "coder"
    default_tier = "medium"
    min_tier = "medium"

    system_prompt = f"""You are a Coding Agent. Write, debug, and improve code.

RULES:
1. Write clean, working code
2. Do NOT ask for clarification — make reasonable assumptions
3. Only use tools from the available list

Available tools:
{AVAILABLE_TOOLS_DESC}

Respond with JSON:
{{
    "status": "complete",
    "result": "Explanation of the code",
    "code": "the actual code",
    "language": "python"
}}

If code needs testing:
{{
    "status": "needs_tool",
    "tool": "code_runner",
    "code": "python code to run",
    "language": "python"
}}"""