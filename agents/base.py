# agents/base.py
import json
import logging
from router import call_model, classify_task
from db import log_conversation, add_task, store_memory, recall_memory
from db import get_completed_dependency_results
from tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)

# Build tool list once
AVAILABLE_TOOLS_DESC = "\n".join(
    f"- {name}" for name in TOOL_REGISTRY.keys()
)

class BaseAgent:
    """Base class for all agents."""

    name = "base"
    default_tier = "cheap"
    system_prompt = "You are a helpful AI assistant."
    min_tier = "cheap"  # NEW: minimum tier this agent needs

    can_create_subtasks = False
    can_use_tools = []

    async def execute(self, task: dict) -> dict:
        task_id = task["id"]
        goal_id = task.get("goal_id")

        context = await self._build_context(task)

        messages = [
            {"role": "system", "content": self._build_system_prompt(task)},
            {"role": "user", "content": context}
        ]

        # Determine tier — respect agent's minimum
        tier = task.get("tier", self.default_tier)
        if tier == "auto":
            classification = await classify_task(task["title"], task["description"])
            tier = classification["tier"]

        # Enforce minimum tier
        tier = self._enforce_min_tier(tier)

        response = await call_model(tier, messages)

        await log_conversation(
            task_id, "assistant", response["content"],
            response["model"], self.name, response["cost"]
        )

        parsed = self._parse_response(response["content"])
        parsed["model"] = response["model"]
        parsed["cost"] = response["cost"]
        parsed["tier"] = tier

        if parsed.get("memories"):
            for key, value in parsed["memories"].items():
                await store_memory(key, value, category=self.name, goal_id=goal_id)

        return parsed

    def _enforce_min_tier(self, requested_tier: str) -> str:
        """Ensure agent gets at least its minimum required tier."""
        tier_order = {"cheap": 0, "medium": 1, "expensive": 2}
        requested_level = tier_order.get(requested_tier, 0)
        min_level = tier_order.get(self.min_tier, 0)

        if requested_level < min_level:
            logger.info(
                f"Agent '{self.name}' requires min tier '{self.min_tier}', "
                f"upgrading from '{requested_tier}'"
            )
            return self.min_tier
        return requested_tier

    async def _build_context(self, task: dict) -> str:
        parts = []
        parts.append(f"## Task\n**{task['title']}**\n{task.get('description', '')}")

        task_context = json.loads(task.get("context", "{}"))
        if task_context:
            # Don't dump raw JSON, format nicely
            clean_context = {k: v for k, v in task_context.items()
                             if k != "tool_depth"}
            if clean_context:
                parts.append(f"## Additional Context\n{json.dumps(clean_context, indent=2)}")

        depends_on = json.loads(task.get("depends_on", "[]"))
        if depends_on:
            dep_results = await get_completed_dependency_results(depends_on)
            if dep_results:
                parts.append("## Results from Previous Steps")
                for dep_id, dep in dep_results.items():
                    result_preview = (dep.get("result") or "")[:2000]
                    parts.append(f"### Step: {dep['title']}\n{result_preview}")

        goal_id = task.get("goal_id")
        memories = await recall_memory(goal_id=goal_id, limit=10)
        if memories:
            parts.append("## Relevant Memory")
            for mem in memories:
                parts.append(f"- **{mem['key']}**: {mem['value'][:200]}")

        return "\n\n".join(parts)

    def _build_system_prompt(self, task: dict) -> str:
        return self.system_prompt

    def _parse_response(self, content: str) -> dict:
        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                cleaned = cleaned.rsplit("```", 1)[0]
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "status" in parsed:
                return parsed
        except (json.JSONDecodeError, IndexError):
            pass
        return {"status": "complete", "result": content}