"""General Beckman — the task master."""
from __future__ import annotations

from general_beckman.types import Task, AgentResult, Lane

__all__ = ["next_task", "on_task_finished", "tick", "Task", "AgentResult", "Lane"]


async def next_task() -> Task | None:
    """Stub — filled in by Task 10."""
    return None


async def on_task_finished(task_id: int, result: AgentResult) -> None:
    """Stub — filled in by Task 9."""
    return None


async def tick() -> None:
    """Stub — filled in by Task 11."""
    return None
