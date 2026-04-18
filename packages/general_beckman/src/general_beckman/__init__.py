"""General Beckman — the task master."""
from __future__ import annotations

from general_beckman.types import Task, AgentResult, Lane
from general_beckman.lifecycle import on_task_finished, set_orchestrator

__all__ = [
    "next_task", "on_task_finished", "tick", "set_orchestrator",
    "Task", "AgentResult", "Lane",
]


async def next_task() -> Task | None:
    """Stub — filled in by Task 10."""
    return None


async def tick() -> None:
    """Stub — filled in by Task 12."""
    return None
