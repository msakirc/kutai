"""Pure function: agent result -> list of actions orchestrator must take.

Phase 1: Action types are lightweight dataclasses consumed by orchestrator's
existing side-effect code (update_task, spawn subtasks, telegram.send, etc.).
Phase 2b: these become Decision emissions consumed by orchestrator's switch.
"""

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass(frozen=True)
class Complete:
    task_id: int
    result: str
    iterations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpawnSubtasks:
    parent_task_id: int
    subtasks: list[dict]


@dataclass(frozen=True)
class RequestClarification:
    task_id: int
    question: str
    chat_id: int | None = None


@dataclass(frozen=True)
class RequestReview:
    task_id: int
    summary: str


@dataclass(frozen=True)
class Exhausted:
    task_id: int
    error: str


@dataclass(frozen=True)
class Failed:
    task_id: int
    error: str


Action = Union[Complete, SpawnSubtasks, RequestClarification, RequestReview, Exhausted, Failed]


def route_result(task: dict, agent_result: dict | None) -> list[Action]:
    """Map (task, agent_result) -> actions orchestrator must execute."""
    task_id = task["id"]

    if agent_result is None:
        return [Failed(task_id=task_id, error="no_result_returned")]

    status = agent_result.get("status")

    if status == "completed":
        return [Complete(
            task_id=task_id,
            result=agent_result.get("result", ""),
            iterations=agent_result.get("iterations", 0),
            metadata=agent_result.get("metadata", {}),
        )]

    if status == "needs_subtasks":
        subtasks = agent_result.get("subtasks", [])
        return [SpawnSubtasks(parent_task_id=task_id, subtasks=subtasks)]

    if status == "needs_clarification":
        return [RequestClarification(
            task_id=task_id,
            question=agent_result.get("question", ""),
            chat_id=task.get("chat_id"),
        )]

    if status == "needs_review":
        return [RequestReview(
            task_id=task_id,
            summary=agent_result.get("summary", ""),
        )]

    if status == "exhausted":
        return [Exhausted(
            task_id=task_id,
            error=agent_result.get("error", "max_iterations_reached"),
        )]

    return [Failed(
        task_id=task_id,
        error=agent_result.get("error", f"unknown_status:{status}"),
    )]
