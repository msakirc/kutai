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
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SpawnSubtasks:
    parent_task_id: int
    subtasks: list[dict]
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RequestClarification:
    task_id: int
    question: str
    chat_id: int | None = None
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RequestReview:
    task_id: int
    summary: str
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Exhausted:
    task_id: int
    error: str
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Failed:
    task_id: int
    error: str
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MissionAdvance:
    """Signal that a mission task completed cleanly — spawn workflow_advance."""
    task_id: int
    mission_id: int
    completed_task_id: int
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CompleteWithReusedAnswer:
    """Complete derived from existing clarification_history (no re-ask)."""
    task_id: int
    result: str
    raw: dict = field(default_factory=dict)


Action = Union[
    Complete, SpawnSubtasks, RequestClarification, RequestReview,
    Exhausted, Failed, MissionAdvance, CompleteWithReusedAnswer,
]


def route_result(task: dict, agent_result: dict | None) -> list[Action]:
    """Map (task, agent_result) -> actions orchestrator must execute."""
    task_id = task["id"]

    if agent_result is None:
        return [Failed(task_id=task_id, error="no_result_returned", raw={})]

    raw = agent_result or {}
    status = agent_result.get("status")

    if status == "completed":
        return [Complete(
            task_id=task_id,
            result=agent_result.get("result", ""),
            iterations=agent_result.get("iterations", 0),
            metadata=agent_result.get("metadata", {}),
            raw=raw,
        )]

    if status == "needs_subtasks":
        subtasks = agent_result.get("subtasks", [])
        return [SpawnSubtasks(parent_task_id=task_id, subtasks=subtasks, raw=raw)]

    if status == "needs_clarification":
        return [RequestClarification(
            task_id=task_id,
            question=agent_result.get("question", ""),
            chat_id=task.get("chat_id"),
            raw=raw,
        )]

    if status == "needs_review":
        return [RequestReview(
            task_id=task_id,
            summary=agent_result.get("summary", ""),
            raw=raw,
        )]

    if status == "exhausted":
        return [Exhausted(
            task_id=task_id,
            error=agent_result.get("error", "max_iterations_reached"),
            raw=raw,
        )]

    return [Failed(
        task_id=task_id,
        error=agent_result.get("error", f"unknown_status:{status}"),
        raw=raw,
    )]
