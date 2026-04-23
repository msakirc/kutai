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


@dataclass(frozen=True)
class RequestPostHook:
    """Spawn a post-hook task (grader or artifact_summarizer) for a source.

    `kind` is either "grade" or "summary:<artifact_name>" (one spawn per
    large output artifact after a grade pass).
    """
    source_task_id: int
    kind: str
    source_ctx: dict


@dataclass(frozen=True)
class PostHookVerdict:
    """Apply the result of a completed post-hook task back to the source."""
    source_task_id: int
    kind: str
    passed: bool
    raw: dict


Action = Union[
    Complete, SpawnSubtasks, RequestClarification, RequestReview,
    Exhausted, Failed, MissionAdvance, CompleteWithReusedAnswer,
    RequestPostHook, PostHookVerdict,
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
        # Agents signal the clarification text via one of several fields
        # depending on how they got here:
        #   "question"      — explicit clarify action at parse time
        #   "clarification" — post_execute_workflow_step (triggers_clarification
        #                     steps) + base.py needs_clarification override
        #   "result"        — some agents stuff it into result when returning
        #                     final_answer that classifies as needs_clarification
        # Before the Phase 2b extraction, the orchestrator's inline handler
        # tried all three. Reading only "question" here dropped real
        # questions into the empty string and DLQ'd the mechanical
        # clarify task that spawned downstream.
        question_text = (
            agent_result.get("question")
            or agent_result.get("clarification")
            or agent_result.get("result")
            or ""
        )
        if not isinstance(question_text, str):
            question_text = str(question_text)
        question_text = question_text.strip()
        # Heuristic quality gate — the triggers_clarification path in
        # post_execute_workflow_step already runs dogru_mu_samet; agents
        # that self-declare needs_clarification bypassed it. Reject
        # degenerate text (empty, low-entropy, repetitive) as a Failed
        # so the retry path kicks in instead of shipping garbage to the
        # user via mechanical clarify.
        if not question_text:
            return [Failed(
                task_id=task_id,
                error="needs_clarification with empty question/clarification",
                raw=raw,
            )]
        try:
            from dogru_mu_samet import assess as _cq_assess
            _cq = _cq_assess(question_text)
            if _cq.is_degenerate:
                return [Failed(
                    task_id=task_id,
                    error=f"clarification rejected: {_cq.summary}",
                    raw=raw,
                )]
        except Exception:
            pass
        return [RequestClarification(
            task_id=task_id,
            question=question_text,
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
