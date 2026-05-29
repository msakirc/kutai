"""CPS SP3 - post-hook continuation handlers (grading / code_review / summarize).

Shape B: the post-hook enqueues the raw_dispatch reviewer/summarizer child
directly with on_complete/on_error; these handlers parse the child output,
build a PostHookVerdict, and re-enter the EXISTING _apply_posthook_verdict.
The grader/code_reviewer/artifact_summarizer agent classes are deleted (SP3).
Handler bodies are filled in T5 (grade), T6 (code_review), T7 (summary).
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("beckman.posthook_continuations")


def _extract_content(result: dict) -> str:
    """Dual-shape decode (matches src/app/interview.py:297-310).

    Normal terminal: result['result']['content']. Restart-reconcile:
    top-level result['content']. List blocks are joined.
    """
    result = result or {}
    inner = result.get("result")
    if isinstance(inner, dict):
        content = inner.get("content", "")
    elif inner is not None:
        content = inner
    else:
        content = result.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content or "")


# Handler bodies filled in T5/T6/T7.
async def _grade_resume(child_task_id: int, result: dict, state: dict) -> None:
    raise NotImplementedError  # Task 5


async def _grade_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    raise NotImplementedError  # Task 5


async def _code_review_resume(child_task_id: int, result: dict, state: dict) -> None:
    raise NotImplementedError  # Task 6


async def _code_review_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    raise NotImplementedError  # Task 6


async def _summary_resume(child_task_id: int, result: dict, state: dict) -> None:
    raise NotImplementedError  # Task 7


async def _summary_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    raise NotImplementedError  # Task 7


def register_continuations() -> None:
    """Register SP3 post-hook CPS handlers. Idempotent."""
    try:
        from general_beckman.continuations import register_resume
        register_resume("posthook.grade.resume", _grade_resume)
        register_resume("posthook.grade.resume_err", _grade_resume_err)
        register_resume("posthook.code_review.resume", _code_review_resume)
        register_resume("posthook.code_review.resume_err", _code_review_resume_err)
        register_resume("posthook.summary.resume", _summary_resume)
        register_resume("posthook.summary.resume_err", _summary_resume_err)
    except Exception as exc:  # noqa: BLE001
        logger.debug("posthook continuation registration deferred", error=str(exc))


# Register at import so handlers are present for restart reconcile.
register_continuations()
