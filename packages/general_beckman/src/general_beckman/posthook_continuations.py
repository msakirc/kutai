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


# ──────────────────────────────────────────────────────────────────────────
# Module-level helpers (patch points + production wiring).
# Tests patch pc._apply_posthook_verdict and pc._enqueue_grade_child, so these
# MUST be plain module-level coroutines/functions (not inlined imports).
# ──────────────────────────────────────────────────────────────────────────
async def _apply_posthook_verdict(child_task: dict, verdict) -> None:
    """Re-enter the EXISTING apply-layer verdict applier (general_beckman.apply)."""
    from general_beckman.apply import _apply_posthook_verdict as _impl
    await _impl(child_task, verdict)


def _parse_ctx(source: dict) -> dict:
    """Decode a task's context column (str JSON | dict | None) to a dict."""
    import json as _json
    ctx = source.get("context") or "{}"
    if isinstance(ctx, str):
        try:
            return _json.loads(ctx)
        except (ValueError, TypeError):
            return {}
    return ctx if isinstance(ctx, dict) else {}


def _grade_raw_dict(verdict) -> dict:
    """Normalize a GradeResult dataclass -> dict matching the deleted
    GraderAgent posthook_verdict['raw'] shape (see src/agents/grader.py
    ~108-141). _apply_posthook_verdict -> _grader_verdict_text reads these
    keys (insight/strategy/situation + the failed-axis booleans), so the
    verdict.raw MUST be a dict carrying them, never a bare string."""
    return {
        "passed": bool(verdict.passed),
        "relevant": getattr(verdict, "relevant", None),
        "complete": getattr(verdict, "complete", None),
        "well_formed": getattr(verdict, "well_formed", None),
        "coherent": getattr(verdict, "coherent", None),
        "situation": getattr(verdict, "situation", None),
        "strategy": getattr(verdict, "strategy", None),
        "tools": getattr(verdict, "tools", None),
        "preference": getattr(verdict, "preference", None),
        "insight": getattr(verdict, "insight", None),
        "raw": getattr(verdict, "raw", ""),
    }


def _make_grade_verdict(source_task_id, passed: bool, raw: dict):
    """Build the PostHookVerdict (kind='grade') applied to the source."""
    from general_beckman.result_router import PostHookVerdict
    return PostHookVerdict(source_task_id=source_task_id, kind="grade",
                           passed=passed, raw=raw)


def _result_model(result: dict) -> str:
    """Pull the reviewer child's chosen model from the child RESULT.

    Normal terminal shape: result['result']['model'] (dispatcher records it).
    Restart-reconcile shape: top-level result['model']. Empty string if absent.
    """
    inner = (result or {}).get("result")
    if isinstance(inner, dict) and inner.get("model"):
        return inner["model"]
    return (result or {}).get("model", "") or ""


async def _enqueue_grade_child(source_task_id: int, *, exclusions: list, attempt: int,
                               mission_id=None) -> None:
    """Chain a follow-up grade reviewer child (attempt 1) via the durable
    continuation substrate, with the failed grader model excluded.

    Builds the grading reviewer spec (src.core.grading.build_grading_spec) for
    the source task and enqueues it with on_complete/on_error pointing back at
    these handlers. cont_state carries source_task_id / attempt / exclusions so
    the resume sees the same parent state. build_grading_spec may return a
    GradeResult instead of a spec (trivial / degenerate source) — in that case
    apply the auto-fail verdict directly without enqueueing.
    """
    import general_beckman
    from src.core.grading import build_grading_spec, GradeResult
    from src.infra.db import get_task

    source = await get_task(source_task_id)
    if source is None:
        logger.warning("grade chain: source missing", source_id=source_task_id)
        return

    built = build_grading_spec(source, list(exclusions))
    if isinstance(built, GradeResult):
        # Short-circuit auto-fail (trivial/empty/degenerate) — apply directly.
        await _apply_posthook_verdict(
            {"id": source_task_id},
            _make_grade_verdict(source_task_id, bool(built.passed),
                                _grade_raw_dict(built)),
        )
        return

    await general_beckman.enqueue(
        built,
        parent_id=source_task_id,
        on_complete="posthook.grade.resume",
        on_error="posthook.grade.resume_err",
        cont_state={
            "source_task_id": source_task_id,
            "attempt": attempt,
            "exclusions": list(exclusions),
            "mission_id": mission_id if mission_id is not None
            else source.get("mission_id"),
        },
    )


async def _grade_resume(child_task_id: int, result: dict, state: dict) -> None:
    """Resume after a grade reviewer child completed.

    Parse the child output. On success → apply the grade verdict. On parse-fail
    of attempt 0 → chain a 2nd reviewer child (failed model excluded, attempt=1).
    On parse-fail of attempt 1 → auto-fail (grader_incapable).
    """
    from src.core.grading import parse_grade_response

    source_task_id = state.get("source_task_id")
    attempt = int(state.get("attempt", 0))
    exclusions = list(state.get("exclusions") or [])
    raw_text = _extract_content(result)

    try:
        verdict = parse_grade_response(raw_text)
        verdict.raw = raw_text
        await _apply_posthook_verdict(
            {"id": child_task_id},
            _make_grade_verdict(source_task_id, verdict.passed,
                                _grade_raw_dict(verdict)),
        )
        return
    except ValueError:
        if attempt == 0:
            model = _result_model(result)
            if model and model not in exclusions:
                exclusions.append(model)
            logger.info(
                "grade parse-fail attempt 0 — chaining 2nd reviewer child",
                source_id=source_task_id, excluded=model,
            )
            await _enqueue_grade_child(
                source_task_id, exclusions=exclusions, attempt=1,
                mission_id=state.get("mission_id"),
            )
            return
        msg = (f"auto-fail: grader_incapable after 2 attempts: "
               f"{raw_text[:300]}")
        logger.warning("grade parse-fail attempt 1 — auto-failing source",
                       source_id=source_task_id)
        await _apply_posthook_verdict(
            {"id": child_task_id},
            _make_grade_verdict(source_task_id, False,
                                {"passed": False, "raw": msg}),
        )


async def _grade_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    """On_error: the grade reviewer child terminally failed (no candidates /
    infra). Auto-fail the source's grade rather than leaving it parked."""
    source_task_id = state.get("source_task_id")
    err = (result or {}).get("error", "unknown")
    msg = f"auto-fail: grader call failed ({err})"
    logger.warning("grade child failed terminally — auto-failing source",
                   source_id=source_task_id, error=str(err)[:200])
    await _apply_posthook_verdict(
        {"id": child_task_id},
        _make_grade_verdict(source_task_id, False,
                            {"passed": False, "raw": msg}),
    )


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
