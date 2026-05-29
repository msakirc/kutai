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

    SP3 single-spawn path: delegates to the canonical apply-layer helper
    ``_enqueue_posthook_llm_child`` rather than duplicating the build+enqueue.
    The helper short-circuits to an auto-fail verdict when the source is
    trivial/degenerate (build_grading_spec returns a GradeResult).
    """
    from general_beckman.apply import _enqueue_posthook_llm_child
    from src.infra.db import get_task

    source = await get_task(source_task_id)
    if source is None:
        logger.warning("grade chain: source missing", source_id=source_task_id)
        return

    await _enqueue_posthook_llm_child(
        "grade", source, _parse_ctx(source),
        exclusions=list(exclusions), attempt=attempt,
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


def _make_cr_verdict(source_task_id, passed: bool, raw: dict):
    """Build a PostHookVerdict for kind='code_review'."""
    from general_beckman.result_router import PostHookVerdict
    return PostHookVerdict(source_task_id=source_task_id, kind="code_review",
                           passed=passed, raw=raw)


async def _code_review_resume(child_task_id: int, result: dict, state: dict) -> None:
    """Resume after a code_review child completed (single-shot — no chaining).

    Parses the child output with parse_code_review_response, builds a
    PostHookVerdict whose raw dict mirrors the original CodeReviewerAgent
    shape (keys: passed / issues / raw) so _apply_code_review_verdict can
    read raw.get("issues") for retry feedback, then re-enters
    _apply_posthook_verdict.
    """
    from src.core.code_review import parse_code_review_response

    source_task_id = state.get("source_task_id")
    raw_text = _extract_content(result)
    cr = parse_code_review_response(raw_text)
    await _apply_posthook_verdict(
        {"id": child_task_id},
        _make_cr_verdict(
            source_task_id,
            cr.passed,
            {"passed": cr.passed, "issues": list(cr.issues), "raw": cr.raw},
        ),
    )


async def _code_review_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    """On_error: the code_review child terminally failed. Auto-fail the source."""
    source_task_id = state.get("source_task_id")
    err = (result or {}).get("error", "unknown")
    logger.warning("code_review child failed terminally — auto-failing source",
                   source_id=source_task_id, error=str(err)[:200])
    await _apply_posthook_verdict(
        {"id": child_task_id},
        _make_cr_verdict(
            source_task_id,
            False,
            {"passed": False, "issues": [],
             "raw": f"auto-fail: code-review call failed ({err})"},
        ),
    )


def _make_summary_verdict(source_task_id, artifact_name: str, passed: bool, summary: str):
    """Build the PostHookVerdict (kind='summary:<artifact_name>') applied to the source."""
    from general_beckman.result_router import PostHookVerdict
    return PostHookVerdict(
        source_task_id=source_task_id, kind=f"summary:{artifact_name}",
        passed=passed, raw={"summary": summary, "artifact_name": artifact_name},
    )


async def _summary_resume(child_task_id: int, result: dict, state: dict) -> None:
    """Resume after an artifact summarizer child completed.

    Parse the child output. passed = len(summary) >= 50 AND not degenerate
    (mirrors the deleted ArtifactSummarizerAgent output check). On fail the
    apply layer drains the pending_posthooks entry but the structural summary
    already stored by post_execute is kept.
    """
    source_task_id = state.get("source_task_id")
    artifact_name = state.get("artifact_name") or ""
    summary = _extract_content(result).strip()
    passed = bool(summary) and len(summary) >= 50
    if passed:
        try:
            from dogru_mu_samet import assess as cq_assess
            if cq_assess(summary).is_degenerate:
                passed = False
        except Exception:  # noqa: BLE001
            pass
    await _apply_posthook_verdict(
        {"id": child_task_id},
        _make_summary_verdict(source_task_id, artifact_name, passed, summary if passed else ""),
    )


async def _summary_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    """On_error: the summarizer child terminally failed. Drain the pending kind with passed=False.

    The structural summary was already stored by post_execute; the apply layer
    just drains the pending_posthooks entry so the source can complete.
    """
    source_task_id = state.get("source_task_id")
    artifact_name = state.get("artifact_name") or ""
    await _apply_posthook_verdict(
        {"id": child_task_id},
        _make_summary_verdict(source_task_id, artifact_name, False, ""),
    )


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
