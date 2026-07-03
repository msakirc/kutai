"""CPS SP3 - post-hook continuation handlers (grading / code_review / summarize).

Shape B: the post-hook enqueues the raw_dispatch reviewer/summarizer child
directly with on_complete/on_error; these handlers parse the child output,
build a PostHookVerdict, and re-enter the EXISTING _apply_posthook_verdict.
The grader/code_reviewer/artifact_summarizer agent classes are deleted (SP3).
Handler bodies are filled in T5 (grade), T6 (code_review), T7 (summary).
"""
from __future__ import annotations

from yazbunu import get_logger

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


async def _advance_posthook_chain(source_task_id) -> None:
    """Spawn the NEXT kind in the source's ordered post-hook cursor.

    SP3b Task 6 — after a rewriter (constrained_emit / self_reflect) resume has
    applied its verdict (rewrite or no-op), advance the ``_posthook_queue``
    cursor so the next kind (self_reflect → grade) is spawned. Re-enters the
    apply-layer cursor walk, which skip-drains any no-op rewriter and lands on
    the terminal grade gate. Module-level so tests can patch it.
    """
    if source_task_id is None:
        return
    from general_beckman.apply import _advance_posthook_chain as _impl
    await _impl(int(source_task_id))


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


def _only_completeness_failed(verdict) -> bool:
    """True when a grade FAIL is driven SOLELY by the completeness axis —
    COMPLETE explicitly NO while RELEVANT / COHERENT / WELL_FORMED are NOT
    explicitly NO.

    Used only when a deterministic shape verifier already proved completeness
    (cont_state shape_verify_passed=True). The grader confabulates COMPLETE:NO on
    structured artifacts — its own prompt forbids judging presence — which DLQ'd
    task 567449 [5.0a]. Overriding that confab is safe because completeness is a
    proven fact; but RELEVANT:NO (wrong product / off-topic) and COHERENT:NO are
    exactly what the shape verifier CANNOT see, so those stay terminal. A
    WELL_FORMED:NO — the shape verifier and grader disagreeing on structure — is a
    rare contradiction that must NOT silently auto-pass, so it stays terminal too.
    """
    return (
        getattr(verdict, "complete", None) is False
        and getattr(verdict, "relevant", None) is not False
        and getattr(verdict, "coherent", None) is not False
        and getattr(verdict, "well_formed", None) is not False
    )


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
    from dabidabi import get_task

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

    shape_verify_passed = bool(state.get("shape_verify_passed"))

    try:
        verdict = parse_grade_response(raw_text)
        verdict.raw = raw_text
        # Advisory-COMPLETE override: a deterministic shape verifier already
        # proved completeness at spawn time, so a grade FAIL whose only failing
        # axis is COMPLETE is a confab (task 567449) — flip it to PASS. A
        # RELEVANT:NO / COHERENT:NO FAIL (topicality the verifier can't see)
        # stays terminal.
        if (not verdict.passed and shape_verify_passed
                and _only_completeness_failed(verdict)):
            logger.info(
                "grade COMPLETE-only FAIL overridden to PASS — shape verifier "
                "already proved completeness; RELEVANT/COHERENT intact",
                source_id=source_task_id,
            )
            verdict.passed = True
            verdict.raw = ("advisory-COMPLETE override (shape verify authoritative "
                           "on completeness; RELEVANT/COHERENT intact): " + raw_text)
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
        # Grader incapable after 2 attempts. When the shape verifier already
        # proved completeness, fall back to auto-PASS rather than punishing a
        # shape-valid producer for a grader that can't emit a parseable verdict
        # (outage-safety parity with the old skip-the-grade fix). No parseable
        # relevance signal is available to keep terminal, so completeness (proven)
        # governs.
        if shape_verify_passed:
            logger.info(
                "grader incapable but shape verify passed — auto-PASS "
                "(completeness proven, LLM grade unavailable)",
                source_id=source_task_id,
            )
            await _apply_posthook_verdict(
                {"id": child_task_id},
                _make_grade_verdict(
                    source_task_id, True,
                    {"passed": True,
                     "raw": "shape verify authoritative; grader incapable — "
                            "completeness proven, LLM grade unavailable"}),
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
    infra). Auto-fail the source's grade rather than leaving it parked — UNLESS a
    deterministic shape verifier already proved completeness at spawn time, in
    which case auto-PASS (outage-safety parity with the old skip-the-grade fix: a
    shape-valid producer must never be punished for grader unavailability)."""
    source_task_id = state.get("source_task_id")
    err = (result or {}).get("error", "unknown")
    if state.get("shape_verify_passed"):
        logger.info(
            "grade child failed terminally but shape verify passed — auto-PASS",
            source_id=source_task_id, error=str(err)[:200],
        )
        await _apply_posthook_verdict(
            {"id": child_task_id},
            _make_grade_verdict(
                source_task_id, True,
                {"passed": True,
                 "raw": f"shape verify authoritative; grade child failed ({err}) "
                        "— completeness proven, LLM grade unavailable"}),
        )
        return
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


# ──────────────────────────────────────────────────────────────────────────
# SP3b — constrained_emit + self_reflect resume handlers (rewrite verdict).
#
# Unlike grade / code_review (which GATE) and summary (which stores), these
# two REWRITE the source's result in place via the Task 4 verdict path:
#   PostHookVerdict(action="rewrite", new_result=<str>).
# A child that produces unusable output (non-JSON emit / ok-or-degenerate
# reflect / terminal error) must NEVER rewrite — the source draft survives.
# ──────────────────────────────────────────────────────────────────────────


def _make_rewrite_verdict(source_task_id, kind: str, new_result: str):
    """Build a PostHookVerdict(action="rewrite") for the source task."""
    from general_beckman.result_router import PostHookVerdict
    return PostHookVerdict(
        source_task_id=source_task_id, kind=kind, passed=True,
        raw={}, action="rewrite", new_result=new_result,
    )


async def _constrained_emit_resume(child_task_id: int, result: dict, state: dict) -> None:
    """Resume after a constrained_emit child completed.

    The child re-emitted the draft as schema-conforming JSON. Rewrite the
    source result with the emitted JSON. On unusable output (empty / non-JSON)
    leave the draft — never corrupt the source.
    """
    import json as _json

    source_task_id = state.get("source_task_id")
    emitted = _extract_content(result).strip()
    if not emitted:
        logger.warning("constrained_emit produced empty output — keeping draft",
                       source_id=source_task_id)
        await _advance_posthook_chain(source_task_id)
        return
    # Cheap shape check: must parse as JSON. The schema-validation hook does
    # the deeper required-field check on the next pass.
    try:
        _json.loads(emitted)
    except (ValueError, TypeError):
        logger.warning("constrained_emit produced non-JSON output — keeping draft",
                       source_id=source_task_id)
        await _advance_posthook_chain(source_task_id)
        return
    await _apply_posthook_verdict(
        {"id": child_task_id},
        _make_rewrite_verdict(source_task_id, "constrained_emit", emitted),
    )
    # SP3b Task 6 — rewrite landed; advance the cursor to self_reflect / grade.
    await _advance_posthook_chain(source_task_id)


async def _constrained_emit_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    """On_error: the emit child failed terminally. Leave the draft untouched —
    constrained_emit is best-effort; a failed emit is never a source failure."""
    source_task_id = state.get("source_task_id")
    err = (result or {}).get("error", "unknown")
    logger.warning("constrained_emit child failed terminally — keeping draft",
                   source_id=source_task_id, error=str(err)[:200])
    # No verdict applied → source draft survives. The cursor must still advance
    # to the next post-hook so the source isn't stranded in 'ungraded'.
    await _advance_posthook_chain(source_task_id)


def _reflect_corrected_or_none(raw_text: str, source_task_id) -> str | None:
    """Return a usable corrected_result for a rewrite, or None for a no-op.

    Rewrite ONLY when verdict=="fix" AND corrected_result is non-empty AND
    non-degenerate (dogru_mu_samet). ok / unparseable / no-correction / degenerate
    → None (keep the draft). Warning severity must NEVER fail the source.
    """
    import json as _json

    try:
        parsed = _json.loads(raw_text)
    except (ValueError, TypeError):
        parsed = None
    if not isinstance(parsed, dict) or parsed.get("verdict") != "fix":
        return None  # ok / unparseable → keep the draft
    corrected = parsed.get("corrected_result")
    if not isinstance(corrected, str) or not corrected.strip():
        return None  # fix without a usable correction → keep the draft
    try:
        from dogru_mu_samet import assess as cq_assess
        if cq_assess(corrected).is_degenerate:
            logger.warning(
                "self_reflect corrected_result degenerate — keeping draft",
                source_id=source_task_id,
            )
            return None
    except Exception:  # noqa: BLE001
        pass
    return corrected


async def _self_reflect_resume(child_task_id: int, result: dict, state: dict) -> None:
    """Resume after a self_reflect child completed.

    Rewrite the source on a usable "fix"; otherwise no-op. EITHER way the Task 6
    cursor must advance to the next post-hook (grade) so the source isn't
    stranded in 'ungraded'.
    """
    source_task_id = state.get("source_task_id")
    raw_text = _extract_content(result).strip()
    corrected = _reflect_corrected_or_none(raw_text, source_task_id)
    if corrected is not None:
        await _apply_posthook_verdict(
            {"id": child_task_id},
            _make_rewrite_verdict(source_task_id, "self_reflect", corrected),
        )
    # SP3b Task 6 — advance the cursor whether or not the rewrite landed.
    await _advance_posthook_chain(source_task_id)


async def _self_reflect_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    """On_error: the reflect child failed terminally. Leave the draft untouched —
    self_reflect is warning severity; a failed reflection is never a source fail."""
    source_task_id = state.get("source_task_id")
    err = (result or {}).get("error", "unknown")
    logger.warning("self_reflect child failed terminally — keeping draft",
                   source_id=source_task_id, error=str(err)[:200])
    # No verdict applied → source draft survives. Advance the cursor so the
    # source isn't stranded in 'ungraded'.
    await _advance_posthook_chain(source_task_id)


# ──────────────────────────────────────────────────────────────────────────
# SP6 — critic_gate posthook resume (admitted LLM child, fail-closed).
# critic_gate ∈ _Z1_MECHANICAL_KINDS → _apply_posthook_verdict_locked routes a
# passed=False verdict to single-shot DLQ. Handler only builds the verdict.
# ──────────────────────────────────────────────────────────────────────────


async def _persist_critic_log(state: dict, verdict: str, reasons: list) -> None:
    from mr_roboto.critic_gate import _persist
    await _persist(
        state.get("mission_id"),
        str(state.get("action_name") or "unknown"),
        verdict,
        list(reasons or []),
        str(state.get("payload_hash") or ""),
    )


def _make_critic_verdict(source_task_id, passed: bool, reasons: list):
    from general_beckman.result_router import PostHookVerdict
    reasons = list(reasons or [])
    raw = {"reasons": reasons}
    # FIX 2 — the blocker DLQ writer (_apply_z1_mechanical_verdict) builds the
    # founder-visible error_detail from raw.get("error"), never raw["reasons"].
    # Surface the veto reason under "error" so it survives into the DLQ row.
    if not passed and reasons:
        raw["error"] = "critic veto: " + "; ".join(reasons)
    return PostHookVerdict(
        source_task_id=source_task_id, kind="critic_gate",
        passed=passed, raw=raw,
    )


async def _critic_resume(child_task_id: int, result: dict, state: dict) -> None:
    # FIX 1 — fail-CLOSED on garbage. parse_verdict_strict returns a VETO for
    # any output that is not an explicit {"verdict": "pass"|"veto"} object, so a
    # broken/garbage critic BLOCKS the irreversible action (fail-closed; there
    # is no default-pass path — a garbage or missing verdict is always a veto).
    from mr_roboto.critic_gate import parse_verdict_strict
    source_task_id = state.get("source_task_id")
    parsed = parse_verdict_strict(_extract_content(result))
    passed = parsed["verdict"] != "veto"
    if not passed:
        logger.warning("critic veto — failing source",
                       source_id=source_task_id, reasons=parsed.get("reasons"))
    await _persist_critic_log(state, parsed["verdict"], parsed.get("reasons") or [])
    await _apply_posthook_verdict(
        {"id": child_task_id},
        _make_critic_verdict(source_task_id, passed, parsed.get("reasons") or []),
    )


async def _critic_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    source_task_id = state.get("source_task_id")
    err = (result or {}).get("error", "unknown")
    reasons = [f"critic verdict unavailable (producer error: {str(err)[:120]}) — fail-closed"]
    logger.warning("critic child failed terminally — failing source (fail-closed)",
                   source_id=source_task_id, reasons=reasons)
    await _persist_critic_log(state, "veto", reasons)
    await _apply_posthook_verdict(
        {"id": child_task_id},
        _make_critic_verdict(source_task_id, False, reasons),
    )


# ──────────────────────────────────────────────────────────────────────────
# Verdict verification (2026-06-26) — Tier-2 adversarial refuter resume.
# The reviewer (source) is parked ungraded on this continuation while the
# admitted refuter child runs. On resume we drop the candidate findings the
# refuter could not support (fail-closed against a refuter that confabulates its
# OWN quote) and hand the survivors back to the apply layer to route/complete.
# ──────────────────────────────────────────────────────────────────────────


async def _finish_review_after_refuter(source_task_id, reviewer_id, mission_id,
                                       final_issues) -> None:
    """Re-enter the apply-layer finisher. Module-level so tests can patch it."""
    from general_beckman.apply import _finish_review_after_refuter as _impl
    await _impl(
        source_task_id=source_task_id, reviewer_id=reviewer_id,
        mission_id=mission_id, final_issues=final_issues,
    )


def _filter_refuted(kept_issues: list, candidates: list, parsed) -> list:
    """Build the surviving issue list. ``parsed`` is the refuter verdict map, or
    None on a whole-output parse failure (then keep ALL candidates — a refuter
    outage must not silently disable the halt). Drops are matched back to
    kept_issues by (target_artifact, problem)."""
    from mr_roboto.verdict_refuter import refuter_keep

    dropped_keys: set = set()
    if parsed is not None:
        # NB: re-grounding the refuter's own quote needs the artifact content;
        # the caller resolves it per candidate and passes it via `_contents`.
        for i, cand in enumerate(candidates):
            content = (cand or {}).get("_content")
            if not refuter_keep(parsed.get(i), content):
                dropped_keys.add(((cand or {}).get("target_artifact"),
                                  (cand or {}).get("problem")))
    return [
        iss for iss in (kept_issues or [])
        if (iss.get("target_artifact"), iss.get("problem")) not in dropped_keys
    ]


async def _verdict_verify_resume(child_task_id: int, result: dict, state: dict) -> None:
    from mr_roboto.verdict_refuter import parse_refuter_output
    from mr_roboto.verify_review_verdict import _resolve_artifact_content

    candidates = list(state.get("candidates") or [])
    kept_issues = list(state.get("kept_issues") or [])
    mission_id = state.get("mission_id")
    raw_text = _extract_content(result)
    parsed = parse_refuter_output(raw_text, len(candidates))

    # Re-resolve each candidate's artifact so refuter_keep can re-ground the
    # refuter's OWN supporting quote (fail-closed against a confabulating refuter).
    if parsed is not None:
        for cand in candidates:
            try:
                cand["_content"] = await _resolve_artifact_content(
                    mission_id, cand.get("target_artifact"))
            except Exception:  # noqa: BLE001
                cand["_content"] = None

    final = _filter_refuted(kept_issues, candidates, parsed)
    logger.info(
        "verdict-verify resume: refuter filtered findings",
        source_id=state.get("source_task_id"),
        kept=len(kept_issues), survivors=len(final),
        parsed=(parsed is not None),
    )
    await _finish_review_after_refuter(
        state.get("source_task_id"), state.get("reviewer_id"),
        mission_id, final,
    )


async def _verdict_verify_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    """The refuter child failed terminally (no candidates / infra). Keep ALL
    candidates and route normally — an outage must NOT silently disable the
    safety halt."""
    err = (result or {}).get("error", "unknown")
    logger.warning(
        "verdict-verify refuter failed terminally — keeping all findings (route)",
        source_id=state.get("source_task_id"), error=str(err)[:200],
    )
    await _finish_review_after_refuter(
        state.get("source_task_id"), state.get("reviewer_id"),
        state.get("mission_id"), list(state.get("kept_issues") or []),
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
        # SP3b — emit/reflect rewrite resumes.
        register_resume("posthook.constrained_emit.resume", _constrained_emit_resume)
        register_resume("posthook.constrained_emit.resume_err", _constrained_emit_resume_err)
        register_resume("posthook.self_reflect.resume", _self_reflect_resume)
        register_resume("posthook.self_reflect.resume_err", _self_reflect_resume_err)
        # SP6 — critic_gate admitted LLM child (fail-closed).
        register_resume("posthook.critic.resume", _critic_resume)
        register_resume("posthook.critic.resume_err", _critic_resume_err)
        # Verdict verification (2026-06-26) — Tier-2 adversarial refuter.
        register_resume("posthook.verdict_verify.resume", _verdict_verify_resume)
        register_resume("posthook.verdict_verify.resume_err", _verdict_verify_resume_err)
    except Exception as exc:  # noqa: BLE001
        logger.debug("posthook continuation registration deferred", error=str(exc))


# Register at import so handlers are present for restart reconcile.
register_continuations()
