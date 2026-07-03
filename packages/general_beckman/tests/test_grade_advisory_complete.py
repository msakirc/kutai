"""Advisory-COMPLETE grade override — resume side.

Companion to ``test_grade_verify_authority.py`` (spawn side). When the artifact
is a pure STRUCTURED value (structured-only schema, no .md produces) whose
authoritative shape check passed, the grade continuation is tagged
``shape_verify_passed=True`` and the grade STILL runs. For those artifacts a
passing structural verifier ≈ substantive completeness, so the grader's COMPLETE
axis (SEMANTIC ADEQUACY per grading.yaml — depth / no stubs) carries no
information the verifier hasn't already proven. This file pins the resume-handler
contract:

  * a grade FAIL whose ONLY failing axis is COMPLETE is overridden to PASS
    (the confab that DLQ'd 567449 [5.0a] design_tokens as a degenerate repeat);
  * a FAIL with RELEVANT:NO or COHERENT:NO stays TERMINAL — topicality/coherence
    is exactly what the shape verifier CANNOT see, so the LLM grade remains
    authoritative there;
  * a bare / unexplained FAIL (no failing axis, or COMPLETE absent) stays
    TERMINAL — the override never fires without positive COMPLETE-only evidence;
  * without the tag, nothing is overridden (unchanged behavior);
  * a grader that can't produce a parseable verdict, or a grade child that dies
    on infra/no-candidates, falls back to auto-PASS ONLY when tagged —
    outage-safety parity with the old skip-the-grade fix (a shape-valid
    structured producer is never punished for grader unavailability).
"""
from __future__ import annotations

import pytest

import general_beckman.posthook_continuations as pc


def _result(grader_text: str) -> dict:
    return {"result": {"content": grader_text, "model": "qwen3.5-9b"}}


def _capture(monkeypatch):
    seen = []

    async def fake_apply(child, verdict):
        seen.append(verdict)

    monkeypatch.setattr(pc, "_apply_posthook_verdict", fake_apply)
    return seen


_COMPLETE_ONLY_FAIL = (
    "RELEVANT: YES\nCOMPLETE: NO\nWELL_FORMED: PASS\nCOHERENT: PASS\nVERDICT: FAIL"
)


# ── override fires (completeness-only) ───────────────────────────────────────

@pytest.mark.asyncio
async def test_complete_only_fail_overridden_when_shape_passed(monkeypatch):
    seen = _capture(monkeypatch)
    await pc._grade_resume(
        1, _result(_COMPLETE_ONLY_FAIL),
        {"source_task_id": 5, "attempt": 0, "shape_verify_passed": True},
    )
    assert seen and seen[0].passed is True


# ── override does NOT fire (semantic axis failed) ────────────────────────────

@pytest.mark.asyncio
async def test_relevant_fail_stays_terminal_when_shape_passed(monkeypatch):
    seen = _capture(monkeypatch)
    text = "RELEVANT: NO\nCOMPLETE: NO\nWELL_FORMED: PASS\nCOHERENT: PASS\nVERDICT: FAIL"
    await pc._grade_resume(
        1, _result(text),
        {"source_task_id": 5, "attempt": 0, "shape_verify_passed": True},
    )
    assert seen and seen[0].passed is False  # topicality is terminal


@pytest.mark.asyncio
async def test_coherent_fail_stays_terminal_when_shape_passed(monkeypatch):
    seen = _capture(monkeypatch)
    text = "RELEVANT: YES\nCOMPLETE: NO\nWELL_FORMED: PASS\nCOHERENT: NO\nVERDICT: FAIL"
    await pc._grade_resume(
        1, _result(text),
        {"source_task_id": 5, "attempt": 0, "shape_verify_passed": True},
    )
    assert seen and seen[0].passed is False


@pytest.mark.asyncio
async def test_bare_fail_no_axes_stays_terminal_when_shape_passed(monkeypatch):
    # Cardinal-sin guard: a FAIL with NO per-axis fields (COMPLETE absent) must
    # NOT auto-pass — the override requires positive COMPLETE-only evidence.
    seen = _capture(monkeypatch)
    await pc._grade_resume(
        1, _result("VERDICT: FAIL"),
        {"source_task_id": 5, "attempt": 0, "shape_verify_passed": True},
    )
    assert seen and seen[0].passed is False


@pytest.mark.asyncio
async def test_well_formed_fail_stays_terminal_when_shape_passed(monkeypatch):
    # Grader and shape verifier disagree on structure — a rare contradiction that
    # must NOT silently auto-pass; the producer re-pends.
    seen = _capture(monkeypatch)
    text = "RELEVANT: YES\nCOMPLETE: NO\nWELL_FORMED: FAIL\nCOHERENT: PASS\nVERDICT: FAIL"
    await pc._grade_resume(
        1, _result(text),
        {"source_task_id": 5, "attempt": 0, "shape_verify_passed": True},
    )
    assert seen and seen[0].passed is False


# ── override does NOT fire (no tag) ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_complete_only_fail_terminal_without_tag(monkeypatch):
    seen = _capture(monkeypatch)
    await pc._grade_resume(
        1, _result(_COMPLETE_ONLY_FAIL),
        {"source_task_id": 5, "attempt": 0},  # no shape_verify_passed
    )
    assert seen and seen[0].passed is False


@pytest.mark.asyncio
async def test_genuine_pass_unchanged_when_shape_passed(monkeypatch):
    seen = _capture(monkeypatch)
    text = "RELEVANT: YES\nCOMPLETE: YES\nWELL_FORMED: PASS\nCOHERENT: PASS\nVERDICT: PASS"
    await pc._grade_resume(
        1, _result(text),
        {"source_task_id": 5, "attempt": 0, "shape_verify_passed": True},
    )
    assert seen and seen[0].passed is True


# ── outage-safety fallback (grader unavailable / incapable) ──────────────────

@pytest.mark.asyncio
async def test_grade_child_terminal_fail_auto_passes_when_shape_passed(monkeypatch):
    seen = _capture(monkeypatch)
    await pc._grade_resume_err(
        1, {"error": "No model candidates available"},
        {"source_task_id": 5, "shape_verify_passed": True},
    )
    assert seen and seen[0].passed is True


@pytest.mark.asyncio
async def test_grade_child_terminal_fail_auto_fails_without_tag(monkeypatch):
    seen = _capture(monkeypatch)
    await pc._grade_resume_err(
        1, {"error": "No model candidates available"},
        {"source_task_id": 5},  # no tag
    )
    assert seen and seen[0].passed is False


@pytest.mark.asyncio
async def test_grader_incapable_auto_passes_when_shape_passed(monkeypatch):
    seen = _capture(monkeypatch)
    # Unparseable grader output on the FINAL attempt (attempt=1).
    await pc._grade_resume(
        1, _result("assorted prose with no structured verdict at all"),
        {"source_task_id": 5, "attempt": 1, "shape_verify_passed": True},
    )
    assert seen and seen[0].passed is True


@pytest.mark.asyncio
async def test_grader_incapable_auto_fails_without_tag(monkeypatch):
    seen = _capture(monkeypatch)
    await pc._grade_resume(
        1, _result("assorted prose with no structured verdict at all"),
        {"source_task_id": 5, "attempt": 1},  # no tag
    )
    assert seen and seen[0].passed is False
