"""A passing shape verifier proves COMPLETENESS only for STRUCTURED artifacts;
prose keeps the full grade.

Mission-90 task 567449 [5.0a] design_tokens_generation: the model converged on a
shape-VALID design_tokens.json (verify_design_tokens_shape → ok), but the LLM
grader emitted `COMPLETE: NO / VERDICT: FAIL`. The producer re-emitted the same
correct artifact byte-identically and the degenerate-repeat detector DLQ'd it —
it was killed *because* it correctly converged.

Correct framing (grading.yaml): the grader's COMPLETE axis is SEMANTIC ADEQUACY
("adequate depth, no stubs or hand-waving; NOT field presence"). A shape verifier
proves STRUCTURE, which ≈ substantive completeness ONLY when the returned
structured value IS the whole artifact — a pure .json config/decision
(design_tokens, ADR, taste_emphasis, surfaces). For a FREE-FORM authored doc (any
.md produces — charter, reverse_pitch, user_flow, premortem, register, …)
"adequate depth" is a real axis the verifier cannot see, so the LLM grade stays
fully authoritative there.

So the override is gated on the codebase's authoritative structured-artifact
predicate ``coulson._write_tools_redundant`` (structured-only schema AND no .md
produces). Structured → the grade spawns with cont_state tagged
``shape_verify_passed=True`` and the resume handler
(``test_grade_advisory_complete.py``) overrides a completeness-only FAIL to PASS
while RELEVANT:NO / COHERENT:NO stays terminal. Prose / .md-authored → NOT tagged,
the verifier is not even probed, and the grade binds all axes.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


_OBJ_SCHEMA = {"design_tokens": {"type": "object", "required_fields": ["_schema_version"]}}
_MD_SCHEMA = {"product_charter": {"type": "markdown"}}
_VALID_RESULT = '{"_schema_version": "1", "mission_id": 9}'
_SHAPE_CHECK = [{
    "kind": "verify_design_tokens_shape",
    "payload": {"action": "verify_design_tokens_shape",
                "path": ".style/design_tokens.json"},
}]
_JSON_PRODUCES = ["mission_9/.style/design_tokens.json"]
_MD_PRODUCES = ["mission_9/.charter/product_charter.md"]


class _FakeAction:
    def __init__(self, status: str):
        self.status = status


def _source():
    return {"id": 5, "mission_id": 9, "result": _VALID_RESULT,
            "title": "design_tokens", "description": "generate design tokens"}


def _cont_state(enq):
    """Pull the cont_state dict the grade child was enqueued with."""
    assert enq.await_args is not None, "grade child was never enqueued"
    return enq.await_args.kwargs["cont_state"]


async def _run(monkeypatch, source_ctx, verify_status="completed"):
    import general_beckman.apply as apply_mod
    monkeypatch.setattr(apply_mod, "_apply_posthook_verdict", AsyncMock())
    probe = AsyncMock(return_value=_FakeAction(verify_status))
    monkeypatch.setattr("mr_roboto.run", probe)
    with patch.object(apply_mod, "enqueue", AsyncMock(return_value=1)) as enq:
        await apply_mod._enqueue_posthook_llm_child("grade", _source(), source_ctx)
    return enq, probe


@pytest.mark.asyncio
async def test_structured_artifact_spawns_tagged(monkeypatch):
    # Pure .json structured artifact: shape ≈ substantive completeness → grade
    # RUNS (RELEVANT/COHERENT kept) tagged so a completeness-only FAIL is later
    # overridden.
    enq, probe = await _run(
        monkeypatch,
        {"artifact_schema": _OBJ_SCHEMA, "checks": _SHAPE_CHECK,
         "produces": _JSON_PRODUCES},
    )
    enq.assert_awaited_once()
    probe.assert_awaited_once()  # verifier probed
    assert _cont_state(enq)["shape_verify_passed"] is True


@pytest.mark.asyncio
async def test_markdown_artifact_never_tagged(monkeypatch):
    # PROSE (.md, markdown schema): COMPLETE is a real adequacy axis the verifier
    # cannot prove → grade stays fully authoritative, verifier not even probed.
    enq, probe = await _run(
        monkeypatch,
        {"artifact_schema": _MD_SCHEMA,
         "checks": [{"kind": "verify_charter_shape",
                     "payload": {"action": "verify_charter_shape"}}],
         "produces": _MD_PRODUCES},
    )
    enq.assert_awaited_once()
    probe.assert_not_awaited()  # prose → verifier never runs
    assert _cont_state(enq)["shape_verify_passed"] is False


@pytest.mark.asyncio
async def test_object_schema_but_md_produces_never_tagged(monkeypatch):
    # user_flow / premortem carry an OBJECT schema to validate markdown
    # frontmatter but AUTHOR a .md doc — the .md produces is the authoritative
    # free-form signal, so depth still matters and the grade stays authoritative.
    enq, probe = await _run(
        monkeypatch,
        {"artifact_schema": {"user_flow": {"type": "object"}},
         "checks": [{"kind": "verify_user_flow_shape",
                     "payload": {"action": "verify_user_flow_shape"}}],
         "produces": ["mission_9/.flow/user_flow.md"]},
    )
    enq.assert_awaited_once()
    probe.assert_not_awaited()
    assert _cont_state(enq)["shape_verify_passed"] is False


@pytest.mark.asyncio
async def test_structured_shape_fail_spawns_untagged(monkeypatch):
    # A real earlier-attempt defect (shape FAIL) → grade fully authoritative,
    # continuation NOT tagged, producer re-pends on a FAIL as before.
    enq, probe = await _run(
        monkeypatch,
        {"artifact_schema": _OBJ_SCHEMA, "checks": _SHAPE_CHECK,
         "produces": _JSON_PRODUCES},
        verify_status="failed",
    )
    enq.assert_awaited_once()
    probe.assert_awaited_once()
    assert _cont_state(enq)["shape_verify_passed"] is False


@pytest.mark.asyncio
async def test_narrow_check_leaves_grade_untagged(monkeypatch):
    # A NARROW check (verify_contains_product_name — one substring) is not a
    # completeness authority: even on a structured artifact it is not probed, so
    # the grade stays fully authoritative.
    enq, probe = await _run(
        monkeypatch,
        {"artifact_schema": _OBJ_SCHEMA,
         "checks": [{"kind": "verify_contains_product_name",
                     "payload": {"action": "verify_contains_product_name"}}],
         "produces": _JSON_PRODUCES},
    )
    enq.assert_awaited_once()
    probe.assert_not_awaited()
    assert _cont_state(enq)["shape_verify_passed"] is False


@pytest.mark.asyncio
async def test_no_shape_check_leaves_grade_untagged(monkeypatch):
    # A structured step without any authoritative check → grade spawns untagged
    # and the verifier is never invoked.
    enq, probe = await _run(
        monkeypatch,
        {"artifact_schema": _OBJ_SCHEMA, "checks": [], "produces": _JSON_PRODUCES},
    )
    enq.assert_awaited_once()
    probe.assert_not_awaited()
    assert _cont_state(enq)["shape_verify_passed"] is False
