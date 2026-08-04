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
async def test_registry_validator_tags_even_with_md_produces(monkeypatch):
    # verify_adr_register is a registry-listed FULL-ARTIFACT completeness proof —
    # register.md is a mechanical index with no depth axis. It authors .md but is
    # override-eligible via the registry seam (prose *_shape checks never are).
    enq, probe = await _run(
        monkeypatch,
        {"artifact_schema": {"adrs": {"type": "array"}},
         "checks": [{"kind": "verify_adr_register",
                     "payload": {"action": "verify_adr_register",
                                 "path": ".adr/register.md"}}],
         "produces": ["mission_9/.adr/register.md"]},
    )
    enq.assert_awaited_once()
    probe.assert_awaited_once()
    assert _cont_state(enq)["shape_verify_passed"] is True


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
async def test_narrow_check_structured_gate_pass_tags(monkeypatch):
    # A NARROW check (verify_contains_product_name — one substring) is not a
    # completeness authority, so it is not probed. But the artifact is a pure
    # structured .json whose deterministic schema gate PASSED — that gate IS the
    # structural completeness proof (residual #2). No authoritative check ran, so
    # the schema-gate pass tags the grade; a COMPLETE-only FAIL is then a confab.
    enq, probe = await _run(
        monkeypatch,
        {"artifact_schema": _OBJ_SCHEMA,
         "checks": [{"kind": "verify_contains_product_name",
                     "payload": {"action": "verify_contains_product_name"}}],
         "produces": _JSON_PRODUCES},
    )
    enq.assert_awaited_once()
    probe.assert_not_awaited()  # narrow check never probed
    assert _cont_state(enq)["shape_verify_passed"] is True


@pytest.mark.asyncio
async def test_no_shape_check_structured_gate_pass_tags(monkeypatch):
    # Residual #2: a structured-only SINGLE-FILE .json step (returned value IS
    # the whole artifact) carrying NO verify_*_shape check. The deterministic
    # schema gate that already passed (incl. via evidence backfill) is the
    # structural completeness proof the absent verifier would provide, so the
    # grade is tagged — killing the confab loop on a schema-valid, on-disk .json
    # artifact. Narrow: structured-only (.md prose AND directory-produces steps
    # like 7.4a/b/d are excluded — their return JSON is not the whole artifact).
    enq, probe = await _run(
        monkeypatch,
        {"artifact_schema": _OBJ_SCHEMA, "checks": [], "produces": _JSON_PRODUCES},
    )
    enq.assert_awaited_once()
    probe.assert_not_awaited()  # no shape check to probe
    assert _cont_state(enq)["shape_verify_passed"] is True


@pytest.mark.asyncio
async def test_no_check_md_produces_gate_pass_not_tagged(monkeypatch):
    # A schema step that AUTHORS a .md doc (object schema over frontmatter) with
    # no shape check is NOT structured-only — prose "adequate depth" is a real
    # axis the schema gate cannot see, so the residual-#2 promotion must NOT fire
    # and the grade stays fully authoritative.
    enq, probe = await _run(
        monkeypatch,
        {"artifact_schema": {"user_flow": {"type": "object"}},
         "checks": [], "produces": ["mission_9/.flow/user_flow.md"]},
    )
    enq.assert_awaited_once()
    probe.assert_not_awaited()
    assert _cont_state(enq)["shape_verify_passed"] is False


@pytest.mark.asyncio
async def test_directory_produces_structured_not_tagged(monkeypatch):
    # A DIRECTORY-produces coder step (m90 7.4a/b/d: produces mission_/backend/)
    # is NOT single-file structured — the returned JSON is not the whole on-disk
    # tree, so a passing schema gate is not a full completeness proof and the
    # step must NOT be tagged. Residual #2 excludes it (_write_tools_redundant is
    # False for any directory produces), keeping the promotion narrow.
    enq, probe = await _run(
        monkeypatch,
        {"artifact_schema": _OBJ_SCHEMA, "checks": [],
         "produces": ["mission_9/backend/"]},
    )
    enq.assert_awaited_once()
    probe.assert_not_awaited()
    assert _cont_state(enq)["shape_verify_passed"] is False


@pytest.mark.asyncio
async def test_no_check_schema_gate_fail_never_tags(monkeypatch):
    # Narrowness floor: a structured-only step whose artifact FAILS the schema
    # gate (missing a required field) short-circuits to a FAIL verdict BEFORE the
    # grade child is enqueued — nothing is ever tagged, so residual #2 can never
    # promote a schema-invalid artifact.
    import general_beckman.apply as apply_mod
    monkeypatch.setattr(apply_mod, "_apply_posthook_verdict", AsyncMock())
    monkeypatch.setattr("mr_roboto.run", AsyncMock(return_value=_FakeAction("completed")))
    bad = {"id": 5, "mission_id": 9, "result": '{"mission_id": 9}',  # no _schema_version
           "title": "design_tokens", "description": "generate design tokens"}
    with patch.object(apply_mod, "enqueue", AsyncMock(return_value=1)) as enq:
        ret = await apply_mod._enqueue_posthook_llm_child(
            "grade", bad,
            {"artifact_schema": _OBJ_SCHEMA, "checks": [], "produces": _JSON_PRODUCES})
    enq.assert_not_awaited()  # schema-FAIL short-circuit → no grade child, no tag
    assert ret is False
