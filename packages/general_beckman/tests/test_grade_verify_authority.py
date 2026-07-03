"""The deterministic shape verifier proves COMPLETENESS; the LLM grade keeps the
TOPICALITY axis.

Mission-90 task 567449 [5.0a] design_tokens_generation: the model converged on a
shape-VALID design_tokens.json (verify_design_tokens_shape → ok), but the
scope-blind LLM grader confabulated `COMPLETE: NO / VERDICT: FAIL` (its own
prompt says "DO NOT JUDGE field/section presence"). The producer re-emitted the
same correct artifact byte-identically, and the degenerate-repeat detector DLQ'd
it as "not converging" — it was killed *because* it correctly converged.

Original fix (2026-06-27) SKIPPED the grade entirely on a shape PASS — which
also dropped the LLM grade's RELEVANT/COHERENT axes (is this artifact about the
RIGHT product, does it hang together) on ~24 steps. That is a real topicality
hole the shape verifier cannot cover.

Advisory-COMPLETE refinement (this file): on a shape PASS the grade STILL RUNS,
but the continuation is tagged ``shape_verify_passed=True``. The resume handler
(``test_grade_advisory_complete.py``) then overrides a *completeness-only* grade
FAIL to PASS (killing the 567449 confab loop) while a RELEVANT:NO / COHERENT:NO
FAIL stays terminal (topicality preserved). On a shape FAIL the tag is False and
the grade is fully authoritative, so the producer re-pends as before.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


_SCHEMA = {"design_tokens": {"type": "object", "required_fields": ["_schema_version"]}}
_VALID_RESULT = '{"_schema_version": "1", "mission_id": 9}'
_CHECKS = [{
    "kind": "verify_design_tokens_shape",
    "payload": {"action": "verify_design_tokens_shape",
                "path": ".style/design_tokens.json"},
}]


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


@pytest.mark.asyncio
async def test_grade_spawns_tagged_when_shape_verifier_passes(monkeypatch):
    # Shape PASS no longer skips the grade — it RUNS (topicality axis kept) with
    # the continuation tagged so a completeness-only FAIL is overridden later.
    import general_beckman.apply as apply_mod

    monkeypatch.setattr(apply_mod, "_apply_posthook_verdict", AsyncMock())
    monkeypatch.setattr("mr_roboto.run",
                        AsyncMock(return_value=_FakeAction("completed")))

    source_ctx = {"artifact_schema": _SCHEMA, "checks": _CHECKS}
    with patch.object(apply_mod, "enqueue", AsyncMock(return_value=1)) as enq:
        await apply_mod._enqueue_posthook_llm_child("grade", _source(), source_ctx)

    enq.assert_awaited_once()  # LLM grade RUNS — RELEVANT/COHERENT preserved
    assert _cont_state(enq)["shape_verify_passed"] is True


@pytest.mark.asyncio
async def test_grade_spawns_untagged_when_shape_verifier_fails(monkeypatch):
    # A real earlier-attempt defect (shape FAIL) → grade is fully authoritative,
    # continuation NOT tagged, producer re-pends on a FAIL as before.
    import general_beckman.apply as apply_mod

    monkeypatch.setattr(apply_mod, "_apply_posthook_verdict", AsyncMock())
    monkeypatch.setattr("mr_roboto.run",
                        AsyncMock(return_value=_FakeAction("failed")))

    source_ctx = {"artifact_schema": _SCHEMA, "checks": _CHECKS}
    with patch.object(apply_mod, "enqueue", AsyncMock(return_value=1)) as enq:
        await apply_mod._enqueue_posthook_llm_child("grade", _source(), source_ctx)

    enq.assert_awaited_once()
    assert _cont_state(enq)["shape_verify_passed"] is False


@pytest.mark.asyncio
async def test_grade_spawns_tagged_on_verify_adr_register(monkeypatch):
    # verify_adr_register is a full-artifact deterministic validator that does
    # NOT carry the *_shape suffix. Authority is a registry, not a naming
    # convention — else step 4.14 (register.md) stays exposed to the 567449
    # confab loop. Passing it tags the continuation exactly like *_shape.
    import general_beckman.apply as apply_mod

    monkeypatch.setattr(apply_mod, "_apply_posthook_verdict", AsyncMock())
    monkeypatch.setattr("mr_roboto.run",
                        AsyncMock(return_value=_FakeAction("completed")))

    source_ctx = {"artifact_schema": _SCHEMA, "checks": [
        {"kind": "verify_adr_register",
         "payload": {"action": "verify_adr_register", "path": ".adr/register.md"}}]}
    with patch.object(apply_mod, "enqueue", AsyncMock(return_value=1)) as enq:
        await apply_mod._enqueue_posthook_llm_child("grade", _source(), source_ctx)

    enq.assert_awaited_once()
    assert _cont_state(enq)["shape_verify_passed"] is True


@pytest.mark.asyncio
async def test_narrow_check_leaves_grade_untagged(monkeypatch):
    # A NARROW check (verify_contains_product_name — one substring) is not a
    # completeness authority: the verifier never runs, so the grade stays fully
    # authoritative (untagged) and RELEVANT/COMPLETE/COHERENT all bind.
    import general_beckman.apply as apply_mod

    monkeypatch.setattr(apply_mod, "_apply_posthook_verdict", AsyncMock())
    monkeypatch.setattr("mr_roboto.run",
                        AsyncMock(return_value=_FakeAction("completed")))

    source_ctx = {"artifact_schema": _SCHEMA, "checks": [
        {"kind": "verify_contains_product_name",
         "payload": {"action": "verify_contains_product_name"}}]}
    with patch.object(apply_mod, "enqueue", AsyncMock(return_value=1)) as enq:
        await apply_mod._enqueue_posthook_llm_child("grade", _source(), source_ctx)

    enq.assert_awaited_once()
    assert _cont_state(enq)["shape_verify_passed"] is False


@pytest.mark.asyncio
async def test_no_shape_check_leaves_grade_untagged(monkeypatch):
    # A step without a verify_*_shape check is unaffected — grade spawns untagged
    # and the verifier is never invoked.
    import general_beckman.apply as apply_mod

    monkeypatch.setattr(apply_mod, "_apply_posthook_verdict", AsyncMock())
    ran = {"called": False}

    async def fake_run(task):
        ran["called"] = True
        return _FakeAction("completed")

    monkeypatch.setattr("mr_roboto.run", fake_run)

    source_ctx = {"artifact_schema": _SCHEMA, "checks": []}
    with patch.object(apply_mod, "enqueue", AsyncMock(return_value=1)) as enq:
        await apply_mod._enqueue_posthook_llm_child("grade", _source(), source_ctx)

    enq.assert_awaited_once()
    assert ran["called"] is False  # no shape check → verifier never invoked
    assert _cont_state(enq)["shape_verify_passed"] is False
