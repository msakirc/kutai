"""The deterministic shape verifier is authoritative; the LLM grade is subordinate.

Mission-90 task 567449 [5.0a] design_tokens_generation: the model converged on a
shape-VALID design_tokens.json (verify_design_tokens_shape → ok), but the
scope-blind LLM grader confabulated `COMPLETE: NO / VERDICT: FAIL` (its own
prompt says "DO NOT JUDGE field/section presence"). The producer re-emitted the
same correct artifact byte-identically, and the degenerate-repeat detector DLQ'd
it as "not converging" — it was killed *because* it correctly converged.

Fix: at grade time, run the step's verify_*_shape check inline on the
materialized artifact. If it PASSES, shape/completeness is a proven
deterministic fact, so the confab-prone LLM grade is skipped (auto-PASS) — the
same short-circuit shape as the empty-scope exemption. When the verifier FAILS
(a real defect, e.g. _schema_version "1.0.0" or an empty variant on earlier
attempts), the LLM grade runs as normal and the producer still re-pends.
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


@pytest.mark.asyncio
async def test_grade_auto_passes_when_shape_verifier_passes(monkeypatch):
    import general_beckman.apply as apply_mod

    verdicts = []

    async def fake_apply(child_task, verdict):
        verdicts.append(verdict)

    monkeypatch.setattr(apply_mod, "_apply_posthook_verdict", fake_apply)
    monkeypatch.setattr("mr_roboto.run",
                        AsyncMock(return_value=_FakeAction("completed")))

    source_ctx = {"artifact_schema": _SCHEMA, "checks": _CHECKS}
    with patch.object(apply_mod, "enqueue", AsyncMock(return_value=1)) as enq:
        await apply_mod._enqueue_posthook_llm_child("grade", _source(), source_ctx)

    enq.assert_not_awaited()  # confab-prone LLM grade SKIPPED
    assert verdicts and verdicts[0].passed is True


@pytest.mark.asyncio
async def test_grade_runs_normally_when_shape_verifier_fails(monkeypatch):
    import general_beckman.apply as apply_mod

    monkeypatch.setattr(apply_mod, "_apply_posthook_verdict", AsyncMock())
    monkeypatch.setattr("mr_roboto.run",
                        AsyncMock(return_value=_FakeAction("failed")))

    source_ctx = {"artifact_schema": _SCHEMA, "checks": _CHECKS}
    with patch.object(apply_mod, "enqueue", AsyncMock(return_value=1)) as enq:
        await apply_mod._enqueue_posthook_llm_child("grade", _source(), source_ctx)

    enq.assert_awaited_once()  # real defect → normal LLM grade still runs


@pytest.mark.asyncio
async def test_grade_auto_passes_on_verify_adr_register(monkeypatch):
    # verify_adr_register is a full-artifact deterministic validator that does
    # NOT carry the *_shape suffix. Authority must be a registry, not a naming
    # convention — else step 4.14 (register.md) stays exposed to the exact
    # 567449 confab-grade → degenerate-repeat loop while Fix 2 already treats
    # verify_adr_register as authoritative (asymmetry the reviewer flagged).
    import general_beckman.apply as apply_mod

    verdicts = []

    async def fake_apply(child_task, verdict):
        verdicts.append(verdict)

    monkeypatch.setattr(apply_mod, "_apply_posthook_verdict", fake_apply)
    monkeypatch.setattr("mr_roboto.run",
                        AsyncMock(return_value=_FakeAction("completed")))

    source_ctx = {"artifact_schema": _SCHEMA, "checks": [
        {"kind": "verify_adr_register",
         "payload": {"action": "verify_adr_register", "path": ".adr/register.md"}}]}
    with patch.object(apply_mod, "enqueue", AsyncMock(return_value=1)) as enq:
        await apply_mod._enqueue_posthook_llm_child("grade", _source(), source_ctx)

    enq.assert_not_awaited()
    assert verdicts and verdicts[0].passed is True


@pytest.mark.asyncio
async def test_narrow_check_does_not_auto_pass_grade(monkeypatch):
    # A NARROW check (verify_contains_product_name — one substring) is not a
    # completeness authority: it must NOT skip the LLM grade.
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


@pytest.mark.asyncio
async def test_no_shape_check_grades_normally(monkeypatch):
    # A step without a verify_*_shape check is unaffected — LLM grade still spawns.
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
