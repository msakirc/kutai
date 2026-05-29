"""SP3 Task 5 - grade resume + chaining."""
import pytest
from unittest.mock import AsyncMock, patch


def _grade_result(text, model="m"):
    return {"result": {"content": text, "model": model}, "status": "completed"}


_PASS_RAW = ("RELEVANT: YES\nCOMPLETE: YES\nVERDICT: PASS\n"
             "WELL_FORMED: PASS\nCOHERENT: PASS\n")


@pytest.mark.asyncio
async def test_grade_resume_pass_applies_verdict():
    from general_beckman import posthook_continuations as pc
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._grade_resume(101, _grade_result(_PASS_RAW),
                               {"source_task_id": 7, "attempt": 0, "exclusions": []})
    assert ap.await_count == 1
    verdict = ap.call_args.args[1]
    assert verdict.kind == "grade" and verdict.source_task_id == 7
    assert verdict.passed is True
    assert isinstance(verdict.raw, dict)


@pytest.mark.asyncio
async def test_grade_resume_parsefail_attempt0_chains_second_child():
    from general_beckman import posthook_continuations as pc
    with patch.object(pc, "_enqueue_grade_child", AsyncMock()) as enq, \
         patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._grade_resume(101, _grade_result("garbage no verdict", model="qwen-thinking"),
                               {"source_task_id": 7, "attempt": 0, "exclusions": []})
    enq.assert_awaited_once()
    _, kwargs = enq.call_args
    assert kwargs["attempt"] == 1
    assert "qwen-thinking" in kwargs["exclusions"]
    ap.assert_not_awaited()


@pytest.mark.asyncio
async def test_grade_resume_parsefail_attempt1_autofails():
    from general_beckman import posthook_continuations as pc
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._grade_resume(102, _grade_result("still garbage"),
                               {"source_task_id": 7, "attempt": 1, "exclusions": ["m"]})
    verdict = ap.call_args.args[1]
    assert verdict.passed is False
    assert "grader_incapable" in str(verdict.raw)


@pytest.mark.asyncio
async def test_grade_resume_err_autofails():
    from general_beckman import posthook_continuations as pc
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._grade_resume_err(103, {"status": "failed", "error": "no candidates"},
                                   {"source_task_id": 7, "attempt": 0, "exclusions": []})
    verdict = ap.call_args.args[1]
    assert verdict.passed is False
    assert "grader call failed" in str(verdict.raw)
