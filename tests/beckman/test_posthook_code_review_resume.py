"""SP3 Task 6 - code_review resume."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_code_review_resume_pass():
    from general_beckman import posthook_continuations as pc
    raw = "ISSUES:\n- NONE\nVERDICT: PASS\n"
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._code_review_resume(201, {"result": {"content": raw}},
                                     {"source_task_id": 9})
    v = ap.call_args.args[1]
    assert v.kind == "code_review" and v.passed is True
    assert isinstance(v.raw, dict)


@pytest.mark.asyncio
async def test_code_review_resume_fail_carries_issues():
    from general_beckman import posthook_continuations as pc
    raw = "ISSUES:\n- missing auth check in a.py:12\nVERDICT: FAIL\n"
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._code_review_resume(202, {"result": {"content": raw}},
                                     {"source_task_id": 9})
    v = ap.call_args.args[1]
    assert v.passed is False


@pytest.mark.asyncio
async def test_code_review_resume_err():
    from general_beckman import posthook_continuations as pc
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._code_review_resume_err(203, {"status": "failed", "error": "x"},
                                         {"source_task_id": 9})
    assert ap.call_args.args[1].passed is False
