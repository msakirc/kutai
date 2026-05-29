"""SP3 Task 7 - summary resume."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_summary_resume_pass_builds_summary_kind_verdict():
    from general_beckman import posthook_continuations as pc
    summary = "This artifact describes the user stories and acceptance flow in detail. " * 3
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._summary_resume(301, {"result": {"content": summary}},
                                 {"source_task_id": 5, "artifact_name": "user_stories"})
    v = ap.call_args.args[1]
    assert v.kind == "summary:user_stories" and v.passed is True
    assert v.raw["summary"].startswith("This artifact")


@pytest.mark.asyncio
async def test_summary_resume_short_fails():
    from general_beckman import posthook_continuations as pc
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._summary_resume(302, {"result": {"content": "tiny"}},
                                 {"source_task_id": 5, "artifact_name": "x"})
    assert ap.call_args.args[1].passed is False


@pytest.mark.asyncio
async def test_summary_resume_err_fails_closed():
    from general_beckman import posthook_continuations as pc
    with patch.object(pc, "_apply_posthook_verdict", AsyncMock()) as ap:
        await pc._summary_resume_err(303, {"status": "failed"},
                                     {"source_task_id": 5, "artifact_name": "x"})
    assert ap.call_args.args[1].passed is False
