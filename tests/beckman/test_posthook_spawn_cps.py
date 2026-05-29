"""SP3 Task 8 - CPS spawn for grade/code_review/summary post-hooks."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_grade_posthook_enqueues_child_with_continuation_no_agent_task():
    from general_beckman import apply as ap
    source = {"id": 7, "title": "T", "description": "D", "result": "x" * 200,
              "context": "{}", "mission_id": 3}
    with patch.object(ap, "enqueue", AsyncMock(return_value=999)) as enq:
        await ap._enqueue_posthook_llm_child("grade", source,
                                             {"generating_model": "gen-m",
                                              "grade_excluded_models": ["bad"]})
    enq.assert_awaited_once()
    args, kwargs = enq.call_args
    spec = args[0]
    assert spec["agent_type"] == "reviewer"
    assert kwargs["on_complete"] == "posthook.grade.resume"
    assert kwargs["on_error"] == "posthook.grade.resume_err"
    cs = kwargs["cont_state"]
    assert cs["source_task_id"] == 7
    assert cs["mission_id"] == 3            # mission_id in cont_state ...
    assert "mission_id" not in spec         # ... NOT on the child spec row
    # exclusions seeded from generating_model + grade_excluded_models
    assert set(["gen-m", "bad"]).issubset(set(spec["context"]["llm_call"]["exclude_models"]))


@pytest.mark.asyncio
async def test_trivial_source_short_circuits_to_verdict_no_child():
    from general_beckman import apply as ap
    source = {"id": 7, "result": "  ", "context": "{}", "mission_id": None}
    with patch.object(ap, "enqueue", AsyncMock()) as enq, \
         patch.object(ap, "_apply_posthook_verdict", AsyncMock()) as apv:
        await ap._enqueue_posthook_llm_child("grade", source, {})
    enq.assert_not_awaited()
    apv.assert_awaited_once()
    verdict = apv.call_args.args[1]
    assert verdict.passed is False


@pytest.mark.asyncio
async def test_code_review_posthook_enqueues_child():
    from general_beckman import apply as ap
    source = {"id": 9, "title": "T", "description": "D",
              "result": "def login(u, p):\n    return check(u, p)\n" * 3,
              "context": '{"produces": ["a.py"]}', "mission_id": None}
    with patch.object(ap, "enqueue", AsyncMock(return_value=998)) as enq:
        await ap._enqueue_posthook_llm_child("code_review", source,
                                             {"generating_model": "g"})
    enq.assert_awaited_once()
    assert enq.call_args.kwargs["on_complete"] == "posthook.code_review.resume"
