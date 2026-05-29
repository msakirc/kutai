"""SP3 Task 9 - grade-pass spawns summary children via CPS helper."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_grade_pass_spawns_summary_child_via_helper():
    from general_beckman import apply as ap
    from general_beckman.result_router import PostHookVerdict
    source = {"id": 5, "mission_id": 3, "status": "ungraded",
              "context": '{"_pending_posthooks": ["grade"]}'}
    with patch("src.infra.db.get_task", AsyncMock(return_value=source)), \
         patch("src.infra.db.update_task", AsyncMock()), \
         patch("src.infra.db.add_task", AsyncMock(return_value=None)), \
         patch.object(ap, "_summary_kinds_for_source",
                      AsyncMock(return_value=["summary:user_stories"])), \
         patch.object(ap, "_enqueue_posthook_llm_child", AsyncMock()) as enq, \
         patch.object(ap, "_record_and_resolve_confidence", AsyncMock()), \
         patch.object(ap, "_spawn_workflow_advance_if_mission", AsyncMock()):
        v = PostHookVerdict(source_task_id=5, kind="grade", passed=True, raw={})
        await ap._apply_posthook_verdict({"id": 999}, v)
    enq.assert_awaited()
    assert enq.call_args.args[0] == "summary:user_stories"
