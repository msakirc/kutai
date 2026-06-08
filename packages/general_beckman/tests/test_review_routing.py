import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_route_repends_tagged_producers_existing_rows():
    from general_beckman.review_routing import route_review_failure
    wf = {"steps": [
        {"id": "3.4", "output_artifacts": ["requirements_spec"]},
        {"id": "3.11", "input_artifacts": ["requirements_spec"], "output_artifacts": ["rr"]},
    ]}
    review_result = {"status": "fail", "issues": [
        {"target_artifact": "requirements_spec", "severity": "blocker", "problem": "no traceability"},
    ]}
    repended = []
    async def fake_repend(mission_id, step_id, feedback):
        repended.append((step_id, feedback))
        return True
    with patch("general_beckman.review_routing._repend_producer", new=fake_repend), \
         patch("general_beckman.review_routing._escalate_to_founder", new=AsyncMock()) as halt:
        outcome = await route_review_failure(
            mission_id=1, reviewer_id="3.11", review_result=review_result, workflow=wf,
        )
    assert tuple(s for s, _ in repended) == ("3.4",)
    assert "no traceability" in repended[0][1]
    assert outcome["routed"] == ["3.4"]
    assert outcome["escalated"] is False
    halt.assert_not_awaited()

@pytest.mark.asyncio
async def test_route_escalates_when_all_unresolved():
    from general_beckman.review_routing import route_review_failure
    wf = {"steps": [{"id": "3.11", "input_artifacts": [], "output_artifacts": ["rr"]}]}
    review_result = {"status": "fail", "issues": [
        {"target_artifact": None, "severity": "blocker", "problem": "systemic"},
    ]}
    with patch("general_beckman.review_routing._assign_unresolved", new=AsyncMock(return_value={})), \
         patch("general_beckman.review_routing._repend_producer", new=AsyncMock()) as rp, \
         patch("general_beckman.review_routing._escalate_to_founder", new=AsyncMock()) as halt:
        outcome = await route_review_failure(
            mission_id=1, reviewer_id="3.11", review_result=review_result, workflow=wf,
        )
    rp.assert_not_awaited()
    halt.assert_awaited_once()
    assert outcome["escalated"] is True

@pytest.mark.asyncio
async def test_route_escalates_when_producer_exhausted():
    from general_beckman.review_routing import route_review_failure
    wf = {"steps": [
        {"id": "3.4", "output_artifacts": ["requirements_spec"]},
        {"id": "3.11", "input_artifacts": ["requirements_spec"], "output_artifacts": ["rr"]},
    ]}
    review_result = {"status": "fail", "issues": [
        {"target_artifact": "requirements_spec", "severity": "blocker", "problem": "x"}]}
    with patch("general_beckman.review_routing._repend_producer", new=AsyncMock(return_value=False)), \
         patch("general_beckman.review_routing._escalate_to_founder", new=AsyncMock()) as halt:
        outcome = await route_review_failure(
            mission_id=1, reviewer_id="3.11", review_result=review_result, workflow=wf)
    halt.assert_awaited_once()  # producer at cap -> escalate
