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
            mission_id=1, reviewer_id="3.11", review_result=review_result,
            workflow=wf, reviewer_task_id=99,
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
            mission_id=1, reviewer_id="3.11", review_result=review_result,
            workflow=wf, reviewer_task_id=99,
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
            mission_id=1, reviewer_id="3.11", review_result=review_result,
            workflow=wf, reviewer_task_id=99)
    halt.assert_awaited_once()  # producer at cap -> escalate


@pytest.mark.asyncio
async def test_route_sends_one_card_when_multiple_producers_exhausted():
    """Multiple exhausted producers in one review must yield exactly ONE
    founder-halt card (not one per producer). Each card already renders the
    FULL issues list + full producer-button set, so N cards = N identical
    duplicates (the mission-89 19:37 triple-card bug)."""
    from general_beckman.review_routing import route_review_failure
    wf = {"steps": [
        {"id": "0.1", "output_artifacts": ["product_charter"]},
        {"id": "1.0c", "output_artifacts": ["prior_art_report"]},
        {"id": "1.6", "output_artifacts": ["market_research_report"]},
        {"id": "1.13", "input_artifacts": [
            "product_charter", "prior_art_report", "market_research_report"],
            "output_artifacts": ["research_review_result"]},
    ]}
    review_result = {"status": "fail", "issues": [
        {"target_artifact": "product_charter", "severity": "blocker", "problem": "a"},
        {"target_artifact": "prior_art_report", "severity": "blocker", "problem": "b"},
        {"target_artifact": "market_research_report", "severity": "major", "problem": "c"},
    ]}
    with patch("general_beckman.review_routing._repend_producer", new=AsyncMock(return_value=False)), \
         patch("general_beckman.review_routing._escalate_to_founder", new=AsyncMock()) as halt:
        outcome = await route_review_failure(
            mission_id=89, reviewer_id="1.13", review_result=review_result,
            workflow=wf, reviewer_task_id=99)
    halt.assert_awaited_once()  # 3 exhausted producers -> still ONE card
    assert outcome["escalated"] is True


@pytest.mark.asyncio
async def test_escalate_parks_reviewer_on_all_unresolved():
    """An all-unresolved fail with a reviewer_task_id must PARK the reviewer
    (update_task status=waiting_human) and report escalated=True — the safety
    fix: an escalated review must not advance unreviewed."""
    from general_beckman.review_routing import route_review_failure
    wf = {"steps": [{"id": "3.11", "input_artifacts": [], "output_artifacts": ["rr"]}]}
    review_result = {"status": "fail", "issues": [
        {"target_artifact": None, "severity": "blocker", "problem": "systemic"},
    ]}
    upd = AsyncMock()
    with patch("general_beckman.review_routing._assign_unresolved",
               new=AsyncMock(return_value={})), \
         patch("general_beckman.review_routing._resolve_founder_chat_id",
               new=AsyncMock(return_value=None)), \
         patch("src.infra.db.update_task", new=upd):
        outcome = await route_review_failure(
            mission_id=5, reviewer_id="3.11", review_result=review_result,
            workflow=wf, reviewer_task_id=777,
        )
    upd.assert_awaited_once_with(777, status="waiting_human")
    assert outcome["escalated"] is True
