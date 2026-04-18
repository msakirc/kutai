"""Tests for general_beckman.queue."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_classify_lane():
    from general_beckman.queue import classify_lane
    assert classify_lane({"agent_type": "mechanical"}) == "mechanical"
    assert classify_lane({"agent_type": "researcher"}) == "cloud_llm"
    assert classify_lane({"agent_type": "planner"}) == "cloud_llm"
    assert classify_lane({"agent_type": "coder"}) == "local_llm"
    assert classify_lane({"agent_type": "unknown"}) == "local_llm"


@pytest.mark.asyncio
async def test_pick_ready_task_returns_first_unsaturated():
    from general_beckman.queue import pick_ready_task
    rows = [
        {"id": 1, "agent_type": "researcher"},   # cloud_llm
        {"id": 2, "agent_type": "coder"},        # local_llm
    ]
    with patch("general_beckman.queue.get_ready_tasks", new=AsyncMock(return_value=rows)), \
         patch("general_beckman.queue.claim_task", new=AsyncMock(return_value=True)):
        task = await pick_ready_task(saturated_lanes={"cloud_llm"})
    assert task["id"] == 2


@pytest.mark.asyncio
async def test_pick_ready_task_returns_none_when_all_saturated():
    from general_beckman.queue import pick_ready_task
    rows = [{"id": 1, "agent_type": "researcher"}]
    with patch("general_beckman.queue.get_ready_tasks", new=AsyncMock(return_value=rows)), \
         patch("general_beckman.queue.claim_task", new=AsyncMock(return_value=True)):
        task = await pick_ready_task(saturated_lanes={"cloud_llm", "local_llm"})
    assert task is None


@pytest.mark.asyncio
async def test_pick_ready_task_returns_none_when_queue_empty():
    from general_beckman.queue import pick_ready_task
    with patch("general_beckman.queue.get_ready_tasks", new=AsyncMock(return_value=[])), \
         patch("general_beckman.queue.claim_task", new=AsyncMock(return_value=True)):
        task = await pick_ready_task(saturated_lanes=set())
    assert task is None


@pytest.mark.asyncio
async def test_pick_ready_task_mechanical_bypasses_llm_saturation():
    from general_beckman.queue import pick_ready_task
    rows = [{"id": 99, "agent_type": "mechanical"}]
    with patch("general_beckman.queue.get_ready_tasks", new=AsyncMock(return_value=rows)), \
         patch("general_beckman.queue.claim_task", new=AsyncMock(return_value=True)):
        task = await pick_ready_task(saturated_lanes={"local_llm", "cloud_llm"})
    assert task["id"] == 99


@pytest.mark.asyncio
async def test_pick_ready_task_skips_when_claim_fails():
    from general_beckman.queue import pick_ready_task
    rows = [
        {"id": 1, "agent_type": "coder"},
        {"id": 2, "agent_type": "coder"},
    ]
    claim_mock = AsyncMock(side_effect=[False, True])
    with patch("general_beckman.queue.get_ready_tasks", new=AsyncMock(return_value=rows)), \
         patch("general_beckman.queue.claim_task", claim_mock):
        task = await pick_ready_task(saturated_lanes=set())
    assert task["id"] == 2


@pytest.mark.asyncio
async def test_count_pending_cloud_tasks():
    from general_beckman.queue import count_pending_cloud_tasks
    rows = [
        {"id": 1, "agent_type": "researcher"},
        {"id": 2, "agent_type": "coder"},
        {"id": 3, "agent_type": "planner"},
    ]
    with patch("general_beckman.queue.get_ready_tasks", new=AsyncMock(return_value=rows)):
        n = await count_pending_cloud_tasks()
    assert n == 2
