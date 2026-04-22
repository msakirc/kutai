"""Tests for general_beckman.queue."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_pick_ready_task_returns_first_eligible():
    from general_beckman.queue import pick_ready_task
    rows = [
        {"id": 1, "agent_type": "researcher"},
        {"id": 2, "agent_type": "coder"},
    ]
    with patch("general_beckman.queue.get_ready_tasks", new=AsyncMock(return_value=rows)), \
         patch("general_beckman.queue.claim_task", new=AsyncMock(return_value=True)):
        task = await pick_ready_task()
    assert task["id"] == 1


@pytest.mark.asyncio
async def test_pick_ready_task_returns_none_when_queue_empty():
    from general_beckman.queue import pick_ready_task
    with patch("general_beckman.queue.get_ready_tasks", new=AsyncMock(return_value=[])), \
         patch("general_beckman.queue.claim_task", new=AsyncMock(return_value=True)):
        task = await pick_ready_task()
    assert task is None


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
        task = await pick_ready_task()
    assert task["id"] == 2
