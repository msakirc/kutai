"""test_task_lifecycle.py — Integration tests for the full task DB lifecycle.

Tests:
- Creating, claiming, completing tasks
- Dependency blocking / unblocking
- Task cancellation (cascading to children)
- Mission creation and task attachment
- Task deduplication
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTaskCreate:
    """Task creation and retrieval."""

    def test_add_and_get_task(self, temp_db):
        """add_task returns an ID; get_task returns the same record."""
        from src.infra.db import add_task, get_task

        async def _run():
            tid = await add_task(
                title="Test task",
                description="A test description",
                agent_type="executor",
                priority=5,
            )
            assert tid is not None, "add_task should return an integer ID"
            task = await get_task(tid)
            assert task is not None
            assert task["title"] == "Test task"
            assert task["status"] == "pending"
            assert task["agent_type"] == "executor"
            return tid

        run_async(_run())

    def test_add_task_with_mission(self, temp_db):
        """Tasks can be attached to a mission."""
        from src.infra.db import add_task, add_mission, get_task

        async def _run():
            mid = await add_mission(
                title="Test mission",
                description="Integration test mission",
            )
            tid = await add_task(
                title="Mission task",
                description="belongs to a mission",
                mission_id=mid,
                agent_type="coder",
            )
            task = await get_task(tid)
            assert task["mission_id"] == mid

        run_async(_run())

    def test_task_deduplication(self, temp_db):
        """Identical pending task is not created twice."""
        from src.infra.db import add_task

        async def _run():
            tid1 = await add_task(
                title="Dup task",
                description="Same description",
                agent_type="executor",
            )
            tid2 = await add_task(
                title="Dup task",
                description="Same description",
                agent_type="executor",
            )
            assert tid1 is not None
            # Second call returns None (deduplication)
            assert tid2 is None, (
                "add_task should return None for a duplicate pending task"
            )

        run_async(_run())

    def test_task_deduplication_different_agent(self, temp_db):
        """Different agent_type produces a separate task (not deduped)."""
        from src.infra.db import add_task

        async def _run():
            tid1 = await add_task(
                title="Task A",
                description="desc",
                agent_type="coder",
            )
            tid2 = await add_task(
                title="Task A",
                description="desc",
                agent_type="researcher",  # different agent type
            )
            assert tid1 is not None
            assert tid2 is not None
            assert tid1 != tid2

        run_async(_run())


@pytest.mark.integration
class TestTaskClaiming:
    """Atomic task claiming (race condition prevention)."""

    def test_claim_pending_task(self, temp_db):
        """claim_task transitions a pending task to processing."""
        from src.infra.db import add_task, claim_task, get_task

        async def _run():
            tid = await add_task(
                title="Claimable",
                description="pending task",
                agent_type="executor",
            )
            success = await claim_task(tid)
            assert success is True

            task = await get_task(tid)
            assert task["status"] == "processing"
            assert task["started_at"] is not None

        run_async(_run())

    def test_claim_twice_fails(self, temp_db):
        """Second claim attempt on the same task returns False."""
        from src.infra.db import add_task, claim_task

        async def _run():
            tid = await add_task(
                title="Once only",
                description="",
                agent_type="executor",
            )
            first = await claim_task(tid)
            second = await claim_task(tid)
            assert first is True
            assert second is False  # already in 'processing', not 'pending'

        run_async(_run())

    def test_claim_nonexistent_task(self, temp_db):
        """Claiming a non-existent task ID returns False (no crash)."""
        from src.infra.db import claim_task

        async def _run():
            result = await claim_task(999999)
            assert result is False

        run_async(_run())


@pytest.mark.integration
class TestTaskCompletion:
    """Task status transitions through the full lifecycle."""

    def test_complete_task(self, temp_db):
        """Task can be moved through pending → processing → completed."""
        from src.infra.db import add_task, claim_task, update_task, get_task
        from datetime import datetime, timezone

        async def _run():
            tid = await add_task(
                title="Complete me",
                description="",
                agent_type="executor",
            )
            await claim_task(tid)
            await update_task(
                tid,
                status="completed",
                result="done",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            task = await get_task(tid)
            assert task["status"] == "completed"
            assert task["result"] == "done"

        run_async(_run())

    def test_fail_task(self, temp_db):
        """Task can be marked as failed with an error message."""
        from src.infra.db import add_task, claim_task, update_task, get_task
        from datetime import datetime, timezone

        async def _run():
            tid = await add_task(
                title="Fail me",
                description="",
                agent_type="executor",
            )
            await claim_task(tid)
            await update_task(
                tid,
                status="failed",
                error="Something went wrong",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            task = await get_task(tid)
            assert task["status"] == "failed"
            assert "Something went wrong" in task["error"]

        run_async(_run())


@pytest.mark.integration
class TestTaskDependencies:
    """Dependency-based task ordering (blocked → unblocked)."""

    def test_blocked_task_not_ready(self, temp_db):
        """A task with an incomplete dependency is not returned by get_ready_tasks."""
        from src.infra.db import add_task, get_ready_tasks

        async def _run():
            # Parent task (not yet complete)
            parent_id = await add_task(
                title="Parent",
                description="must finish first",
                agent_type="executor",
            )
            # Child task depends on parent
            child_id = await add_task(
                title="Child",
                description="blocked until parent done",
                agent_type="executor",
                depends_on=[parent_id],
            )
            assert child_id is not None

            ready = await get_ready_tasks(limit=10)
            ready_ids = [t["id"] for t in ready]

            assert parent_id in ready_ids, "Parent (no deps) should be ready"
            assert child_id not in ready_ids, "Child (blocked) should NOT be ready"

        run_async(_run())

    def test_unblocked_task_becomes_ready(self, temp_db):
        """After parent completes, child becomes available via get_ready_tasks."""
        from src.infra.db import add_task, claim_task, update_task, get_ready_tasks
        from datetime import datetime, timezone

        async def _run():
            parent_id = await add_task(
                title="Parent step",
                description="",
                agent_type="executor",
            )
            child_id = await add_task(
                title="Child step",
                description="",
                agent_type="executor",
                depends_on=[parent_id],
            )

            # Complete the parent
            await claim_task(parent_id)
            await update_task(
                parent_id,
                status="completed",
                result="parent done",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )

            ready = await get_ready_tasks(limit=10)
            ready_ids = [t["id"] for t in ready]
            assert child_id in ready_ids, (
                "Child should appear in ready tasks after parent completes"
            )

        run_async(_run())

    def test_multi_level_dependency_chain(self, temp_db):
        """A → B → C chain: only A is initially ready; B and C are blocked."""
        from src.infra.db import add_task, claim_task, update_task, get_ready_tasks
        from datetime import datetime, timezone

        async def _run():
            a_id = await add_task(title="A", description="", agent_type="executor")
            b_id = await add_task(title="B", description="", agent_type="executor",
                                   depends_on=[a_id])
            c_id = await add_task(title="C", description="", agent_type="executor",
                                   depends_on=[b_id])

            ready = await get_ready_tasks(limit=10)
            ready_ids = {t["id"] for t in ready}
            assert a_id in ready_ids
            assert b_id not in ready_ids
            assert c_id not in ready_ids

            # Complete A
            await claim_task(a_id)
            await update_task(a_id, status="completed", result="A done",
                               completed_at=datetime.now(timezone.utc).isoformat())

            ready2 = await get_ready_tasks(limit=10)
            ready_ids2 = {t["id"] for t in ready2}
            assert b_id in ready_ids2
            assert c_id not in ready_ids2

        run_async(_run())


@pytest.mark.integration
class TestTaskCancellation:
    """Task cancellation, including cascading to children."""

    def test_cancel_pending_task(self, temp_db):
        """cancel_task marks a pending task as cancelled."""
        from src.infra.db import add_task, cancel_task, get_task

        async def _run():
            tid = await add_task(
                title="Cancel me",
                description="",
                agent_type="executor",
            )
            result = await cancel_task(tid)
            assert result is True

            task = await get_task(tid)
            assert task["status"] == "cancelled"

        run_async(_run())

    def test_cancel_cascades_to_children(self, temp_db):
        """Cancelling a parent also cancels its pending children."""
        from src.infra.db import add_task, cancel_task, get_task

        async def _run():
            parent_id = await add_task(
                title="Parent",
                description="",
                agent_type="executor",
                parent_task_id=None,
            )
            # Add children via add_subtasks_atomically to set parent_task_id
            child1_id = await add_task(
                title="Child 1",
                description="",
                agent_type="executor",
                parent_task_id=parent_id,
            )
            child2_id = await add_task(
                title="Child 2",
                description="",
                agent_type="executor",
                parent_task_id=parent_id,
            )

            await cancel_task(parent_id)

            parent = await get_task(parent_id)
            child1 = await get_task(child1_id)
            child2 = await get_task(child2_id)

            assert parent["status"] == "cancelled"
            assert child1["status"] == "cancelled"
            assert child2["status"] == "cancelled"

        run_async(_run())

    def test_cancel_completed_task_returns_false(self, temp_db):
        """Cannot cancel a completed task; cancel_task returns False."""
        from src.infra.db import add_task, claim_task, update_task, cancel_task
        from datetime import datetime, timezone

        async def _run():
            tid = await add_task(
                title="Already done",
                description="",
                agent_type="executor",
            )
            await claim_task(tid)
            await update_task(
                tid,
                status="completed",
                result="done",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )

            result = await cancel_task(tid)
            assert result is False, "Cancelling a completed task should return False"

        run_async(_run())


@pytest.mark.integration
class TestMissionLifecycle:
    """Mission creation, task attachment, and status tracking."""

    def test_create_mission(self, temp_db):
        """add_mission returns ID; get_mission returns it."""
        from src.infra.db import add_mission, get_mission

        async def _run():
            mid = await add_mission(
                title="Build a todo app",
                description="Full stack todo application",
                priority=7,
            )
            assert mid is not None
            mission = await get_mission(mid)
            assert mission is not None
            assert mission["title"] == "Build a todo app"
            assert mission["status"] == "active"

        run_async(_run())

    def test_mission_with_tasks(self, temp_db):
        """get_tasks_for_mission returns all tasks added to the mission."""
        from src.infra.db import add_mission, add_task, get_tasks_for_mission

        async def _run():
            mid = await add_mission(
                title="Test mission",
                description="",
            )
            for i in range(3):
                await add_task(
                    title=f"Task {i}",
                    description=f"step {i}",
                    mission_id=mid,
                    agent_type="executor",
                )

            tasks = await get_tasks_for_mission(mid)
            assert len(tasks) == 3
            assert all(t["mission_id"] == mid for t in tasks)

        run_async(_run())

    def test_mission_context_stored(self, temp_db):
        """Context dict is stored and retrieved correctly."""
        from src.infra.db import add_mission, get_mission
        import json

        async def _run():
            ctx = {"user_id": 123, "workflow": "i2p"}
            mid = await add_mission(
                title="Context test",
                description="",
                context=ctx,
            )
            mission = await get_mission(mid)
            stored = mission.get("context")
            if isinstance(stored, str):
                stored = json.loads(stored)
            assert stored.get("user_id") == 123
            assert stored.get("workflow") == "i2p"

        run_async(_run())
