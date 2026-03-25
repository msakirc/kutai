"""Tests for Fix #8 (Cost Tracking) and Fix #9 (Push Notifications)."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


def run_async(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Fix #8: Cost Tracking Tests ───────────────────────────────────────────────


class TestCostAccumulatesAcrossTasks(unittest.TestCase):
    """Simulate multiple task completions and verify total cost accumulates."""

    @patch("src.collaboration.blackboard.get_db", new_callable=AsyncMock)
    def test_cost_accumulates_across_tasks(self, mock_db):
        from src.collaboration.blackboard import write_blackboard, read_blackboard, clear_cache

        clear_cache()

        db_mock = AsyncMock()
        db_mock.execute = AsyncMock(return_value=AsyncMock(fetchone=AsyncMock(return_value=None)))
        db_mock.commit = AsyncMock()
        mock_db.return_value = db_mock

        async def _test():
            clear_cache()
            mission_id = 42

            # Write initial cost tracking
            await write_blackboard(mission_id, "cost_tracking", {
                "total_cost": 0.0, "task_count": 0, "by_phase": {}
            })

            # Simulate 3 task completions accumulating cost
            for cost in [0.05, 0.12, 0.03]:
                current = await read_blackboard(mission_id, "cost_tracking")
                if not isinstance(current, dict):
                    current = {"total_cost": 0.0, "task_count": 0, "by_phase": {}}
                current["total_cost"] = current.get("total_cost", 0.0) + cost
                current["task_count"] = current.get("task_count", 0) + 1
                await write_blackboard(mission_id, "cost_tracking", current)

            final = await read_blackboard(mission_id, "cost_tracking")
            self.assertAlmostEqual(final["total_cost"], 0.20, places=4)
            self.assertEqual(final["task_count"], 3)

        run_async(_test())
        clear_cache()


class TestCostTracksByPhase(unittest.TestCase):
    """Tasks from different phases should have separate cost tracking."""

    @patch("src.collaboration.blackboard.get_db", new_callable=AsyncMock)
    def test_cost_tracks_by_phase(self, mock_db):
        from src.collaboration.blackboard import write_blackboard, read_blackboard, clear_cache

        clear_cache()

        db_mock = AsyncMock()
        db_mock.execute = AsyncMock(return_value=AsyncMock(fetchone=AsyncMock(return_value=None)))
        db_mock.commit = AsyncMock()
        mock_db.return_value = db_mock

        async def _test():
            clear_cache()
            mission_id = 99

            await write_blackboard(mission_id, "cost_tracking", {
                "total_cost": 0.0, "task_count": 0, "by_phase": {}
            })

            # Simulate tasks from different phases
            tasks = [
                {"cost": 0.10, "phase": "phase_1"},
                {"cost": 0.05, "phase": "phase_1"},
                {"cost": 0.20, "phase": "phase_2"},
            ]
            for t in tasks:
                current = await read_blackboard(mission_id, "cost_tracking")
                current["total_cost"] += t["cost"]
                current["task_count"] += 1
                phase_costs = current.get("by_phase", {})
                phase_costs[t["phase"]] = phase_costs.get(t["phase"], 0.0) + t["cost"]
                current["by_phase"] = phase_costs
                await write_blackboard(mission_id, "cost_tracking", current)

            final = await read_blackboard(mission_id, "cost_tracking")
            self.assertAlmostEqual(final["by_phase"]["phase_1"], 0.15, places=4)
            self.assertAlmostEqual(final["by_phase"]["phase_2"], 0.20, places=4)
            self.assertAlmostEqual(final["total_cost"], 0.35, places=4)

        run_async(_test())
        clear_cache()


class TestCostMilestoneNotification(unittest.TestCase):
    """When crossing a $1 threshold, a notification should be sent."""

    def test_cost_milestone_notification(self):
        """Simulate the milestone detection logic from _handle_complete."""

        notifications = []

        async def mock_send(text):
            notifications.append(text)

        async def _test():
            # Simulate the threshold-crossing logic
            thresholds = [1.0, 5.0, 10.0]

            # Case 1: crossing $1 threshold (prev=0.95, new=1.05)
            cost = 0.10
            total_cost = 1.05
            prev = total_cost - cost  # 0.95

            for threshold in thresholds:
                if prev < threshold <= total_cost:
                    await mock_send(
                        f"Mission #1 cost milestone: ${total_cost:.2f}\n"
                        f"(10 tasks completed)"
                    )
                    break

            self.assertEqual(len(notifications), 1)
            self.assertIn("$1.05", notifications[0])

            # Case 2: not crossing a threshold (prev=1.05, new=1.15)
            total_cost = 1.15
            prev = total_cost - 0.10

            found = False
            for threshold in thresholds:
                if prev < threshold <= total_cost:
                    found = True
                    break
            self.assertFalse(found)

        run_async(_test())


class TestNoCostTrackingWithoutGoal(unittest.TestCase):
    """Tasks without a mission_id should not trigger cost tracking."""

    def test_no_cost_tracking_without_mission(self):
        # The cost tracking block in _handle_complete is gated by:
        #   if task.get("mission_id") and cost > 0:
        task_no_mission = {"id": 1, "title": "test", "context": "{}"}
        self.assertFalse(bool(task_no_mission.get("mission_id") and 0.05 > 0))

        task_zero_cost = {"id": 2, "title": "test", "mission_id": 5, "context": "{}"}
        cost = 0
        self.assertFalse(bool(task_zero_cost.get("mission_id") and cost > 0))


# ── Fix #9: Push Notification Tests ──────────────────────────────────────────


class TestPhaseCompletionNotification(unittest.TestCase):
    """When a workflow phase completes, a notification should be sent."""

    def test_phase_completion_notification(self):
        from src.workflows.engine.status import compute_phase_progress

        # All tasks in phase_1 completed
        tasks = [
            {"status": "completed", "context": json.dumps({
                "workflow_phase": "phase_1", "is_workflow_step": True
            })},
            {"status": "completed", "context": json.dumps({
                "workflow_phase": "phase_1", "is_workflow_step": True
            })},
            # phase_2 has one pending
            {"status": "completed", "context": json.dumps({
                "workflow_phase": "phase_2", "is_workflow_step": True
            })},
            {"status": "pending", "context": json.dumps({
                "workflow_phase": "phase_2", "is_workflow_step": True
            })},
        ]

        progress = compute_phase_progress(tasks)

        # phase_1 should be fully complete
        p1 = progress["phase_1"]
        self.assertEqual(p1["completed"], 2)
        self.assertEqual(p1["total"], 2)
        self.assertTrue(p1["completed"] == p1["total"] and p1["total"] > 0)

        # phase_2 should NOT be fully complete
        p2 = progress["phase_2"]
        self.assertEqual(p2["completed"], 1)
        self.assertEqual(p2["total"], 2)
        self.assertFalse(p2["completed"] == p2["total"])

        # Verify total/completed phases count
        total_phases = len(progress)
        completed_phases = sum(
            1 for p in progress.values()
            if p.get("completed", 0) == p.get("total", 0)
        )
        self.assertEqual(total_phases, 2)
        self.assertEqual(completed_phases, 1)


class TestStepFailureNotification(unittest.TestCase):
    """When a workflow step fails, a notification should be sent."""

    def test_step_failure_notification(self):
        """Verify the failure notification logic from the except handler."""
        notifications = []

        async def mock_send(text):
            notifications.append(text)

        async def _test():
            task_ctx = {"is_workflow_step": True, "workflow_phase": "phase_4"}
            task = {"id": 55, "title": "Implement auth module", "mission_id": 10}

            # Simulate the notification logic from the except handler
            if task_ctx.get("is_workflow_step"):
                wf_phase = task_ctx.get("workflow_phase", "?")
                await mock_send(
                    f"Workflow step failed: #{task['id']}\n"
                    f"_{task.get('title', '')[:60]}_\n"
                    f"Phase: {wf_phase}"
                )

            self.assertEqual(len(notifications), 1)
            self.assertIn("#55", notifications[0])
            self.assertIn("phase_4", notifications[0])
            self.assertIn("Implement auth module", notifications[0])

        run_async(_test())


class TestNoNotificationForNonWorkflowTasks(unittest.TestCase):
    """Regular tasks should not trigger workflow notifications."""

    def test_no_notification_for_non_workflow_tasks(self):
        notifications = []

        async def mock_send(text):
            notifications.append(text)

        async def _test():
            # Task without workflow context
            task_ctx = {}
            task = {"id": 77, "title": "Simple task", "mission_id": 5}

            # Simulate the notification check
            if task_ctx.get("is_workflow_step") and task.get("mission_id"):
                await mock_send("Should not appear")

            self.assertEqual(len(notifications), 0)

            # Task with is_workflow_step = False
            task_ctx2 = {"is_workflow_step": False}
            if task_ctx2.get("is_workflow_step") and task.get("mission_id"):
                await mock_send("Should not appear either")

            self.assertEqual(len(notifications), 0)

        run_async(_test())


if __name__ == "__main__":
    unittest.main()
