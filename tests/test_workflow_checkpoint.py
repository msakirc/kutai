"""Tests for the Workflow Checkpoint/Resume system.

Covers:
- DB functions: upsert_workflow_checkpoint, get_workflow_checkpoint
- Phase completion detection: _check_phase_completion
- Runner resume: find_resumable, resume
"""

import asyncio
import json
import unittest
from unittest.mock import patch, AsyncMock, MagicMock


# ── DB function tests ─────────────────────────────────────────────────────


class TestUpsertWorkflowCheckpoint(unittest.TestCase):
    """Tests for upsert_workflow_checkpoint."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    @patch("src.infra.db.get_db")
    def test_creates_new_checkpoint(self, mock_get_db):
        """Inserts a new checkpoint record."""
        from src.infra.db import upsert_workflow_checkpoint

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        self._run(upsert_workflow_checkpoint(
            mission_id=1,
            workflow_name="i2p_v2",
            current_phase="phase_1",
            completed_phases=["phase_0"],
            metadata={"key": "value"},
        ))

        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]

        self.assertIn("INSERT OR REPLACE", sql)
        self.assertEqual(params[0], 1)  # mission_id
        self.assertEqual(params[1], "i2p_v2")  # workflow_name
        self.assertEqual(params[2], "phase_1")  # current_phase
        self.assertEqual(json.loads(params[3]), ["phase_0"])  # completed_phases
        self.assertIsNone(params[4])  # failed_step_id
        self.assertEqual(json.loads(params[6]), {"key": "value"})  # metadata
        mock_db.commit.assert_called_once()

    @patch("src.infra.db.get_db")
    def test_defaults_for_optional_params(self, mock_get_db):
        """Defaults: completed_phases=[], metadata={}."""
        from src.infra.db import upsert_workflow_checkpoint

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        self._run(upsert_workflow_checkpoint(
            mission_id=5,
            workflow_name="test_wf",
        ))

        params = mock_db.execute.call_args[0][1]
        self.assertEqual(json.loads(params[3]), [])  # completed_phases
        self.assertEqual(json.loads(params[6]), {})  # metadata


# ── Phase completion tests ────────────────────────────────────────────────


class TestCheckPhaseCompletion(unittest.TestCase):
    """Tests for _check_phase_completion in hooks.py."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    @patch("src.workflows.engine.hooks.upsert_workflow_checkpoint", new_callable=AsyncMock, create=True)
    @patch("src.workflows.engine.hooks.get_workflow_checkpoint", new_callable=AsyncMock, create=True)
    @patch("src.workflows.engine.hooks.get_tasks_for_mission", new_callable=AsyncMock, create=True)
    def test_phase_complete_when_all_done(self, mock_get_tasks, mock_get_cp, mock_upsert):
        """Returns True and checkpoints when all phase tasks are terminal."""
        # We need to patch at the import point inside the function
        with patch("src.infra.db.get_tasks_for_mission", new_callable=AsyncMock) as m_tasks, \
             patch("src.infra.db.get_workflow_checkpoint", new_callable=AsyncMock) as m_cp, \
             patch("src.infra.db.upsert_workflow_checkpoint", new_callable=AsyncMock) as m_upsert:

            from src.workflows.engine.hooks import _check_phase_completion

            m_tasks.return_value = [
                {"id": 1, "status": "completed", "context": json.dumps({"workflow_phase": "phase_1"})},
                {"id": 2, "status": "skipped", "context": json.dumps({"workflow_phase": "phase_1"})},
                {"id": 3, "status": "pending", "context": json.dumps({"workflow_phase": "phase_2"})},
            ]
            m_cp.return_value = {
                "workflow_name": "test_wf",
                "completed_phases": [],
            }

            result = self._run(_check_phase_completion(1, "phase_1"))
            self.assertTrue(result)
            m_upsert.assert_called_once()
            call_kwargs = m_upsert.call_args[1]
            self.assertIn("phase_1", call_kwargs["completed_phases"])

    def test_phase_incomplete_when_tasks_pending(self):
        """Returns False when some phase tasks are still pending."""
        with patch("src.infra.db.get_tasks_for_mission", new_callable=AsyncMock) as m_tasks, \
             patch("src.infra.db.get_workflow_checkpoint", new_callable=AsyncMock), \
             patch("src.infra.db.upsert_workflow_checkpoint", new_callable=AsyncMock) as m_upsert:

            from src.workflows.engine.hooks import _check_phase_completion

            m_tasks.return_value = [
                {"id": 1, "status": "completed", "context": json.dumps({"workflow_phase": "phase_1"})},
                {"id": 2, "status": "pending", "context": json.dumps({"workflow_phase": "phase_1"})},
            ]

            result = self._run(_check_phase_completion(1, "phase_1"))
            self.assertFalse(result)
            m_upsert.assert_not_called()

    def test_phase_no_tasks_returns_false(self):
        """Returns False when no tasks match the phase."""
        with patch("src.infra.db.get_tasks_for_mission", new_callable=AsyncMock) as m_tasks, \
             patch("src.infra.db.get_workflow_checkpoint", new_callable=AsyncMock), \
             patch("src.infra.db.upsert_workflow_checkpoint", new_callable=AsyncMock):

            from src.workflows.engine.hooks import _check_phase_completion

            m_tasks.return_value = [
                {"id": 1, "status": "completed", "context": json.dumps({"workflow_phase": "phase_2"})},
            ]

            result = self._run(_check_phase_completion(1, "phase_1"))
            self.assertFalse(result)

    def test_cancelled_tasks_count_as_terminal(self):
        """Cancelled tasks are considered terminal."""
        with patch("src.infra.db.get_tasks_for_mission", new_callable=AsyncMock) as m_tasks, \
             patch("src.infra.db.get_workflow_checkpoint", new_callable=AsyncMock) as m_cp, \
             patch("src.infra.db.upsert_workflow_checkpoint", new_callable=AsyncMock):

            from src.workflows.engine.hooks import _check_phase_completion

            m_tasks.return_value = [
                {"id": 1, "status": "cancelled", "context": json.dumps({"workflow_phase": "phase_1"})},
            ]
            m_cp.return_value = {"workflow_name": "test_wf", "completed_phases": []}

            result = self._run(_check_phase_completion(1, "phase_1"))
            self.assertTrue(result)


# ── Runner resume tests ──────────────────────────────────────────────────


class TestFindResumable(unittest.TestCase):
    """Tests for WorkflowRunner.find_resumable."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    @patch("src.infra.db.get_workflow_checkpoint", new_callable=AsyncMock)
    @patch("src.infra.db.get_active_missions", new_callable=AsyncMock)
    def test_returns_none_when_no_active_workflow(self, mock_missions, mock_cp):
        """Returns None when no active missions."""
        from src.workflows.engine.runner import WorkflowRunner

        mock_missions.return_value = []
        runner = WorkflowRunner()
        result = self._run(runner.find_resumable("i2p_v2"))
        self.assertIsNone(result)

    @patch("src.infra.db.get_workflow_checkpoint", new_callable=AsyncMock)
    @patch("src.infra.db.get_active_missions", new_callable=AsyncMock)
    def test_returns_mission_id_when_checkpoint_exists(self, mock_missions, mock_cp):
        """Returns mission_id when an active mission has a checkpoint."""
        from src.workflows.engine.runner import WorkflowRunner

        mock_missions.return_value = [
            {"id": 42, "context": json.dumps({"workflow_name": "i2p_v2"})},
        ]
        mock_cp.return_value = {"workflow_name": "i2p_v2", "completed_phases": []}
        runner = WorkflowRunner()
        result = self._run(runner.find_resumable("i2p_v2"))
        self.assertEqual(result, 42)

    @patch("src.infra.db.get_workflow_checkpoint", new_callable=AsyncMock)
    @patch("src.infra.db.get_active_missions", new_callable=AsyncMock)
    def test_skips_missions_without_checkpoint(self, mock_missions, mock_cp):
        """Skips active missions that have no checkpoint."""
        from src.workflows.engine.runner import WorkflowRunner

        mock_missions.return_value = [
            {"id": 10, "context": json.dumps({"workflow_name": "i2p_v2"})},
        ]
        mock_cp.return_value = None
        runner = WorkflowRunner()
        result = self._run(runner.find_resumable("i2p_v2"))
        self.assertIsNone(result)

    @patch("src.infra.db.get_workflow_checkpoint", new_callable=AsyncMock)
    @patch("src.infra.db.get_active_missions", new_callable=AsyncMock)
    def test_skips_wrong_workflow_name(self, mock_missions, mock_cp):
        """Skips active missions with a different workflow_name."""
        from src.workflows.engine.runner import WorkflowRunner

        mock_missions.return_value = [
            {"id": 10, "context": json.dumps({"workflow_name": "other_workflow"})},
        ]
        runner = WorkflowRunner()
        result = self._run(runner.find_resumable("i2p_v2"))
        self.assertIsNone(result)
        mock_cp.assert_not_called()


class TestResume(unittest.TestCase):
    """Tests for WorkflowRunner.resume."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    @patch("src.infra.db.add_task", new_callable=AsyncMock)
    @patch("src.infra.db.update_task", new_callable=AsyncMock)
    @patch("src.infra.db.get_workflow_checkpoint", new_callable=AsyncMock)
    @patch("src.infra.db.get_tasks_for_mission", new_callable=AsyncMock)
    def test_resume_resets_failed_tasks(self, mock_tasks, mock_cp, mock_update, mock_add):
        """Failed tasks are reset to pending."""
        from src.workflows.engine.runner import WorkflowRunner

        mock_cp.return_value = {
            "workflow_name": "i2p_v2",
            "completed_phases": ["phase_1"],
            "metadata": {},
        }
        mock_tasks.return_value = [
            {"id": 10, "status": "completed", "context": json.dumps({"workflow_step_id": "1.1"})},
            {"id": 11, "status": "failed", "context": json.dumps({"workflow_step_id": "1.2"})},
        ]

        mock_wf = MagicMock()
        mock_wf.steps = [
            {"id": "1.1", "name": "Step 1"},
            {"id": "1.2", "name": "Step 2"},
        ]

        runner = WorkflowRunner()

        with patch("src.workflows.engine.runner.load_workflow", return_value=mock_wf):
            with patch.object(runner.artifact_store, "warm_cache", new_callable=AsyncMock):
                result = self._run(runner.resume(1))

        self.assertEqual(result, 1)
        mock_update.assert_called_once_with(11, status="pending", retry_count=0, error=None)

    @patch("src.infra.db.get_workflow_checkpoint", new_callable=AsyncMock)
    @patch("src.infra.db.get_tasks_for_mission", new_callable=AsyncMock)
    def test_resume_raises_on_missing_checkpoint(self, mock_tasks, mock_cp):
        """Raises ValueError if no checkpoint exists."""
        from src.workflows.engine.runner import WorkflowRunner

        mock_tasks.return_value = []
        mock_cp.return_value = None
        runner = WorkflowRunner()

        with self.assertRaises(ValueError) as ctx:
            self._run(runner.resume(999))
        self.assertIn("no workflow checkpoint", str(ctx.exception))

    @patch("src.infra.db.add_task", new_callable=AsyncMock)
    @patch("src.infra.db.update_task", new_callable=AsyncMock)
    @patch("src.infra.db.get_workflow_checkpoint", new_callable=AsyncMock)
    @patch("src.infra.db.get_tasks_for_mission", new_callable=AsyncMock)
    def test_resume_warms_artifact_cache(self, mock_tasks, mock_cp, mock_update, mock_add):
        """warm_cache is called during resume."""
        from src.workflows.engine.runner import WorkflowRunner

        mock_cp.return_value = {
            "workflow_name": "i2p_v2",
            "completed_phases": [],
            "metadata": {},
        }
        mock_tasks.return_value = [
            {"id": 10, "status": "completed", "context": json.dumps({"workflow_step_id": "1.1"})},
        ]

        mock_wf = MagicMock()
        mock_wf.steps = [{"id": "1.1", "name": "Step 1"}]

        runner = WorkflowRunner()
        mock_warm = AsyncMock()

        with patch("src.workflows.engine.runner.load_workflow", return_value=mock_wf):
            with patch.object(runner.artifact_store, "warm_cache", mock_warm):
                self._run(runner.resume(1))

        mock_warm.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
