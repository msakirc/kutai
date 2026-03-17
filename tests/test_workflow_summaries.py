"""Tests for phase summary generation and injection (Gap 2).

Covers:
- get_phase_summaries retrieval
- _generate_phase_summary artifact creation
- Phase summary injection in pre_execute_workflow_step
"""

import asyncio
import json
import unittest

from src.workflows.engine.artifacts import ArtifactStore, get_phase_summaries


class TestGetPhaseSummaries(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.store = ArtifactStore(use_db=False)

    def tearDown(self):
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_returns_empty_for_phase_0(self):
        result = self._run(get_phase_summaries(self.store, 1, "phase_0"))
        self.assertEqual(result, {})

    def test_returns_phase_neg1_summary(self):
        self._run(self.store.store(1, "phase_-1_summary", "Onboarding summary"))
        result = self._run(get_phase_summaries(self.store, 1, "phase_0"))
        self.assertEqual(result, {"phase_-1_summary": "Onboarding summary"})

    def test_returns_summaries_for_earlier_phases(self):
        self._run(self.store.store(1, "phase_0_summary", "Phase 0 done"))
        self._run(self.store.store(1, "phase_1_summary", "Phase 1 done"))
        self._run(self.store.store(1, "phase_2_summary", "Phase 2 done"))
        self._run(self.store.store(1, "phase_3_summary", "Phase 3 done"))
        result = self._run(get_phase_summaries(self.store, 1, "phase_3"))
        self.assertIn("phase_0_summary", result)
        self.assertIn("phase_1_summary", result)
        self.assertIn("phase_2_summary", result)
        self.assertNotIn("phase_3_summary", result)

    def test_skips_missing_summaries(self):
        self._run(self.store.store(1, "phase_1_summary", "Phase 1 done"))
        result = self._run(get_phase_summaries(self.store, 1, "phase_3"))
        self.assertEqual(len(result), 1)
        self.assertIn("phase_1_summary", result)

    def test_returns_empty_for_invalid_phase(self):
        result = self._run(get_phase_summaries(self.store, 1, "invalid"))
        self.assertEqual(result, {})


class TestGeneratePhaseSummary(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_generates_summary_artifact(self):
        from src.workflows.engine.hooks import _generate_phase_summary, get_artifact_store
        store = get_artifact_store()
        self._run(store.store(1, "spec_doc", "The full specification document"))
        phase_tasks = [{"id": 10, "status": "completed",
            "context": json.dumps({"workflow_phase": "phase_1", "output_artifacts": ["spec_doc"]})}]
        self._run(_generate_phase_summary(1, "phase_1", phase_tasks))
        summary = self._run(store.retrieve(1, "phase_1_summary"))
        self.assertIsNotNone(summary)
        self.assertIn("Phase 1", summary)
        self.assertIn("spec_doc", summary)

    def test_generates_summary_with_no_outputs(self):
        from src.workflows.engine.hooks import _generate_phase_summary, get_artifact_store
        store = get_artifact_store()
        phase_tasks = [{"id": 11, "status": "completed",
            "context": json.dumps({"workflow_phase": "phase_2"})}]
        self._run(_generate_phase_summary(1, "phase_2", phase_tasks))
        summary = self._run(store.retrieve(1, "phase_2_summary"))
        self.assertIsNotNone(summary)
        self.assertIn("Phase 2", summary)

    def test_truncates_long_artifacts(self):
        from src.workflows.engine.hooks import _generate_phase_summary, get_artifact_store
        store = get_artifact_store()
        self._run(store.store(2, "long_doc", "x" * 500))
        phase_tasks = [{"id": 12, "status": "completed",
            "context": json.dumps({"workflow_phase": "phase_1", "output_artifacts": ["long_doc"]})}]
        self._run(_generate_phase_summary(2, "phase_1", phase_tasks))
        summary = self._run(store.retrieve(2, "phase_1_summary"))
        self.assertIn("...", summary)
        self.assertLess(len(summary), 500)


class TestPhaseSummaryInjection(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_injects_phase_summaries(self):
        from src.workflows.engine.hooks import pre_execute_workflow_step, get_artifact_store
        store = get_artifact_store()
        goal_id = 200
        self._run(store.store(goal_id, "phase_0_summary", "Phase 0 decisions: use React"))
        task = {"description": "Build the frontend",
            "context": json.dumps({"is_workflow_step": True, "goal_id": goal_id,
                "workflow_phase": "phase_1", "input_artifacts": []})}
        result = self._run(pre_execute_workflow_step(task))
        self.assertIn("Phase 0 decisions", result["description"])

    def test_no_injection_when_no_summaries(self):
        from src.workflows.engine.hooks import pre_execute_workflow_step
        task = {"description": "Initial step",
            "context": json.dumps({"is_workflow_step": True, "goal_id": 201,
                "workflow_phase": "phase_0", "input_artifacts": []})}
        result = self._run(pre_execute_workflow_step(task))
        self.assertIn("Initial step", result["description"])


if __name__ == "__main__":
    unittest.main()
