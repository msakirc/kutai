"""Tests for workflow progress tracking and status reporting."""

import json
import unittest

from src.workflows.engine.status import (
    PHASE_NAMES,
    compute_phase_progress,
    format_status_message,
)


class TestComputePhaseProgress(unittest.TestCase):
    """Tests for compute_phase_progress."""

    def test_compute_phase_progress(self):
        """Counts completed/total per phase correctly."""
        tasks = [
            {"context": {"workflow_phase": "phase_0", "status": "completed"}},
            {"context": {"workflow_phase": "phase_0", "status": "completed"}},
            {"context": {"workflow_phase": "phase_0", "status": "pending"}},
            {"context": {"workflow_phase": "phase_1", "status": "pending"}},
        ]
        progress = compute_phase_progress(tasks)
        self.assertEqual(progress["phase_0"]["completed"], 2)
        self.assertEqual(progress["phase_0"]["total"], 3)
        self.assertEqual(progress["phase_0"]["name"], "Idea Capture & Clarification")
        self.assertEqual(progress["phase_1"]["completed"], 0)
        self.assertEqual(progress["phase_1"]["total"], 1)

    def test_compute_handles_string_context(self):
        """Handles context as a JSON string."""
        tasks = [
            {"context": json.dumps({"workflow_phase": "phase_2", "status": "completed"})},
            {"context": json.dumps({"workflow_phase": "phase_2", "status": "pending"})},
        ]
        progress = compute_phase_progress(tasks)
        self.assertEqual(progress["phase_2"]["completed"], 1)
        self.assertEqual(progress["phase_2"]["total"], 2)

    def test_compute_skips_non_workflow_tasks(self):
        """Tasks without workflow_phase are skipped."""
        tasks = [
            {"context": {"status": "completed"}},
            {"context": {"workflow_phase": "phase_3", "status": "completed"}},
            {"context": "not json"},
            {"context": {}},
        ]
        progress = compute_phase_progress(tasks)
        self.assertNotIn("", progress)
        self.assertEqual(len(progress), 1)
        self.assertIn("phase_3", progress)


class TestFormatStatusMessage(unittest.TestCase):
    """Tests for format_status_message."""

    def test_format_status_message(self):
        """Output has header with workflow_id and goal_id, and phase names."""
        progress = {
            "phase_0": {"completed": 1, "total": 2, "name": "Idea Capture & Clarification"},
        }
        msg = format_status_message("wf_abc", 42, progress)
        self.assertIn("wf_abc", msg)
        self.assertIn("42", msg)
        self.assertIn("Idea Capture & Clarification", msg)

    def test_format_completed_phase(self):
        """Completed phase gets checkmark icon."""
        progress = {
            "phase_0": {"completed": 3, "total": 3, "name": "Idea Capture & Clarification"},
        }
        msg = format_status_message("wf_1", 1, progress)
        self.assertIn("\u2705", msg)  # ✅

    def test_format_in_progress_phase(self):
        """Partially completed phase gets spinner icon."""
        progress = {
            "phase_1": {"completed": 1, "total": 3, "name": "Market & Competitive Research"},
        }
        msg = format_status_message("wf_1", 1, progress)
        self.assertIn("\U0001f504", msg)  # 🔄

    def test_format_pending_phase(self):
        """Phase with no completions gets empty icon."""
        progress = {
            "phase_2": {"completed": 0, "total": 5, "name": "Product Strategy & Definition"},
        }
        msg = format_status_message("wf_1", 1, progress)
        self.assertIn("\u2b1c", msg)  # ⬜

    def test_format_phase_sorting(self):
        """Phases are sorted by numeric ID, including negative."""
        progress = {
            "phase_1": {"completed": 0, "total": 1, "name": "Market & Competitive Research"},
            "phase_-1": {"completed": 1, "total": 1, "name": "Existing Project Onboarding"},
            "phase_0": {"completed": 0, "total": 1, "name": "Idea Capture & Clarification"},
        }
        msg = format_status_message("wf_1", 1, progress)
        lines = msg.strip().split("\n")
        # Find lines with phase names and check order
        phase_lines = [l for l in lines if "phase" in l.lower() or "Onboarding" in l or "Capture" in l or "Market" in l]
        idx_neg1 = next(i for i, l in enumerate(phase_lines) if "Onboarding" in l)
        idx_0 = next(i for i, l in enumerate(phase_lines) if "Capture" in l)
        idx_1 = next(i for i, l in enumerate(phase_lines) if "Market" in l)
        self.assertLess(idx_neg1, idx_0)
        self.assertLess(idx_0, idx_1)


class TestPhaseNames(unittest.TestCase):
    """Tests for PHASE_NAMES constant."""

    def test_phase_names_complete(self):
        """All 17 phases (phase_-1 through phase_15) are present."""
        self.assertEqual(len(PHASE_NAMES), 17)
        for i in range(-1, 16):
            self.assertIn(f"phase_{i}", PHASE_NAMES, f"Missing phase_{i}")


if __name__ == "__main__":
    unittest.main()
