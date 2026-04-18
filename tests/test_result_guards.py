"""Unit tests for src/core/result_guards.py.

These are lightweight tests: guards mutate DB/telegram side-effects, which
are all mocked.  The happy/terminal/retry outcomes are what matter.
"""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.result_guards import (
    GuardHandled,
    guard_clarification_suppression,
    guard_pipeline_artifacts,
    guard_subtasks_blocked_for_workflow,
    guard_ungraded_post_hook,
    guard_workflow_step_post_hook,
)


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _task(**over):
    d = {"id": 1, "title": "t", "agent_type": "executor"}
    d.update(over)
    return d


def _self_mock():
    s = MagicMock()
    s.telegram = MagicMock()
    s.telegram.send_notification = AsyncMock()
    s._validate_clarification = AsyncMock(return_value=True)
    s._handle_clarification = AsyncMock()
    s._handle_complete = AsyncMock()
    s._assess_timeout_progress = AsyncMock(return_value=0.0)
    return s


class TestGuardPipelineArtifacts(unittest.TestCase):
    def test_non_pipeline_noop(self):
        s = _self_mock()
        out = run_async(guard_pipeline_artifacts(s, _task(), {}, {}, "executor"))
        self.assertIsNone(out)

    def test_non_workflow_noop(self):
        s = _self_mock()
        with patch("src.workflows.engine.hooks.is_workflow_step", return_value=False):
            out = run_async(guard_pipeline_artifacts(s, _task(), {}, {}, "pipeline"))
        self.assertIsNone(out)


class TestGuardSubtasksBlockedForWorkflow(unittest.TestCase):
    def test_non_workflow_falls_through(self):
        s = _self_mock()
        with patch("src.workflows.engine.hooks.is_workflow_step", return_value=False):
            out = run_async(guard_subtasks_blocked_for_workflow(s, _task(), {}, {"status": "needs_subtasks"}))
        self.assertIsNone(out)

    def test_workflow_step_blocks_and_retries(self):
        s = _self_mock()
        with patch("src.workflows.engine.hooks.is_workflow_step", return_value=True), \
             patch("src.core.result_guards.update_task", new_callable=AsyncMock) as mock_update:
            out = run_async(guard_subtasks_blocked_for_workflow(
                s, _task(), {"is_workflow_step": True}, {"status": "needs_subtasks"}))
        self.assertIsInstance(out, GuardHandled)
        self.assertEqual(out.reason, "subtasks_blocked_for_workflow")
        mock_update.assert_called_once()


class TestGuardClarificationSuppression(unittest.TestCase):
    def test_silent_task_suppressed(self):
        s = _self_mock()
        with patch("src.core.result_guards.update_task", new_callable=AsyncMock) as mock_update:
            out = run_async(guard_clarification_suppression(
                s, _task(), {"silent": True}, {"status": "needs_clarification"}))
        self.assertIsInstance(out, GuardHandled)
        self.assertEqual(out.reason, "clarification_silent")
        mock_update.assert_called_once()

    def test_may_not_need_clarification_retries(self):
        s = _self_mock()
        with patch("src.core.result_guards.update_task", new_callable=AsyncMock) as mock_update:
            out = run_async(guard_clarification_suppression(
                s, _task(), {"may_need_clarification": False},
                {"status": "needs_clarification"}))
        self.assertIsInstance(out, GuardHandled)
        self.assertEqual(out.reason, "clarification_not_allowed")
        mock_update.assert_called_once()

    def test_history_reuses_completed(self):
        s = _self_mock()
        ctx = {"clarification_history": [{"question": "q?", "answer": "a"}]}
        result = {"status": "needs_clarification", "result": ""}
        with patch("src.workflows.engine.hooks.is_workflow_step", return_value=False):
            out = run_async(guard_clarification_suppression(s, _task(), ctx, result))
        self.assertIsInstance(out, GuardHandled)
        self.assertEqual(out.reason, "clarification_history_reused")
        self.assertEqual(result["status"], "completed")
        s._handle_complete.assert_called_once()

    def test_happy_path_falls_through(self):
        s = _self_mock()
        with patch("src.workflows.engine.hooks.is_workflow_step", return_value=False):
            out = run_async(guard_clarification_suppression(
                s, _task(), {}, {"status": "needs_clarification"}))
        self.assertIsNone(out)


class TestGuardUngradedPostHook(unittest.TestCase):
    def test_non_workflow_noop(self):
        s = _self_mock()
        with patch("src.workflows.engine.hooks.is_workflow_step", return_value=False):
            out = run_async(guard_ungraded_post_hook(s, _task(), {}, {"status": "ungraded"}))
        self.assertIsNone(out)


class TestGuardWorkflowStepPostHook(unittest.TestCase):
    def test_non_workflow_noop(self):
        s = _self_mock()
        with patch("src.workflows.engine.hooks.is_workflow_step", return_value=False):
            out = run_async(guard_workflow_step_post_hook(
                s, _task(), {}, {"status": "completed"}))
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
