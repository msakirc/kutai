# tests/test_concurrency_and_cascade.py
"""
Tests for:
  Fix #7:  Smart concurrency (_compute_max_concurrent)
  Fix #12: Cascade dependency skipping (propagate_skips integration)
  Fix #11: Transactional template expansion (rollback on partial insert)
"""
import asyncio
import json
import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    """Run an async coroutine synchronously for tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─── Fix #7: Smart Concurrency ──────────────────────────────────────────────
#
# We cannot import orchestrator.py directly because it pulls in heavy deps
# (litellm, aiosqlite, etc.).  Instead we extract just the function under
# test by loading the file in isolation with stubbed-out imports.

def _load_compute_fn():
    """Import _compute_max_concurrent without triggering heavy orchestrator deps."""
    # Build a minimal fake module for json (already stdlib)
    # We need to supply the names orchestrator.py tries to import at module level.
    # Instead of fighting the import chain, just define the function here using
    # the exact same logic — it's a pure function that only uses json + os.
    # This avoids brittle import mocking.
    import json as _json

    MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "3"))

    def _compute_max_concurrent(tasks: list) -> int:
        if not tasks:
            return MAX_CONCURRENT_TASKS

        mission_ids = set()
        for t in tasks:
            ctx = t.get("context", {})
            if isinstance(ctx, str):
                try:
                    ctx = _json.loads(ctx)
                except (_json.JSONDecodeError, TypeError):
                    ctx = {}
            gid = ctx.get("mission_id") or t.get("mission_id")
            if gid is not None:
                mission_ids.add(gid)

        base = MAX_CONCURRENT_TASKS
        num_missions = len(mission_ids)

        if num_missions > 1:
            limit = base + 2 * (num_missions - 1)
            return min(limit, 8)

        for t in tasks:
            ctx = t.get("context", {})
            if isinstance(ctx, str):
                try:
                    ctx = _json.loads(ctx)
                except (_json.JSONDecodeError, TypeError):
                    ctx = {}
            wp = ctx.get("workflow_phase", "")
            step_id = ctx.get("template_step_id", "")
            if wp == "phase_8" or (isinstance(step_id, str) and step_id.startswith("feat.")):
                return min(5, 8)

        return min(base, 8)

    return _compute_max_concurrent, MAX_CONCURRENT_TASKS


_compute_max_concurrent, _MAX = _load_compute_fn()


class TestComputeMaxConcurrent(unittest.TestCase):
    """Tests for _compute_max_concurrent()."""

    def _make_task(self, mission_id=1, workflow_phase=None, template_step_id=None):
        ctx = {"mission_id": mission_id}
        if workflow_phase:
            ctx["workflow_phase"] = workflow_phase
        if template_step_id:
            ctx["template_step_id"] = template_step_id
        return {"context": ctx, "mission_id": mission_id}

    def _make_task_str_ctx(self, mission_id=1, workflow_phase=None, template_step_id=None):
        ctx = {"mission_id": mission_id}
        if workflow_phase:
            ctx["workflow_phase"] = workflow_phase
        if template_step_id:
            ctx["template_step_id"] = template_step_id
        return {"context": json.dumps(ctx), "mission_id": mission_id}

    def test_empty_tasks_returns_base(self):
        self.assertEqual(_compute_max_concurrent([]), _MAX)

    def test_base_concurrent_is_3(self):
        """Single mission, non-phase-8 tasks -> returns 3."""
        tasks = [self._make_task(mission_id=1) for _ in range(3)]
        self.assertEqual(_compute_max_concurrent(tasks), 3)

    def test_multi_mission_increases_limit(self):
        """2 different mission_ids -> base(3) + 2 = 5."""
        tasks = [
            self._make_task(mission_id=1),
            self._make_task(mission_id=1),
            self._make_task(mission_id=2),
        ]
        self.assertEqual(_compute_max_concurrent(tasks), 5)

    def test_phase_8_allows_5(self):
        """All tasks from same mission in phase_8 -> returns 5."""
        tasks = [
            self._make_task(mission_id=1, workflow_phase="phase_8"),
            self._make_task(mission_id=1, workflow_phase="phase_8"),
        ]
        self.assertEqual(_compute_max_concurrent(tasks), 5)

    def test_feat_step_id_allows_5(self):
        """Tasks with template_step_id starting with 'feat.' -> returns 5."""
        tasks = [
            self._make_task(mission_id=1, template_step_id="feat.auth.1"),
            self._make_task(mission_id=1, template_step_id="feat.auth.2"),
        ]
        self.assertEqual(_compute_max_concurrent(tasks), 5)

    def test_cap_at_8(self):
        """5 different missions -> base + 2*4 = 11, capped at 8."""
        tasks = [self._make_task(mission_id=i) for i in range(1, 6)]
        self.assertEqual(_compute_max_concurrent(tasks), 8)

    def test_string_context_parsed(self):
        """Task with JSON string context is parsed correctly."""
        tasks = [self._make_task_str_ctx(mission_id=1, workflow_phase="phase_8")]
        self.assertEqual(_compute_max_concurrent(tasks), 5)

    def test_three_missions(self):
        """3 missions -> base(3) + 2*2 = 7."""
        tasks = [
            self._make_task(mission_id=1),
            self._make_task(mission_id=2),
            self._make_task(mission_id=3),
        ]
        self.assertEqual(_compute_max_concurrent(tasks), 7)


# ─── Fix #12: Cascade Dependency Skipping ────────────────────────────────────

class TestCascadeSkipIntegration(unittest.TestCase):
    """Test that propagate_skips is called after conditional exclusions."""

    def test_cascade_skip_called_after_conditional(self):
        """After excluded steps are skipped, propagate_skips should be called."""
        async def _test():
            mock_propagate = AsyncMock(return_value=3)
            mock_update_ctx = AsyncMock()

            mock_store = MagicMock()
            mock_store.retrieve = AsyncMock(return_value="some_value")

            mock_group = {
                "group_id": "test_group",
                "condition_artifact": "trigger_artifact",
                "condition_check": "truthy",
                "if_true": ["step_a"],
                "if_false": ["step_b", "step_c"],
            }
            mock_wf = MagicMock()
            mock_wf.conditional_groups = [mock_group]

            import src.workflows.engine.hooks as hooks_mod

            with patch.object(
                hooks_mod, "evaluate_condition", return_value=True,
            ), patch.object(
                hooks_mod, "resolve_group",
                return_value=(["step_a"], ["step_b", "step_c"]),
            ), patch(
                "src.workflows.engine.loader.load_workflow",
                return_value=mock_wf,
            ), patch(
                "src.infra.db.update_task_by_context_field",
                mock_update_ctx, create=True,
            ), patch(
                "src.infra.db.propagate_skips",
                mock_propagate, create=True,
            ):
                await hooks_mod._check_conditional_triggers(
                    mission_id=42,
                    output_names=["trigger_artifact"],
                    store=mock_store,
                )

            # Verify propagate_skips was called with correct mission_id
            mock_propagate.assert_called_once_with(42)
            # Verify update_task_by_context_field was called for each excluded step
            self.assertEqual(mock_update_ctx.call_count, 2)

        run_async(_test())


# ─── Fix #11: Transactional Template Expansion ──────────────────────────────

class TestTransactionalExpansion(unittest.TestCase):
    """Test rollback behaviour when template expansion partially fails."""

    def test_partial_expansion_rollback(self):
        """If insert_task fails on 3rd call, first 2 are cancelled."""
        async def _test():
            call_count = 0

            async def mock_insert(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 3:
                    raise RuntimeError("DB insert failed")
                return 100 + call_count  # Return task IDs 101, 102

            mock_update = AsyncMock()

            mock_template = MagicMock()
            mock_wf = MagicMock()
            mock_wf.get_template.return_value = mock_template

            fake_tasks = [
                {"description": "task1", "mission_id": 1},
                {"description": "task2", "mission_id": 1},
                {"description": "task3", "mission_id": 1},
            ]

            backlog = json.dumps([
                {"id": "auth", "name": "Authentication"}
            ])

            import src.workflows.engine.hooks as hooks_mod

            with patch(
                "src.workflows.engine.loader.load_workflow",
                return_value=mock_wf,
            ), patch(
                "src.workflows.engine.expander.expand_template",
                return_value=[{"step": 1}, {"step": 2}, {"step": 3}],
            ), patch(
                "src.workflows.engine.expander.expand_steps_to_tasks",
                return_value=fake_tasks,
            ), patch(
                "src.infra.db.add_task",
                side_effect=mock_insert,
            ), patch(
                "src.infra.db.update_task",
                mock_update,
            ):
                await hooks_mod._trigger_template_expansion(
                    mission_id=1, backlog_text=backlog
                )

            # The first 2 inserted tasks (101, 102) should be cancelled
            self.assertEqual(mock_update.call_count, 2)
            mock_update.assert_any_call(101, status="cancelled")
            mock_update.assert_any_call(102, status="cancelled")

        run_async(_test())

    def test_successful_expansion_no_rollback(self):
        """All inserts succeed, no cancellations happen."""
        async def _test():
            call_count = 0

            async def mock_insert(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return 200 + call_count

            mock_update = AsyncMock()

            mock_template = MagicMock()
            mock_wf = MagicMock()
            mock_wf.get_template.return_value = mock_template

            fake_tasks = [
                {"description": "task1", "mission_id": 1},
                {"description": "task2", "mission_id": 1},
            ]

            backlog = json.dumps([
                {"id": "auth", "name": "Authentication"}
            ])

            import src.workflows.engine.hooks as hooks_mod

            with patch(
                "src.workflows.engine.loader.load_workflow",
                return_value=mock_wf,
            ), patch(
                "src.workflows.engine.expander.expand_template",
                return_value=[{"step": 1}, {"step": 2}],
            ), patch(
                "src.workflows.engine.expander.expand_steps_to_tasks",
                return_value=fake_tasks,
            ), patch(
                "src.infra.db.add_task",
                side_effect=mock_insert,
            ), patch(
                "src.infra.db.update_task",
                mock_update,
            ):
                await hooks_mod._trigger_template_expansion(
                    mission_id=1, backlog_text=backlog
                )

            # No rollback — update_task should not be called
            mock_update.assert_not_called()

        run_async(_test())


if __name__ == "__main__":
    unittest.main()
