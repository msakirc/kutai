"""test_workflow_pipeline.py — Integration tests for the workflow/mission pipeline.

Tests:
- Mission creation from a user-style message
- Workflow task expansion (idea → subtasks via workflow JSON)
- mission_id propagation into task context
- Workflow checkpoint save/restore
- Workflow JSON loading
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Workflow JSON loading
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestWorkflowJsonLoading:
    """The i2p workflow JSON files parse correctly."""

    def test_load_i2p_v3(self):
        """i2p_v3.json is valid JSON and has expected structure."""
        workflow_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "src", "workflows", "i2p", "i2p_v3.json"
        )
        assert os.path.exists(workflow_path), f"Workflow file missing: {workflow_path}"

        with open(workflow_path, encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict), "Workflow JSON should be a dict"
        assert "phases" in data or "steps" in data or "workflow" in data, (
            f"Workflow JSON lacks expected keys. Keys found: {list(data.keys())}"
        )


# ---------------------------------------------------------------------------
# Mission creation and task attachment
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMissionCreation:
    """Mission → task creation pipeline at the DB level."""

    def test_create_mission_and_subtasks(self, temp_db):
        """A mission can be created and subtasks added atomically."""
        from src.infra.db import add_mission, add_subtasks_atomically, get_tasks_for_mission, add_task

        async def _run():
            mid = await add_mission(
                title="Build a todo app",
                description="Idea-to-product pipeline test",
                priority=7,
                workflow="i2p",
            )

            # Add a root task to act as the parent
            root_task_id = await add_task(
                title="Root planning task",
                description="Decompose the idea",
                mission_id=mid,
                agent_type="planner",
            )

            subtasks = [
                {
                    "title": "Research market",
                    "description": "Understand the todo app market",
                    "agent_type": "researcher",
                    "tier": "auto",
                    "priority": 5,
                    "depends_on": [],
                },
                {
                    "title": "Design architecture",
                    "description": "Design the system architecture",
                    "agent_type": "architect",
                    "tier": "auto",
                    "priority": 5,
                    "depends_on": [],
                },
            ]

            created_ids = await add_subtasks_atomically(
                parent_task_id=root_task_id,
                subtasks=subtasks,
                mission_id=mid,
            )

            assert len(created_ids) == 2
            assert all(tid != -1 for tid in created_ids)

            tasks = await get_tasks_for_mission(mid)
            # Should have root task + 2 subtasks
            assert len(tasks) == 3

        run_async(_run())

    def test_mission_id_in_task_context(self, temp_db):
        """mission_id is accessible from task context (propagation test)."""
        from src.infra.db import add_mission, add_task, get_task
        import json

        async def _run():
            mid = await add_mission(
                title="Mission context test",
                description="",
            )
            ctx = {"mission_id": mid, "user_id": 42, "step": "planning"}
            tid = await add_task(
                title="Context propagation task",
                description="test",
                mission_id=mid,
                agent_type="executor",
                context=ctx,
            )

            task = await get_task(tid)
            # mission_id should be a column
            assert task["mission_id"] == mid

            # context should also contain mission_id for agents that read it
            stored_ctx = task.get("context")
            if isinstance(stored_ctx, str):
                stored_ctx = json.loads(stored_ctx)
            assert stored_ctx.get("mission_id") == mid, (
                "mission_id must be present in task context so agents "
                "can propagate it to their subtasks (regression fix test)"
            )

        run_async(_run())

    def test_get_active_missions(self, temp_db):
        """get_active_missions returns only missions with status='active'."""
        from src.infra.db import add_mission, get_active_missions, update_mission

        async def _run():
            mid1 = await add_mission(title="Active mission 1", description="")
            mid2 = await add_mission(title="Active mission 2", description="")
            mid3 = await add_mission(title="Completed mission", description="")

            # Complete the third
            await update_mission(mid3, status="completed")

            active = await get_active_missions()
            active_ids = [m["id"] for m in active]
            assert mid1 in active_ids
            assert mid2 in active_ids
            assert mid3 not in active_ids

        run_async(_run())


# ---------------------------------------------------------------------------
# Workflow checkpoint save/restore
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestWorkflowCheckpoints:
    """Workflow checkpoint mechanism (save and restore state)."""

    def test_save_and_load_task_checkpoint(self, temp_db):
        """A task checkpoint can be saved and retrieved."""
        from src.infra.db import add_task, save_task_checkpoint, load_task_checkpoint

        async def _run():
            tid = await add_task(
                title="Checkpoint test task",
                description="",
                agent_type="coder",
            )

            checkpoint_data = {
                "iteration": 3,
                "messages": [
                    {"role": "system", "content": "You are a coder"},
                    {"role": "user", "content": "Write hello world"},
                    {"role": "assistant", "content": '{"action": "tool_call", "tool": "shell"}'},
                ],
                "tools_used": True,
                "total_cost": 0.001,
                "used_model": "local/test-model",
            }

            await save_task_checkpoint(tid, checkpoint_data)
            loaded = await load_task_checkpoint(tid)

            assert loaded is not None
            assert loaded["iteration"] == 3
            assert loaded["tools_used"] is True
            assert len(loaded["messages"]) == 3
            assert loaded["used_model"] == "local/test-model"

        run_async(_run())

    def test_clear_task_checkpoint(self, temp_db):
        """Checkpoint can be cleared after task completion."""
        from src.infra.db import add_task, save_task_checkpoint, load_task_checkpoint, clear_task_checkpoint

        async def _run():
            tid = await add_task(
                title="Clear checkpoint test",
                description="",
                agent_type="executor",
            )

            await save_task_checkpoint(tid, {"iteration": 1, "messages": []})
            assert await load_task_checkpoint(tid) is not None

            await clear_task_checkpoint(tid)
            assert await load_task_checkpoint(tid) is None

        run_async(_run())

    def test_checkpoint_nonexistent_task(self, temp_db):
        """Loading a checkpoint for a task with no checkpoint returns None."""
        from src.infra.db import add_task, load_task_checkpoint

        async def _run():
            tid = await add_task(
                title="No checkpoint",
                description="",
                agent_type="executor",
            )
            result = await load_task_checkpoint(tid)
            assert result is None

        run_async(_run())


# ---------------------------------------------------------------------------
# Workflow task expansion (integration-level, no LLM)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestWorkflowExpansion:
    """Workflow step expansion at the runner utility level."""

    def test_resolve_dependencies_basic(self):
        """Workflow step IDs are correctly mapped to DB task IDs."""
        from src.workflows.engine.runner import resolve_dependencies

        step_to_task = {"0.1": 10, "0.2": 20, "1.1": 30}
        result = resolve_dependencies(["0.1", "1.1"], step_to_task)
        assert result == [10, 30]

    def test_resolve_dependencies_skips_missing(self):
        """Unknown step IDs are skipped with a warning (not an error)."""
        from src.workflows.engine.runner import resolve_dependencies

        step_to_task = {"0.1": 10}
        result = resolve_dependencies(["0.1", "MISSING_STEP"], step_to_task)
        assert result == [10]

    def test_build_step_description_combines_content(self):
        """build_step_description includes instruction and artifact content."""
        from src.workflows.engine.runner import build_step_description

        desc = build_step_description(
            instruction="Analyze the idea carefully.",
            input_artifacts=["idea", "market_data"],
            artifact_contents={
                "idea": "Build a smart calendar app",
                "market_data": "Calendar market: $5B TAM",
            },
        )
        assert "Analyze the idea carefully." in desc
        assert "Build a smart calendar app" in desc
        assert "Calendar market" in desc
