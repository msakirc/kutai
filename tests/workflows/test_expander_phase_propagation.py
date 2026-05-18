"""z10-wire-fixes F5 — expander propagates step.phase → context.workflow_phase.

The pacing breakdown / B-table rollup depends on every workflow task row
landing with ``tasks.phase_id`` populated. The chain is:

  workflow step ``phase`` field
   → expander writes ``context.workflow_phase``
   → add_task extracts ``workflow_phase`` into ``tasks.phase_id`` column

Verify the first leg here so a future expander refactor doesn't silently
drop the field (T3A pacing dashboards would render empty phase tables).
"""
from __future__ import annotations

from src.workflows.engine.expander import expand_steps_to_tasks


def test_expander_writes_workflow_phase_into_each_task_context():
    steps = [
        {"id": "1.1", "phase": "phase_1", "name": "a"},
        {"id": "7.3", "phase": "phase_7", "name": "b"},
        {"id": "15.10b_record_demo", "phase": "phase_15", "name": "c"},
    ]
    tasks = expand_steps_to_tasks(steps, mission_id="m-1")
    assert len(tasks) == 3
    for t, expected_phase in zip(tasks, ("phase_1", "phase_7", "phase_15")):
        ctx = t.get("context") or {}
        assert ctx.get("workflow_phase") == expected_phase, (
            f"expander failed to propagate phase for step {t.get('id')}: "
            f"got {ctx.get('workflow_phase')!r}, want {expected_phase!r}"
        )
        # Sanity: step_id round-tripped too — without it the pacing
        # breakdown can't link a task back to a workflow step.
        assert ctx.get("workflow_step_id"), (
            f"expander dropped workflow_step_id for {t}"
        )


def test_expander_defaults_phase_when_missing():
    """Defensive: legacy step dicts without a phase still land in
    phase_0 so the column is never NULL via the expander path."""
    tasks = expand_steps_to_tasks(
        [{"id": "x.0", "name": "legacy"}], mission_id="m-1",
    )
    assert tasks[0]["context"]["workflow_phase"] == "phase_0"
