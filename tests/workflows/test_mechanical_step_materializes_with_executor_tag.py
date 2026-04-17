"""Verify the workflow engine propagates executor='mechanical' into task context.

Because tasks are serialized through the DB (no executor column), the expander
must stash executor + payload inside the task context. The orchestrator later
reads these from context to route to salako.
"""

from src.workflows.engine.expander import expand_steps_to_tasks


def _steps_with_mechanical() -> list[dict]:
    return [
        {
            "id": "7.3",
            "phase": "phase_7",
            "name": "backend_scaffold",
            "agent": "coder",
            "instruction": "scaffold it",
        },
        {
            "id": "7.3.git_commit",
            "phase": "phase_7",
            "name": "backend_scaffold_git_commit",
            "agent": "mechanical",
            "executor": "mechanical",
            "payload": {"action": "git_commit"},
            "depends_on": ["7.3"],
            "instruction": "auto-commit",
        },
    ]


def test_mechanical_step_materializes_with_executor_and_payload_in_context():
    tasks = expand_steps_to_tasks(_steps_with_mechanical(), mission_id="m1")
    mech_tasks = [t for t in tasks if t["agent_type"] == "mechanical"]
    assert len(mech_tasks) == 1
    mt = mech_tasks[0]
    ctx = mt["context"]
    assert ctx.get("executor") == "mechanical"
    assert ctx.get("payload") == {"action": "git_commit"}


def test_non_mechanical_step_has_no_executor_tag_in_context():
    tasks = expand_steps_to_tasks(_steps_with_mechanical(), mission_id="m1")
    coder_tasks = [t for t in tasks if t["agent_type"] == "coder"]
    assert len(coder_tasks) == 1
    assert "executor" not in coder_tasks[0]["context"]
    assert "payload" not in coder_tasks[0]["context"]
