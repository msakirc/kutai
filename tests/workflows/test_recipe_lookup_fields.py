"""recipe_lookup / recipe_hint carry from step JSON into task context."""
import json

import pytest

from src.workflows.engine.loader import load_workflow
from src.workflows.engine.expander import build_step_context, expand_steps_to_tasks


def test_loader_preserves_recipe_fields():
    """A step declaring recipe_lookup/recipe_hint keeps them on get_step."""
    wf = load_workflow("i2p_v3")
    # Find any step that declares recipe_lookup explicitly.
    found = None
    for sid in wf.all_step_ids():
        step = wf.get_step(sid)
        if step and "recipe_lookup" in step:
            found = step
            break
    assert found is not None, "no i2p step declares recipe_lookup"
    assert isinstance(found["recipe_lookup"], bool)


def test_expander_propagates_recipe_lookup_true():
    step = {
        "id": "3.2", "agent": "coder", "title": "Scaffold",
        "recipe_lookup": True, "recipe_hint": "python package scaffold",
    }
    ctx = build_step_context(step, mission_id=1)
    assert ctx["recipe_lookup"] is True
    assert ctx["recipe_hint"] == "python package scaffold"


def test_expander_default_recipe_lookup_for_scaffold_phase():
    """A scaffold-phase step with no explicit flag defaults to True."""
    step = {"id": "3.1", "agent": "coder", "title": "Scaffold the repo",
            "phase": "scaffold"}
    ctx = build_step_context(step, mission_id=1)
    assert ctx["recipe_lookup"] is True


def test_expander_default_recipe_lookup_false_for_design_phase():
    step = {"id": "2.1", "agent": "architect", "title": "Design architecture",
            "phase": "architecture"}
    ctx = build_step_context(step, mission_id=1)
    assert ctx["recipe_lookup"] is False


def test_expander_explicit_flag_overrides_phase_default():
    step = {"id": "2.9", "agent": "architect", "title": "x",
            "phase": "architecture", "recipe_lookup": True}
    ctx = build_step_context(step, mission_id=1)
    assert ctx["recipe_lookup"] is True


# ---------------------------------------------------------------------------
# P2-4 regression: Cause 2 — expander must thread mission payload into
# per-step task context under a 'payload' key.
# ---------------------------------------------------------------------------

def test_expander_threads_mission_payload_into_step_context():
    """expand_steps_to_tasks must put initial_context (mission payload) under
    context['payload'] so flash._build_task_ctx can expose task.payload.*.

    This FAILS before the Cause-2 fix because the expander only puts the
    initial_context into context['workflow_context'] — the 'payload' key is
    absent and ctx.get('payload') returns {} in flash._build_task_ctx.
    """
    step = {
        "id": "3.2",
        "name": "Scaffold the Python package",
        "instruction": "Create the package",
        "agent": "coder",
        "phase": "scaffold",
    }
    mission_payload = {"project_name": "wt", "author_name": "alice"}

    tasks = expand_steps_to_tasks([step], mission_id=1,
                                  initial_context=mission_payload)
    assert tasks, "expected at least one task from expansion"
    ctx = tasks[0]["context"]

    assert "payload" in ctx, (
        f"context is missing 'payload' key; keys={list(ctx.keys())} — "
        "Cause 2 unfixed: expander never threads mission payload into step context"
    )
    assert ctx["payload"] == mission_payload, (
        f"context['payload']={ctx['payload']!r} != mission_payload={mission_payload!r}"
    )
