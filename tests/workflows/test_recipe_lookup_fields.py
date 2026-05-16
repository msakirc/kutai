"""recipe_lookup / recipe_hint carry from step JSON into task context."""
import json

import pytest

from src.workflows.engine.loader import load_workflow
from src.workflows.engine.expander import build_step_context


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
