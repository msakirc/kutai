"""Unit tests for coulson.skill_render."""
from coulson import skill_render


def test_render_empty_envelope_is_empty():
    assert skill_render.render_skill_envelope([]) == ""


def test_render_prose_inject_block():
    env = [{
        "artifact_id": 1, "name": "anthropics-pdf",
        "exposure_class": "inject", "applies_to": "execution",
        "render": "prose",
        "payload": {"body": "Use the PDF skill to extract text.",
                    "kind": "prompt_skill"},
        "confidence": 0.8,
    }]
    block = skill_render.render_skill_envelope(env)
    assert "## Relevant Skills from Library" in block
    assert "anthropics-pdf" in block
    assert "Use the PDF skill to extract text." in block


def test_render_prebind_shows_concrete_call():
    env = [{
        "artifact_id": 2, "name": "cc-pypackage",
        "exposure_class": "inject", "applies_to": "execution",
        "render": "prebind",
        "payload": {"body": "Scaffold a Python package.",
                    "kind": "shell_recipe",
                    "bound_args": {"project_name": "wt"}},
        "confidence": 0.7,
    }]
    block = skill_render.render_skill_envelope(env)
    assert "cc-pypackage" in block
    # prebind renders the concrete call with bound args.
    assert "project_name" in block and "wt" in block


def test_render_filters_to_execution_only():
    env = [{
        "artifact_id": 3, "name": "rubric-x", "exposure_class": "inject",
        "applies_to": "grading", "render": "prose",
        "payload": {"body": "grading rubric"}, "confidence": 0.9,
    }]
    # grading-tagged skills are not for the agent prompt (Phase 2: none
    # exist, but the filter must hold).
    assert skill_render.render_skill_envelope(env) == ""


def test_render_skips_tool_class():
    env = [{
        "artifact_id": 4, "name": "api-coingecko", "exposure_class": "tool",
        "applies_to": "execution", "render": "prose",
        "payload": {"body": "x"}, "confidence": 0.6,
    }]
    # tool-class entries are not prose — they feed allowed_tools, not text.
    assert skill_render.render_skill_envelope(env) == ""


def test_tool_names_from_envelope():
    env = [
        {"artifact_id": 5, "name": "api-coingecko", "exposure_class": "tool",
         "applies_to": "execution", "render": "prose",
         "payload": {"body": "x"}, "confidence": 0.6},
        {"artifact_id": 6, "name": "anthropics-pdf",
         "exposure_class": "inject", "applies_to": "execution",
         "render": "prose", "payload": {"body": "x"}, "confidence": 0.8},
    ]
    tools = skill_render.tool_names_from_envelope(env)
    assert tools == ["api-coingecko"]
