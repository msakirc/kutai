"""build_user_context reads task['skills'] envelope, not skills.py."""
import json

import pytest


class _Profile:
    name = "coder"
    allowed_tools = ["read_file", "write_file"]
    max_iterations = 5
    _prompt_version_override = None
    _suppress_clarification = False

    def get_system_prompt(self, task):
        return "You are a coder."


@pytest.mark.asyncio
async def test_context_renders_envelope_inject(monkeypatch):
    from coulson import context

    task = {
        "id": 7, "title": "Build a thing", "description": "do it",
        "agent_type": "coder",
        "context": json.dumps({}),
        "skills": [{
            "artifact_id": 1, "name": "anthropics-pdf",
            "exposure_class": "inject", "applies_to": "execution",
            "render": "prose",
            "payload": {"body": "PDF extraction guidance.",
                        "kind": "prompt_skill"},
            "confidence": 0.8,
        }],
    }
    text, injected_tools = await context.build_user_context(
        _Profile(), task, model_ctx=4096)
    assert "anthropics-pdf" in text
    assert "PDF extraction guidance." in text


@pytest.mark.asyncio
async def test_context_envelope_tool_class_injects_tool(monkeypatch):
    from coulson import context

    task = {
        "id": 8, "title": "Fetch a price", "description": "",
        "agent_type": "researcher", "context": json.dumps({}),
        "skills": [{
            "artifact_id": 5, "name": "api-coingecko",
            "exposure_class": "tool", "applies_to": "execution",
            "render": "prose", "payload": {"body": "x"}, "confidence": 0.6,
        }],
    }
    _, injected_tools = await context.build_user_context(
        _Profile(), task, model_ctx=4096)
    assert "api-coingecko" in injected_tools


@pytest.mark.asyncio
async def test_context_no_envelope_renders_nothing_extra(monkeypatch):
    from coulson import context

    task = {
        "id": 9, "title": "Plain task", "description": "",
        "agent_type": "coder", "context": json.dumps({}),
    }
    text, injected = await context.build_user_context(
        _Profile(), task, model_ctx=4096)
    assert "Relevant Skills from Library" not in text
    assert injected == []


# ---------------------------------------------------------------------------
# P2-2 regression: envelope must be rendered for WORKFLOW-STEP tasks too.
# A task whose title contains a "[X.Y]" step id triggers the exemplar branch
# (if not _step_id: is False) and the `if not _step_id:` guard was silently
# skipping the envelope block — so intersect-matched skills were discarded.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_context_envelope_rendered_for_workflow_step_task(monkeypatch):
    """Skill envelope must appear in the prompt even when the task title carries
    a [X.Y] workflow-step prefix (i.e. extract_step_id returns truthy)."""
    from coulson import context

    # Patch extract_step_id so it finds a step id (simulating a workflow task).
    try:
        import src.memory.workflow_exemplars as _we
        monkeypatch.setattr(_we, "extract_step_id", lambda title: "3.2")
        monkeypatch.setattr(_we, "lookup_exemplars",
                            lambda **kw: None)  # no exemplars — keep it simple
    except Exception:
        pass  # module may not be importable in unit context; that's fine

    task = {
        "id": 42,
        "title": "[3.2] Scaffold the package",   # workflow-step title
        "description": "Build the skeleton",
        "agent_type": "coder",
        "context": json.dumps({}),
        "skills": [{
            "artifact_id": 7,
            "name": "anthropics-python-packaging",
            "exposure_class": "inject",
            "applies_to": "execution",
            "render": "prose",
            "payload": {"body": "Use src layout for Python packages.",
                        "kind": "prompt_skill"},
            "confidence": 0.85,
        }],
    }
    text, _ = await context.build_user_context(_Profile(), task, model_ctx=4096)

    assert "anthropics-python-packaging" in text, (
        "Skill envelope was NOT rendered for workflow-step task — "
        "the 'if not _step_id:' guard is suppressing it"
    )
    assert "Use src layout for Python packages." in text, (
        "Skill body was NOT rendered for workflow-step task"
    )
