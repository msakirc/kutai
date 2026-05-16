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
