"""skills.py shim: envelope-aware find_relevant_skills + byte-identical
injection output."""
import json

import pytest

from src.memory import skills


def test_format_skills_byte_identical_for_legacy_dicts():
    """The legacy formatter output must not shift — the shim keeps
    format_skills_for_prompt verbatim. Frozen golden string."""
    legacy = [{
        "name": "shopping_search",
        "description": "Route shopping queries to the advisor",
        "injection_count": 0,
        "injection_success": 0,
        "strategies": [],
    }]
    out = skills.format_skills_for_prompt(legacy, context_budget=4096)
    expected = (
        "## Relevant Skills from Library\n\n"
        "- shopping_search: Route shopping queries to the advisor "
        "(tools: none, 0% success)\n"
    )
    assert out == expected


@pytest.mark.asyncio
async def test_find_relevant_skills_returns_envelope_inject_slice():
    """When a task carries the Phase 2 envelope, the shim returns the
    inject slice as legacy-shaped skill dicts — no vector search."""
    task = {
        "id": 11,
        "skills": [
            {"artifact_id": 1, "name": "anthropics-pdf",
             "exposure_class": "inject", "applies_to": "execution",
             "render": "prose",
             "payload": {"body": "PDF guidance"}, "confidence": 0.8},
            {"artifact_id": 2, "name": "api-coingecko",
             "exposure_class": "tool", "applies_to": "execution",
             "render": "prose", "payload": {"body": "x"}, "confidence": 0.5},
        ],
    }
    result = await skills.find_relevant_skills("anything", task=task)
    # tool-class excluded; only inject survives.
    assert len(result) == 1
    assert result[0]["name"] == "anthropics-pdf"
    assert result[0]["description"] == "PDF guidance"


@pytest.mark.asyncio
async def test_find_relevant_skills_empty_when_no_envelope():
    """No envelope → shim returns [] (the vector path is retired; coulson
    now drives matching via intersect)."""
    result = await skills.find_relevant_skills("anything", task={"id": 12})
    assert result == []
