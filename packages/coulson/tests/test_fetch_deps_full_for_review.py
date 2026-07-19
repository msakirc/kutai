"""Review steps must fetch FULL input artifacts, not the lossy `_summary` form.

Mission 90 task 567399 ([1.13] research_quality_review): the reviewer was fed
`_summary` stubs for all 6 input artifacts (charter cut after 3/5 solutions,
prior-art missing most of 20 entries, etc.), then ran 17 line-by-line checks
against them and hallucinated 7/9 "missing content" blockers — halting a mission
whose full artifacts were actually complete. The reviewer (whose whole job is to
verify the full documents) must receive the full form; other agents keep the
summary preference (general context-saving).
"""
from __future__ import annotations

import json
import pytest


class _Store:
    async def retrieve(self, mid, name):
        if name.endswith("_summary"):
            return "SUMMARY-STUB (lossy)"
        return "FULL-DOC-BODY with every section present"


class _Profile:
    def __init__(self, name):
        self.name = name
        self.allowed_tools = []


def _task():
    ctx = {
        "is_workflow_step": True,
        "input_artifacts": ["market_research_report"],
        "mission_id": 90,
    }
    return {"id": 1, "mission_id": 90, "context": json.dumps(ctx)}


@pytest.mark.asyncio
async def test_reviewer_gets_full_artifact_not_summary(monkeypatch):
    monkeypatch.setattr(
        "src.workflows.engine.hooks.get_artifact_store", lambda: _Store()
    )
    from coulson.context import fetch_deps

    out = await fetch_deps(_Profile("reviewer"), _task(), max_tokens=8000)
    assert "FULL-DOC-BODY" in out          # full form injected
    assert "SUMMARY-STUB" not in out        # lossy summary NOT used
    assert "(full)" in out                  # form label reflects full


@pytest.mark.asyncio
async def test_reviewer_gets_full_even_when_input_declares_summary_name(monkeypatch):
    """When a workflow step declares the SUMMARY form in input_artifacts (e.g.
    ``requirements_spec_summary``), the reviewer must still receive the full
    bare artifact. Otherwise ``_wants_full`` fetches the summary-named stub
    verbatim and the reviewer hallucinates 'missing' content (m90 567426:
    requirements_spec_summary=836 chars vs full requirements_spec=9769 →
    halted a COMPLETE spec with bogus 'FR/User-Stories/Traceability tables
    empty' blockers)."""
    monkeypatch.setattr(
        "src.workflows.engine.hooks.get_artifact_store", lambda: _Store()
    )
    from coulson.context import fetch_deps

    ctx = {
        "is_workflow_step": True,
        "input_artifacts": ["requirements_spec_summary"],
        "mission_id": 90,
    }
    task = {"id": 2, "mission_id": 90, "context": json.dumps(ctx)}
    out = await fetch_deps(_Profile("reviewer"), task, max_tokens=8000)
    assert "FULL-DOC-BODY" in out
    assert "SUMMARY-STUB" not in out


@pytest.mark.asyncio
async def test_non_reviewer_still_prefers_summary(monkeypatch):
    monkeypatch.setattr(
        "src.workflows.engine.hooks.get_artifact_store", lambda: _Store()
    )
    from coulson.context import fetch_deps

    out = await fetch_deps(_Profile("analyst"), _task(), max_tokens=8000)
    assert "SUMMARY-STUB" in out            # summary preference unchanged
    assert "(summary)" in out
