"""The object/array schema text-fallback must NOT run on a markdown artifact.

Mission-90 task 567452 [5.0c] user_flow_lock: the step declares
`artifact_schema = {"user_flow": {"type":"object","required_fields":
["surfaces","mermaid_per_surface"]}}` for a MARKDOWN produces
(`.flow/user_flow.md`). When the object value can't be extracted (markdown is
not JSON), `validate_artifact_schema` falls back to a literal substring search
for the field NAMES against the prose — which is meaningless: `mermaid_per_surface`
never appears verbatim in a real flow doc (false-reject), while `per` matches
`persona` (false-pass). Every such step (4.14/5.0c/5.0d/6.5z) carries a
`verify_*_shape` mechanical check that IS the authoritative validator, so the
text-fallback is pure noise. When the artifact is a markdown produces, defer to
that check.
"""
from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from src.workflows.engine.hooks import (
    validate_artifact_schema,
    _post_execute_workflow_step_impl,
)


_USER_FLOW_SCHEMA = {
    "user_flow": {"type": "object",
                  "required_fields": ["surfaces", "mermaid_per_surface"]}
}

# A real, valid flow doc — has surfaces + a mermaid diagram, but of course does
# NOT contain the literal token "mermaid_per_surface".
_FLOW_MD = (
    "---\nmission_id: 90\nsurfaces: [web]\n---\n\n"
    "# HabitHub User Flow\n\n## Web\nUser lands, signs up, onboards.\n\n"
    "```mermaid\nflowchart TD\n  Landing --> Signup --> Home\n```\n"
)


def test_object_text_fallback_false_rejects_markdown_without_flag():
    # Documents the bug: field-NAME substring search rejects a valid flow doc.
    ok, err = validate_artifact_schema(_FLOW_MD, _USER_FLOW_SCHEMA)
    assert not ok
    assert "mermaid_per_surface" in err


def test_markdown_produces_defers_to_verify_shape():
    # With produces_markdown=True the object text-fallback is skipped; the
    # step's verify_*_shape check owns validation.
    ok, err = validate_artifact_schema(
        _FLOW_MD, _USER_FLOW_SCHEMA, produces_markdown=True
    )
    assert ok, err


def test_produces_markdown_still_validates_a_real_json_object_strictly():
    # The flag only bypasses the PROSE fallback. A genuine JSON object that is
    # missing a required field must still fail — produces_markdown must not be
    # a blanket "skip validation".
    schema = {"cfg": {"type": "object", "required_fields": ["a", "b"]}}
    bad = json.dumps({"a": 1})  # missing "b"
    ok, err = validate_artifact_schema(bad, schema, produces_markdown=True)
    assert not ok


def test_array_text_fallback_also_skipped_for_markdown_produces():
    schema = {"items": {"type": "array", "min_items": 5}}
    md = "# Register\n\n- only\n- two items\n"
    ok_no, _ = validate_artifact_schema(md, schema)
    assert not ok_no  # array text-fallback counts 2 < 5 → reject
    ok_yes, err = validate_artifact_schema(md, schema, produces_markdown=True)
    assert ok_yes, err


# ── call-site wiring: the flag is derived from ctx.produces (all *.md) ─────

def _step(produces: list[str]) -> tuple[dict, dict]:
    # A SYNTHETIC step_id (not a live i2p_v3 step): post_execute refreshes the
    # artifact_schema from the live workflow JSON by step_id, so a real id (5.0c
    # is now `markdown` after the m90 fix) would override the object schema this
    # test controls. An unknown id makes the refresh a no-op, isolating the CODE
    # under test (produces_markdown derived from the produces FORM).
    ctx = {
        "is_workflow_step": True, "workflow_step_id": "unit_synthetic_object_md",
        "mission_id": 999,
        "output_artifacts": ["user_flow"], "produces": produces,
        "artifact_schema": _USER_FLOW_SCHEMA,
    }
    task = {"id": 1, "agent_type": "analyst", "context": json.dumps(ctx),
            "mission_id": 999, "worker_attempts": 0}
    result = {"status": "completed", "result": _FLOW_MD}
    return task, result


@pytest.mark.asyncio
async def test_post_execute_markdown_produces_skips_object_text_fallback():
    task, result = _step(["mission_999/.flow/user_flow.md"])
    with patch("src.infra.db.update_task", AsyncMock()), \
         patch("src.infra.db.get_db", AsyncMock(return_value=AsyncMock())), \
         patch("src.workflows.engine.hooks.materialize_produces",
               AsyncMock(return_value=_FLOW_MD)):
        await _post_execute_workflow_step_impl(task, result)
    assert "missing content about" not in (result.get("error") or "")


@pytest.mark.asyncio
async def test_post_execute_non_markdown_produces_still_text_validates():
    # Same object schema, but a .json produces — the text-fallback must still
    # run (the flag activates only when EVERY produces path is *.md).
    task, result = _step(["mission_999/.flow/user_flow.json"])
    with patch("src.infra.db.update_task", AsyncMock()), \
         patch("src.infra.db.get_db", AsyncMock(return_value=AsyncMock())), \
         patch("src.workflows.engine.hooks.materialize_produces",
               AsyncMock(return_value=_FLOW_MD)):
        await _post_execute_workflow_step_impl(task, result)
    assert "missing content about" in (result.get("error") or "")
    assert "mermaid_per_surface" in (result.get("error") or "")


# ── invariant: deferring is only safe because a verify_* check exists ─────────

def test_every_markdown_object_step_declares_a_verify_check():
    """A markdown produces with an object/array schema relies ENTIRELY on its
    verify_* check once the prose text-fallback is skipped. Enforce that every
    such step actually declares one — a future step that forgot would otherwise
    silently lose its only schema gate."""
    path = os.path.join(os.path.dirname(__file__), "..", "src", "workflows",
                        "i2p", "i2p_v3.json")
    wf = json.load(open(path, encoding="utf-8"))
    offenders = []
    for s in wf.get("steps", []):
        produces = s.get("produces") or []
        if not (produces and all(str(p).endswith(".md") for p in produces)):
            continue
        schema = s.get("artifact_schema") or {}
        has_structured = any(
            isinstance(r, dict) and r.get("type") in ("object", "array")
            for r in schema.values()
        )
        if not has_structured:
            continue
        checks = s.get("checks") or []
        if not any(str(c.get("kind", "")).startswith("verify_") for c in checks):
            offenders.append(s.get("id"))
    assert not offenders, (
        "markdown+object/array steps whose only schema gate is now deferred but "
        f"that declare NO verify_* check (silent gate drop): {offenders}"
    )
