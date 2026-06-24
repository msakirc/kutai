"""_apply_auto_strip: write tools are stripped only for STRUCTURED schemas.

Task #524995, [0.0c] interview_script_generation: a markdown produce-step
(agent=analyst) had write_file auto-stripped because it carried an
artifact_schema. The analyst then either looped on the unavailable
write_file or emitted a narration-wrapped ('## Analysis …') final_answer
that poisoned the materialized file → schema/shape gate DLQ. The model
demonstrably writes a CLEAN file via write_file. So: keep write tools for
free-form (markdown/string) schemas; strip only for object/array.
"""
from __future__ import annotations

import types

from coulson import _apply_auto_strip


def _prof(tools):
    return types.SimpleNamespace(
        name="analyst", allowed_tools=list(tools),
        _original_allowed_tools=None, _tools_overridden=False,
    )


def test_markdown_schema_keeps_write_file():
    prof = _prof(["write_file"])
    ctx = {"artifact_schema": {"interview_script": {"type": "markdown"}}}
    _apply_auto_strip(prof, ctx)
    assert "write_file" in prof.allowed_tools


def test_string_schema_keeps_write_file():
    prof = _prof(["write_file", "read_file"])
    ctx = {"artifact_schema": {"blurb": {"type": "string"}}}
    _apply_auto_strip(prof, ctx)
    assert "write_file" in prof.allowed_tools


def test_object_schema_still_strips_write_file():
    prof = _prof(["write_file", "read_file"])
    ctx = {"artifact_schema": {"prior_art_queries": {"type": "object"}}}
    _apply_auto_strip(prof, ctx)
    assert "write_file" not in prof.allowed_tools
    assert "read_file" in prof.allowed_tools


def test_typeless_schema_still_strips_write_file():
    """Object schemas frequently omit an explicit type — default structured."""
    prof = _prof(["write_file", "read_file"])
    ctx = {"artifact_schema": {"draft": {"required_fields": ["x"]}}}
    _apply_auto_strip(prof, ctx)
    assert "write_file" not in prof.allowed_tools


def test_mixed_schema_with_any_freeform_keeps_write_file():
    prof = _prof(["write_file"])
    ctx = {"artifact_schema": {
        "data": {"type": "object"},
        "report": {"type": "markdown"},
    }}
    _apply_auto_strip(prof, ctx)
    assert "write_file" in prof.allowed_tools


def test_allow_write_optout_keeps_write_file_for_object():
    prof = _prof(["write_file"])
    ctx = {"artifact_schema": {"x": {"type": "object"}}, "_allow_write_tools": True}
    _apply_auto_strip(prof, ctx)
    assert "write_file" in prof.allowed_tools
