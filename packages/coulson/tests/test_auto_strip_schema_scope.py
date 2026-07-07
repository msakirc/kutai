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

from coulson import _apply_auto_strip, _ensure_write_tools_for_markdown_produces


def _prof(tools):
    return types.SimpleNamespace(
        name="analyst",
        allowed_tools=(None if tools is None else list(tools)),
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


# ── produces-form is the authoritative artifact-form signal (m90 narration-clobber) ──
# The schema ``type`` is a VALIDATION concern (which fields/shape to check); the
# ``produces`` extension is the AUTHORING concern (how the agent emits the file).
# A step may carry an OBJECT/ARRAY schema (to validate structured markdown
# frontmatter fields like surfaces/mermaid_per_surface) yet still author a ``.md``
# document. Conflating the two stripped write_file on 4 analyst steps
# (5.0c user_flow, 5.0d screen_inventory/shared_shell, 4.14 register, 6.5z
# premortem), forcing the narration-prone analyst down the final_answer path →
# "## Analysis …" wrapper clobbered the materialized file. A ``.md`` produces
# must keep write_file regardless of schema type.


def test_md_produces_object_schema_keeps_write_file():
    """m90 5.0c user_flow — object schema validates the doc's frontmatter, but
    the artifact is a markdown file the analyst authors via write_file."""
    prof = _prof(["write_file", "read_file"])
    ctx = {
        "artifact_schema": {"user_flow": {"type": "object",
                                          "required_fields": ["surfaces"]}},
        "produces": ["mission_90/.flow/user_flow.md"],
    }
    _apply_auto_strip(prof, ctx)
    assert "write_file" in prof.allowed_tools


def test_md_produces_array_schema_keeps_write_file():
    """m90 4.14 register.md — array schema + .md produces, same class."""
    prof = _prof(["write_file"])
    ctx = {
        "artifact_schema": {"register": {"type": "array"}},
        "produces": ["mission_90/.adr/register.md"],
    }
    _apply_auto_strip(prof, ctx)
    assert "write_file" in prof.allowed_tools


def test_object_schema_no_produces_still_strips_write_file():
    """Regression guard: an object schema with NO ``.md`` produces (a verdict/
    reviewer step whose final_answer JSON IS the artifact) keeps stripping."""
    prof = _prof(["write_file", "read_file"])
    ctx = {"artifact_schema": {"verdict": {"type": "object"}}}  # no produces
    _apply_auto_strip(prof, ctx)
    assert "write_file" not in prof.allowed_tools


def test_json_produces_object_schema_still_strips_write_file():
    """Regression guard: a ``.json`` produces is structured — the returned JSON
    IS the artifact ([0.0a] intake_todo_draft). Only ``.md`` flips to free-form."""
    prof = _prof(["write_file", "read_file"])
    ctx = {
        "artifact_schema": {"draft": {"type": "object"}},
        "produces": ["mission_90/.intake/draft.json"],
    }
    _apply_auto_strip(prof, ctx)
    assert "write_file" not in prof.allowed_tools


# ── INVARIANT: a .md produces MUST have write_file, whatever tools_hint did ──
# tools_hint OVERRIDES allowed_tools (``profile.allowed_tools = tools_hint``);
# tools_hint=[] leaves the agent with NO tools, so it cannot author its declared
# .md file and dumps the doc into final_answer, where a narration-prone analyst
# wraps it → materialized clobber (m90 4.14/5.0c/5.0d all shipped tools_hint=[]).
# auto_strip only REMOVES tools, so it cannot restore write_file. This final
# reconciliation guarantees the produces contract: you can't declare a file you
# have no tool to write. Runs AFTER _refresh_workflow_step_config so it sees the
# live produces list.


def test_md_produces_markdown_schema_restores_write_file_after_empty_tools_hint():
    """m90 5.0c/5.0d/6.5z — tools_hint=[] stripped every tool; a markdown-AUTHORING
    .md produces must restore write_file so the analyst can author a clean file."""
    prof = _prof([])                       # what _apply_tools_hint([]) leaves
    ctx = {"produces": ["mission_90/.flow/user_flow.md"],
           "artifact_schema": {"user_flow": {"type": "markdown"}}}
    _ensure_write_tools_for_markdown_produces(prof, ctx)
    assert "write_file" in prof.allowed_tools


def test_md_produces_no_schema_restores_write_file():
    """A .md produces with NO schema is an authoring step — restore write_file."""
    prof = _prof([])
    ctx = {"produces": ["mission_90/notes.md"]}
    _ensure_write_tools_for_markdown_produces(prof, ctx)
    assert "write_file" in prof.allowed_tools


def test_md_produces_structured_return_schema_does_not_restore_write_file():
    """4.14 ADR register.md — a NON-EMPTY object/array schema on a .md produces is
    a structured-RETURN step (agent returns JSON; engine rebuilds the .md). It must
    stay write-stripped; the invariant must NOT hand it write_file."""
    prof = _prof([])
    ctx = {"produces": ["mission_90/.adr/register.md"],
           "artifact_schema": {"adrs": {"type": "array", "min_items": 3}}}
    _ensure_write_tools_for_markdown_produces(prof, ctx)
    assert "write_file" not in prof.allowed_tools


def test_json_produces_does_not_restore_write_file():
    """A .json produces is structured — the final_answer JSON IS the artifact;
    no write tool needed, so the invariant must not add one."""
    prof = _prof([])
    ctx = {"produces": ["mission_90/.intake/draft.json"],
           "artifact_schema": {"draft": {"type": "object"}}}
    _ensure_write_tools_for_markdown_produces(prof, ctx)
    assert "write_file" not in prof.allowed_tools


def test_no_produces_leaves_tools_untouched():
    prof = _prof([])
    ctx = {"artifact_schema": {"verdict": {"type": "object"}}}   # no produces
    _ensure_write_tools_for_markdown_produces(prof, ctx)
    assert prof.allowed_tools == []


def test_none_allowed_tools_means_all_tools_unchanged():
    """allowed_tools=None ⇒ all tools available (write_file already present)."""
    prof = _prof(None)
    ctx = {"produces": ["mission_90/.flow/user_flow.md"]}
    _ensure_write_tools_for_markdown_produces(prof, ctx)
    assert prof.allowed_tools is None


def test_write_file_already_present_is_idempotent():
    prof = _prof(["write_file", "read_file"])
    ctx = {"produces": ["mission_90/.flow/user_flow.md"]}
    _ensure_write_tools_for_markdown_produces(prof, ctx)
    assert prof.allowed_tools.count("write_file") == 1
