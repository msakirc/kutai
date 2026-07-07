"""Tests for the schema_gate mechanical validator (fix #1).

schema_gate promotes the existing src.workflows.engine.hooks.validate_artifact_schema
to a *gating* mechanical post-hook: given a produced artifact string and the
step's artifact_schema, return {"passed", "error"} so a FAIL deterministically
retries the producer with a precise reason (the validator's own message) —
instead of relying on the prose-reading LLM grader.

Regression anchors:
  #289735 (2.8 user_stories): schema demands a JSON array (min_items>=5);
           producer emitted a single object -> must FAIL.
  #289737 (2.10 monetization): schema required_fields=[3 keys]; an artifact
           carrying those 3 keys -> must PASS (the 2.10 prose/schema drift is
           fix #4's job, NOT the gate's).
"""
import json

from mr_roboto.schema_gate import schema_gate


_ARRAY_SCHEMA = {
    "user_stories": {
        "type": "array",
        "min_items": 5,
        "item_fields": ["story_id", "epic", "title", "story",
                        "acceptance_criteria", "priority"],
    }
}

_OBJECT_SCHEMA = {
    "monetization_strategy": {
        "type": "object",
        "required_fields": ["pricing_model", "tiers", "revenue_projections"],
    }
}


def _story(i):
    return {
        "story_id": f"US-00{i}", "epic": "core", "title": f"t{i}",
        "story": "As a user, I want X, so that Y",
        "acceptance_criteria": "Given/When/Then", "priority": "High",
    }


def test_single_object_against_array_schema_fails():
    # #289735 reproduction: one story object where an array of >=5 is required.
    out = json.dumps(_story(1))
    res = schema_gate(output_value=out, schema=_ARRAY_SCHEMA)
    assert res["passed"] is False
    assert res["error"]  # non-empty, actionable reason


def test_array_of_five_stories_passes():
    out = json.dumps([_story(i) for i in range(1, 6)])
    res = schema_gate(output_value=out, schema=_ARRAY_SCHEMA)
    assert res["passed"] is True
    assert not res["error"]


def test_object_with_required_fields_passes():
    # #289737: artifact carries exactly the 3 schema-required keys -> PASS.
    out = json.dumps({
        "pricing_model": "Freemium",
        "tiers": {"free": {}, "pro": {}},
        "revenue_projections": {"y1": 1000},
    })
    res = schema_gate(output_value=out, schema=_OBJECT_SCHEMA)
    assert res["passed"] is True
    assert not res["error"]


def test_object_missing_required_field_fails():
    out = json.dumps({"pricing_model": "Freemium", "tiers": {}})
    res = schema_gate(output_value=out, schema=_OBJECT_SCHEMA)
    assert res["passed"] is False
    assert "revenue_projections" in res["error"] or res["error"]


def test_no_schema_is_vacuous_pass():
    res = schema_gate(output_value="anything", schema={})
    assert res["passed"] is True
    assert not res["error"]


# ── Conditional-empty exemption anchored to an upstream input ────────────

_OVERLAY_SCHEMA = {
    "compliance_overlay": {
        "type": "object",
        "fields": {
            "required_documents": {
                "type": "array",
                "empty_ok_when_input_empty": "compliance_fingerprint.jurisdictions",
            },
        },
        "_schema_version": "1",
    }
}


def test_empty_overlay_passes_when_fingerprint_has_no_jurisdictions():
    # A hobby app: fingerprint has zero jurisdictions, so zero required docs
    # is the CORRECT answer. The gate must accept it when given the upstream.
    overlay = json.dumps({"required_documents": []})
    inputs = {"compliance_fingerprint": {"jurisdictions": []}}
    res = schema_gate(output_value=overlay, schema=_OVERLAY_SCHEMA, inputs=inputs)
    assert res["passed"] is True, res["error"]
    assert not res["error"]


def test_empty_overlay_fails_when_fingerprint_has_jurisdictions():
    # Real scope upstream → an empty list is a LAZY placeholder. Reject.
    overlay = json.dumps({"required_documents": []})
    inputs = {"compliance_fingerprint": {"jurisdictions": ["US", "EU"]}}
    res = schema_gate(output_value=overlay, schema=_OVERLAY_SCHEMA, inputs=inputs)
    assert res["passed"] is False
    assert "required_documents" in res["error"]


def test_empty_overlay_fails_without_inputs():
    # No upstream proof supplied → conservative reject (back-compat default).
    overlay = json.dumps({"required_documents": []})
    res = schema_gate(output_value=overlay, schema=_OVERLAY_SCHEMA)
    assert res["passed"] is False


# ── produces_markdown: an object schema on a .md artifact must defer to
#    verify_*_shape, NOT run the field-NAME substring fallback (which is
#    meaningless on prose). Mirrors the producer gate (hooks.py:1928); closes
#    the grade-path asymmetry (apply.py grade gate omitted the flag). ──────────

_UF_OBJECT_SCHEMA = {
    "user_flow": {"type": "object",
                  "required_fields": ["surfaces", "mermaid_per_surface"]}
}
# A real markdown doc: has `surfaces:` frontmatter + a ```mermaid block, but
# NOT the literal field name `mermaid_per_surface` (nor any word the loose
# substring fallback would match its parts against — e.g. no "per*" word).
_UF_MARKDOWN = (
    "---\nsurfaces: ['web']\n---\n\n## Web\n\n"
    "```mermaid\ngraph TD\n  A --> B\n```\n"
)


def test_object_schema_on_markdown_defers_when_produces_markdown():
    res = schema_gate(output_value=_UF_MARKDOWN, schema=_UF_OBJECT_SCHEMA,
                      produces_markdown=True)
    assert res["passed"] is True, res["error"]
    assert not res["error"]


def test_object_schema_on_markdown_false_rejects_without_flag():
    # Back-compat: default (produces_markdown=False) still runs the fallback and
    # false-rejects the markdown for the absent field NAME.
    res = schema_gate(output_value=_UF_MARKDOWN, schema=_UF_OBJECT_SCHEMA)
    assert res["passed"] is False
    assert "mermaid_per_surface" in res["error"]
