"""Fix #4 — instruction<->schema field-drift lint.

After fix #1, a step's ``artifact_schema`` is enforced deterministically at the
grade boundary. That makes a *drifted* schema dangerous: if the instruction
tells the producer to emit fields the schema does NOT list (#289737 2.10:
instruction named 6, schema required 3), the gate and the prose grader judge
different contracts. This lint extracts the fields a step's instruction
*enumerates* (the low-noise signal: comma-separated snake_case runs) and asserts
they are all present in the step's schema.
"""
from __future__ import annotations

import json

from src.workflows.engine.field_drift import (
    instruction_declared_fields,
    lint_step_field_drift,
    schema_field_names,
)


# ── extractor ───────────────────────────────────────────────────────────────

def test_extracts_comma_enumerated_snake_case_fields():
    instr = (
        "Define business model: pricing_model, tiers, free_tier_strategy, "
        "upgrade_triggers, billing_implementation, revenue_projections. If "
        "multiple viable models exist, present with trade-offs."
    )
    got = instruction_declared_fields(instr)
    # Only snake_case (underscore) tokens are collected — the low-noise field
    # signal. ``tiers`` (single word) is intentionally NOT collected; it cannot
    # cause a false drift since drift = declared - schema_names.
    assert got == {
        "pricing_model", "free_tier_strategy",
        "upgrade_triggers", "billing_implementation", "revenue_projections",
    }


def test_prose_without_enumeration_declares_nothing():
    assert instruction_declared_fields("Write a concise market summary.") == set()


def test_single_field_mention_is_not_an_enumeration():
    # A lone snake_case token in prose is not a declared output-field list.
    assert instruction_declared_fields(
        "Update the user_profile based on findings."
    ) == set()


# ── schema field names ──────────────────────────────────────────────────────

def test_schema_field_names_unions_required_and_item_fields():
    schema = {
        "user_stories": {"type": "array",
                         "item_fields": ["story_id", "title"]},
        "monetization_strategy": {"type": "object",
                                  "required_fields": ["pricing_model", "tiers"]},
    }
    assert schema_field_names(schema) == {
        "story_id", "title", "pricing_model", "tiers",
    }


def test_schema_field_names_recurses_nested_fields_and_items():
    # Canonical dialect nests sub-fields under fields/items — the gate enforces
    # them, so the lint must count them or it false-flags nested field names.
    schema = {
        "design": {
            "type": "object",
            "fields": {
                "variants": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "fields": {"body_default": {"type": "string"},
                                   "page_title": {"type": "string"}},
                    },
                },
            },
        },
    }
    assert schema_field_names(schema) == {
        "variants", "body_default", "page_title",
    }


# ── per-step lint ───────────────────────────────────────────────────────────

def test_step_with_enumerated_fields_absent_from_schema_drifts():
    step = {
        "id": "x",
        "instruction": "Define: a_field, b_field, c_field.",
        "artifact_schema": {"out": {"type": "object",
                                    "required_fields": ["a_field", "b_field"]}},
    }
    assert lint_step_field_drift(step) == {"c_field"}


def test_step_in_agreement_has_no_drift():
    step = {
        "id": "x",
        "instruction": "Define: a_field, b_field.",
        "artifact_schema": {"out": {"type": "object",
                                    "required_fields": ["a_field", "b_field"]}},
    }
    assert lint_step_field_drift(step) == set()


def test_step_without_schema_is_not_linted():
    step = {"id": "x", "instruction": "Define: a_field, b_field."}
    assert lint_step_field_drift(step) == set()


def test_input_and_output_artifact_refs_are_not_drift():
    # Instructions routinely name upstream input artifacts and the step's own
    # output artifact in a comma list — those are references, not missing
    # output fields, so they must not register as drift.
    step = {
        "id": "x",
        "instruction": "From idea_brief_final and product_charter, define a_field, leftover_field.",
        "input_artifacts": ["idea_brief_final", "product_charter"],
        "output_artifacts": ["the_doc"],
        "artifact_schema": {"out": {"type": "object",
                                    "required_fields": ["a_field"]}},
    }
    assert lint_step_field_drift(step) == {"leftover_field"}


# ── regression anchor: 2.10 reconciled ──────────────────────────────────────

def test_i2p_v3_step_2_10_has_no_field_drift():
    d = json.load(open(r"src/workflows/i2p/i2p_v3.json", encoding="utf-8"))

    def find(o):
        if isinstance(o, dict):
            if str(o.get("id")) == "2.10":
                return o
            for v in o.values():
                r = find(v)
                if r:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = find(v)
                if r:
                    return r
        return None

    step = find(d)
    assert step is not None
    assert lint_step_field_drift(step) == set()
