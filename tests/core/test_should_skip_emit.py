"""Unit tests for ``src.core.reflection_posthook.should_skip_emit``.

Contract (fix #3, 2026-06-05): the constrained-emit pass must fire EXACTLY
when the draft would fail the deterministic artifact-schema gate (#1). The old
predicate only checked top-level artifact-name presence, so a draft that had
the right top-level key but was internally incomplete (missing nested
``required_fields``, or a single object where an array is required) was
SKIPPED — then flowed straight to the grade/schema-gate and DLQ'd on a blind
retry (#289735 object-vs-array, #289737 field completeness).

New contract: skip iff the draft already PASSES full schema validation.
"""
from __future__ import annotations

from src.core.reflection_posthook import should_skip_emit


# --- object with nested required_fields (the #289737 / 2.10 class) ----------

_OBJ_SCHEMA = {
    "monetization_strategy": {
        "type": "object",
        "required_fields": [
            "pricing_model", "tiers", "free_tier_strategy",
            "upgrade_triggers", "billing_implementation", "revenue_projections",
        ],
    }
}


def test_skips_when_object_has_all_required_nested_fields():
    draft = (
        '{"monetization_strategy": {"pricing_model": "freemium", '
        '"tiers": ["free", "pro"], "free_tier_strategy": "limited", '
        '"upgrade_triggers": "usage cap", "billing_implementation": "stripe", '
        '"revenue_projections": "10k/mo"}}'
    )
    assert should_skip_emit(draft, _OBJ_SCHEMA) is True


def test_emits_when_object_missing_nested_required_fields():
    # Top-level key present, but only 3 of 6 nested fields — the 2.10 bug.
    # Old shallow check skipped this; new check must EMIT (skip == False).
    draft = (
        '{"monetization_strategy": {"pricing_model": "freemium", '
        '"tiers": ["free", "pro"], "revenue_projections": "10k/mo"}}'
    )
    assert should_skip_emit(draft, _OBJ_SCHEMA) is False


# --- array with min_items (the #289735 / 2.8 class) -------------------------

_ARR_SCHEMA = {
    "user_stories": {
        "type": "array",
        "min_items": 5,
        "item_fields": ["id", "title"],
    }
}


def test_skips_when_array_meets_min_items():
    items = ", ".join(
        f'{{"id": {i}, "title": "story {i}"}}' for i in range(5)
    )
    draft = f'{{"user_stories": [{items}]}}'
    assert should_skip_emit(draft, _ARR_SCHEMA) is True


def test_emits_when_single_object_where_array_required():
    # Producer emitted a single object instead of an 8-15 item array (#289735).
    draft = '{"user_stories": {"id": 1, "title": "the one story"}}'
    assert should_skip_emit(draft, _ARR_SCHEMA) is False


def test_emits_when_array_below_min_items():
    draft = '{"user_stories": [{"id": 1, "title": "only one"}]}'
    assert should_skip_emit(draft, _ARR_SCHEMA) is False


# --- non-JSON draft ---------------------------------------------------------

def test_emits_when_draft_not_json():
    assert should_skip_emit("Here is my analysis: ...", _OBJ_SCHEMA) is False
