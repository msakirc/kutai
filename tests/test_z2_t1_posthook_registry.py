"""Z2 T1A — Post-hook registry tests.

Verifies:
- Registry contains the expected 3 starter kinds with correct verbs.
- POST_HOOK_KINDS derived frozenset equals the legacy set.
- determine_posthooks round-trips through the registry correctly.
"""
from __future__ import annotations

import pytest
from general_beckman.posthooks import (
    POST_HOOK_REGISTRY,
    POST_HOOK_KINDS,
    _KNOWN_EXTRA_KINDS,
    PostHookSpec,
    determine_posthooks,
)


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------

def test_registry_contains_starter_kinds():
    # Starter set; T2/T3 agents extend this — assert superset, not equality.
    assert {"verify_artifacts", "code_review", "grounding"}.issubset(
        set(POST_HOOK_REGISTRY.keys())
    )


def test_registry_values_are_posthookspec():
    for kind, spec in POST_HOOK_REGISTRY.items():
        assert isinstance(spec, PostHookSpec), f"{kind} spec is not PostHookSpec"


def test_verify_artifacts_spec_fields():
    spec = POST_HOOK_REGISTRY["verify_artifacts"]
    assert spec.kind == "verify_artifacts"
    assert spec.verb == "verify_artifacts"
    assert spec.default_severity == "blocker"
    # No auto-wire triggers — must be declared explicitly on step
    assert spec.auto_wire_triggers == []


def test_code_review_spec_fields():
    spec = POST_HOOK_REGISTRY["code_review"]
    assert spec.kind == "code_review"
    assert spec.verb == "code_reviewer"
    assert spec.default_severity == "blocker"
    assert spec.auto_wire_triggers == []


def test_grounding_spec_fields():
    spec = POST_HOOK_REGISTRY["grounding"]
    assert spec.kind == "grounding"
    assert spec.verb == "check_grounding"
    assert spec.default_severity == "blocker"
    # Must have a wildcard trigger to auto-wire on all produces
    assert "*" in spec.auto_wire_triggers


# ---------------------------------------------------------------------------
# Back-compat aliases
# ---------------------------------------------------------------------------

def test_post_hook_kinds_derived_equals_registry_keys():
    assert POST_HOOK_KINDS == frozenset(POST_HOOK_REGISTRY.keys())


def test_known_extra_kinds_alias_equals_post_hook_kinds():
    assert _KNOWN_EXTRA_KINDS == POST_HOOK_KINDS


# ---------------------------------------------------------------------------
# determine_posthooks round-trip
# ---------------------------------------------------------------------------

def test_determine_posthooks_returns_grade_by_default():
    task = {"agent_type": "writer"}
    result = determine_posthooks(task, {}, {})
    assert result == ["grade"]


def test_determine_posthooks_accepts_registered_kind():
    task = {"agent_type": "coder"}
    ctx = {"post_hooks": ["verify_artifacts"]}
    result = determine_posthooks(task, ctx, {})
    assert "verify_artifacts" in result
    assert "grade" in result


def test_determine_posthooks_rejects_unknown_kind():
    """A kind not in the registry must be silently dropped."""
    task = {"agent_type": "coder"}
    ctx = {"post_hooks": ["nonexistent_kind_xyz"]}
    result = determine_posthooks(task, ctx, {})
    assert "nonexistent_kind_xyz" not in result
    assert "grade" in result


def test_determine_posthooks_all_three_registry_kinds_accepted():
    """All three registered kinds survive the filter."""
    task = {"agent_type": "coder"}
    ctx = {
        "post_hooks": ["verify_artifacts", "code_review", "grounding"],
        "requires_grading": False,
    }
    result = determine_posthooks(task, ctx, {})
    assert "verify_artifacts" in result
    assert "code_review" in result
    assert "grounding" in result


def test_determine_posthooks_no_duplicates():
    """If a kind appears both in post_hooks and via requires_grading path,
    grade won't be duplicated (grade is separate but extras shouldn't dup)."""
    task = {"agent_type": "coder"}
    ctx = {"post_hooks": ["verify_artifacts", "verify_artifacts"]}
    result = determine_posthooks(task, ctx, {})
    assert result.count("verify_artifacts") == 1
