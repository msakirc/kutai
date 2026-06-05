"""i2p_v3 drift-backlog reconciliation (2026-06-05).

The #1 schema gate hard-checks artifact_schema while the LLM grader separately
checks COMPLETE vs the full instruction. Where a step's instruction enumerated
output fields its schema did not list, the producer could omit them: the gate
passed (looser schema) but the grader said COMPLETE:NO -> blind-retry DLQ (the
2.10 class). 16 genuine drifts were reconciled here; 22 lint hits were
confirmed false positives (item values, verbs, enum options, reviewer-read
upstream fields, frontmatter, intentionally-loose token maps) and left alone.

Empty-placeholder safety: the gate rejects []/{}/"" for REQUIRED fields, so
any added field that can legitimately be empty is added as ``optional`` to
avoid manufacturing a new false-DLQ.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from src.workflows.engine.hooks import validate_artifact_schema
from src.workflows.engine.field_drift import lint_step_field_drift, schema_field_names

_I2P = pathlib.Path(__file__).resolve().parents[2] / "src" / "workflows" / "i2p" / "i2p_v3.json"


def _steps() -> dict[str, dict]:
    data = json.loads(_I2P.read_text(encoding="utf-8"))
    return {s["id"]: s for s in data["steps"]}


# (step_id, artifact, [fields that must now be declared in the schema])
_RECONCILED = [
    ("1.14", "go_no_go_decision", ["market_attractiveness", "competitive_feasibility",
                                    "technical_feasibility", "differentiation_potential",
                                    "regulatory_risk"]),
    ("2.7", "mvp_scope", ["estimated_scope_size"]),
    ("1.9", "market_gap_analysis", ["opportunity_id", "estimated_impact",
                                    "difficulty_to_address", "our_ability_to_exploit"]),
    ("1.8", "competitor_ux_evaluation", ["visual_consistency", "information_density",
                                         "navigation_clarity", "onboarding_quality",
                                         "accessibility_signals", "mobile_quality"]),
    ("1.7", "competitor_reviews_raw", ["source_name", "review_count_sampled", "average_rating"]),
    ("1.7", "competitor_sentiment_analysis", ["top_5_praised_aspects", "top_5_pain_points",
                                              "overall_sentiment"]),
    ("-1.6", "project_conventions", ["git_conventions", "state_management_patterns"]),
    ("4.7", "frontend_architecture", ["code_splitting", "data_fetching", "error_boundaries"]),
    ("3.7", "business_rules", ["category", "source", "affected_features"]),
    ("3.6", "i18n_requirements", ["future_languages", "string_extraction_strategy",
                                  "text_expansion_budget"]),
    ("-1.4", "project_state_map", ["completion_percentage", "quality_gaps"]),
    ("2.4", "user_journey_map", ["user_actions", "touchpoints", "emotions",
                                 "drop_off_reasons", "metrics"]),
    ("1.6", "competitor_pricing_analysis", ["pricing_model_type", "tiers", "free_tier_scope",
                                            "upgrade_triggers", "enterprise_offering",
                                            "pricing_page_url"]),
    ("6.2", "dependency_graph", ["parallelizable_groups", "bottleneck_tasks"]),
    ("8.0", "implementation_backlog", ["user_story_ids"]),
    ("8.sprint_ritual", "sprint_review_result", ["next_sprint_adjustments"]),
    ("1.5", "competitor_features_raw", ["feature_description", "which_competitor"]),
]


# Fields genuinely ADDED to a schema (vs reconciled by aligning the instruction
# to the existing schema name — those carry no schema add). Used to prove the
# schema gained the field, regardless of required/optional.
_SCHEMA_ADDS = [
    ("1.14", "go_no_go_decision", ["market_attractiveness", "regulatory_risk"]),
    ("2.7", "mvp_scope", ["estimated_scope_size"]),
    ("1.9", "market_gap_analysis", ["opportunity_id", "our_ability_to_exploit"]),
    ("1.8", "competitor_ux_evaluation", ["visual_consistency", "mobile_quality"]),
    ("1.7", "competitor_reviews_raw", ["review_count_sampled", "average_rating"]),
    ("1.7", "competitor_sentiment_analysis", ["top_5_praised_aspects", "overall_sentiment"]),
    ("-1.6", "project_conventions", ["git_conventions", "state_management_patterns"]),
    ("4.7", "frontend_architecture", ["code_splitting", "data_fetching", "error_boundaries"]),
    ("3.7", "business_rules", ["category", "source", "affected_features"]),
    ("3.6", "i18n_requirements", ["future_languages", "string_extraction_strategy",
                                  "text_expansion_budget"]),
    ("-1.4", "project_state_map", ["completion_percentage", "quality_gaps"]),
    ("2.4", "user_journey_map", ["user_actions", "drop_off_reasons", "metrics"]),
    ("1.6", "competitor_pricing_analysis", ["pricing_model_type", "free_tier_scope",
                                            "enterprise_offering", "pricing_page_url"]),
    ("6.2", "dependency_graph", ["bottleneck_tasks"]),
    ("8.0", "implementation_backlog", ["user_story_ids"]),
    ("8.sprint_ritual", "sprint_review_result", ["next_sprint_adjustments"]),
]

# Every reconciled step (incl. 1.5 / 6.2 reconciled via instruction edit) must
# carry ZERO remaining instruction<->schema drift.
_RECONCILED_STEP_IDS = sorted({r[0] for r in _RECONCILED} | {"1.5", "6.2"})


@pytest.mark.parametrize("step_id,artifact,fields", _SCHEMA_ADDS)
def test_schema_gained_field(step_id, artifact, fields):
    rule = _steps()[step_id]["artifact_schema"][artifact]
    declared = schema_field_names({artifact: rule})
    missing = [f for f in fields if f not in declared]
    assert not missing, f"{step_id}/{artifact} schema still omits {missing}"


# Residual lint hits that are genuine FALSE POSITIVES (not output fields), so
# they must NOT be forced into the schema:
#   2.4 first_value — a journey STAGE NAME enumerated in the stage list.
#   8.0 mvp_scope   — an upstream input artifact referenced as mvp_scope.<path>.
_ALLOWED_FP_LEFTOVER = {
    "2.4": {"first_value"},
    "8.0": {"mvp_scope"},
}


@pytest.mark.parametrize("step_id", _RECONCILED_STEP_IDS)
def test_reconciled_step_has_no_genuine_drift(step_id):
    drift = lint_step_field_drift(_steps()[step_id])
    leftover = drift - _ALLOWED_FP_LEFTOVER.get(step_id, set())
    assert leftover == set(), f"{step_id} still drifts on genuine fields: {sorted(leftover)}"


def test_go_no_go_scores_require_five_dimensions():
    schema = {"go_no_go_decision": _steps()["1.14"]["artifact_schema"]["go_no_go_decision"]}
    good = json.dumps({"go_no_go_decision": {
        "scores": {"market_attractiveness": 7, "competitive_feasibility": 6,
                   "technical_feasibility": 8, "differentiation_potential": 5,
                   "regulatory_risk": 4},
        "weighted_score": 6.2, "recommendation": "Go"}})
    ok, err = validate_artifact_schema(good, schema)
    assert ok, err
    bad = json.dumps({"go_no_go_decision": {
        "scores": {"market_attractiveness": 7},  # 4 dimensions missing
        "weighted_score": 6.2, "recommendation": "Go"}})
    ok2, _ = validate_artifact_schema(bad, schema)
    assert not ok2, "scores missing 4 dimensions must fail"


def test_optional_empty_fields_do_not_false_dlq():
    # -1.4: a fully-complete phase legitimately has empty blocking_gaps/quality_gaps.
    schema = {"project_state_map": _steps()["-1.4"]["artifact_schema"]["project_state_map"]}
    doc = json.dumps({"project_state_map": {
        "phases": [{"completion_percentage": 100, "blocking_gaps": [], "quality_gaps": []}],
        "effective_starting_phase": "phase_3"}})
    ok, err = validate_artifact_schema(doc, schema)
    assert ok, f"empty per-phase gaps must pass (optional): {err}"

    # 2.4: a journey stage with no drop_off_reasons / metrics must still pass.
    sc2 = {"user_journey_map": _steps()["2.4"]["artifact_schema"]["user_journey_map"]}
    doc2 = json.dumps({"user_journey_map": {
        "stages": [{"user_actions": ["a"], "touchpoints": ["t"], "emotions": ["e"]}],
        "aha_moment": "first export"}})
    ok2, err2 = validate_artifact_schema(doc2, sc2)
    assert ok2, f"stage without optional fields must pass: {err2}"
