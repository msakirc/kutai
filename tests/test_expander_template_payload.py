"""Template payload propagation + parameter substitution.

Phase-1 real-tools wiring (feat.14 staging_validation) needs the template
expander to (a) propagate `payload` from a template step into the expanded
step dict and (b) substitute ``{feature_id}`` placeholders inside payload
values so each per-feature instance gets its own concrete artifact name.
Without these, every feature would either lose its payload or share an
unsubstituted artifact name across all features.
"""
from __future__ import annotations

from src.workflows.engine.expander import expand_template, _substitute_payload


def _template():
    return {
        "context_artifacts": [],
        "steps": [
            {
                "template_step_id": "feat.14",
                "name": "staging_validation",
                "agent": "mechanical",
                "instruction": "Smoke {feature_name}",
                "input_artifacts": ["staging_deployment_result"],
                "output_artifacts": ["staging_validation_result"],
                "payload": {
                    "action": "staging_smoke_check",
                    "artifact": "{feature_id}__staging_deployment_result",
                    "method": "GET",
                    "max_attempts": 5,
                },
            },
        ],
    }


def test_substitute_payload_basic():
    out = _substitute_payload(
        {"k": "{feature_id}-thing", "n": 5},
        {"feature_id": "F-001"},
    )
    assert out == {"k": "F-001-thing", "n": 5}


def test_substitute_payload_unknown_placeholder_passes_through():
    out = _substitute_payload(
        {"k": "{nope}/x"},
        {"feature_id": "F-001"},
    )
    assert out == {"k": "{nope}/x"}


def test_substitute_payload_walks_nested():
    out = _substitute_payload(
        {"deep": {"list": ["{feature_id}", "lit"]}, "leaf": 7},
        {"feature_id": "F-9"},
    )
    assert out == {"deep": {"list": ["F-9", "lit"]}, "leaf": 7}


def test_template_payload_propagated_with_substitution():
    expanded = expand_template(
        _template(),
        params={"feature_id": "F-001", "feature_name": "Login"},
        prefix="8.F-001",
    )
    assert len(expanded) == 1
    step = expanded[0]
    assert step["agent"] == "mechanical"
    assert step["payload"]["action"] == "staging_smoke_check"
    assert step["payload"]["artifact"] == "F-001__staging_deployment_result"
    assert step["payload"]["max_attempts"] == 5


def test_template_payload_per_feature_independent():
    """Two different features must get separate substituted artifact names."""
    p1 = expand_template(_template(),
                         params={"feature_id": "F-001", "feature_name": "A"},
                         prefix="8.F-001")[0]["payload"]
    p2 = expand_template(_template(),
                         params={"feature_id": "F-002", "feature_name": "B"},
                         prefix="8.F-002")[0]["payload"]
    assert p1["artifact"] == "F-001__staging_deployment_result"
    assert p2["artifact"] == "F-002__staging_deployment_result"
    assert p1 is not p2
