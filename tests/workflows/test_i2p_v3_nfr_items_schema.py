"""i2p_v3 NFR schema reconciliation (3.2 / 3.3) — fix surfaced by the #1
schema-gate smoke test, 2026-06-05.

The phase-3 NFR steps carried a pre-Z1 schema: ``required_fields`` listed flat
metrics (api_response_time, page_load_time, ...) AND ``items``. But the
instruction + the ``verify_falsification_present`` post-hook only consume an
``items`` array of ``{name, target, risk_if_wrong, validation_method,
falsification_signal}``. Producers emitted the flat metrics and omitted
``items`` (or the reverse), and the constrained-emit pass could not force the
nested triple from a flat ``items`` placeholder — so the step blind-retried to
DLQ (#289776 mission, 18:19/18:20).

Reconciled: each NFR artifact is now ``{items: <non-empty array of triple
objects>}`` — one structure that the schema gate, the falsification post-hook,
and the constrained emitter all agree on.
"""
from __future__ import annotations

import json
import pathlib

from src.workflows.engine.hooks import validate_artifact_schema

_I2P = pathlib.Path(__file__).resolve().parents[2] / "src" / "workflows" / "i2p" / "i2p_v3.json"

# (step_id, artifact_name) pairs reconciled by this fix.
_NFR_ARTIFACTS = [
    ("3.2", "nfr_performance"),
    ("3.2", "nfr_scalability"),
    ("3.3", "nfr_availability"),
    ("3.3", "security_requirements"),
]


def _load_steps() -> dict[str, dict]:
    data = json.loads(_I2P.read_text(encoding="utf-8"))
    steps = data.get("steps") or data.get("tasks") or []
    return {s.get("id"): s for s in steps}


def _schema_for(step_id: str, artifact: str) -> dict:
    step = _load_steps()[step_id]
    return {artifact: step["artifact_schema"][artifact]}


def _triple_item(name: str) -> dict:
    return {
        "name": name,
        "target": "p95 < 500ms",
        "risk_if_wrong": "high",
        "validation_method": "monitor p95 over 7d",
        "falsification_signal": "p95 breaches target 3 days running",
    }


def _good_artifact(artifact: str) -> str:
    return json.dumps({artifact: {"items": [
        _triple_item("metric_a"), _triple_item("metric_b"), _triple_item("metric_c"),
    ]}})


def test_items_only_artifact_passes_for_each_nfr():
    # The reconciled shape: an items array of triple objects, NO flat fields.
    # Under the OLD schema this FAILS (missing api_response_time etc.) — that is
    # the RED discriminator.
    for step_id, artifact in _NFR_ARTIFACTS:
        schema = _schema_for(step_id, artifact)
        ok, err = validate_artifact_schema(_good_artifact(artifact), schema)
        assert ok, f"{step_id}/{artifact} should pass with items-only shape: {err}"


def test_empty_items_fails_for_each_nfr():
    for step_id, artifact in _NFR_ARTIFACTS:
        schema = _schema_for(step_id, artifact)
        ok, _ = validate_artifact_schema(json.dumps({artifact: {"items": []}}), schema)
        assert not ok, f"{step_id}/{artifact} must reject empty items"


def test_item_missing_triple_fails_for_each_nfr():
    for step_id, artifact in _NFR_ARTIFACTS:
        schema = _schema_for(step_id, artifact)
        bad = json.dumps({artifact: {"items": [
            {"name": "a", "target": "y"},  # 3 items (min_items met) but
            {"name": "b", "target": "y"},  # NONE carry the falsification
            {"name": "c", "target": "y"},  # triple — isolates item_fields check
        ]}})
        ok, _ = validate_artifact_schema(bad, schema)
        assert not ok, f"{step_id}/{artifact} must reject items lacking the triple"


def test_nfr_schemas_have_no_flat_residue():
    # The flat metric fields (api_response_time, ...) are gone — items is the
    # single contract.
    for step_id, artifact in _NFR_ARTIFACTS:
        rule = _load_steps()[step_id]["artifact_schema"][artifact]
        fields = rule.get("fields") or {}
        assert set(fields) == {"items"}, (
            f"{step_id}/{artifact} should declare only 'items', got {sorted(fields)}"
        )
        assert "required_fields" not in rule, (
            f"{step_id}/{artifact} still uses legacy required_fields"
        )
