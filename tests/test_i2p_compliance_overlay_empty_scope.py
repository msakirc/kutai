"""[1.11a] compliance_overlay — empty-scope must not DLQ.

When the upstream compliance_fingerprint has no jurisdictions (a low-scope
app, e.g. a hobby todo/habit tracker), there are legitimately ZERO required
legal documents. The step's own ``done_when`` blesses this
("...or fingerprint declares jurisdictions=[]"), but the mechanical schema
gate historically rejected the empty overlay as an "empty placeholder value",
DLQ-ing a correct answer (mission 87 task 524377).

The fix anchors an empty-exemption to the REAL upstream fingerprint, so:
  - empty fingerprint  → empty overlay PASSES
  - populated fingerprint → empty overlay still FAILS (lazy placeholder)
"""
import io
import json

from mr_roboto.schema_gate import schema_gate

_WF = "src/workflows/i2p/i2p_v3.json"


def _step(step_id):
    j = json.load(io.open(_WF, encoding="utf-8"))
    steps = j["steps"] if isinstance(j, dict) and "steps" in j else j
    for s in steps:
        if s.get("id") == step_id:
            return s
    raise AssertionError(f"step {step_id} not found")


def test_1_11a_schema_carries_input_anchored_exemption():
    schema = _step("1.11a")["artifact_schema"]
    blob = json.dumps(schema)
    # The doc-bearing fields are exempted, anchored to the fingerprint.
    assert "empty_ok_when_input_empty" in blob
    assert "compliance_fingerprint.jurisdictions" in blob


def test_empty_overlay_passes_gate_when_fingerprint_has_no_jurisdictions():
    schema = _step("1.11a")["artifact_schema"]
    overlay = json.dumps({
        "required_documents": [],
        "monitoring_obligations": [],
        "data_subject_rights_implementation": [],
    })
    inputs = {"compliance_fingerprint": {"jurisdictions": []}}
    res = schema_gate(output_value=overlay, schema=schema, inputs=inputs)
    assert res["passed"] is True, res["error"]


def test_empty_overlay_fails_gate_when_fingerprint_has_jurisdictions():
    schema = _step("1.11a")["artifact_schema"]
    overlay = json.dumps({
        "required_documents": [],
        "monitoring_obligations": [],
        "data_subject_rights_implementation": [],
    })
    inputs = {"compliance_fingerprint": {"jurisdictions": ["US", "EU"]}}
    res = schema_gate(output_value=overlay, schema=schema, inputs=inputs)
    assert res["passed"] is False
