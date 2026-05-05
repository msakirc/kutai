"""Coverage: ``blockers`` array rule for severity-aware findings gates.

Used by 10.1 owasp_audit and 10.5 encryption_logging_review: the audit can
ship findings of all severities, but `severity in {critical, high}` entries
block the gate until resolved. Avoids the false-positive trap of forcing
zero-finding output.
"""
from __future__ import annotations

import pytest

from src.workflows.engine.schema_dialect import validate_value
from src.workflows.engine.loader import load_workflow


_FINDING_RULE = {
    "type": "array",
    "items": {
        "type": "object",
        "fields": {
            "category":    {"type": "string"},
            "severity":    {"type": "string", "equals": ["critical", "high", "medium", "low", "info"]},
            "description": {"type": "string"},
            "remediation": {"type": "string"},
        },
    },
    "blockers": {"field": "severity", "levels": ["critical", "high"]},
}


def _finding(severity: str, **overrides) -> dict:
    base = {
        "category": "auth",
        "severity": severity,
        "description": "x",
        "remediation": "y",
    }
    base.update(overrides)
    return base


def test_empty_findings_pass():
    err = validate_value(_FINDING_RULE, [_finding("low")])
    assert err is None


def test_medium_low_info_pass():
    err = validate_value(_FINDING_RULE, [
        _finding("medium"), _finding("low"), _finding("info"),
    ])
    assert err is None


def test_critical_finding_blocks():
    err = validate_value(_FINDING_RULE, [_finding("critical")])
    assert err is not None
    assert "BLOCKER" in err
    assert "critical" in err


def test_high_finding_blocks():
    err = validate_value(_FINDING_RULE, [_finding("low"), _finding("high")])
    assert err is not None
    assert "BLOCKER" in err
    assert "[1]" in err  # path includes index of blocker


def test_uppercase_severity_still_blocks():
    """Case-insensitive severity match — agents emit varying case."""
    err = validate_value(_FINDING_RULE, [_finding("CRITICAL")])
    assert err is not None
    assert "BLOCKER" in err


def test_blocker_message_warns_against_downgrade():
    """Retry feedback must steer agent toward fix, not silent downgrade."""
    err = validate_value(_FINDING_RULE, [_finding("critical")])
    assert err is not None
    assert "downgrade" in err.lower() or "do NOT" in err


def test_invalid_severity_value_rejected():
    """Severity must be in the equals list — random labels fail before the
    blocker check (caught by the items.severity rule)."""
    err = validate_value(_FINDING_RULE, [_finding("scary")])
    assert err is not None
    # Either equals-rejection or blocker, both are valid rejection paths.
    assert "scary" in err or "BLOCKER" in err


def test_non_dict_items_skipped_for_blocker_check():
    """Mixed array (not a typical case) shouldn't crash the blocker walk."""
    rule = {"type": "array", "blockers": {"field": "severity", "levels": ["critical"]}}
    err = validate_value(rule, ["string", 1, None])
    # No items_rule = no per-item validation; non-dict items skipped by blocker.
    assert err is None


def test_missing_field_or_levels_disables_rule():
    rule_no_field = {"type": "array", "blockers": {"levels": ["critical"]}}
    rule_no_levels = {"type": "array", "blockers": {"field": "severity"}}
    rule_empty_levels = {"type": "array", "blockers": {"field": "severity", "levels": []}}
    for r in (rule_no_field, rule_no_levels, rule_empty_levels):
        err = validate_value(r, [{"severity": "critical"}])
        assert err is None, f"rule {r} should be a no-op"


# ── i2p_v3 wiring smoke tests ────────────────────────────────────────────

def _step(step_id: str) -> dict:
    wf = load_workflow("i2p_v3")
    s = wf.get_step(step_id)
    assert s is not None
    return s


def test_owasp_audit_findings_gate_wired():
    schema = (_step("10.1").get("artifact_schema") or {}).get("owasp_audit_result")
    assert schema is not None
    fields = schema.get("fields") or {}
    findings = fields.get("findings")
    assert findings is not None
    blockers = findings.get("blockers")
    assert blockers == {"field": "severity", "levels": ["critical", "high"]}


def test_encryption_review_both_findings_gated():
    schema = _step("10.5").get("artifact_schema") or {}
    for artifact in ("encryption_verification_result", "logging_security_result"):
        sub = schema.get(artifact)
        assert sub is not None, f"{artifact} missing"
        findings = (sub.get("fields") or {}).get("findings")
        assert findings is not None, f"{artifact}.findings missing"
        assert findings.get("blockers") == {"field": "severity", "levels": ["critical", "high"]}


def test_owasp_critical_finding_rejected_by_full_artifact_validation():
    """End-to-end: feeding a critical finding through the live i2p_v3 schema
    rejects via the blocker rule, mirroring what the grader does at runtime."""
    schema = (_step("10.1").get("artifact_schema") or {})["owasp_audit_result"]
    bad = {
        "categories": {"injection": "tested"},
        "findings": [
            {"category": "auth", "severity": "critical",
             "description": "bcrypt cost too low", "remediation": "raise to 12"},
        ],
    }
    err = validate_value(schema, bad)
    assert err is not None
    assert "BLOCKER" in err

    good = dict(bad)
    good["findings"] = [dict(f, severity="medium") for f in bad["findings"]]
    err = validate_value(schema, good)
    assert err is None
