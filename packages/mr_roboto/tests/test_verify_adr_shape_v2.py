"""Z3 T4A — ADR schema v2: structured falsification_signal tests.

Covers the new v2 object form, null judgment-only path, back-compat for v1
string form, and type/presence validation rules.
"""
from __future__ import annotations

import copy

import pytest

from mr_roboto.verify_adr_shape import verify_adr_shape


# ── Minimal valid ADR fixtures ───────────────────────────────────────────────

def _base_v1() -> dict:
    """Minimal well-formed v1 ADR with string falsification_signal."""
    return {
        "adr_id": "ADR-2026-05-11-001",
        "title": "Choose DB engine",
        "status": "accepted",
        "context": "Need a database.",
        "decision": "Use PostgreSQL.",
        "consequences": "Operational overhead.",
        "options_considered": [
            {
                "id": "OPT-A",
                "name": "PostgreSQL",
                "rationale_for": "Mature, reliable.",
                "rationale_against": "Ops burden.",
            },
            {
                "id": "OPT-B",
                "name": "SQLite",
                "rationale_for": "Zero-ops.",
                "rationale_against": "No concurrent writes.",
            },
        ],
        "chosen_option_id": "OPT-A",
        "falsification_signal": "if query latency p99 > 500ms by week 4",
        "reversal_cost": "high",
        "supersedes_adr_id": None,
        "_schema_version": "1",
    }


def _base_v2() -> dict:
    """Minimal well-formed v2 ADR with object falsification_signal."""
    adr = _base_v1()
    adr["_schema_version"] = "2"
    adr["falsification_signal"] = {
        "forbidden_imports": ["raw_psycopg2"],
        "forbidden_patterns": ["SELECT \\*"],
        "required_test_coverage": True,
    }
    return adr


# ── v1 back-compat ───────────────────────────────────────────────────────────


def test_v1_string_falsification_accepted_by_v1_verifier():
    """v1 ADR with string falsification_signal passes expected_schema_version='1'."""
    res = verify_adr_shape(adr_obj=_base_v1(), expected_schema_version="1")
    assert res["ok"] is True, res
    assert res["falsification_missing"] is False
    assert res["falsification_invalid"] is False


def test_v1_adr_object_falsification_also_accepted_in_v1_mode():
    """v1 ADR carrying a v2 object form is forward-compatible (lenient v1 mode)."""
    adr = _base_v1()
    adr["falsification_signal"] = {"forbidden_imports": ["os.system"]}
    res = verify_adr_shape(adr_obj=adr, expected_schema_version="1")
    assert res["ok"] is True, res


def test_missing_schema_version_defaults_to_lenient_v1():
    """ADR with no _schema_version treated as v1 (lenient falsification rules)."""
    adr = _base_v1()
    adr.pop("_schema_version")
    # Missing field makes ok=False, but falsification_missing should be False
    # (the string form is still valid).
    res = verify_adr_shape(adr_obj=adr, expected_schema_version="1")
    assert res["falsification_missing"] is False
    assert res["falsification_invalid"] is False
    # overall ok=False because _schema_version is missing
    assert "_schema_version" in res["missing_fields"]


# ── v2 happy paths ───────────────────────────────────────────────────────────


def test_v2_full_object_accepted():
    """v2 ADR with all three keys passes expected_schema_version='2'."""
    res = verify_adr_shape(adr_obj=_base_v2(), expected_schema_version="2")
    assert res["ok"] is True, res
    assert res["falsification_missing"] is False
    assert res["falsification_invalid"] is False


def test_v2_only_forbidden_imports():
    """v2 object with only forbidden_imports (other keys absent) is valid."""
    adr = _base_v2()
    adr["falsification_signal"] = {"forbidden_imports": ["raw_psycopg2"]}
    res = verify_adr_shape(adr_obj=adr, expected_schema_version="2")
    assert res["ok"] is True, res


def test_v2_only_forbidden_patterns():
    """v2 object with only forbidden_patterns is valid."""
    adr = _base_v2()
    adr["falsification_signal"] = {"forbidden_patterns": ["SELECT \\*"]}
    res = verify_adr_shape(adr_obj=adr, expected_schema_version="2")
    assert res["ok"] is True, res


def test_v2_only_required_test_coverage():
    """v2 object with only required_test_coverage=True is valid."""
    adr = _base_v2()
    adr["falsification_signal"] = {"required_test_coverage": True}
    res = verify_adr_shape(adr_obj=adr, expected_schema_version="2")
    assert res["ok"] is True, res


def test_v2_required_test_coverage_false_is_valid():
    """required_test_coverage=False is a valid bool (counts as present)."""
    adr = _base_v2()
    adr["falsification_signal"] = {"required_test_coverage": False}
    res = verify_adr_shape(adr_obj=adr, expected_schema_version="2")
    assert res["ok"] is True, res


# ── v2 null (judgment-only) ──────────────────────────────────────────────────


def test_v2_null_falsification_accepted():
    """null falsification_signal is always valid — judgment-only ADR."""
    adr = _base_v2()
    adr["falsification_signal"] = None
    res = verify_adr_shape(adr_obj=adr, expected_schema_version="2")
    assert res["ok"] is True, res
    assert res["falsification_missing"] is False
    assert res["falsification_invalid"] is False


def test_v1_null_falsification_accepted():
    """null is also valid in v1 lenient mode."""
    adr = _base_v1()
    adr["falsification_signal"] = None
    res = verify_adr_shape(adr_obj=adr, expected_schema_version="1")
    assert res["ok"] is True, res


# ── v2 rejection cases ───────────────────────────────────────────────────────


def test_v2_empty_object_rejected():
    """v2 ADR with {} falsification_signal fails (no keys present)."""
    adr = _base_v2()
    adr["falsification_signal"] = {}
    res = verify_adr_shape(adr_obj=adr, expected_schema_version="2")
    assert res["ok"] is False
    assert res["falsification_invalid"] is True


def test_v2_string_form_rejected_in_strict_v2():
    """v2 mode does not accept a plain string for falsification_signal."""
    adr = _base_v2()
    adr["falsification_signal"] = "if latency > 500ms"
    res = verify_adr_shape(adr_obj=adr, expected_schema_version="2")
    assert res["ok"] is False
    assert res["falsification_invalid"] is True


def test_v2_forbidden_imports_wrong_type_rejected():
    """forbidden_imports must be list[str]; int values are rejected."""
    adr = _base_v2()
    adr["falsification_signal"] = {"forbidden_imports": [1, 2, 3]}
    res = verify_adr_shape(adr_obj=adr, expected_schema_version="2")
    assert res["ok"] is False
    assert res["falsification_invalid"] is True


def test_v2_forbidden_patterns_wrong_type_rejected():
    """forbidden_patterns must be list[str]; a plain string is rejected."""
    adr = _base_v2()
    adr["falsification_signal"] = {"forbidden_patterns": "SELECT *"}
    res = verify_adr_shape(adr_obj=adr, expected_schema_version="2")
    assert res["ok"] is False
    assert res["falsification_invalid"] is True


def test_v2_required_test_coverage_wrong_type_rejected():
    """required_test_coverage must be bool; a string "true" is rejected."""
    adr = _base_v2()
    adr["falsification_signal"] = {"required_test_coverage": "true"}
    res = verify_adr_shape(adr_obj=adr, expected_schema_version="2")
    assert res["ok"] is False
    assert res["falsification_invalid"] is True


def test_v2_all_keys_present_but_all_empty_lists_rejected():
    """All list keys present but empty — no non-empty key, should be invalid."""
    adr = _base_v2()
    adr["falsification_signal"] = {
        "forbidden_imports": [],
        "forbidden_patterns": [],
    }
    res = verify_adr_shape(adr_obj=adr, expected_schema_version="2")
    assert res["ok"] is False
    assert res["falsification_invalid"] is True


# ── existing v1 behaviour preserved ─────────────────────────────────────────


def test_v1_empty_string_still_rejected():
    """Empty string falsification_signal still triggers falsification_missing in v1 mode."""
    adr = _base_v1()
    adr["falsification_signal"] = ""
    res = verify_adr_shape(adr_obj=adr, expected_schema_version="1")
    assert res["ok"] is False
    assert res["falsification_missing"] is True


def test_v1_whitespace_only_string_rejected():
    adr = _base_v1()
    adr["falsification_signal"] = "   "
    res = verify_adr_shape(adr_obj=adr, expected_schema_version="1")
    assert res["ok"] is False
    assert res["falsification_missing"] is True


# ── output dict always includes new key ─────────────────────────────────────


def test_result_always_has_falsification_invalid_key():
    """falsification_invalid is always present in the result dict."""
    res = verify_adr_shape(adr_obj=_base_v1(), expected_schema_version="1")
    assert "falsification_invalid" in res

    res2 = verify_adr_shape(adr_obj=_base_v2(), expected_schema_version="2")
    assert "falsification_invalid" in res2
