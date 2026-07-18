"""`_parse_review_result` turns a reviewer's stored result into a dict/list for
the verify_review_verdict classifier.

Defense-in-depth for m90 task 567426: even after the parsing.py root fix stores
verdicts as JSON, a stored Python-repr string (single quotes) must still parse —
the chokepoint falls back to ast.literal_eval so a real verdict is never dropped
as "no parseable review verdict".
"""
from __future__ import annotations

from general_beckman.apply import _parse_review_result


def test_dict_passthrough():
    d = {"status": "fail", "issues": []}
    assert _parse_review_result(d) is d


def test_valid_json_string():
    out = _parse_review_result('{"status": "pass", "issues": []}')
    assert out == {"status": "pass", "issues": []}


def test_python_repr_string_recovered():
    # The exact 567426 shape: repr(dict) with single quotes, double only where
    # the value contains an apostrophe.
    repr_str = (
        "{'status': 'fail', 'issues': [{'target_artifact': 'requirements_spec.md', "
        "'severity': 'blocker', 'problem': 'Missing triples', "
        "'suggested_fix': \"Add 'Risk if Wrong' column\"}]}"
    )
    out = _parse_review_result(repr_str)
    assert isinstance(out, dict)
    assert out["status"] == "fail"
    assert out["issues"][0]["severity"] == "blocker"


def test_garbage_returns_none():
    assert _parse_review_result("not a verdict, just prose !!!") is None


def test_empty_returns_none():
    assert _parse_review_result("") is None
    assert _parse_review_result(None) is None
