# test_autopersist_candidate.py
"""Auto-persist recovery for inline final_answer artifacts (mission 75 DLQ).

Step 0.0a.draft (intake_todo_draft) uses the "return JSON as final_answer,
engine persists to the produces path" contract with write_file disabled.
The recovery in react.py only persisted `.md` produces, so the `.json`
artifact was never written, the grounding guard never cleared, and the
writer agent looped to max_iterations → DLQ (task 165064).

`autopersist_candidate(produces, written, result)` is the pure decision:
return (relative_path, content_to_write) when a single still-unwritten
text artifact (.md / .json) is dumped inline, else None.
"""
from __future__ import annotations

import json

from coulson.grounding import autopersist_candidate


# ─── .json (the bug) ──────────────────────────────────────────────────────────

def test_json_produces_persists_valid_json_string():
    produces = ["mission_75/.intake/intake_todo_draft.json"]
    result = '{"_schema_version": "1", "items": [{"n": 1, "category": "Audience"}]}'
    got = autopersist_candidate(produces, set(), result)
    assert got == (produces[0], result)


def test_json_produces_serializes_dict_result():
    produces = ["m/x.json"]
    result = {"_schema_version": "1", "items": [{"n": 1}]}
    got = autopersist_candidate(produces, set(), result)
    assert got is not None
    path, content = got
    assert path == "m/x.json"
    assert json.loads(content) == result  # round-trips


def test_json_produces_rejects_invalid_json():
    # Truncated / malformed → do NOT persist garbage; let the guard re-prompt.
    got = autopersist_candidate(["m/x.json"], set(), '{"items": [')
    assert got is None


def test_json_produces_rejects_empty_result():
    assert autopersist_candidate(["m/x.json"], set(), "") is None
    assert autopersist_candidate(["m/x.json"], set(), "   ") is None


# ─── .md (regression — existing behavior preserved) ──────────────────────────

def test_md_produces_persists_long_markdown():
    produces = ["m/charter.md"]
    result = "# Charter\n\n" + ("lorem ipsum " * 60)  # >500 chars
    got = autopersist_candidate(produces, set(), result)
    assert got == (produces[0], result)


def test_md_produces_rejects_short_markdown():
    assert autopersist_candidate(["m/charter.md"], set(), "too short") is None


# ─── guards: only when the single produces is still unwritten ─────────────────

def test_skips_when_already_written():
    produces = ["m/x.json"]
    written = {"m/x.json"}
    valid = '{"items": []}'
    assert autopersist_candidate(produces, written, valid) is None


def test_skips_multi_file_produces():
    got = autopersist_candidate(["a.json", "b.json"], set(), '{"x": 1}')
    assert got is None


def test_skips_non_text_extension():
    assert autopersist_candidate(["m/logo.png"], set(), "x" * 600) is None
