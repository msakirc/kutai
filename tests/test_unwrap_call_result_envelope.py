"""structured_emit / CallResult envelope unwrap + schema validation.

Root cause of the mission_79 schema-validation DLQ cluster (2026-05-30):
artifacts were persisted as the raw HaLLederiz Kadir ``CallResult`` envelope —

    {"content": "<stringified JSON artifact>", "model": "...",
     "usage": {...}, "task": "structured_emit", ...}

— i.e. the real array/object is DOUBLE-ENCODED as a JSON string under
``content``.  ``_unwrap_envelope`` only knew the ``final_answer`` and
``write_file`` envelopes, so it returned the CallResult envelope unchanged;
``_extract_artifact_value`` then saw a dict whose keys are
content/model/usage (not the artifact key), so:
  - array artifacts  -> returns None -> text fallback counts ~0 items
                        ("'technology_trends' has ~0 list items, need >= 1")
  - object artifacts -> returns the ENVELOPE dict -> dialect reports the
                        first required field missing
                        ("audience_research_data.demographics: missing ...")

Both are false negatives: the content is complete. These tests pin the peel.
"""
from __future__ import annotations

import json

from src.workflows.engine.hooks import _unwrap_envelope, validate_artifact_schema


def _call_result_envelope(inner) -> str:
    """Build the exact on-disk CallResult envelope shape (content double-encoded)."""
    return json.dumps({
        "content": json.dumps(inner) if not isinstance(inner, str) else inner,
        "model": "gemini/gemini-2.5-flash",
        "model_name": "gemini/gemini-2.5-flash",
        "cost": 0.0,
        "usage": {"prompt_tokens": 1136, "completion_tokens": 800},
        "tool_calls": None,
        "latency": 3.95,
        "thinking": None,
        "is_local": False,
        "ran_on": "gemini",
        "provider": "gemini",
        "task": "structured_emit",
        "capability_score": 0.0,
        "difficulty": 5,
    })


# ── _unwrap_envelope peels the CallResult envelope ──────────────────────────

def test_unwrap_peels_call_result_array():
    arr = [{"name": "A"}, {"name": "B"}]
    out = _unwrap_envelope(_call_result_envelope(arr))
    assert json.loads(out) == arr


def test_unwrap_peels_call_result_object():
    obj = {"demographics": {"age_range": "18-45"}, "psychographics": {}}
    out = _unwrap_envelope(_call_result_envelope(obj))
    assert json.loads(out) == obj


def test_unwrap_does_not_touch_legit_object_with_content_field():
    """An artifact that legitimately has a ``content`` field but no CallResult
    sibling markers must NOT be mistaken for an envelope."""
    legit = {"content": "the body text", "title": "My Doc"}
    out = _unwrap_envelope(json.dumps(legit))
    # No model/usage markers → not an envelope → returned as-is (round-trips).
    assert json.loads(out) == legit


def test_unwrap_final_answer_still_works():
    out = _unwrap_envelope(json.dumps({"action": "final_answer", "result": "hello"}))
    assert out == "hello"


# ── end-to-end: validate_artifact_schema accepts the wrapped artifact ───────

_ARRAY_SCHEMA = {
    "technology_trends": {
        "type": "array", "min_items": 1,
        "item_fields": ["name", "description"],
    }
}

_OBJECT_SCHEMA = {
    "audience_research_data": {
        "type": "object",
        "required_fields": ["demographics", "psychographics"],
    }
}


def test_array_artifact_in_envelope_validates():
    inner = [
        {"name": "LLM Coaching", "description": "x"},
        {"name": "Wearable APIs", "description": "y"},
    ]
    ok, err = validate_artifact_schema(_call_result_envelope(inner), _ARRAY_SCHEMA)
    assert ok, err


def test_object_artifact_in_envelope_validates():
    inner = {"demographics": {"age_range": "18-45"}, "psychographics": {"x": 1}}
    ok, err = validate_artifact_schema(_call_result_envelope(inner), _OBJECT_SCHEMA)
    assert ok, err


def test_array_too_few_items_still_fails_through_envelope():
    """The peel must not paper over a genuinely-short artifact."""
    schema = {"trends": {"type": "array", "min_items": 5}}
    ok, err = validate_artifact_schema(_call_result_envelope([{"a": 1}]), schema)
    assert not ok
    assert "trends" in err
