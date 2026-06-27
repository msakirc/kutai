"""verify_falsification_present must re-pend the PRODUCER, not single-shot DLQ.

Mission-90 task 567413 [3.1] functional_requirements_extraction: the analyst
emitted a ~12k requirements array that was valid JSON EXCEPT one corrupt seam
(an unterminated string between FR-010 and FR-011 — a localized generation
glitch, NOT truncation). The Z1 post-hook wiring strict-`json.loads` failed,
silently set `artifacts={}`, the verifier returned `empty=True`, and because
`verify_falsification_present` was routed through the single-shot
`_apply_z1_mechanical_verdict` rail it DLQ'd the producer at wa=1 with the
misleading message `missing=[] critical_underspecified=[] empty=True` — never
giving the model a chance to fix its own JSON.

`verify_falsification_present` judges LLM PRODUCER output (the requirements
array in tasks.result, produces=None), exactly like `prior_art_min_coverage`,
so it belongs in `_PRODUCER_QUALITY_Z1_BLOCKERS` (re-pend-with-feedback rail).
And a parse failure must surface as actionable feedback ("re-emit valid JSON"),
not a silent `empty`.
"""
from __future__ import annotations

import json

from general_beckman.apply import (
    _posthook_agent_and_payload,
    _PRODUCER_QUALITY_Z1_BLOCKERS,
)
from general_beckman.result_router import RequestPostHook


# ── routing: producer-quality, so re-pend instead of single-shot DLQ ──────

def test_falsification_routes_through_producer_repend_rail():
    assert "verify_falsification_present" in _PRODUCER_QUALITY_Z1_BLOCKERS


# ── payload builder surfaces the JSON parse error (no silent empty) ───────

# The real 567413 seam: FR-010's falsification_signal string is never closed
# before FR-011's object opens.
_MALFORMED = (
    '[\n'
    '  {"req_id": "FR-010", "title": "Export",\n'
    '   "falsification_signal": "rate falls below 95% or QA finds a discrepancy\n'
    '  {"req_id": "FR-011", "title": "Search", "falsification_signal": "x"}\n'
    ']'
)

_VALID = json.dumps([
    {"req_id": "FR-1", "title": "t", "description": "d",
     "risk_if_wrong": "high", "validation_method": "weekly audit of logs",
     "falsification_signal": "error rate > 5%"},
])


def test_payload_builder_captures_parse_error_on_malformed_json():
    a = RequestPostHook(source_task_id=3, kind="verify_falsification_present",
                        source_ctx={})
    source_ctx = {"output_artifacts": ["functional_requirements"]}
    _, spec = _posthook_agent_and_payload(
        a, {"id": 3, "result": _MALFORMED}, source_ctx
    )
    payload = spec["payload"]
    assert payload.get("parse_error"), "malformed JSON must surface a parse_error"
    assert payload["artifacts"] == {}, "no items could be parsed"


def test_payload_builder_no_parse_error_on_valid_array():
    a = RequestPostHook(source_task_id=3, kind="verify_falsification_present",
                        source_ctx={})
    source_ctx = {"output_artifacts": ["functional_requirements"]}
    _, spec = _posthook_agent_and_payload(
        a, {"id": 3, "result": _VALID}, source_ctx
    )
    payload = spec["payload"]
    assert "parse_error" not in payload
    assert list(payload["artifacts"].keys()) == ["functional_requirements"]
