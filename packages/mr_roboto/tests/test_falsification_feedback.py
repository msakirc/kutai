"""verify_falsification_present must give the producer ACTIONABLE feedback when
it can't find any items — not a bare `empty=True`.

When the wiring can't parse the producer's output (mission-90 567413: a corrupt
JSON seam), `artifacts` arrives empty and the verifier reports `empty=True`. For
the producer-re-pend rail to converge, the result must carry an `error` that
tells the model what to fix ("re-emit a valid JSON array"), optionally quoting
the JSON parse error. `_adapt_shape_findings` turns that `error` into the retry
feedback the model sees.
"""
from __future__ import annotations

import pytest

import mr_roboto


_GOOD_ITEM = {
    "req_id": "FR-1", "risk_if_wrong": "high",
    "validation_method": "weekly audit of export logs",
    "falsification_signal": "export success rate < 95%",
}


@pytest.mark.asyncio
async def test_empty_artifacts_yield_actionable_error():
    action = await mr_roboto.run(
        {"id": 1, "payload": {"action": "verify_falsification_present",
                              "artifacts": {}}}
    )
    assert action.status == "failed"
    res = action.result
    assert res["empty"] is True
    assert res.get("error"), "empty result must carry actionable feedback"
    assert "JSON" in res["error"]


@pytest.mark.asyncio
async def test_parse_error_is_quoted_in_feedback():
    action = await mr_roboto.run(
        {"id": 1, "payload": {"action": "verify_falsification_present",
                              "artifacts": {},
                              "parse_error": "Expecting ',' delimiter: line 111 col 6"}}
    )
    res = action.result
    assert "Expecting ',' delimiter" in res.get("error", "")


@pytest.mark.asyncio
async def test_passes_when_items_carry_triple():
    action = await mr_roboto.run(
        {"id": 1, "payload": {"action": "verify_falsification_present",
                              "artifacts": {"functional_requirements": [_GOOD_ITEM]}}}
    )
    assert action.status == "completed"
    assert action.result["ok"] is True
