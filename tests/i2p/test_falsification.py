"""Z1 Tier 2 (P4) — verify_falsification_present contract tests."""
from __future__ import annotations

import asyncio

from mr_roboto.verify_falsification_present import (
    REQUIRED_TRIPLE,
    verify_falsification_present,
)
from mr_roboto import run as mr_roboto_run


def _good_item(req_id: str = "FR-001", risk: str = "high") -> dict:
    return {
        "req_id": req_id,
        "title": "Daily price scan triggers cron",
        "description": "Cron job at 02:00 TR-time pulls prices.",
        "priority": "must",
        "risk_if_wrong": risk,
        "validation_method": (
            "monitor founder-reported missed-window complaints over 30d"
        ),
        "falsification_signal": "5+ missed-window complaints in 30 days",
    }


# ────────────────────────────────────────────────────────────────────────────
# Pure-function unit tests
# ────────────────────────────────────────────────────────────────────────────


def test_accepts_populated_triples():
    res = verify_falsification_present(
        artifacts={
            "functional_requirements": [
                _good_item("FR-001"),
                _good_item("FR-002", risk="medium"),
            ]
        }
    )
    assert res["ok"] is True, res
    assert res["checked"] == 2
    assert res["missing"] == []
    assert res["critical_underspecified"] == []
    assert res["empty"] is False


def test_rejects_missing_triple_field():
    bad = _good_item("FR-003")
    bad.pop("validation_method")
    res = verify_falsification_present(
        artifacts={"functional_requirements": [bad]}
    )
    assert res["ok"] is False
    assert res["missing"][0]["item_id"] == "FR-003"
    assert "validation_method" in res["missing"][0]["missing_fields"]


def test_rejects_bad_risk_enum():
    bad = _good_item("FR-004", risk="catastrophic")
    res = verify_falsification_present(
        artifacts={"functional_requirements": [bad]}
    )
    assert res["ok"] is False
    msg = " ".join(res["missing"][0]["missing_fields"])
    assert "catastrophic" in msg or "risk_if_wrong" in msg


def test_critical_requires_specific_validation():
    bad = _good_item("FR-005", risk="critical")
    bad["validation_method"] = "users will tell us"
    res = verify_falsification_present(
        artifacts={"functional_requirements": [bad]}
    )
    assert res["ok"] is False
    assert res["critical_underspecified"]
    assert res["critical_underspecified"][0]["item_id"] == "FR-005"


def test_critical_with_specific_validation_passes():
    good = _good_item("FR-006", risk="critical")
    good["validation_method"] = "monitor latency p95 < 500ms over 7d"
    res = verify_falsification_present(
        artifacts={"functional_requirements": [good]}
    )
    assert res["ok"] is True


def test_critical_accepts_concrete_methods_without_keywords():
    """Lenient heuristic: real measurable methods that match no keyword and
    carry no digit must still pass (mission-81 3.3/3.7 false-DLQ). Full rigor
    lives at reviewer 3.11 — the pre-gate only rejects vague rationalizations.
    """
    for method in (
        "Automated security penetration test targeting session bypass.",
        "Quarterly full-scale disaster recovery (DR) drill simulation",
        "Monthly backup integrity testing and point-in-time recovery validation",
    ):
        item = _good_item("BR-X", risk="critical")
        item["validation_method"] = method
        res = verify_falsification_present(
            artifacts={"business_rules": [item]}
        )
        assert res["ok"] is True, (method, res)
        assert res["critical_underspecified"] == [], (method, res)


def test_critical_rejects_terse_nonanswer():
    """A critical method too short to name a concrete check still fails."""
    bad = _good_item("FR-007", risk="critical")
    bad["validation_method"] = "we test"
    res = verify_falsification_present(
        artifacts={"functional_requirements": [bad]}
    )
    assert res["ok"] is False
    assert res["critical_underspecified"]


def test_walks_dict_with_items_array():
    res = verify_falsification_present(
        artifacts={
            "nfr_performance": {
                "api_response_time": "p95<500ms",
                "items": [
                    _good_item("NFR-perf-001"),
                ],
            }
        }
    )
    assert res["ok"] is True
    assert res["checked"] == 1


def test_walks_personas_array():
    res = verify_falsification_present(
        artifacts={
            "personas": {
                "personas": [
                    _good_item("persona-1"),
                ]
            }
        }
    )
    assert res["ok"] is True


def test_empty_signals_wiring_bug():
    res = verify_falsification_present(
        artifacts={"functional_requirements": []}
    )
    assert res["ok"] is False
    assert res["empty"] is True



# ────────────────────────────────────────────────────────────────────────────
# Mechanical dispatch tests
# ────────────────────────────────────────────────────────────────────────────


def test_dispatch_completed_on_good():
    task = {
        "id": 0,
        "mission_id": 0,
        "payload": {
            "action": "verify_falsification_present",
            "artifacts": {
                "functional_requirements": [_good_item()],
            },
        },
    }
    res = asyncio.run(mr_roboto_run(task))
    assert res.status == "completed"
    assert res.result["ok"] is True


def test_dispatch_failed_on_bad():
    bad = _good_item()
    bad.pop("falsification_signal")
    task = {
        "id": 0,
        "mission_id": 0,
        "payload": {
            "action": "verify_falsification_present",
            "artifacts": {"functional_requirements": [bad]},
        },
    }
    res = asyncio.run(mr_roboto_run(task))
    assert res.status == "failed"
    assert "verify_falsification_present" in (res.error or "")


def _wire(result: str, output_names=("functional_requirements",)):
    """Drive the apply.py post-hook wiring that builds the `artifacts`
    payload from a source task's raw `result` string."""
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    a = RequestPostHook(
        source_task_id=1,
        kind="verify_falsification_present",
        source_ctx={"output_artifacts": list(output_names)},
    )
    source = {"id": 1, "result": result}
    source_ctx = {"output_artifacts": list(output_names)}
    _agent, spec = _posthook_agent_and_payload(a, source, source_ctx)
    return spec["payload"]["artifacts"]


def test_wiring_unwraps_fenced_json_array():
    """A ```json-fenced array of items must reach the validator, not be
    swallowed as an unparseable string (empty=True wiring failure)."""
    import json

    body = json.dumps([_good_item("BR-001"), _good_item("BR-002")])
    fenced = f"```json\n{body}\n```"
    artifacts = _wire(fenced)
    res = verify_falsification_present(artifacts=artifacts)
    assert res["empty"] is False, artifacts
    assert res["checked"] == 2


def test_wiring_unwraps_narration_wrapped_artifact():
    """LLM prose preamble + a fenced artifact (mission-81 3.2 nfr) must be
    unwrapped before json.loads — the artifact is complete, only buried."""
    import json

    inner = {"items": [_good_item("NFR-001"), _good_item("NFR-002")]}
    result = (
        "Okay, I will analyze the provided information to define the NFRs.\n\n"
        "Here is the artifact:\n\n```json\n" + json.dumps(inner) + "\n```"
    )
    artifacts = _wire(result, output_names=("nfr_performance",))
    res = verify_falsification_present(artifacts=artifacts)
    assert res["empty"] is False, artifacts
    assert res["checked"] == 2
    assert res["ok"] is True


def test_wiring_plain_json_still_works():
    """Regression: an un-fenced JSON object must still parse."""
    import json

    inner = {"functional_requirements": [_good_item("FR-001")]}
    artifacts = _wire(json.dumps(inner))
    res = verify_falsification_present(artifacts=artifacts)
    assert res["empty"] is False
    assert res["checked"] == 1


def test_required_triple_constant():
    assert REQUIRED_TRIPLE == (
        "risk_if_wrong",
        "validation_method",
        "falsification_signal",
    )


def test_workflow_step_carries_falsification_post_hook():
    """i2p_v3.json must wire verify_falsification_present on requirement steps."""
    import json
    from pathlib import Path

    wf_path = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "workflows"
        / "i2p"
        / "i2p_v3.json"
    )
    wf = json.loads(wf_path.read_text(encoding="utf-8"))
    by_id = {s["id"]: s for s in wf["steps"]}

    for sid in ("3.1", "3.2", "3.3", "3.7"):
        step = by_id[sid]
        assert "verify_falsification_present" in (step.get("post_hooks") or []), (
            f"step {sid} missing verify_falsification_present post_hook"
        )

    # The post_hook IS the gate now. Standalone `.verify` sibling steps were
    # removed (the post-hook runs the same check earlier + cheaper); assert
    # they stay gone so the wiring isn't silently double-run.
    for sid in ("3.1.verify", "3.2.verify", "3.3.verify", "3.7.verify"):
        assert sid not in by_id, (
            f"step {sid} reappeared — post_hook is the sole falsification gate"
        )
