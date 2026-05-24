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


def test_legacy_pre_falsification_short_circuits():
    res = verify_falsification_present(
        artifacts={"functional_requirements": [_good_item()]},
        legacy_pre_falsification=True,
    )
    assert res["ok"] is True
    assert res["legacy_pre_falsification"] is True


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

    # Sibling verify steps exist; legacy_pre_falsification gate removed — now unconditional.
    for sid in ("3.1.verify", "3.2.verify", "3.3.verify", "3.7.verify"):
        step = by_id[sid]
        assert step["agent"] == "mechanical"
        sw = step.get("skip_when") or ""
        assert not sw or "legacy_pre_" not in sw, (
            f"step {sid} still has a legacy_pre_ gate: {sw!r}"
        )
        assert step["payload"]["action"] == "verify_falsification_present"
