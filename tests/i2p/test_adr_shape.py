"""Z1 Tier 2 (P3 + C7 + A8) — ADR shape / register / cost-curve tests.

Locks the universal-shape ADR validator + register consistency check +
cost-curve presence guard. Pure unit tests: no LLM, no DB, no async
beyond the mr_roboto dispatch smoke checks.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path

import pytest

from mr_roboto import run as mr_roboto_run
from mr_roboto.verify_adr_shape import verify_adr_shape
from mr_roboto.verify_adr_register import verify_adr_register
from mr_roboto.verify_cost_curve_present import verify_cost_curve_present


_FIXTURES = (
    Path(__file__).resolve().parent
    / "reviewer_regression"
    / "fixtures"
    / "v1"
    / "4_16"
)


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


# ────────────────────────────────────────────────────────────────────────────
# verify_adr_shape — accept happy path + reject every failure mode
# ────────────────────────────────────────────────────────────────────────────


def test_verify_adr_shape_accepts_good_adr_set():
    fx = _load("good_adr_set.json")
    for adr in fx["adrs"]:
        res = verify_adr_shape(adr_obj=adr, expected_schema_version="1")
        assert res["ok"], (adr["adr_id"], res)


@pytest.mark.parametrize(
    "field",
    [
        "adr_id",
        "title",
        "status",
        "context",
        "decision",
        "consequences",
        "options_considered",
        "chosen_option_id",
        "falsification_signal",
        "reversal_cost",
        "supersedes_adr_id",
        "_schema_version",
    ],
)
def test_verify_adr_shape_rejects_missing_required_field(field: str):
    fx = _load("good_adr_set.json")
    adr = copy.deepcopy(fx["adrs"][0])
    adr.pop(field, None)
    res = verify_adr_shape(adr_obj=adr)
    assert res["ok"] is False
    assert field in res["missing_fields"], res


def test_verify_adr_shape_rejects_orphan_chosen_option_id():
    fx = _load("good_adr_set.json")
    adr = copy.deepcopy(fx["adrs"][0])
    adr["chosen_option_id"] = "OPT-DOES-NOT-EXIST"
    res = verify_adr_shape(adr_obj=adr)
    assert res["ok"] is False
    assert res["orphan_chosen_option_id"] is True


def test_verify_adr_shape_rejects_invalid_status():
    fx = _load("good_adr_set.json")
    adr = copy.deepcopy(fx["adrs"][0])
    adr["status"] = "wishful"
    res = verify_adr_shape(adr_obj=adr)
    assert res["ok"] is False
    assert res["status_invalid"] is True


def test_verify_adr_shape_rejects_invalid_reversal_cost():
    fx = _load("good_adr_set.json")
    adr = copy.deepcopy(fx["adrs"][0])
    adr["reversal_cost"] = "ginormous"
    res = verify_adr_shape(adr_obj=adr)
    assert res["ok"] is False
    assert res["reversal_cost_invalid"] is True


def test_verify_adr_shape_rejects_unresolved_supersedes_adr_id():
    fx = _load("good_adr_set.json")
    adr = copy.deepcopy(fx["adrs"][0])
    adr["supersedes_adr_id"] = "not-a-real-adr-id"
    res = verify_adr_shape(adr_obj=adr)
    assert res["ok"] is False
    assert res["supersedes_invalid"] is True


def test_verify_adr_shape_rejects_empty_falsification_signal():
    fx = _load("good_adr_set.json")
    adr = copy.deepcopy(fx["adrs"][0])
    adr["falsification_signal"] = ""
    res = verify_adr_shape(adr_obj=adr)
    assert res["ok"] is False
    assert res["falsification_missing"] is True


def test_verify_adr_shape_rejects_schema_version_mismatch():
    fx = _load("good_adr_set.json")
    adr = copy.deepcopy(fx["adrs"][0])
    adr["_schema_version"] = "2"
    res = verify_adr_shape(adr_obj=adr, expected_schema_version="1")
    assert res["ok"] is False
    assert res["schema_version_mismatch"] == {"found": "2", "expected": "1"}


def test_verify_adr_shape_bad_adr_set_fails_shape_failure_modes():
    """bad_adr_set ADR[0] = empty falsification, ADR[2] = orphan chosen.
    ADR[1] is a cost-curve failure caught by verify_cost_curve_present
    (a separate concern; shape validation alone is allowed to pass it)."""
    fx = _load("bad_adr_set.json")
    results = [verify_adr_shape(adr_obj=adr) for adr in fx["adrs"]]
    # Shape failures.
    assert results[0]["ok"] is False
    assert results[0]["falsification_missing"] is True
    assert results[2]["ok"] is False
    assert results[2]["orphan_chosen_option_id"] is True
    # ADR[1] is a cost-curve failure — exercise that path explicitly.
    cost_res = verify_cost_curve_present(adr_obj=fx["adrs"][1])
    assert cost_res["ok"] is False
    assert cost_res["options_missing_curve"]


def test_verify_adr_shape_via_text():
    fx = _load("good_adr_set.json")
    adr = fx["adrs"][0]
    res = verify_adr_shape(adr_text=json.dumps(adr))
    assert res["ok"], res


def test_verify_adr_shape_via_markdown_fence():
    fx = _load("good_adr_set.json")
    adr = fx["adrs"][0]
    md = "Here is the ADR:\n\n```json\n" + json.dumps(adr) + "\n```\n"
    res = verify_adr_shape(adr_text=md)
    assert res["ok"], res


# ────────────────────────────────────────────────────────────────────────────
# verify_adr_register — register vs on-disk consistency
# ────────────────────────────────────────────────────────────────────────────


def test_verify_adr_register_accepts_consistent_register(tmp_path: Path):
    adr_dir = tmp_path / ".adr"
    adr_dir.mkdir()
    for aid in ("ADR-2026-05-10-001", "ADR-2026-05-10-002"):
        (adr_dir / f"{aid}.json").write_text("{}", encoding="utf-8")
    register = adr_dir / "register.md"
    register.write_text(
        "# ADR Register\n\n"
        "- ADR-2026-05-10-001 — A — status:accepted\n"
        "- ADR-2026-05-10-002 — B — status:accepted\n",
        encoding="utf-8",
    )
    res = verify_adr_register(register_path=str(register))
    assert res["ok"], res
    assert sorted(res["referenced"]) == [
        "ADR-2026-05-10-001",
        "ADR-2026-05-10-002",
    ]


def test_verify_adr_register_rejects_when_referenced_file_missing(tmp_path: Path):
    adr_dir = tmp_path / ".adr"
    adr_dir.mkdir()
    # Note: only -001 written, register references -001 AND -002.
    (adr_dir / "ADR-2026-05-10-001.json").write_text("{}", encoding="utf-8")
    register = adr_dir / "register.md"
    register.write_text(
        "- ADR-2026-05-10-001 — A — status:accepted\n"
        "- ADR-2026-05-10-002 — B — status:accepted\n",
        encoding="utf-8",
    )
    res = verify_adr_register(register_path=str(register))
    assert res["ok"] is False
    assert res["missing_files"] == ["ADR-2026-05-10-002"]


def test_verify_adr_register_rejects_orphan_files(tmp_path: Path):
    adr_dir = tmp_path / ".adr"
    adr_dir.mkdir()
    for aid in ("ADR-2026-05-10-001", "ADR-2026-05-10-002"):
        (adr_dir / f"{aid}.json").write_text("{}", encoding="utf-8")
    register = adr_dir / "register.md"
    register.write_text(
        "- ADR-2026-05-10-001 — A — status:accepted\n",
        encoding="utf-8",
    )
    res = verify_adr_register(register_path=str(register))
    assert res["ok"] is False
    assert res["orphan_files"] == ["ADR-2026-05-10-002"]


def test_verify_adr_register_rejects_empty_register():
    res = verify_adr_register(register_text="")
    assert res["ok"] is False


# ────────────────────────────────────────────────────────────────────────────
# verify_cost_curve_present — A8 cost-curve guard for stack ADRs
# ────────────────────────────────────────────────────────────────────────────


def test_verify_cost_curve_present_accepts_populated_curve():
    fx = _load("good_adr_set.json")
    # Second ADR is a stack ADR (database) with cost_curve on every option.
    res = verify_cost_curve_present(adr_obj=fx["adrs"][1])
    assert res["ok"], res


def test_verify_cost_curve_present_rejects_missing_curve_on_option():
    fx = _load("good_adr_set.json")
    adr = copy.deepcopy(fx["adrs"][1])
    adr["options_considered"][1].pop("monthly_cost_curve", None)
    res = verify_cost_curve_present(adr_obj=adr)
    assert res["ok"] is False
    assert res["options_missing_curve"]


def test_verify_cost_curve_present_rejects_partial_curve():
    fx = _load("good_adr_set.json")
    adr = copy.deepcopy(fx["adrs"][1])
    adr["options_considered"][0]["monthly_cost_curve"] = {"at_mvp": "$0"}
    res = verify_cost_curve_present(adr_obj=adr)
    assert res["ok"] is False


def test_verify_cost_curve_present_rejects_missing_top_level_fields():
    fx = _load("good_adr_set.json")
    adr = copy.deepcopy(fx["adrs"][1])
    adr.pop("cost_at_target_users_usd", None)
    adr.pop("cost_mitigation_plan", None)
    res = verify_cost_curve_present(adr_obj=adr)
    assert res["ok"] is False
    assert res["cost_at_target_missing"] is True
    assert res["cost_mitigation_field_missing"] is True


def test_verify_cost_curve_present_allows_null_mitigation_plan():
    fx = _load("good_adr_set.json")
    # Default fixture sets cost_mitigation_plan: null and curve fits ceiling
    res = verify_cost_curve_present(adr_obj=fx["adrs"][1])
    assert res["ok"], res


# ────────────────────────────────────────────────────────────────────────────
# Component-library ADR (C7) — universal shape with stack_adr_ref
# ────────────────────────────────────────────────────────────────────────────


def test_component_library_adr_passes_universal_shape():
    fx = _load("good_component_library.json")
    res = verify_adr_shape(adr_obj=fx["adr"], expected_schema_version="1")
    assert res["ok"], res
    # Stack-adr ref is a non-universal extra field — must not break shape.
    assert fx["adr"]["stack_adr_ref"] == "ADR-2026-05-10-001"


def test_component_library_adr_bad_fails_universal_shape():
    fx = _load("bad_component_library.json")
    res = verify_adr_shape(adr_obj=fx["adr"], expected_schema_version="1")
    assert res["ok"] is False
    assert res["orphan_chosen_option_id"] is True
    assert res["falsification_missing"] is True


# ────────────────────────────────────────────────────────────────────────────
# Mechanical dispatch — exercise mr_roboto.run() arms
# ────────────────────────────────────────────────────────────────────────────


def _run_mr(payload: dict) -> object:
    task = {"id": 0, "mission_id": 0, "payload": payload}
    return asyncio.run(mr_roboto_run(task))


def test_mr_roboto_dispatch_verify_adr_shape_accept():
    fx = _load("good_adr_set.json")
    res = _run_mr({"action": "verify_adr_shape", "adr_obj": fx["adrs"][0]})
    assert res.status == "completed", (res.status, res.error, res.result)


def test_mr_roboto_dispatch_verify_adr_shape_reject():
    fx = _load("bad_adr_set.json")
    res = _run_mr({"action": "verify_adr_shape", "adr_obj": fx["adrs"][0]})
    assert res.status == "failed"


def test_mr_roboto_dispatch_verify_adr_register_accept(tmp_path: Path):
    adr_dir = tmp_path / ".adr"
    adr_dir.mkdir()
    (adr_dir / "ADR-2026-05-10-001.json").write_text("{}", encoding="utf-8")
    (adr_dir / "register.md").write_text(
        "- ADR-2026-05-10-001 — A — status:accepted\n",
        encoding="utf-8",
    )
    res = _run_mr(
        {
            "action": "verify_adr_register",
            "register_path": str(adr_dir / "register.md"),
        }
    )
    assert res.status == "completed", (res.status, res.error)


def test_mr_roboto_dispatch_verify_cost_curve_present_accept():
    fx = _load("good_adr_set.json")
    res = _run_mr(
        {"action": "verify_cost_curve_present", "adr_obj": fx["adrs"][1]}
    )
    assert res.status == "completed", (res.status, res.error)


def test_mr_roboto_dispatch_verify_cost_curve_present_reject():
    fx = _load("good_adr_set.json")
    adr = copy.deepcopy(fx["adrs"][1])
    adr.pop("cost_at_target_users_usd", None)
    res = _run_mr(
        {"action": "verify_cost_curve_present", "adr_obj": adr}
    )
    assert res.status == "failed"


# ────────────────────────────────────────────────────────────────────────────
# i2p_v3.json — Z1 Tier 2 wiring smoke checks
# ────────────────────────────────────────────────────────────────────────────


_WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "workflows"
    / "i2p"
    / "i2p_v3.json"
)


def _load_wf() -> dict:
    return json.loads(_WORKFLOW.read_text(encoding="utf-8"))


def _step(wf: dict, sid: str) -> dict | None:
    return next((s for s in wf["steps"] if s.get("id") == sid), None)


def test_i2p_v3_has_component_library_step():
    wf = _load_wf()
    s = _step(wf, "4.2a")
    assert s is not None, "step 4.2a missing"
    assert s["name"] == "component_library_decision"
    assert s["agent"] == "analyst"
    assert "mission_{mission_id}/.adr/component_library_decision.json" in (
        s.get("produces") or []
    )


def test_i2p_v3_phase4_adr_steps_have_verify_siblings():
    wf = _load_wf()
    for sid in ("4.1", "4.2", "4.4", "4.6", "4.8", "4.9", "4.10", "4.2a"):
        v = _step(wf, f"{sid}.verify")
        assert v is not None, f"missing {sid}.verify"
        assert v["agent"] == "mechanical"
        assert v["payload"]["action"] == "verify_adr_shape"


def test_i2p_v3_stack_steps_have_cost_curve_verify():
    wf = _load_wf()
    for sid in ("4.2", "4.4", "4.6", "4.8", "4.9", "4.10"):
        v = _step(wf, f"{sid}.verify_cost_curve")
        assert v is not None, f"missing {sid}.verify_cost_curve"
        assert v["payload"]["action"] == "verify_cost_curve_present"


def test_i2p_v3_4_14_has_register_verify():
    wf = _load_wf()
    v = _step(wf, "4.14.verify_register")
    assert v is not None
    assert v["payload"]["action"] == "verify_adr_register"


def test_i2p_v3_4_16_reviewer_instruction_extended():
    wf = _load_wf()
    s = _step(wf, "4.16")
    assert s is not None
    instr = s["instruction"]
    # All seven checks must be referenced in the prompt.
    for needle in (
        "options_considered",
        "chosen_option_id",
        "falsification_signal",
        "reversal_cost",
        "monthly_cost_curve",
        "cost_at_target_users_usd",
        "component-library ADR from step 4.2a",
        "charter `solution.id`",
    ):
        assert needle in instr, f"reviewer instruction missing: {needle!r}"


def test_i2p_v3_adr_steps_have_no_legacy_gate():
    # legacy_pre_adr gate was removed; ADR steps are now unconditional
    wf = _load_wf()
    for sid in ("4.1", "4.2", "4.2a", "4.4", "4.6", "4.8", "4.9", "4.10", "4.14"):
        s = _step(wf, sid)
        assert s is not None, sid
        sw = s.get("skip_when") or ""
        assert not sw or "legacy_pre_" not in sw, (
            f"step {sid} still has a legacy_pre_ gate: {sw!r}"
        )


def test_i2p_v3_adr_artifact_schema_carries_universal_fields():
    wf = _load_wf()
    expected_required = {
        "adr_id",
        "title",
        "status",
        "context",
        "decision",
        "consequences",
        "options_considered",
        "chosen_option_id",
        "falsification_signal",
        "reversal_cost",
        "supersedes_adr_id",
    }
    for sid, label in (
        ("4.1", "architecture_pattern_decision"),
        ("4.2", "tech_stack_decision"),
        ("4.4", "database_schema_decision"),
        ("4.6", "auth_design_decision"),
        ("4.8", "third_party_selections_decision"),
        ("4.9", "infrastructure_designs_decision"),
        ("4.10", "communication_designs_decision"),
        ("4.2a", "component_library_decision"),
    ):
        s = _step(wf, sid)
        sch = s["artifact_schema"][label]
        # Z3 T4A: step 4.4 bumped to v2 (structured falsification_signal).
        # Other steps remain on v1 pending follow-up migration.
        expected_version = "2" if sid == "4.4" else "1"
        assert sch["_schema_version"] == expected_version, sid
        assert expected_required.issubset(set(sch["required_fields"])), sid
