"""Z1 Tier 6C (T6C / C18) — i2p_v3 wiring tests for end-of-phase-6
GitHub repo init."""
from __future__ import annotations

from src.workflows.engine.loader import load_workflow


def _step(step_id: str) -> dict:
    wf = load_workflow("i2p_v3")
    s = wf.get_step(step_id)
    assert s is not None, f"step {step_id!r} missing in i2p_v3"
    return s


def test_step_6_7_exists_and_is_mechanical():
    s = _step("6.7")
    assert s["agent"] == "mechanical"
    assert s["phase"] == "phase_6"
    assert s["payload"]["action"] == "init_mission_github_repo"


def test_step_6_7_depends_on_6_6():
    s = _step("6.7")
    assert "6.6" in s["depends_on"], (
        "6.7 must wait for project_plan_review (which carries the "
        "compliance_blocker_check post-hook)."
    )


def test_step_6_7_no_legacy_gate():
    # legacy_pre_github_init gate was removed; step is now unconditional
    s = _step("6.7")
    sw = s.get("skip_when") or ""
    assert not sw or "legacy_pre_" not in sw


def test_step_6_7_produces_status_file():
    s = _step("6.7")
    produces = s.get("produces") or []
    assert any("github_init_status.md" in p for p in produces)


def test_step_6_7_default_visibility_public():
    s = _step("6.7")
    assert s["payload"].get("repo_visibility", "public") == "public"
