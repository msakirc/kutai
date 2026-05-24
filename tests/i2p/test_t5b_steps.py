"""Z1 Tier 5B (T5B) — i2p_v3 wiring tests for premortem (A6) and
spec-stays-alive wave-start steps (B5)."""
from __future__ import annotations

import pytest

from src.workflows.engine.loader import load_workflow


def _step(step_id: str) -> dict:
    wf = load_workflow("i2p_v3")
    s = wf.get_step(step_id)
    assert s is not None, f"step {step_id!r} missing in i2p_v3"
    return s


# ── A6 premortem ──────────────────────────────────────────────────────────────


def test_premortem_step_exists_and_depends_on_6_5():
    s = _step("6.5z")
    assert s["agent"] == "analyst"
    assert "6.5" in s["depends_on"]
    # legacy_pre_premortem gate was removed; step is now unconditional
    assert not s.get("skip_when") or "legacy_pre_" not in s.get("skip_when", "")
    produces = s.get("produces") or []
    assert any("premortem.md" in p for p in produces)


def test_premortem_verify_shape_sibling_exists():
    s = _step("6.5z.verify_shape")
    assert s["agent"] == "mechanical"
    assert s["payload"]["action"] == "verify_premortem_shape"
    assert "6.5z" in s["depends_on"]


def test_6_6_reviewer_depends_on_premortem_and_mentions_check():
    s = _step("6.6")
    assert "6.5z" in s["depends_on"], (
        "6.6 must wait for the premortem so the reviewer's premortem-coverage "
        "augmentation has the artifact available."
    )
    instr = s["instruction"]
    # Augmentation: must mention premortem + plausibility threshold.
    assert "premortem" in instr.lower()
    assert "plausibility" in instr.lower()
    assert "monitoring" in instr.lower()


# ── B5 spec_consistency_check wave-start steps ───────────────────────────────


WAVE_PHASES = (
    ("7.0z", "phase_7", "6.6"),
    ("8.0z", "phase_8", "7.17"),
    # 9.0z was updated to depend on 8.spike.git_commit (not 8.arch_check which is recurring)
    ("9.0z", "phase_9", "8.spike.git_commit"),
    ("10.0z", "phase_10", "9.11"),
    ("11.0z", "phase_11", "10.9"),
    ("12.0z", "phase_12", "11.5"),
)


@pytest.mark.parametrize("step_id,phase,prev_step", WAVE_PHASES)
def test_wave_start_step_exists_and_correct_shape(step_id, phase, prev_step):
    s = _step(step_id)
    assert s["agent"] == "mechanical"
    assert s["phase"] == phase
    assert prev_step in s["depends_on"], (
        f"{step_id} must depend on prior phase's last step {prev_step}"
    )
    assert s["payload"]["action"] == "spec_consistency_check"
    assert s["payload"].get("current_phase") == phase
    # legacy_pre_spec_alive gate was removed; step is now unconditional
    sw = s.get("skip_when", "")
    assert not sw or "legacy_pre_" not in sw
    produces = s.get("produces") or []
    assert any("spec_drift_report.md" in p for p in produces)


def test_existing_phase_7_first_step_unmodified():
    """We must NOT have changed 7.1's existing depends_on (T5B brief)."""
    s = _step("7.1")
    # 7.1 still depends on 6.6 — wave-start runs in parallel, doesn't gate.
    assert "6.6" in s["depends_on"]
    # We should not have inserted 7.0z into 7.1's depends_on.
    assert "7.0z" not in s["depends_on"]
