"""Tests for the i2p dry-run simulator."""
import json
from pathlib import Path

import pytest

from fatih_hoca.simulate_i2p import (
    DIFFICULTY_MAP,
    simulate,
    build_report,
    _FakeNerdHerd,
)


def test_difficulty_mapping():
    assert DIFFICULTY_MAP["easy"] == 3
    assert DIFFICULTY_MAP["medium"] == 5
    assert DIFFICULTY_MAP["hard"] == 8


def test_fake_nerd_herd_snapshot_is_stable():
    nh = _FakeNerdHerd()
    s1 = nh.snapshot()
    s2 = nh.snapshot()
    assert s1.vram_available_mb == s2.vram_available_mb
    assert s1.vram_available_mb == 7000


def test_fake_nerd_herd_idle_seconds_default():
    """Default idle_seconds=300 so local urgency = 0.5 in simulations."""
    nh = _FakeNerdHerd()
    snap = nh.snapshot()
    assert snap.local.idle_seconds == 300.0


def test_fake_nerd_herd_idle_seconds_override():
    """idle_seconds can be overridden for explicit zero-urgency tests."""
    nh = _FakeNerdHerd(idle_seconds=0.0)
    snap = nh.snapshot()
    assert snap.local.idle_seconds == 0.0


def test_simulate_with_loaded_model(tmp_path):
    """loaded_model kwarg is accepted without error."""
    workflow = {
        "steps": [
            {"id": "1.1", "name": "s1", "agent": "coder", "difficulty": "easy"},
        ]
    }
    workflow_path = tmp_path / "wf.json"
    workflow_path.write_text(json.dumps(workflow))
    # loaded_model not in registry for this minimal workflow → should warn, not crash
    records = simulate(workflow_path, loaded_model="nonexistent-model")
    assert len(records) == 1


def test_simulate_returns_one_record_per_step(tmp_path):
    """Given a minimal workflow of 3 steps, simulator produces 3 records."""
    workflow = {
        "steps": [
            {"id": "1.1", "name": "s1", "agent": "coder", "difficulty": "easy",
             "tools_hint": ["shell"]},
            {"id": "1.2", "name": "s2", "agent": "researcher", "difficulty": "medium",
             "tools_hint": []},
            {"id": "1.3", "name": "s3", "agent": "analyst", "difficulty": "hard",
             "tools_hint": []},
        ]
    }
    workflow_path = tmp_path / "wf.json"
    workflow_path.write_text(json.dumps(workflow))

    records = simulate(workflow_path)
    assert len(records) == 3
    assert {r["step_id"] for r in records} == {"1.1", "1.2", "1.3"}
    for r in records:
        assert "picked_model" in r
        assert "picked_score" in r
        assert "agent" in r
        assert "difficulty" in r


def test_build_report_aggregates_picks():
    records = [
        {"step_id": "1", "task_name": "a", "agent": "coder", "difficulty": 5,
         "picked_model": "m1", "picked_score": 8.0, "top3": []},
        {"step_id": "2", "task_name": "b", "agent": "coder", "difficulty": 5,
         "picked_model": "m1", "picked_score": 7.5, "top3": []},
        {"step_id": "3", "task_name": "c", "agent": "researcher", "difficulty": 5,
         "picked_model": "m2", "picked_score": 9.0, "top3": []},
    ]
    report = build_report(records)
    assert report["total_steps"] == 3
    assert report["coverage"] == 3
    dist = {row[0]: row[1] for row in report["distribution"]}
    assert dist["m1"] == 2
    assert dist["m2"] == 1


def test_simulate_real_i2p_smoke():
    """Smoke: simulate real i2p_v3.json, assert non-empty, most steps pick something."""
    # Walk up from the test file to find the repo root, then into src/workflows.
    here = Path(__file__).resolve()
    # Test file: packages/fatih_hoca/tests/test_simulate_i2p.py
    # parents: [0]=tests [1]=fatih_hoca [2]=packages [3]=repo root
    wf = here.parents[3] / "src" / "workflows" / "i2p" / "i2p_v3.json"
    if not wf.exists():
        wf = None
    if wf is None:
        pytest.skip("i2p_v3.json not found in expected locations")
    records = simulate(wf)
    assert len(records) > 100
    picked = [r for r in records if r["picked_model"] != "<none>"]
    # Require >=60% coverage. Registry may not cover every exotic agent type.
    assert len(picked) / len(records) > 0.6
