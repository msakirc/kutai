"""Z10 T4A — i2p_v3.json demo step insertion validation."""
from __future__ import annotations

import json
import os

import pytest


def _load_v3():
    path = os.path.join("src", "workflows", "i2p", "i2p_v3.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _by_id(steps, sid):
    for s in steps:
        if s.get("id") == sid:
            return s
    return None


def test_record_demo_step_exists():
    wf = _load_v3()
    step = _by_id(wf["steps"], "15.10b_record_demo")
    assert step is not None, "15.10b_record_demo missing"
    assert step["agent"] == "mechanical"
    assert step["executor"] == "record_demo"
    assert "demo.mp4" in (step.get("produces") or [])
    # depends_on must include the prior phase-15 production step
    assert "15.10" in (step.get("depends_on") or [])
    # skip_when token for the no-e2e case
    assert "no_e2e_specs" in (step.get("skip_when") or [])
    # z10-wire-fixes F3: scenario_path is no longer hardcoded — record_demo
    # resolves it from missions.demo_scenario_path / newest e2e spec at run
    # time. Payload should NOT pin it any more.
    pay = step.get("payload") or {}
    assert "scenario_path" not in pay, (
        "F3: scenario_path must not be hardcoded — record_demo resolves at runtime"
    )
    assert int(pay.get("max_seconds")) == 90


def test_verify_demo_step_exists_and_blocks():
    wf = _load_v3()
    verify = _by_id(wf["steps"], "15.10c_verify_demo")
    assert verify is not None, "15.10c_verify_demo missing"
    assert verify["agent"] == "mechanical"
    assert verify["executor"] == "verify_demo_artifact"
    # verify_demo must depend on record_demo so it sits between record and bundle
    assert "15.10b_record_demo" in (verify.get("depends_on") or [])

    # roadmap_update (15.14) must transitively depend on verify_demo so the
    # demo gate blocks mission-final roadmap step.
    roadmap = _by_id(wf["steps"], "15.14")
    assert roadmap is not None
    assert "15.10c_verify_demo" in (roadmap.get("depends_on") or [])


def test_deliverable_bundle_step_exists():
    wf = _load_v3()
    bundle = _by_id(wf["steps"], "15.14b_deliverable_bundle")
    assert bundle is not None, "15.14b_deliverable_bundle missing"
    assert bundle["agent"] == "mechanical"
    assert bundle["executor"] == "mission_deliverable_bundle"
    deps = bundle.get("depends_on") or []
    # Must come after both verify_demo and roadmap_update
    assert "15.10c_verify_demo" in deps
    assert "15.14" in deps


def test_existing_step_ids_unchanged():
    """Verify the insertion didn't renumber any prior phase-15 step."""
    wf = _load_v3()
    expected = {
        "15.1", "15.2", "15.3", "15.4", "15.5", "15.6", "15.7", "15.8",
        "15.9", "15.10", "15.11", "15.12", "15.13", "15.14",
    }
    have = {s.get("id") for s in wf["steps"]}
    missing = expected - have
    assert not missing, f"original phase-15 ids dropped: {missing}"
