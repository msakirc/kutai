"""Verify the i2p_v3 workflow has explicit mechanical git_commit steps.

Phase 2a re-enables auto_commit by wiring it in as an explicit workflow
step rather than a dormant orchestrator hook.
"""

import json
from pathlib import Path


WF_PATH = Path(__file__).resolve().parents[2] / "src" / "workflows" / "i2p" / "i2p_v3.json"


def _load_workflow():
    with open(WF_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_i2p_v3_has_at_least_one_mechanical_commit_step():
    wf = _load_workflow()
    mech_steps = [
        s for s in wf["steps"]
        if s.get("agent") == "mechanical"
        and s.get("payload", {}).get("action") == "git_commit"
    ]
    assert mech_steps, "expected ≥1 mechanical git_commit step in i2p_v3.json"


def test_mechanical_commit_steps_depend_on_their_coder_parent():
    wf = _load_workflow()
    for s in wf["steps"]:
        if s.get("agent") != "mechanical":
            continue
        if s.get("payload", {}).get("action") != "git_commit":
            continue
        sid = s["id"]
        deps = s.get("depends_on", [])
        assert deps, f"mechanical step {sid} must declare depends_on"
        # step id looks like "<parent>.git_commit"
        expected_parent = sid.rsplit(".git_commit", 1)[0]
        assert expected_parent in deps, (
            f"step {sid} should depend on its parent {expected_parent}, got {deps}"
        )


def test_mechanical_commit_steps_carry_executor_tag():
    wf = _load_workflow()
    for s in wf["steps"]:
        if s.get("agent") != "mechanical":
            continue
        assert s.get("executor") == "mechanical", (
            f"mechanical step {s['id']} missing executor=mechanical"
        )
