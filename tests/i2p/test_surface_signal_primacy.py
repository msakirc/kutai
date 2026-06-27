"""Guard: i2p step 3.6 must honor the founder-grounded surface_signal and not
let PRD prose flip target_platform — the 'app' → web build-rail bug (2026-06-27).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "workflows" / "i2p" / "i2p_v3.json"
)


@pytest.fixture(scope="module")
def workflow():
    with open(WORKFLOW_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _step_by_id(workflow: dict, step_id: str) -> dict:
    for s in workflow["steps"]:
        if s.get("id") == step_id:
            return s
    raise KeyError(f"step {step_id!r} not found")


def test_step_3_6_does_not_let_prose_override_founder_signal(workflow):
    instr = _step_by_id(workflow, "3.6")["instruction"].lower()
    # The old escape that let PRD prose override a medium founder signal is gone.
    assert "unless the prd explicitly contradicts" not in instr
    # Founder-words primacy is explicit for the medium case.
    assert "do not override it from prd prose" in instr
    assert "founder themselves" in instr
