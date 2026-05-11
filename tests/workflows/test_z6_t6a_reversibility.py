"""Z6 T6A — every i2p_v3 step has a reversibility tag and the audit
script is idempotent against the committed labels."""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / "src" / "workflows" / "i2p" / "i2p_v3.json"
_SCRIPT = _REPO / "scripts" / "z6_reversibility_audit.py"

_VALID = {"full", "partial", "irreversible"}


def _load_workflow() -> dict:
    with _WORKFLOW.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "z6_reversibility_audit", _SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_step_has_reversibility():
    wf = _load_workflow()
    missing: list[str] = []
    invalid: list[tuple[str, str]] = []
    for step in wf["steps"]:
        rev = step.get("reversibility")
        if rev is None:
            missing.append(step.get("id", "?"))
            continue
        if rev not in _VALID:
            invalid.append((step.get("id", "?"), rev))
    assert not missing, f"steps missing reversibility: {missing[:10]}"
    assert not invalid, f"invalid reversibility values: {invalid[:10]}"


def test_needs_real_tools_steps_are_irreversible():
    wf = _load_workflow()
    nrt = [
        s for s in wf["steps"]
        if s.get("needs_real_tools")
        or (isinstance(s.get("context"), dict)
            and s["context"].get("needs_real_tools"))
    ]
    assert nrt, "expected at least one needs_real_tools step"
    bad = [s["id"] for s in nrt if s.get("reversibility") != "irreversible"]
    assert not bad, f"needs_real_tools steps not irreversible: {bad}"


def test_audit_script_is_idempotent_against_committed_labels():
    """Re-running the script proposes the exact tags currently in JSON."""
    mod = _load_script_module()
    wf = _load_workflow()
    diffs: list[tuple[str, str, str]] = []
    for sid, current, proposed in mod.walk(wf):
        if current != proposed:
            diffs.append((sid, current, proposed))
    assert not diffs, (
        f"audit script disagrees with committed tags: {diffs[:10]}"
    )


def test_propose_helpers():
    mod = _load_script_module()
    # needs_real_tools → irreversible regardless of text.
    assert mod.propose({"needs_real_tools": True, "instruction": ""}) == \
        "irreversible"
    assert mod.propose(
        {"context": {"needs_real_tools": True}, "instruction": ""}
    ) == "irreversible"
    # Trigger word → partial.
    assert mod.propose({"instruction": "Deploy to production"}) == "partial"
    assert mod.propose({"instruction": "git push origin main"}) == "partial"
    # Plain read step → full.
    assert mod.propose({"instruction": "Read the file and lint it"}) == "full"
    assert mod.propose({"name": "snapshot"}) == "full"


def test_workflow_is_valid_json_after_apply():
    """Sanity — the committed workflow still parses (no trailing-comma
    issues from the apply path)."""
    wf = _load_workflow()
    assert isinstance(wf, dict)
    assert isinstance(wf.get("steps"), list)
    assert len(wf["steps"]) > 0
