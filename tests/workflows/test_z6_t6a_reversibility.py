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
    """Real-tool steps with write side-effects must be irreversible.
    Steps marked ``read_only: true`` are exempt — they call vendor APIs for
    status polling or metadata authoring without mutating remote state, so
    their reversibility is correctly ``full``.
    """
    wf = _load_workflow()
    nrt = [
        s for s in wf["steps"]
        if (
            s.get("needs_real_tools")
            or (isinstance(s.get("context"), dict)
                and s["context"].get("needs_real_tools"))
        )
        and not s.get("read_only")  # read-only vendor calls are exempt
    ]
    assert nrt, "expected at least one needs_real_tools (non-read-only) step"
    bad = [s["id"] for s in nrt if s.get("reversibility") != "irreversible"]
    assert not bad, f"needs_real_tools steps not irreversible: {bad}"


def test_audit_script_never_downgrades_committed_labels():
    """The audit may be more lenient than a manual override (a maintainer
    can tag a step ``irreversible`` even when the heuristic would say
    ``full``). The rule we enforce is: the script's proposal must never
    be *less restrictive* than what's already in the JSON.
    """
    mod = _load_script_module()
    wf = _load_workflow()
    rank = {"full": 0, "partial": 1, "irreversible": 2}
    # ``locked`` steps are deliberate manual safety overrides (the Z0
    # dangerous-step audit + the app-store upload sub-steps). The heuristic
    # is not expected to reproduce them — by definition a maintainer tagged
    # them stricter than any text/needs_real_tools signal would. Exempt them
    # so the threshold below still catches *accidental* over-tightening on
    # ordinary steps.
    locked_ids = {s.get("id") for s in wf["steps"] if s.get("locked")}
    # Non-locked steps the heuristic can't reason about but a maintainer
    # deliberately tightened. Each is well-understood; the heuristic only
    # sees needs_real_tools + a few trigger words, so it can't infer these.
    # New entries here require a human deciding the manual tag is correct —
    # that is the bite of this test: any downgrade NOT on this list (and not
    # locked) is an accidental over-tightening and fails.
    ALLOWED_NONLOCKED_DOWNGRADES = {
        "8.0ab",                  # spec_consistency wave-start (multi-file spec)
        "8.0b",                   # spec_consistency wave-start (multi-file spec)
        "13.11b",                 # manual irreversible
        "13.12",                  # manual partial
        "13.demo_distribute",     # uploads demo to YouTube (external, partial)
        "14.8.preview",           # emits a tunneled preview URL (external)
        "15.14z_kill_preview_url",  # preview-URL lifecycle cleanup
    }
    unexpected: list[tuple[str, str, str]] = []
    for sid, current, proposed in mod.walk(wf):
        if not current or sid in locked_ids or sid in ALLOWED_NONLOCKED_DOWNGRADES:
            continue
        if rank[proposed] < rank[current]:
            unexpected.append((sid, current, proposed))
    assert not unexpected, (
        f"audit would downgrade {len(unexpected)} tag(s) not on the locked "
        f"or allowed lists (accidental over-tightening?): {unexpected[:10]}"
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
