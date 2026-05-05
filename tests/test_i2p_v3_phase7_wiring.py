"""Coverage: phase-7 scaffold steps now declare ``produces`` + ``post_hooks``
so the verify_artifacts (and code_review where applicable) post-hook runs
after each top-level scaffold completes. Top-level expander propagates
both fields into task ctx — symmetric to the feature-template wiring.

Stack-variant steps (7.4 database_setup, 7.6 test_infrastructure) are
intentionally NOT wired: the file paths are too tech-stack-dependent to
pin without false positives. Document via test exclusion.
"""
from __future__ import annotations

import pytest

from src.workflows.engine.loader import load_workflow


# step_id -> (path fragments expected in produces, expected post_hooks)
PHASE7_WIRED = {
    "7.2":  ([".pre-commit-config.yaml"],            ["verify_artifacts"]),
    "7.3":  (["backend/.env.example"],               ["verify_artifacts"]),
    "7.5":  (["frontend/package.json"],              ["verify_artifacts"]),
    "7.7":  (["Dockerfile", "docker-compose.yml"],   ["verify_artifacts"]),
    "7.8":  ([".github/workflows/ci.yml"],           ["verify_artifacts"]),
    "7.9":  (["frontend/src/styles/tokens.css"],     ["verify_artifacts", "code_review"]),
    "7.10": (["frontend/src/components/ui/"],        ["verify_artifacts", "code_review"]),
    "7.11": (["frontend/src/components/ui/"],        ["verify_artifacts", "code_review"]),
}

# Skipped intentionally: too stack-variant for a single canonical path.
PHASE7_DEFERRED = {"7.4", "7.6"}


def _step(step_id: str) -> dict:
    wf = load_workflow("i2p_v3")
    s = wf.get_step(step_id)
    assert s is not None, f"step {step_id!r} missing"
    return s


@pytest.mark.parametrize("step_id,fragments,hooks", [
    (sid, frags, hks) for sid, (frags, hks) in PHASE7_WIRED.items()
])
def test_phase7_step_declares_produces_and_post_hooks(step_id, fragments, hooks):
    step = _step(step_id)
    produces = step.get("produces") or []
    assert produces, f"{step_id} missing produces"
    for frag in fragments:
        assert any(frag in p for p in produces), (
            f"{step_id} produces {produces} missing fragment {frag!r}"
        )
    assert step.get("post_hooks") == hooks, (
        f"{step_id} post_hooks {step.get('post_hooks')} != {hooks}"
    )


@pytest.mark.parametrize("step_id", sorted(PHASE7_DEFERRED))
def test_phase7_deferred_steps_have_no_produces(step_id):
    """Stack-variant scaffolds left unwired by design — tracked here so
    follow-up work that wires them can drop the entry from PHASE7_DEFERRED."""
    step = _step(step_id)
    assert "produces" not in step, (
        f"{step_id} now has produces — move it from PHASE7_DEFERRED to PHASE7_WIRED"
    )


def test_launch_go_no_go_gates_on_approved():
    step = _step("13.14")
    schema = (step.get("artifact_schema") or {}).get("launch_go_no_go")
    assert schema is not None
    fields = schema.get("fields") or {}
    approved = fields.get("approved")
    assert approved == {"type": "boolean", "equals": True}, (
        f"13.14 approved field rule unexpected: {approved}"
    )
