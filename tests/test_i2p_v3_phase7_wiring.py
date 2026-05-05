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


# step_id -> (path fragments expected somewhere in produces, expected post_hooks)
PHASE7_WIRED = {
    "7.2":  ([".pre-commit-config.yaml"],            ["verify_artifacts"]),
    "7.3":  (["backend/.env.example"],               ["verify_artifacts"]),
    "7.4":  (["migrations/"],                        ["verify_artifacts"]),
    "7.5":  (["frontend/package.json"],              ["verify_artifacts"]),
    "7.6":  (["backend/", "frontend/"],              ["verify_artifacts"]),
    "7.7":  (["Dockerfile", "docker-compose.yml"],   ["verify_artifacts"]),
    "7.8":  ([".github/workflows/ci.yml"],           ["verify_artifacts"]),
    "7.9":  (["frontend/src/styles/tokens.css"],     ["verify_artifacts", "code_review"]),
    "7.10": (["frontend/src/components/ui/"],        ["verify_artifacts", "code_review"]),
    "7.11": (["frontend/src/components/ui/"],        ["verify_artifacts", "code_review"]),
}

# Phase-7 wiring complete (was 8 of 10; 7.4/7.6 moved to PHASE7_WIRED via
# any_of + glob support in salako.verify_artifacts).
PHASE7_DEFERRED: set[str] = set()


def _step(step_id: str) -> dict:
    wf = load_workflow("i2p_v3")
    s = wf.get_step(step_id)
    assert s is not None, f"step {step_id!r} missing"
    return s


def _flatten_produces(produces) -> list[str]:
    """Collect every string anywhere in produces (top-level or nested any_of
    list). Lets the fragment matcher look inside any_of alternatives without
    knowing the entry's shape up front."""
    out: list[str] = []
    for entry in produces or []:
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, list):
            for alt in entry:
                if isinstance(alt, str):
                    out.append(alt)
    return out


@pytest.mark.parametrize("step_id,fragments,hooks", [
    (sid, frags, hks) for sid, (frags, hks) in PHASE7_WIRED.items()
])
def test_phase7_step_declares_produces_and_post_hooks(step_id, fragments, hooks):
    step = _step(step_id)
    produces = step.get("produces") or []
    assert produces, f"{step_id} missing produces"
    flat = _flatten_produces(produces)
    for frag in fragments:
        assert any(frag in p for p in flat), (
            f"{step_id} produces {produces} missing fragment {frag!r}"
        )
    assert step.get("post_hooks") == hooks, (
        f"{step_id} post_hooks {step.get('post_hooks')} != {hooks}"
    )


def test_phase7_deferred_set_empty():
    """All Phase-7 candidates from the handoff are now wired."""
    assert PHASE7_DEFERRED == set()


def test_launch_go_no_go_gates_on_approved():
    step = _step("13.14")
    schema = (step.get("artifact_schema") or {}).get("launch_go_no_go")
    assert schema is not None
    fields = schema.get("fields") or {}
    approved = fields.get("approved")
    assert approved == {"type": "boolean", "equals": True}, (
        f"13.14 approved field rule unexpected: {approved}"
    )
