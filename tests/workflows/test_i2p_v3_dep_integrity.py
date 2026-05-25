"""i2p_v3 workflow dependency-integrity guards.

Mission #71 (first end-to-end run, 2026-05-22) exposed a class of bug invisible
to grep: a one-shot gate step (``9.0z`` spec_consistency_check_phase_9) declared
``depends_on: ["8.arch_check"]``, but ``8.arch_check`` is a ``type:recurring``
step that never materialises as a one-shot task. ``resolve_dependencies``
(runner.py) warns-and-drops the unresolved ref, leaving the gate with empty
deps → Beckman admits it in Phase 0 → the consistency check runs before any
spec exists and passes vacuously.

These guards keep the static DAG honest: every hard dependency must point at a
step that will actually become a task.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

WF_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "workflows" / "i2p" / "i2p_v3.json"
)


def _load_steps() -> list[dict]:
    wf = json.load(io.open(WF_PATH, encoding="utf-8"))  # noqa: SIM115 (utf-8 required)
    return wf.get("steps", [])


def test_no_missing_dependency_refs():
    """Every depends_on entry must reference a defined step id."""
    steps = _load_steps()
    by_id = {s["id"]: s for s in steps}
    offenders = [
        (s["id"], dep)
        for s in steps
        for dep in (s.get("depends_on") or [])
        if dep not in by_id
    ]
    assert not offenders, f"depends_on points at undefined steps: {offenders}"


def test_no_forward_reference_dependencies():
    """Every depends_on must point at an EARLIER step in the list.

    The runner resolves depends_on step-ids to DB task-ids in creation
    (list) order; a forward reference (depending on a step defined later)
    resolves against a not-yet-created task, the dep is silently dropped,
    and the dependent dispatches immediately. Production 2026-05-25 mission
    76 step 0.6a (non_goals_confirm) was placed before its dependency
    0.6a.verify → the confirm fired with no draft on disk. The whole
    workflow must stay topologically ordered.
    """
    steps = _load_steps()
    pos = {s["id"]: i for i, s in enumerate(steps)}
    offenders = [
        (s["id"], dep)
        for s in steps
        for dep in (s.get("depends_on") or [])
        if dep in pos and pos[dep] > pos[s["id"]]
    ]
    assert not offenders, (
        f"forward-reference depends_on (dependency defined AFTER the "
        f"dependent — will dispatch unguarded): {offenders}"
    )


def test_intake_draft_uses_a_file_writing_agent():
    """`0.0a.draft` must produce intake_todo_draft.json on disk. The `analyst`
    agent could not (its toolset/prompt is research/report-oriented → it
    narrated "## Analysis…" instead of writing the file, mission #73). It must
    use a file-writing agent (`writer`), mirroring reverse_pitch_draft."""
    by_id = {s["id"]: s for s in _load_steps()}
    step = by_id["0.0a.draft"]
    assert step["agent"] == "writer", (
        f"0.0a.draft must use a file-writing agent, got {step['agent']!r}"
    )
    # It declares a produces file, so it genuinely needs write capability.
    assert any(".intake/intake_todo_draft.json" in p for p in step.get("produces", []))


def test_consistency_gates_depend_on_concrete_steps():
    """Each spec_consistency_check wave-start gate must depend on a present,
    non-recurring step — otherwise it admits unguarded (the #71 9.0z bug)."""
    steps = _load_steps()
    by_id = {s["id"]: s for s in steps}
    recurring = {s["id"] for s in steps if s.get("type") == "recurring"}
    gates = [s for s in steps if s.get("name", "").startswith("spec_consistency_check")]
    assert gates, "no spec_consistency_check gates found — workflow shape changed"
    for g in gates:
        deps = g.get("depends_on") or []
        assert deps, f"{g['id']} has no dependency (would admit at mission start)"
        for d in deps:
            assert d in by_id, f"{g['id']} -> missing step {d}"
            assert d not in recurring, (
                f"{g['id']} depends on recurring step {d}; recurring steps never "
                f"materialise as one-shot tasks, so the gate admits unguarded"
            )
