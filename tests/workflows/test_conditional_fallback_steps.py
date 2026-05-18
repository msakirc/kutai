"""Conditional-group fallback_steps must become real tasks.

Regression guard for a latent workflow-engine bug surfaced during Z5:

``_check_conditional_triggers`` only ever marks the *excluded* branch of a
conditional group as skipped — it never *creates* tasks. The runner builds
its task pool from ``wf.steps`` alone, but ``fallback_steps`` are defined
*inside* the conditional group, not in ``wf.steps``. So a fallback-only step
(``competitor_deep_dive``'s ``1.5_lite``) never got a task and silently never
ran.

``runner.merge_fallback_steps`` folds fallback steps into the pool. These
tests pin that behaviour.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from src.workflows.engine.conditions import resolve_group
from src.workflows.engine.loader import load_workflow
from src.workflows.engine.runner import merge_fallback_steps


def _wf(steps, conditional_groups):
    """A minimal stand-in carrying just the attributes the helper reads."""
    return SimpleNamespace(steps=steps, conditional_groups=conditional_groups)


# ── merge_fallback_steps unit behaviour ────────────────────────────────────


def test_fallback_only_step_is_merged_into_pool():
    wf = _wf(
        steps=[{"id": "1.5"}, {"id": "1.6"}],
        conditional_groups=[
            {
                "group_id": "g",
                "if_true": ["1.5", "1.6"],
                "if_false": ["1.5_lite"],
                "fallback_steps": [{"id": "1.5_lite", "phase": "phase_1"}],
            }
        ],
    )
    merged_ids = [s["id"] for s in merge_fallback_steps(wf)]
    assert "1.5_lite" in merged_ids
    assert merged_ids.count("1.5_lite") == 1


def test_fallback_id_already_in_steps_is_not_duplicated():
    wf = _wf(
        steps=[{"id": "1.5"}, {"id": "7.5m"}],
        conditional_groups=[
            {
                "group_id": "frontend_platform",
                "if_true": ["7.5m"],
                "if_false": ["1.5"],
                # 7.5m is already a real step — must not be appended twice.
                "fallback_steps": [{"id": "7.5m"}],
            }
        ],
    )
    merged_ids = [s["id"] for s in merge_fallback_steps(wf)]
    assert merged_ids.count("7.5m") == 1


def test_no_conditional_groups_returns_steps_unchanged():
    wf = _wf(steps=[{"id": "a"}, {"id": "b"}], conditional_groups=[])
    assert [s["id"] for s in merge_fallback_steps(wf)] == ["a", "b"]


def test_fallback_step_definition_is_preserved_whole():
    fb = {"id": "1.5_lite", "phase": "phase_1", "agent": "researcher",
          "depends_on": ["1.3"]}
    wf = _wf(steps=[], conditional_groups=[{"fallback_steps": [fb]}])
    merged = merge_fallback_steps(wf)
    assert merged[0] is fb  # whole dict carried through, not a stripped copy


# ── i2p_v3.json real-data invariant ────────────────────────────────────────


def test_i2p_v3_competitor_deep_dive_fallback_now_registers():
    """competitor_deep_dive's 1.5_lite is fallback-only — the bug's victim."""
    wf = load_workflow("i2p_v3")
    main_ids = {s.get("id") for s in wf.steps}

    group = next(
        (cg for cg in wf.conditional_groups
         if cg.get("group_id") == "competitor_deep_dive"),
        None,
    )
    assert group is not None, "competitor_deep_dive group missing"

    fallback_ids = {fb.get("id") for fb in group.get("fallback_steps", [])}
    assert "1.5_lite" in fallback_ids, "1.5_lite no longer a fallback step"
    # Precondition for the bug: it is NOT a top-level step.
    assert "1.5_lite" not in main_ids, "1.5_lite is a main step — test stale"

    # The fix: after the merge it is in the pool the runner expands to tasks.
    merged_ids = {s.get("id") for s in merge_fallback_steps(wf)}
    assert "1.5_lite" in merged_ids


def test_every_i2p_v3_fallback_step_lands_in_the_pool():
    """No conditional-group fallback step may be left out of the task pool."""
    wf = load_workflow("i2p_v3")
    merged_ids = {s.get("id") for s in merge_fallback_steps(wf)}
    for cg in wf.conditional_groups:
        for fb in cg.get("fallback_steps", []):
            assert fb.get("id") in merged_ids, (
                f"fallback step {fb.get('id')} in group "
                f"{cg.get('group_id')} not in task pool"
            )


def test_resolve_group_includes_fallback_on_false_branch():
    """When the condition is false, resolve_group puts the fallback in
    `included` — which is now backed by a real task thanks to the merge."""
    wf = load_workflow("i2p_v3")
    group = next(
        cg for cg in wf.conditional_groups
        if cg.get("group_id") == "competitor_deep_dive"
    )
    # Fewer than 3 competitors → condition false → lite path included.
    included, excluded = resolve_group(group, json.dumps(["only_one"]))
    assert "1.5_lite" in included
    assert "1.5_lite" not in excluded
