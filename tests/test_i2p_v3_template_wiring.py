"""End-to-end smoke test: i2p_v3 feature_implementation_template now declares
``produces`` + ``post_hooks`` on build steps, expander interpolates the
{feature_id} placeholder per-feature, and the values land in step context.

Replaces mission-57's silent post-hook absence: every feature instance
now ships verify_artifacts (and code_review where applicable) post-hooks
when expanded.
"""
from __future__ import annotations

import json

from src.workflows.engine.expander import (
    expand_template, expand_steps_to_tasks,
)
from src.workflows.engine.loader import load_workflow


_BUILD_STEPS_EXPECTED = {
    # name -> (expected produces glob fragments, expected post_hooks)
    "database_migration": (
        ["migrations/versions/", "_initial.py"],
        ["verify_artifacts"],
    ),
    "backend_service": (
        ["backend/app/models/", "backend/app/services/"],
        ["verify_artifacts", "code_review"],
    ),
    "backend_endpoints": (
        ["backend/app/routes/"],
        ["verify_artifacts", "code_review"],
    ),
    "backend_tests": (
        ["backend/tests/test_"],
        ["verify_artifacts"],
    ),
    "frontend_state": (
        ["frontend/src/types/", "frontend/src/api/", "frontend/src/state/"],
        ["verify_artifacts", "code_review"],
    ),
    "frontend_components": (
        ["frontend/src/components/", "/index.tsx"],
        ["verify_artifacts", "code_review"],
    ),
    "frontend_pages": (
        ["frontend/src/app/", "/page.tsx"],
        ["verify_artifacts", "code_review"],
    ),
    "frontend_tests": (
        ["frontend/src/__tests__/"],
        ["verify_artifacts"],
    ),
    "e2e_tests": (
        ["e2e/", ".spec.ts"],
        ["verify_artifacts"],
    ),
}


def _expanded_steps_by_name(feature_id: str = "F-001"):
    wf = load_workflow("i2p_v3")
    template = wf.get_template("feature_implementation_template")
    assert template is not None
    expanded = expand_template(
        template,
        params={"feature_id": feature_id, "feature_name": "Real-Time Sync"},
        prefix=f"8.{feature_id}.",
    )
    return {s.get("name"): s for s in expanded}


def test_each_build_step_declares_produces_and_post_hooks():
    by_name = _expanded_steps_by_name("F-001")
    for name, (path_fragments, expected_hooks) in _BUILD_STEPS_EXPECTED.items():
        step = by_name.get(name)
        assert step is not None, f"missing template step {name!r}"
        produces = step.get("produces") or []
        assert produces, f"{name!r} missing produces declaration"
        for fragment in path_fragments:
            assert any(fragment in p for p in produces), (
                f"{name!r} produces {produces} missing fragment {fragment!r}"
            )
        assert step.get("post_hooks") == expected_hooks, (
            f"{name!r} post_hooks {step.get('post_hooks')} != {expected_hooks}"
        )


def test_produces_paths_have_feature_id_interpolated():
    """{feature_id} placeholder must be substituted with the params value;
    no raw braces should remain in the expanded paths."""
    by_name = _expanded_steps_by_name("F-042")
    for name in _BUILD_STEPS_EXPECTED:
        step = by_name.get(name)
        produces = step.get("produces") or []
        for p in produces:
            assert "{feature_id}" not in p, (
                f"{name!r} produces {p!r} retains unsubstituted placeholder"
            )
            assert "F-042" in p or "{feature_id}" not in p, (
                f"{name!r} produces {p!r} did not get feature_id baked in"
            )


def test_two_features_get_distinct_produces_paths():
    """Each feature instance gets its own concrete produces list."""
    a = _expanded_steps_by_name("F-001")["backend_service"]["produces"]
    b = _expanded_steps_by_name("F-099")["backend_service"]["produces"]
    assert a != b
    assert any("F-001" in p for p in a)
    assert any("F-099" in p for p in b)
    # Same template structure, different feature ids.
    assert len(a) == len(b)


def test_expanded_to_tasks_carries_produces_and_post_hooks_in_ctx():
    """expand_steps_to_tasks puts produces + post_hooks into task context so
    determine_posthooks (Beckman) can read them at dispatch time."""
    wf = load_workflow("i2p_v3")
    template = wf.get_template("feature_implementation_template")
    expanded = expand_template(
        template,
        params={"feature_id": "F-001", "feature_name": "Sync"},
        prefix="8.F-001.",
    )
    tasks = expand_steps_to_tasks(expanded, mission_id=999, initial_context={})
    by_title = {}
    for t in tasks:
        title = t.get("title") or ""
        by_title[title] = t

    # Find a task by its template-step suffix.
    backend_service_task = None
    for title, t in by_title.items():
        if "backend_service" in title:
            backend_service_task = t
            break
    assert backend_service_task is not None
    ctx = backend_service_task.get("context") or {}
    if isinstance(ctx, str):
        ctx = json.loads(ctx)
    assert "produces" in ctx
    assert "F-001" in ctx["produces"][0]
    # Auto-wire prepends "grounding" because the step declares produces;
    # see expander.expand_steps_to_tasks (Layer 2 of G).
    assert ctx.get("post_hooks") == ["grounding", "verify_artifacts", "code_review"]


def test_design_only_steps_unchanged():
    """spec_review, implementation_plan, code_review, staging_*, quality_checks
    must NOT have produces/post_hooks added — they're not build steps."""
    by_name = _expanded_steps_by_name("F-001")
    for name in (
        "spec_review", "implementation_plan",
        "code_review", "staging_deploy", "staging_validation", "quality_checks",
    ):
        step = by_name.get(name)
        assert step is not None, f"step {name!r} missing"
        assert "produces" not in step, f"{name!r} should not declare produces"
        assert "post_hooks" not in step, f"{name!r} should not declare post_hooks"
