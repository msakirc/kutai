"""Coverage: NEEDS-REAL-TOOLS marker on i2p_v3 steps that fundamentally
require deploy/infra/observability adapters mr_roboto does not yet have.

Marker semantics:
- ``needs_real_tools: true`` on the step
- instruction prefixed with ``[NEEDS-REAL-TOOLS]`` so the agent prompt
  surfaces the constraint

Marker is informational + drift-guard for now; flow-control hookup
(skip / human-gate routing) is left to the real-tools workstream.

Until mr_roboto gains real adapters (Vercel/Railway/Supabase/Datadog/Sentry
+ Playwright runner), these steps must NOT claim success autonomously.
"""
from __future__ import annotations

import pytest

from src.workflows.engine.loader import load_workflow


# Top-level steps from the 2026-05-05 handoff.
# 9.4 (e2e_test_suite) and 13.11 (social_preview_test) were unblocked by
# Phase 1 of the real-tools workstream (2026-05-05) — 9.4 paired with a
# mechanical 9.4a runner via mr_roboto.run_pytest, 13.11 swapped to
# mr_roboto.social_preview_check (parse_og_tags + image HEAD).
TOP_LEVEL_NEEDS_REAL_TOOLS = [
    "7.13",   # staging_environment   — needs vendor provision API
    "13.1",   # production_infrastructure — needs vendor deploy API
    "13.3",   # monitoring_setup      — needs Sentry/UptimeRobot adapters
]

# Template steps (feature_implementation_template).
# feat.14 (staging_validation) unblocked Phase 1 via mr_roboto.staging_smoke_check
# (http_check on the per-feature staging URL). feat.13 still needs a real CD
# trigger — visual / multi-breakpoint smoke remains deferred.
TEMPLATE_NEEDS_REAL_TOOLS = [
    "staging_deploy",      # feat.13 — needs CD trigger + image-diff runner
]


def _step(step_id: str) -> dict:
    wf = load_workflow("i2p_v3")
    s = wf.get_step(step_id)
    assert s is not None, f"step {step_id!r} missing"
    return s


def _template_step(name: str) -> dict:
    wf = load_workflow("i2p_v3")
    template = wf.get_template("feature_implementation_template")
    assert template is not None
    for s in template.get("steps", []):
        if s.get("name") == name:
            return s
    raise AssertionError(f"template step {name!r} missing")


@pytest.mark.parametrize("step_id", TOP_LEVEL_NEEDS_REAL_TOOLS)
def test_top_level_step_marker(step_id):
    step = _step(step_id)
    assert step.get("needs_real_tools") is True, (
        f"{step_id} missing needs_real_tools: true"
    )
    instruction = step.get("instruction") or ""
    assert instruction.startswith("[NEEDS-REAL-TOOLS]"), (
        f"{step_id} instruction missing [NEEDS-REAL-TOOLS] prefix"
    )


@pytest.mark.parametrize("name", TEMPLATE_NEEDS_REAL_TOOLS)
def test_template_step_marker(name):
    step = _template_step(name)
    assert step.get("needs_real_tools") is True, (
        f"template step {name!r} missing needs_real_tools: true"
    )
    instruction = step.get("instruction") or ""
    assert instruction.startswith("[NEEDS-REAL-TOOLS]"), (
        f"template step {name!r} instruction missing [NEEDS-REAL-TOOLS] prefix"
    )


def test_no_extra_steps_silently_marked():
    """Catch drift: if someone adds the marker to a step not on the
    handoff list, the test fails so the marker stays a deliberate set,
    not an emergent one."""
    wf = load_workflow("i2p_v3")
    seen_top = {s["id"] for s in wf.steps if s.get("needs_real_tools")}
    expected_top = set(TOP_LEVEL_NEEDS_REAL_TOOLS)
    extras = seen_top - expected_top
    missing = expected_top - seen_top
    assert not extras, f"unexpected needs_real_tools markers: {sorted(extras)}"
    assert not missing, f"missing needs_real_tools markers: {sorted(missing)}"

    template = wf.get_template("feature_implementation_template")
    seen_tmpl = {s["name"] for s in template.get("steps", []) if s.get("needs_real_tools")}
    expected_tmpl = set(TEMPLATE_NEEDS_REAL_TOOLS)
    extras_t = seen_tmpl - expected_tmpl
    missing_t = expected_tmpl - seen_tmpl
    assert not extras_t, f"unexpected template needs_real_tools markers: {sorted(extras_t)}"
    assert not missing_t, f"missing template needs_real_tools markers: {sorted(missing_t)}"
