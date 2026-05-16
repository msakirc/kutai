"""Z5 T1 — platform branching rails.

Covers the mobile-track foundation landed in Z5 Tier 1:

1. The ``frontend_platform`` conditional group routes the frontend scaffold
   by ``platform_requirements.target_platform`` — web missions to ``7.5``,
   mobile missions to ``7.5m``, ``both`` missions run both.
2. The feature-implementation template expands the web frontend steps
   (``feat.7``–``feat.10``) or the Expo variants (``feat.7m``–``feat.10m``)
   depending on the mission's ``target_platform``.

These assertions guard the branching primitive that every later Z5 tier
(recipes, mobile adapters, device QA, distribution) depends on.
"""
from __future__ import annotations

import json

from src.workflows.engine.conditions import evaluate_condition, resolve_group
from src.workflows.engine.expander import (
    expand_template,
    select_platform_variants,
)
from src.workflows.engine.loader import load_workflow


# ── fixtures ───────────────────────────────────────────────────────────────


def _platform_requirements(target_platform: str) -> str:
    """A minimal but schema-valid platform_requirements artifact JSON."""
    return json.dumps(
        {
            "primary_platform": "mobile" if target_platform != "web" else "web",
            "browser_support": ["chrome", "safari"],
            "mobile_support": target_platform != "web",
            "screen_sizes": ["phone", "tablet", "desktop"],
            "os_versions": {"ios": "16+", "android": "13+"},
            "device_types": ["phone", "tablet"],
            "target_platform": target_platform,
        }
    )


def _frontend_platform_group() -> dict:
    """The frontend_platform conditional group, loaded from i2p_v3."""
    wf = load_workflow("i2p_v3")
    group = wf.get_conditional_group("frontend_platform")
    assert group is not None, "frontend_platform conditional group missing from i2p_v3"
    return group


# ── 1. conditional-group routing ───────────────────────────────────────────


def test_frontend_platform_group_exists_and_well_formed():
    """The group references real step ids — no orphans.

    Note: ``7.5m`` is a *real step* in the steps list, not a
    fallback-only alternate. It is intentionally NOT in ``fallback_steps``
    — ``resolve_group`` always merges fallback ids into the included set
    on the False branch, which would wrongly include the Expo scaffold
    for web missions.
    """
    group = _frontend_platform_group()
    assert group["condition_artifact"] == "platform_requirements"
    assert group["condition_check"] == "target_platform in ('mobile','both')"
    assert group["if_true"] == ["7.5m"]
    assert group["if_false"] == ["7.5"]

    wf = load_workflow("i2p_v3")
    step_ids = {s["id"] for s in wf.steps}
    fallback_ids = {fb["id"] for fb in group.get("fallback_steps", [])}
    # Every id in if_true / if_false resolves to a real step or a fallback.
    for sid in group["if_true"] + group["if_false"]:
        assert sid in step_ids or sid in fallback_ids, (
            f"frontend_platform references orphan step id {sid!r}"
        )
    # Both scaffolds must be real steps so they are inserted as tasks at
    # mission start; the conditional group skips the branch that doesn't
    # apply.
    assert "7.5m" in step_ids, "7.5m must be a real step in the steps list"
    assert "7.5" in step_ids


def test_web_mission_routes_to_7_5_not_7_5m():
    """target_platform=web → 7.5 included, 7.5m excluded."""
    group = _frontend_platform_group()
    artifact = _platform_requirements("web")

    assert evaluate_condition(group["condition_check"], artifact) is False

    included, excluded = resolve_group(group, artifact)
    assert "7.5" in included
    assert "7.5m" not in included
    assert "7.5m" in excluded
    assert "7.5" not in excluded


def test_mobile_mission_routes_to_7_5m_not_7_5():
    """target_platform=mobile → 7.5m included, 7.5 excluded."""
    group = _frontend_platform_group()
    artifact = _platform_requirements("mobile")

    assert evaluate_condition(group["condition_check"], artifact) is True

    included, excluded = resolve_group(group, artifact)
    assert "7.5m" in included
    assert "7.5" not in included
    assert "7.5" in excluded
    assert "7.5m" not in excluded


def test_both_mission_runs_the_mobile_branch():
    """target_platform=both satisfies the condition (mobile branch runs).

    The conditional group is binary: ``both`` evaluates the condition True,
    so the Expo scaffold 7.5m runs and the web scaffold 7.5 is skipped by
    the group. (A 'both' mission that wants a web app too gets it through
    the feature-template variants — see the template tests below — and the
    web stack token from step 4.2; the standalone web *scaffold* is not
    re-added here.)
    """
    group = _frontend_platform_group()
    artifact = _platform_requirements("both")

    assert evaluate_condition(group["condition_check"], artifact) is True

    included, excluded = resolve_group(group, artifact)
    assert "7.5m" in included
    assert "7.5" in excluded


def test_missing_target_platform_defaults_to_web_routing():
    """A platform_requirements artifact without target_platform must not
    crash and must route to the web branch (safe default)."""
    group = _frontend_platform_group()
    artifact = json.dumps({"primary_platform": "web", "mobile_support": False})

    assert evaluate_condition(group["condition_check"], artifact) is False
    included, excluded = resolve_group(group, artifact)
    assert "7.5" in included
    assert "7.5m" in excluded


# ── 2. mobile_app_submission group is no longer dangling ───────────────────


def test_mobile_app_submission_group_not_dangling():
    """The mobile_app_submission group's if_true must not point at the
    dead 14.10 id (Z9 reused it). T1 sets it to [] until T5 repoints it."""
    wf = load_workflow("i2p_v3")
    group = wf.get_conditional_group("mobile_app_submission")
    assert group is not None
    assert group["if_true"] == [], (
        "mobile_app_submission.if_true must be [] until Z5 T5 adds the "
        "real submit step"
    )
    step_ids = {s["id"] for s in wf.steps}
    fallback_ids = {fb["id"] for fb in group.get("fallback_steps", [])}
    for sid in group["if_true"] + group["if_false"]:
        assert sid in step_ids or sid in fallback_ids


# ── 3. platform_requirements schema carries target_platform ────────────────


def test_platform_requirements_schema_has_target_platform():
    """Step 3.6's platform_requirements artifact schema must require
    target_platform and be bumped to schema version 2."""
    wf = load_workflow("i2p_v3")
    step = wf.get_step("3.6")
    assert step is not None
    schema = step["artifact_schema"]["platform_requirements"]
    assert "target_platform" in schema["required_fields"]
    assert schema["_schema_version"] == "2"


# ── 4. feature-template variant selection ──────────────────────────────────


def test_select_platform_variants_web_keeps_web_steps():
    """web → keep feat.7-10, drop every …m variant."""
    steps = [
        {"template_step_id": "feat.6"},
        {"template_step_id": "feat.7"},
        {"template_step_id": "feat.7m"},
        {"template_step_id": "feat.10"},
        {"template_step_id": "feat.10m"},
        {"template_step_id": "feat.11"},
    ]
    out = {s["template_step_id"] for s in select_platform_variants(steps, "web")}
    assert {"feat.6", "feat.7", "feat.10", "feat.11"} <= out
    assert "feat.7m" not in out
    assert "feat.10m" not in out


def test_select_platform_variants_mobile_keeps_expo_steps():
    """mobile → drop feat.7-10, keep the …m variants."""
    steps = [
        {"template_step_id": "feat.7"},
        {"template_step_id": "feat.7m"},
        {"template_step_id": "feat.9"},
        {"template_step_id": "feat.9m"},
    ]
    out = {s["template_step_id"] for s in select_platform_variants(steps, "mobile")}
    assert out == {"feat.7m", "feat.9m"}


def test_select_platform_variants_both_keeps_everything():
    steps = [
        {"template_step_id": "feat.7"},
        {"template_step_id": "feat.7m"},
        {"template_step_id": "feat.8"},
        {"template_step_id": "feat.8m"},
    ]
    out = {s["template_step_id"] for s in select_platform_variants(steps, "both")}
    assert out == {"feat.7", "feat.7m", "feat.8", "feat.8m"}


def test_select_platform_variants_unknown_value_defaults_to_web():
    steps = [
        {"template_step_id": "feat.7"},
        {"template_step_id": "feat.7m"},
    ]
    for bad in ("", "ios", "garbage", None):
        out = {
            s["template_step_id"]
            for s in select_platform_variants(steps, bad)  # type: ignore[arg-type]
        }
        assert out == {"feat.7"}, f"value {bad!r} should fall back to web routing"


def test_feature_template_mobile_picks_expo_variants():
    """expand_template with target_platform=mobile emits the Expo feat.*m
    frontend steps and NOT the web feat.7-10 steps."""
    wf = load_workflow("i2p_v3")
    template = wf.get_template("feature_implementation_template")
    assert template is not None

    expanded = expand_template(
        template,
        params={
            "feature_id": "F-001",
            "feature_name": "User Login",
            "target_platform": "mobile",
        },
        prefix="8.F-001.",
    )
    ids = {s["id"] for s in expanded}
    # Expo variants present
    for m in ("feat.7m", "feat.8m", "feat.9m", "feat.10m"):
        assert f"8.F-001.{m}" in ids, f"mobile expansion missing {m}"
    # Web frontend steps absent
    for w in ("feat.7", "feat.8", "feat.9", "feat.10"):
        assert f"8.F-001.{w}" not in ids, f"mobile expansion kept web step {w}"
    # Expo screen step writes under app/ (RN), not frontend/src/app
    screen = next(s for s in expanded if s["id"] == "8.F-001.feat.9m")
    assert any(p.startswith("app/") for p in screen["produces"])


def test_feature_template_web_picks_dom_variants():
    """expand_template with target_platform=web (the default) keeps the
    web feat.7-10 steps and drops the Expo variants."""
    wf = load_workflow("i2p_v3")
    template = wf.get_template("feature_implementation_template")
    assert template is not None

    expanded = expand_template(
        template,
        params={"feature_id": "F-002", "feature_name": "Dashboard"},
        prefix="8.F-002.",
    )
    ids = {s["id"] for s in expanded}
    for w in ("feat.7", "feat.8", "feat.9", "feat.10"):
        assert f"8.F-002.{w}" in ids, f"web expansion missing {w}"
    for m in ("feat.7m", "feat.8m", "feat.9m", "feat.10m"):
        assert f"8.F-002.{m}" not in ids, f"web expansion kept Expo step {m}"


def test_feature_template_both_picks_all_frontend_variants():
    wf = load_workflow("i2p_v3")
    template = wf.get_template("feature_implementation_template")
    assert template is not None

    expanded = expand_template(
        template,
        params={
            "feature_id": "F-003",
            "feature_name": "Notifications",
            "target_platform": "both",
        },
        prefix="8.F-003.",
    )
    ids = {s["id"] for s in expanded}
    for s in ("feat.7", "feat.7m", "feat.10", "feat.10m"):
        assert f"8.F-003.{s}" in ids, f"both expansion missing {s}"
