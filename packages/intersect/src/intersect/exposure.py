"""Exposure-class decision: (tier × kind × confidence) → exposure_class.

Phase 2 exposure classes: inject, tool, preempt, quarantine. NO sandbox
(v1.1). T2 artifacts are treated like T3 — quarantined until a founder
promotes them. The tier ceiling is the hard cap; intersect floors the
exposure based on the task confidence against the θ thresholds.

Tier → eligible exposure ceiling (spec Tier-classifier table):
  T0 → inject, tool, preempt
  T1 → inject, tool          (no preempt)
  T2 → quarantine            (Phase 2; sandbox is v1.1)
  T3 → quarantine
"""
from __future__ import annotations

# θ_preempt > θ_inject > θ_tool > θ_min. Conservative defaults; lowered
# later based on yalayut_usage success-rate telemetry.
THETA_PREEMPT: float = 0.80
THETA_INJECT: float = 0.55
THETA_TOOL: float = 0.45
THETA_MIN: float = 0.30

# Artifact types / kinds that are callable (tool exposure).
_CALLABLE_TYPES = frozenset({"api", "mcp"})
# Skill kinds that are mechanizable recipe shapes.
_RECIPE_KINDS = frozenset({"shell_recipe", "procedure"})


def classify(artifact, *, confidence: float) -> str:
    """Decide the exposure class for one matched artifact.

    Returns one of: 'inject', 'tool', 'preempt', 'quarantine'.
    """
    _tier_raw = getattr(artifact, "vet_tier", None)
    tier = int(_tier_raw) if _tier_raw is not None else 3

    # Tier ceiling — T2/T3 never surface in Phase 2.
    if tier >= 2:
        return "quarantine"

    # Below the floor — not worth exposing.
    if confidence < THETA_MIN:
        return "quarantine"

    artifact_type = getattr(artifact, "artifact_type", "skill")
    kind = getattr(artifact, "kind", None)
    mechanizable = bool(getattr(artifact, "mechanizable", False))

    # preempt — T0 only, mechanizable recipe, high confidence.
    if (tier == 0
            and artifact_type == "skill"
            and kind in _RECIPE_KINDS
            and mechanizable
            and confidence >= THETA_PREEMPT):
        return "preempt"

    # tool — callable artifacts (api verbs, mcp tools).
    if artifact_type in _CALLABLE_TYPES:
        if confidence >= THETA_TOOL:
            return "tool"
        return "quarantine"

    # inject — everything skill-shaped above θ_inject.
    if confidence >= THETA_INJECT:
        return "inject"
    return "quarantine"


def render_variant(artifact, *, bound_args: dict | None) -> str:
    """Pick the inject render sub-variant: 'prose' | 'prebind'.

    'prebind' only when the artifact is parametric (has inputs_schema)
    AND every required field is statically bound. Otherwise 'prose'.
    """
    inputs_schema = getattr(artifact, "inputs_schema", None) or {}
    if not inputs_schema:
        return "prose"
    if not bound_args:
        return "prose"
    # All schema fields must be present in bound_args for a prebind render.
    for field in inputs_schema:
        if field not in bound_args or bound_args[field] is None:
            return "prose"
    return "prebind"
