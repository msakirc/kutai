"""
Context gating — decides which context layers to inject per task type.

Each task profile maps to a set of layer IDs. Heuristic overrides
adjust based on task metadata. Budget calculator distributes tokens
across active layers by priority weight.
"""
from __future__ import annotations

import json
import os

CONTEXT_POLICIES: dict[str, set[str]] = {
    "executor":         {"deps", "skills", "api"},
    "coder":            {"deps", "skills", "profile", "rag"},
    "implementer":      {"deps", "prior", "skills", "profile", "board"},
    "fixer":            {"deps", "skills", "rag", "profile"},
    "researcher":       {"skills", "rag", "api", "convo"},
    "shopping_advisor": {"skills", "convo"},
    "assistant":        {"convo", "rag", "memory", "prefs"},
    "writer":           {"deps", "convo", "memory", "prefs"},
    "planner":          {"deps", "board", "ambient", "memory"},
    "architect":        {"deps", "profile", "board", "rag"},
    "reviewer":         set(),
    "summarizer":       {"deps"},
    "analyst":          {"deps", "rag", "board"},
    "router":           set(),
    "visual_reviewer":  set(),
    "test_generator":   {"deps", "skills", "profile"},
}

DEFAULT_POLICY: set[str] = {"deps", "skills", "rag"}

LAYER_WEIGHTS: dict[str, int] = {
    "deps": 5, "prior": 4, "skills": 3, "rag": 3,
    "convo": 2, "board": 2, "profile": 1, "ambient": 1,
    "api": 1, "memory": 1, "prefs": 1,
}

CONTEXT_FRACTION = 0.40


def _int_env(name: str, default: int) -> int:
    """Parse an int env var, falling back to ``default`` on absent/garbage.

    This constant is read at import time; a malformed value must not crash the
    context builder (and everything importing it)."""
    try:
        return int(os.environ[name])
    except (KeyError, ValueError, TypeError):
        return default


# Absolute ceiling on the per-build context-layer budget pool, independent of
# the target model's window. Without it, ``available = model_ctx * FRACTION``
# scales unbounded: a gemini-class 1M-token window yields a 400k pool, so the
# deps + board layers fill with the legacy completed-results dump and the
# (102k-token) mission blackboard — observed ~190k prompt tokens on mission 86
# / step 1.4a (2026-06-18). The B-table then learns that p90, the estimator
# forces a ~226k ctx_needed, every model is filtered (ctx + free-tier TPM), and
# the task DLQs — self-reinforcing because each oversized run re-teaches the
# estimate. The cap restores fleet eligibility: 32k sits above the observed
# legit per-step max (~26k) yet keeps the total prompt (~pool + ~10k overhead)
# under the 60k-class free-tier TPM caps and the 64k/128k model windows that
# were being filtered out. Small local models (8k * 0.40 = 3.3k) are
# unaffected (available already below the cap). Env-overridable.
CONTEXT_ABS_CAP = _int_env("KUTAI_CONTEXT_ABS_CAP", 32768)


def get_context_policy(agent_type: str) -> set[str]:
    """Return the context layer set for a given agent/task profile."""
    return set(CONTEXT_POLICIES.get(agent_type, DEFAULT_POLICY))


def apply_heuristics(task: dict, policy: set[str]) -> set[str]:
    """Adjust policy based on task metadata. Returns new set, never mutates input."""
    p = set(policy)

    ctx = task.get("context", {})
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    if not isinstance(ctx, dict):
        ctx = {}

    if ctx.get("tools_hint"):
        p.add("skills")
        p.add("api")

    if task.get("depends_on"):
        p.add("deps")

    if ctx.get("is_followup"):
        p.add("convo")

    if task.get("mission_id"):
        p.add("board")

    return p


def compute_layer_budgets(model_context: int, active_layers: set[str]) -> dict[str, int]:
    """Distribute token budget across active layers by priority weight."""
    if not active_layers:
        return {}

    available = min(int(model_context * CONTEXT_FRACTION), CONTEXT_ABS_CAP)

    active_weights = {k: LAYER_WEIGHTS[k] for k in active_layers if k in LAYER_WEIGHTS}
    total_weight = sum(active_weights.values()) or 1

    return {
        layer: int(available * w / total_weight)
        for layer, w in active_weights.items()
    }
