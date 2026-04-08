"""
Context gating — decides which context layers to inject per task type.

Each task profile maps to a set of layer IDs. Heuristic overrides
adjust based on task metadata. Budget calculator distributes tokens
across active layers by priority weight.
"""
from __future__ import annotations

import json

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
    "error_recovery":   {"deps", "rag", "skills"},
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

    available = int(model_context * CONTEXT_FRACTION)

    active_weights = {k: LAYER_WEIGHTS[k] for k in active_layers if k in LAYER_WEIGHTS}
    total_weight = sum(active_weights.values()) or 1

    return {
        layer: int(available * w / total_weight)
        for layer, w in active_weights.items()
    }
