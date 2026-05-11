"""pick_recipe — mechanical verb: select the best matching recipe for a mission feature.

Z2 T5A substrate.

Return shape
------------
{
    "ok": True,
    "picked": {"name": str, "version": str, "fit_score": float} | None,
    "candidates": [{"name": str, "version": str, "fit_score": float}, ...],
    "reason": str,
}

``picked`` is the top candidate when ``fit_score >= min_fit``; None otherwise.
Planner reads ``picked is None`` as a signal to fall back to scratch implementation.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.pick_recipe")


async def pick_recipe(
    mission_id: int | None,
    feature_decl: str,
    stack: str,
    *,
    recipes_dir: str = "recipes",
    min_fit: float = 0.7,
) -> dict[str, Any]:
    """Select the best matching recipe for ``feature_decl`` on ``stack``.

    Parameters
    ----------
    mission_id:
        Mission context (used for logging; not written here — caller calls
        pin_recipe separately if it wants to persist the choice).
    feature_decl:
        Free-text feature name from the planner (e.g. "auth",
        "user registration with email verify").
    stack:
        '+'-delimited tech stack (e.g. "fastapi+postgres+nextjs").
    recipes_dir:
        Root directory of the recipe library (default: "recipes").
    min_fit:
        Minimum fit_score to consider a recipe picked. Default 0.7 (superset
        match or better). Below threshold → picked=None.

    Returns
    -------
    dict with keys: ok, picked, candidates, reason.
    """
    from src.infra.recipes import list_recipes, match_recipe

    try:
        all_recipes = list_recipes(recipes_dir)
    except Exception as exc:
        logger.error("pick_recipe: list_recipes failed: %s", exc)
        return {
            "ok": False,
            "picked": None,
            "candidates": [],
            "reason": f"list_recipes error: {exc}",
        }

    ranked = match_recipe(stack, all_recipes)

    # Apply feature_decl bonus: substring match of feature_decl against
    # recipe.name or recipe.lessons_domain → +0.1 (capped at 1.0).
    # Track bonus separately so it can break stack-score ties when both
    # recipes hit fit=1.0 already.
    feature_lower = feature_decl.lower().strip()
    adjusted: list[tuple, float] = []
    for recipe, base_score in ranked:
        bonus = 0.0
        if feature_lower and (
            feature_lower in recipe.name.lower()
            or recipe.name.lower() in feature_lower
            or (recipe.lessons_domain and feature_lower in recipe.lessons_domain.lower())
        ):
            bonus = 0.1
        final_score = min(1.0, base_score + bonus)
        adjusted.append((recipe, final_score, bonus))

    # Re-sort: score DESC, then bonus DESC (feature-matched first on ties),
    # then name/version alphabetical for determinism.
    adjusted.sort(key=lambda x: (-x[1], -x[2], x[0].name, x[0].version))

    candidates = [
        {"name": r.name, "version": r.version, "fit_score": score}
        for r, score, _bonus in adjusted
    ]

    if adjusted and adjusted[0][1] >= min_fit:
        top_recipe, top_score, _ = adjusted[0]
        picked = {
            "name": top_recipe.name,
            "version": top_recipe.version,
            "fit_score": top_score,
        }
        reason = (
            f"recipe '{top_recipe.name}/{top_recipe.version}' matched "
            f"stack='{stack}' with fit_score={top_score:.2f}"
        )
        logger.info(
            "pick_recipe: picked %s/%s fit=%.2f mission_id=%s feature='%s'",
            top_recipe.name, top_recipe.version, top_score, mission_id, feature_decl,
        )
    else:
        picked = None
        if not adjusted:
            reason = f"no recipe matched stack='{stack}' (0 candidates)"
        else:
            top_score = adjusted[0][1]
            reason = (
                f"below min_fit threshold: best candidate "
                f"'{adjusted[0][0].name}/{adjusted[0][0].version}' "
                f"fit_score={top_score:.2f} < min_fit={min_fit:.2f}"
            )
        logger.info(
            "pick_recipe: no pick (below threshold) mission_id=%s stack='%s' feature='%s'",
            mission_id, stack, feature_decl,
        )

    return {
        "ok": True,
        "picked": picked,
        "candidates": candidates,
        "reason": reason,
    }
