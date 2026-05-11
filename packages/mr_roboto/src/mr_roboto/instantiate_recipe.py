"""instantiate_recipe — mechanical verb: instantiate a recipe into a target directory.

Z2 T5C.

Payload shape
-------------
{
    "action": "instantiate_recipe",
    "recipe_name": str,        # e.g. "auth"
    "version": str,            # e.g. "v1"
    "target_dir": str,         # destination directory
    "params": dict[str, str],  # optional override params (empty dict if omitted)
    "recipes_dir": str,        # optional, default "recipes"
}

Return shape (Action.result)
----------------------------
{
    "ok": bool,
    "files_written": list[str],
    "params_used": dict[str, str],
    "skipped": list[str],
    "error": str | None,
}
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.instantiate_recipe")


async def instantiate_recipe_verb(
    recipe_name: str,
    version: str,
    target_dir: str,
    params: "dict[str, str] | None" = None,
    recipes_dir: str = "recipes",
) -> "dict[str, Any]":
    """Load recipe ``recipe_name/version`` and instantiate it into ``target_dir``.

    Parameters
    ----------
    recipe_name:
        Recipe name (e.g. "auth").
    version:
        Recipe version (e.g. "v1").
    target_dir:
        Destination directory for instantiated files.
    params:
        Optional parameter overrides to substitute ``<<KEY>>`` tokens.
    recipes_dir:
        Root of the recipe library (default: "recipes").

    Returns
    -------
    dict with keys: ok, files_written, params_used, skipped, error.
    """
    from pathlib import Path
    from src.infra.recipes import load_recipe, instantiate_recipe

    if not recipe_name or not version or not target_dir:
        return {
            "ok": False,
            "files_written": [],
            "params_used": {},
            "skipped": [],
            "error": "recipe_name, version, and target_dir are required",
        }

    recipe_yaml = Path(recipes_dir) / recipe_name / version / "recipe.yaml"
    if not recipe_yaml.exists():
        return {
            "ok": False,
            "files_written": [],
            "params_used": {},
            "skipped": [],
            "error": f"recipe.yaml not found: {recipe_yaml}",
        }

    try:
        recipe = load_recipe(str(recipe_yaml))
    except ValueError as exc:
        return {
            "ok": False,
            "files_written": [],
            "params_used": {},
            "skipped": [],
            "error": f"load_recipe failed: {exc}",
        }

    result = instantiate_recipe(
        recipe=recipe,
        target_dir=target_dir,
        params=dict(params or {}),
    )

    logger.info(
        "instantiate_recipe_verb: recipe=%s/%s target=%s ok=%s files=%d",
        recipe_name, version, target_dir,
        result.get("ok"), len(result.get("files_written") or []),
    )
    return result
