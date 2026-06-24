"""instantiate_picked_recipes — Z2 Item-2 followup.

Reads mission/recipe_picks.json (emitted by step 8.0a recipe_pick_all),
calls instantiate_recipe for each non-null pick, and writes a manifest
to mission/recipe_instantiations.json.

Payload
-------
{
    "action": "instantiate_picked_recipes",
    "recipe_picks_path": str,  # relative to mission workspace
    "manifest_path": str,      # relative to mission workspace
    "recipes_dir": str,        # optional, default "recipes"
}

Return
------
{
    "ok": bool,
    "instantiations": [
        {"feature": str, "recipe": str, "version": str, "target_dir": str,
         "files_written": list[str]},
        ...
    ],
    "failures": [{"feature": str, "error": str}, ...],
    "manifest_path": str,
}
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.instantiate_picked_recipes")


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "feature"


async def instantiate_picked_recipes(
    mission_id: int | None,
    recipe_picks_path: str,
    manifest_path: str,
    recipes_dir: str = "recipes",
) -> dict[str, Any]:
    """Walk recipe_picks.json, instantiate each pick, write manifest."""
    from src.tools.workspace import get_mission_workspace
    from src.infra.recipes import load_recipe, instantiate_recipe

    if mission_id is None:
        return {"ok": False, "instantiations": [], "failures": [],
                "error": "mission_id required", "manifest_path": ""}

    ws = get_mission_workspace(int(mission_id))
    picks_abs = recipe_picks_path
    if not os.path.isabs(picks_abs):
        picks_abs = os.path.join(ws, recipe_picks_path)

    if not os.path.isfile(picks_abs):
        return {"ok": False, "instantiations": [], "failures": [],
                "error": f"recipe_picks.json not found: {picks_abs}",
                "manifest_path": ""}

    try:
        with open(picks_abs, "r", encoding="utf-8") as fh:
            picks = json.load(fh)
    except Exception as exc:
        return {"ok": False, "instantiations": [], "failures": [],
                "error": f"recipe_picks parse failed: {exc}",
                "manifest_path": ""}

    # Accept {feature_name: pick_result} OR {"picks": [...]} shape.
    items: list[tuple[str, dict | None]] = []
    if isinstance(picks, dict):
        if "picks" in picks and isinstance(picks["picks"], list):
            for entry in picks["picks"]:
                if isinstance(entry, dict):
                    items.append((str(entry.get("feature") or ""), entry.get("pick")))
        else:
            for feat, pick in picks.items():
                items.append((str(feat), pick if isinstance(pick, dict) else None))

    instantiations: list[dict] = []
    failures: list[dict] = []

    for feature, pick in items:
        if not pick:
            continue
        name = str(pick.get("name") or pick.get("recipe") or "")
        version = str(pick.get("version") or "v1")
        if not name:
            failures.append({"feature": feature, "error": "pick missing name"})
            continue
        recipe_yaml = os.path.join(recipes_dir, name, version, "recipe.yaml")
        try:
            recipe = load_recipe(recipe_yaml)
        except Exception as exc:
            failures.append({"feature": feature, "error": f"load_recipe failed: {exc}"})
            continue
        target_dir = os.path.join(ws, _slug(feature))
        try:
            res = await instantiate_recipe(recipe, target_dir, params={})
            instantiations.append({
                "feature": feature, "recipe": name, "version": version,
                "target_dir": target_dir,
                "files_written": list(res.get("files_written") or []),
            })
        except Exception as exc:
            failures.append({"feature": feature, "error": str(exc)})

    manifest_abs = manifest_path
    if not os.path.isabs(manifest_abs):
        manifest_abs = os.path.join(ws, manifest_path)
    os.makedirs(os.path.dirname(manifest_abs) or ".", exist_ok=True)
    manifest_data = {
        "ok": not failures,
        "instantiations": instantiations,
        "failures": failures,
    }
    with open(manifest_abs, "w", encoding="utf-8") as fh:
        json.dump(manifest_data, fh, indent=2)

    logger.info(
        "instantiate_picked_recipes: %d instantiations, %d failures",
        len(instantiations), len(failures),
    )
    return {
        "ok": not failures,
        "instantiations": instantiations,
        "failures": failures,
        "manifest_path": manifest_abs,
    }


async def run(task: dict) -> dict:
    """mr_roboto mechanical executor entry point."""
    payload = (task.get("context") or {}).get("payload") or {}
    return await instantiate_picked_recipes(
        mission_id=task.get("mission_id"),
        recipe_picks_path=str(payload.get("recipe_picks_path") or "mission/recipe_picks.json"),
        manifest_path=str(payload.get("manifest_path") or "mission/recipe_instantiations.json"),
        recipes_dir=str(payload.get("recipes_dir") or "recipes"),
    )
