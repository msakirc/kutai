"""Mr. Roboto — yalayut_recipe mechanical executor.

intersect (Phase 2) routes a ``preempt`` task to the mechanical lane with
``context.payload`` shaped::

    {"action": "yalayut_recipe", "recipe_id": <int>, "args": {...}}

This executor is a thin leaf shim: it unpacks the payload and calls
``yalayut.run_recipe``. It imports yalayut lazily so mr_roboto carries no
import-time coupling to the catalog.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.yalayut_recipe")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = (task.get("context") or {}).get("payload") or task.get("payload") or {}
    recipe_id = payload.get("recipe_id")
    args = payload.get("args") or {}

    if recipe_id is None:
        return {"ok": False, "reason": "yalayut_recipe: payload missing recipe_id"}
    try:
        recipe_id = int(recipe_id)
    except (TypeError, ValueError):
        return {"ok": False, "reason": f"yalayut_recipe: bad recipe_id {recipe_id!r}"}

    # Propagate the mission workspace so the recipe scaffolds in the right dir.
    if "workspace_path" not in args:
        from src.tools.workspace import get_mission_workspace
        mission_id = task.get("mission_id")
        if mission_id is not None:
            try:
                args = {**args, "workspace_path": get_mission_workspace(mission_id)}
            except Exception as e:
                logger.warning("workspace resolve failed", err=str(e))

    try:
        import yalayut
        result = await yalayut.run_recipe(recipe_id, args)
    except Exception as e:
        logger.warning("run_recipe raised", recipe_id=recipe_id, err=str(e))
        return {"ok": False, "reason": f"yalayut_recipe: run_recipe raised: {e}"}

    if not isinstance(result, dict):
        return {"ok": False, "reason": "yalayut_recipe: run_recipe returned non-dict"}
    return result
