"""§1.D — Z8 monitoring_kit recipes authored.

The Z8 deferred list called for monitoring_kit recipes for the common
stacks; none were ever authored, so i2p step 13.3 monitoring_setup had no
reusable scaffold to fall back on. This adds monitoring_kit_{fastapi,
nextjs,django}/v1 — free-first observability kits (health endpoints +
env-gated Sentry + structured logging).

Coverage: the recipes load + validate, are discoverable by list_recipes,
match their stacks via pick_recipe, and their .py templates parse.
"""
from __future__ import annotations

import ast
import os

import pytest

_STACKS = {
    "monitoring_kit_fastapi": "fastapi",
    "monitoring_kit_nextjs": "nextjs",
    "monitoring_kit_django": "django",
}


def test_all_three_monitoring_kits_discovered():
    from src.infra.recipes import list_recipes
    names = {r.name for r in list_recipes()}
    for name in _STACKS:
        assert name in names, f"{name} not discovered by list_recipes"


def test_monitoring_kits_validate():
    """load_recipe enforces the required-field schema — a load failure means
    list_recipes silently skips the recipe."""
    from src.infra.recipes import load_recipe
    for name in _STACKS:
        recipe = load_recipe(f"recipes/{name}/v1/recipe.yaml")
        assert recipe.version == "v1"
        assert recipe.requires.get("tech_stack"), f"{name} has no tech_stack"
        assert recipe.templates, f"{name} declares no templates"


@pytest.mark.parametrize("name,stack", list(_STACKS.items()))
def test_pick_recipe_matches_each_stack(name, stack):
    from src.infra.recipes import list_recipes, match_recipe
    ranked = match_recipe(stack, list_recipes())
    hit = next((s for r, s in ranked if r.name == name), 0.0)
    assert hit >= 0.7, f"{name} did not match stack '{stack}' (score {hit})"


@pytest.mark.asyncio
async def test_pick_recipe_picks_monitoring_kit_for_feature():
    """The recipe_pick_all i2p step calls pick_recipe(feature, stack) — a
    'monitoring' feature on a fastapi stack must pick the kit."""
    from mr_roboto.pick_recipe import pick_recipe
    res = await pick_recipe(
        mission_id=None,
        feature_decl="monitoring",
        stack="fastapi+postgres+nextjs",
    )
    picked = res.get("picked")
    assert picked is not None, f"no recipe picked: {res.get('reason')}"
    assert picked["name"] == "monitoring_kit_fastapi"


def test_python_templates_parse():
    """RECIPE_PARAM markers must survive ast.parse() — they live in comments
    and docstrings, so every .py template must still be valid Python."""
    for name in ("monitoring_kit_fastapi", "monitoring_kit_django"):
        d = f"recipes/{name}/v1"
        for fn in os.listdir(d):
            if fn.endswith(".py"):
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    ast.parse(f.read())  # raises SyntaxError on failure
