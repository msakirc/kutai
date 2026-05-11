"""recipes — recipe library substrate (Z2 T5A).

Public API
----------
  load_recipe(path)                  -> Recipe
  list_recipes(recipes_dir)          -> list[Recipe]
  match_recipe(stack, recipes)       -> list[tuple[Recipe, float]]
  pin_recipe(mission_id, ...)        -> None
  get_pinned_recipes(mission_id)     -> list[dict]

Schema lives in src/infra/db.py (registered via apply_migration in init_db).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("infra.recipes")

# ---------------------------------------------------------------------------
# YAML loading — prefer pyyaml, fall back to hand-parser for the flat schema
# ---------------------------------------------------------------------------

try:
    import yaml as _yaml  # type: ignore[import]

    def _load_yaml(text: str) -> dict:
        return _yaml.safe_load(text) or {}

except ImportError:  # pragma: no cover
    # Hand-parser fallback: handles top-level scalars + 1-level-deep lists/maps.
    # Covers the recipe.yaml schema (no nesting deeper than requires.tech_stack).
    # Documented choice: pyyaml is a standard KutAI dep; this path is a safety
    # net for environments where the venv is missing it.
    logger.warning(
        "pyyaml not available; using hand-parser for recipe.yaml. "
        "Only top-level keys + 1-level lists/maps are supported."
    )

    def _load_yaml(text: str) -> dict:  # type: ignore[misc]
        """Minimal YAML subset parser: scalars + lists + one-level maps."""
        result: dict = {}
        lines = text.splitlines()
        i = 0
        current_key: Optional[str] = None
        current_list: Optional[list] = None
        current_map: Optional[dict] = None

        def _flush():
            nonlocal current_key, current_list, current_map
            if current_key is None:
                return
            if current_list is not None:
                result[current_key] = current_list
            elif current_map is not None:
                result[current_key] = current_map
            current_key = None
            current_list = None
            current_map = None

        while i < len(lines):
            raw = lines[i]
            stripped = raw.strip()
            i += 1

            if not stripped or stripped.startswith("#"):
                continue

            indent = len(raw) - len(raw.lstrip())

            if stripped.startswith("- ") and indent > 0 and current_list is not None:
                current_list.append(stripped[2:].strip().strip('"').strip("'"))
                continue

            if ":" in stripped and indent > 0 and current_map is not None:
                k, _, v = stripped.partition(":")
                v = v.strip().strip('"').strip("'")
                current_map[k.strip()] = v
                continue

            if indent == 0 and ":" in stripped:
                _flush()
                k, _, v = stripped.partition(":")
                k = k.strip()
                v = v.strip()
                if not v:
                    # Value on subsequent lines
                    current_key = k
                    current_list = None
                    current_map = None
                    # Peek at next non-empty line to decide list vs map
                    j = i
                    while j < len(lines):
                        peek = lines[j].strip()
                        j += 1
                        if not peek or peek.startswith("#"):
                            continue
                        if peek.startswith("- "):
                            current_list = []
                        else:
                            current_map = {}
                        break
                else:
                    result[k] = v.strip('"').strip("'")
                continue

        _flush()
        return result


# ---------------------------------------------------------------------------
# Recipe dataclass
# ---------------------------------------------------------------------------

@dataclass
class Recipe:
    name: str
    version: str
    description: str
    requires: dict = field(default_factory=dict)
    conflicts_with: list = field(default_factory=list)
    post_hooks: list = field(default_factory=list)
    templates: dict = field(default_factory=dict)
    prompts: dict = field(default_factory=dict)
    lessons_domain: str = ""

    # Path to the directory containing this recipe.yaml (set by load_recipe).
    _recipe_dir: Optional[str] = field(default=None, repr=False, compare=False)


_REQUIRED_FIELDS = ("name", "version", "description", "requires", "post_hooks", "templates")


def load_recipe(path: str) -> Recipe:
    """Load and validate a recipe.yaml file.

    Parameters
    ----------
    path:
        Absolute or relative path to `recipe.yaml`.

    Returns
    -------
    Recipe dataclass instance.

    Raises
    ------
    ValueError
        When required fields are missing or the file cannot be read/parsed.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"Cannot read recipe file {path}: {exc}") from exc

    try:
        data = _load_yaml(text)
    except Exception as exc:
        raise ValueError(f"YAML parse error in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"recipe.yaml must be a YAML mapping, got {type(data).__name__}: {path}")

    missing = [f for f in _REQUIRED_FIELDS if f not in data or data[f] is None]
    if missing:
        raise ValueError(
            f"recipe.yaml missing required field(s) {missing!r} in {path}"
        )

    requires = data.get("requires") or {}
    if not isinstance(requires, dict):
        raise ValueError(f"'requires' must be a mapping in {path}")
    if not requires.get("tech_stack"):
        raise ValueError(f"'requires.tech_stack' is empty or missing in {path}")

    return Recipe(
        name=str(data["name"]),
        version=str(data["version"]),
        description=str(data["description"]),
        requires=requires,
        conflicts_with=list(data.get("conflicts_with") or []),
        post_hooks=list(data.get("post_hooks") or []),
        templates=dict(data.get("templates") or {}),
        prompts=dict(data.get("prompts") or {}),
        lessons_domain=str(data.get("lessons_domain") or ""),
        _recipe_dir=str(p.parent),
    )


def list_recipes(recipes_dir: str = "recipes") -> list[Recipe]:
    """Scan `recipes/<name>/<version>/recipe.yaml` and return sorted list.

    Skips versions whose recipe.yaml is missing or fails validation (log + continue).

    Returns recipes sorted by (name, version).
    """
    root = Path(recipes_dir)
    if not root.is_dir():
        logger.warning("recipes_dir not found: %s", recipes_dir)
        return []

    recipes: list[Recipe] = []
    for name_dir in sorted(root.iterdir()):
        if not name_dir.is_dir():
            continue
        for version_dir in sorted(name_dir.iterdir()):
            if not version_dir.is_dir():
                continue
            yaml_path = version_dir / "recipe.yaml"
            if not yaml_path.exists():
                continue
            try:
                recipe = load_recipe(str(yaml_path))
            except ValueError as exc:
                logger.warning("Skipping invalid recipe %s: %s", yaml_path, exc)
                continue
            recipes.append(recipe)

    return sorted(recipes, key=lambda r: (r.name, r.version))


# ---------------------------------------------------------------------------
# Stack matching
# ---------------------------------------------------------------------------

def _parse_stack(stack: str) -> set[str]:
    """Split a stack string on '+' and return lowercase components."""
    return {c.strip().lower() for c in stack.split("+") if c.strip()}


def match_recipe(
    stack: str,
    recipes: list[Recipe],
) -> list[tuple[Recipe, float]]:
    """Return ranked (recipe, fit_score) pairs for the given stack.

    Fit score
    ---------
    - Exact match (input stack == one of recipe's tech_stacks)        → 1.0
    - Superset (every recipe component appears in input stack)        → 0.7
    - 50%+ component overlap                                          → 0.4
    - Otherwise                                                       → 0.0 (filtered)

    Returned list is sorted by fit_score descending, then recipe name.
    """
    input_parts = _parse_stack(stack)
    results: list[tuple[Recipe, float]] = []

    for recipe in recipes:
        tech_stacks = recipe.requires.get("tech_stack") or []
        best_score = 0.0
        for ts in tech_stacks:
            recipe_parts = _parse_stack(ts)
            if not recipe_parts:
                continue
            # Exact match: normalised sets are equal
            if recipe_parts == input_parts:
                best_score = 1.0
                break
            # Superset: every recipe component found in input
            if recipe_parts.issubset(input_parts):
                best_score = max(best_score, 0.7)
                continue
            # Overlap >= 50%
            overlap = len(recipe_parts & input_parts) / len(recipe_parts)
            if overlap >= 0.5:
                best_score = max(best_score, 0.4)

        if best_score > 0.0:
            results.append((recipe, best_score))

    results.sort(key=lambda x: (-x[1], x[0].name, x[0].version))
    return results


# ---------------------------------------------------------------------------
# DB helpers — recipe_pin_log
# ---------------------------------------------------------------------------

async def pin_recipe(
    mission_id: int,
    recipe_name: str,
    version: str,
    fit_score: float = 1.0,
) -> None:
    """Insert a recipe pin for a mission. Idempotent (INSERT OR IGNORE)."""
    from src.infra.db import get_db

    db = await get_db()
    await db.execute(
        """
        INSERT OR IGNORE INTO recipe_pin_log
            (mission_id, recipe_name, version, fit_score)
        VALUES (?, ?, ?, ?)
        """,
        (mission_id, recipe_name, version, fit_score),
    )
    await db.commit()
    logger.info(
        "pin_recipe: mission_id=%s name=%s version=%s fit=%.2f",
        mission_id, recipe_name, version, fit_score,
    )


async def get_pinned_recipes(mission_id: int) -> list[dict]:
    """Return all pinned recipes for a mission, ordered by pinned_at."""
    from src.infra.db import get_db

    db = await get_db()
    cursor = await db.execute(
        """
        SELECT id, mission_id, recipe_name, version, fit_score, pinned_at
        FROM recipe_pin_log
        WHERE mission_id = ?
        ORDER BY pinned_at ASC
        """,
        (mission_id,),
    )
    rows = await cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in rows]
