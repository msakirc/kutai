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

    # T5B optional fields — all have safe defaults so T5A tests keep passing.
    # dependencies.backend: list of Python package names required by the backend template.
    # dependencies.frontend: list of npm package names required by the frontend template.
    dependencies: dict = field(default_factory=dict)
    # param_defaults: default values for RECIPE_PARAM markers in the templates.
    param_defaults: dict = field(default_factory=dict)
    # entry_points: role -> file mapping for planner consumption.
    entry_points: dict = field(default_factory=dict)

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

    # T5B optional fields — validated with safe defaults to stay backward-compatible.
    raw_deps = data.get("dependencies") or {}
    if not isinstance(raw_deps, dict):
        raw_deps = {}
    raw_params = data.get("param_defaults") or {}
    if not isinstance(raw_params, dict):
        raw_params = {}
    raw_entry = data.get("entry_points") or {}
    if not isinstance(raw_entry, dict):
        raw_entry = {}

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
        dependencies={
            "backend": list(raw_deps.get("backend") or []),
            "frontend": list(raw_deps.get("frontend") or []),
        },
        param_defaults={str(k): str(v) for k, v in raw_params.items()},
        entry_points={str(k): str(v) for k, v in raw_entry.items()},
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


async def pin_recipes_from_artifact(
    mission_id: int,
    recipe_picks_path: str,
) -> int:
    """Read recipe_picks.json artifact and write pin_recipe rows for each non-null pick.

    The artifact is a JSON object mapping feature_name → pick_result where
    pick_result is ``{"name": ..., "version": ..., "fit_score": ...}`` or null.

    Returns count of pins written. Idempotent via UNIQUE(mission_id, recipe_name)
    (uses INSERT OR IGNORE, so re-running is safe).

    Raises ValueError when the artifact is missing or unparseable.
    """
    import json

    p = Path(recipe_picks_path)
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"Cannot read recipe_picks artifact {recipe_picks_path}: {exc}") from exc

    try:
        picks: dict = json.loads(text)
    except Exception as exc:
        raise ValueError(f"JSON parse error in {recipe_picks_path}: {exc}") from exc

    if not isinstance(picks, dict):
        raise ValueError(
            f"recipe_picks.json must be a JSON object, got {type(picks).__name__}"
        )

    count = 0
    for feature_name, pick in picks.items():
        if pick is None:
            continue
        if not isinstance(pick, dict):
            logger.warning(
                "pin_recipes_from_artifact: skipping non-dict pick for feature=%s",
                feature_name,
            )
            continue
        recipe_name = pick.get("name")
        version = pick.get("version")
        fit_score = float(pick.get("fit_score") or 1.0)
        if not recipe_name or not version:
            logger.warning(
                "pin_recipes_from_artifact: pick missing name/version for feature=%s",
                feature_name,
            )
            continue
        await pin_recipe(
            mission_id=mission_id,
            recipe_name=str(recipe_name),
            version=str(version),
            fit_score=fit_score,
        )
        count += 1

    logger.info(
        "pin_recipes_from_artifact: mission_id=%s pinned=%d from %s",
        mission_id, count, recipe_picks_path,
    )
    return count


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


# ---------------------------------------------------------------------------
# Recipe instantiation engine — Z2 T5C
# ---------------------------------------------------------------------------

_RECIPE_PARAM_RE = None  # compiled on first use


def _get_param_re():
    """Compile and cache the RECIPE_PARAM comment pattern."""
    import re
    global _RECIPE_PARAM_RE
    if _RECIPE_PARAM_RE is None:
        # Matches: # RECIPE_PARAM:KEY=default  or  // RECIPE_PARAM:KEY=default
        # Captures: key_group, default_group
        _RECIPE_PARAM_RE = re.compile(
            r"(?:#|//)\s*RECIPE_PARAM:([A-Z_][A-Z0-9_]*)=([^\n]*)"
        )
    return _RECIPE_PARAM_RE


def _collect_param_defaults(text: str) -> dict[str, str]:
    """Scan source text for RECIPE_PARAM markers and return {KEY: default}."""
    result: dict[str, str] = {}
    for m in _get_param_re().finditer(text):
        key = m.group(1)
        default = m.group(2).strip()
        if key not in result:
            result[key] = default
    return result


def _substitute_tokens(text: str, resolved: dict[str, str]) -> str:
    """Replace ``<<KEY>>`` tokens in ``text`` using ``resolved``.

    V1 substitution rule
    --------------------
    - RECIPE_PARAM comment markers are left intact (they document knobs).
    - ``<<KEY>>`` tokens elsewhere in the file are replaced with the resolved
      value (``params.get(KEY, default_from_marker)``).
    - Unknown ``<<KEY>>`` tokens (no RECIPE_PARAM declaration) are left as-is.

    Design note: comment-style markers stay so ast.parse() passes before
    AND after substitution — they are Python comments, not syntax.
    """
    import re
    def _replacer(m: "re.Match") -> str:
        key = m.group(1)
        return resolved.get(key, m.group(0))  # leave unknown tokens as-is

    return re.sub(r"<<([A-Z_][A-Z0-9_]*)>>", _replacer, text)


def _substitute_inline_params(text: str, resolved: dict[str, str]) -> str:
    """Replace literal values annotated with inline RECIPE_PARAM markers.

    V2 — numeric / non-substitutable-literal sites.

    Matches:
        NAME = <literal>  # RECIPE_PARAM:KEY=default
        NAME = <literal>    // RECIPE_PARAM:KEY=default
    And replaces ``<literal>`` with ``resolved[KEY]`` when present.
    Preserves the comment marker. Leaves rows alone when KEY is not in
    ``resolved``. Works for int / float / quoted-string literals.
    """
    import re
    pat = re.compile(
        r"(?P<lhs>=\s*)"
        r"(?P<lit>\"[^\"\n]*\"|'[^'\n]*'|-?\d+(?:\.\d+)?)"
        r"(?P<tail>\s+(?:#|//)\s*RECIPE_PARAM:"
        r"(?P<key>[A-Z_][A-Z0-9_]*)=[^\n]*)"
    )

    def _replacer(m: "re.Match") -> str:
        key = m.group("key")
        if key not in resolved:
            return m.group(0)
        lit = m.group("lit")
        new_val = resolved[key]
        # Preserve literal quoting style for string sites.
        if lit.startswith('"'):
            new_val = '"' + new_val.replace('"', '\\"') + '"'
        elif lit.startswith("'"):
            new_val = "'" + new_val.replace("'", "\\'") + "'"
        return m.group("lhs") + new_val + m.group("tail")

    return pat.sub(_replacer, text)


def instantiate_recipe(
    recipe: Recipe,
    target_dir: str,
    params: "dict[str, str]",
) -> dict:
    """Instantiate recipe template files into ``target_dir``.

    Substitution algorithm (V1)
    ---------------------------
    1. For each template file, scan for ``RECIPE_PARAM:KEY=default`` comment
       markers to build the defaults table.
    2. Resolve each KEY: ``params.get(KEY) or recipe.param_defaults.get(KEY)
       or marker_default``.
    3. Replace ``<<KEY>>`` tokens (the body substitution sites) using resolved
       values.  The original RECIPE_PARAM comment lines are preserved so
       generated files are still self-documenting.
    4. Write the result to ``target_dir/<original_filename>``.

    Parameters
    ----------
    recipe:
        Recipe instance (must have ``_recipe_dir`` set by ``load_recipe``).
    target_dir:
        Directory where instantiated files are written.  Created if absent.
    params:
        Caller-supplied parameter overrides.

    Returns
    -------
    dict with keys:
        ``ok`` (bool), ``files_written`` (list[str]), ``params_used`` (dict),
        ``skipped`` (list[str]), ``error`` (str | None).
    """
    import shutil

    recipe_dir = recipe._recipe_dir
    if not recipe_dir:
        return {
            "ok": False,
            "files_written": [],
            "params_used": {},
            "skipped": [],
            "error": "recipe._recipe_dir is not set — load recipe via load_recipe()",
        }

    # Collect all template paths from both `templates` and `entry_points`.
    # Use sets to avoid double-copying when entry_points re-declares a template.
    template_files: set[str] = set()
    for rel in recipe.templates.values():
        template_files.add(rel)
    for rel in recipe.entry_points.values():
        template_files.add(rel)

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    files_written: list[str] = []
    skipped: list[str] = []
    params_used: dict[str, str] = {}

    for rel in sorted(template_files):
        src_path = Path(recipe_dir) / rel
        if not src_path.exists():
            logger.warning("instantiate_recipe: template file missing: %s", src_path)
            skipped.append(rel)
            continue

        if src_path.is_dir():
            # Recurse into directories (e.g. `migrations/`, `tests/`)
            for child in sorted(src_path.rglob("*")):
                if not child.is_file():
                    continue
                child_rel = child.relative_to(Path(recipe_dir))
                dst = target / child_rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                _instantiate_file(child, dst, recipe, params, params_used)
                files_written.append(str(child_rel))
            continue

        dst = target / src_path.name
        _instantiate_file(src_path, dst, recipe, params, params_used)
        files_written.append(src_path.name)

    logger.info(
        "instantiate_recipe: recipe=%s/%s target=%s files=%d skipped=%d",
        recipe.name, recipe.version, target_dir,
        len(files_written), len(skipped),
    )
    return {
        "ok": True,
        "files_written": files_written,
        "params_used": params_used,
        "skipped": skipped,
        "error": None,
    }


def _instantiate_file(
    src: Path,
    dst: Path,
    recipe: Recipe,
    params: dict,
    params_used: dict,
) -> None:
    """Read src, substitute tokens, write to dst.  Updates params_used in-place."""
    try:
        text = src.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("instantiate_recipe: cannot read %s: %s", src, exc)
        dst.write_bytes(src.read_bytes())
        return

    # Collect defaults from RECIPE_PARAM markers in this file.
    marker_defaults = _collect_param_defaults(text)

    # Resolve final values: caller params > recipe.param_defaults > marker default.
    resolved: dict[str, str] = {}
    for key, marker_default in marker_defaults.items():
        value = (
            params.get(key)
            or recipe.param_defaults.get(key)
            or marker_default
        )
        resolved[key] = str(value)
        params_used[key] = str(value)

    # Also resolve any params supplied by caller that don't have markers.
    for key, value in params.items():
        if key not in resolved:
            resolved[key] = str(value)
            params_used[key] = str(value)

    substituted = _substitute_tokens(text, resolved)
    # V2 — inline literal substitution for numeric / quoted-string sites
    # that can't host <<KEY>> tokens without breaking ast.parse.
    substituted = _substitute_inline_params(substituted, resolved)
    dst.write_text(substituted, encoding="utf-8")
