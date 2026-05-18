# Recipe Library

## Convention

Each recipe lives at `recipes/<name>/<version>/`.

- `name` is a lowercase identifier (e.g. `auth`, `upload`, `search`).
- `version` matches the `version:` field in `recipe.yaml` (e.g. `v1`, `v2`).
- A recipe version directory MUST contain a valid `recipe.yaml` to be discovered.

## Layout

```
recipes/
  <name>/
    <version>/
      recipe.yaml               # metadata + schema (required; parsed by src/infra/recipes.py)
      spec.md                   # human-readable description of what the recipe implements
      prompts/
        <stack>.md              # stack-specific prompt fragment injected into agent context
      backend.template.py       # placeholder or real template for backend code
      frontend.template.tsx     # placeholder or real template for frontend code
      tests/
        backend_smoke.template.py
      migrations/
        001_<name>.sql.template
      lessons.md                # known pitfalls + fixes to seed into mission_lessons on instantiation
```

## Pin-version-at-mission-start

When a mission pins a recipe (via the `pick_recipe` mr_roboto verb), the
`recipe_pin_log` table records `(mission_id, recipe_name, version, fit_score)`.
The pinned version is frozen for the duration of the mission — phase-boundary
upgrades (i.e. switching to a newer version mid-mission) require an explicit
`pick_recipe` call with the new version, which will be blocked by the
`UNIQUE(mission_id, recipe_name)` constraint until the old pin is removed.

## Phase-boundary upgrades

Upgrading a recipe mid-mission is intentionally manual. The workflow engine
should never silently upgrade a pinned recipe. To upgrade:

1. Remove the old pin via direct DB write (founder action).
2. Re-run `pick_recipe` with `min_fit` lowered if needed.
3. The new pin is recorded with the new version and fit_score.

## Adding a new recipe

1. Create `recipes/<name>/<version>/`.
2. Add a valid `recipe.yaml` (see schema in `src/infra/recipes.py`).
3. Fill `spec.md` with what the recipe implements.
4. Add stack-specific prompt fragments under `prompts/`.
5. Add template files (backend, frontend, tests, migrations).
6. Add `lessons.md` with known pitfalls.
7. Run `python -c "from src.infra.recipes import list_recipes; print(list_recipes())"` to verify.
