"""Z2 T5A — recipe library v1 substrate tests.

Tests cover:
- Recipe dataclass schema validation + round-trip
- list_recipes scanning the seeded recipes/auth/v1/
- match_recipe fit scoring
- pick_recipe verb
- recipe_pin_log DB helpers
"""
from __future__ import annotations

import os
import sys
import asyncio
import tempfile
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WORKTREE_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestRecipeSchema:
    def test_load_recipe_round_trip(self):
        """load_recipe parses the seeded auth/v1/recipe.yaml correctly."""
        from src.infra.recipes import load_recipe

        yaml_path = WORKTREE_ROOT / "recipes" / "auth" / "v1" / "recipe.yaml"
        assert yaml_path.exists(), f"recipe.yaml not found at {yaml_path}"

        recipe = load_recipe(str(yaml_path))
        assert recipe.name == "auth"
        assert recipe.version == "v1"
        assert recipe.description
        assert "fastapi+postgres+nextjs" in recipe.requires.get("tech_stack", [])
        assert "imports_check" in recipe.post_hooks
        assert "backend" in recipe.templates
        assert recipe.lessons_domain == "auth"

    def test_missing_required_field_raises(self, tmp_path):
        """Missing required field raises ValueError with field name."""
        from src.infra.recipes import load_recipe

        bad_yaml = textwrap.dedent("""\
            name: auth
            version: v1
            # description is missing
            requires:
              tech_stack:
                - fastapi+postgres+nextjs
            post_hooks:
              - imports_check
            templates:
              backend: backend.template.py
        """)
        yaml_file = tmp_path / "recipe.yaml"
        yaml_file.write_text(bad_yaml, encoding="utf-8")

        with pytest.raises(ValueError, match="description"):
            load_recipe(str(yaml_file))

    def test_missing_tech_stack_raises(self, tmp_path):
        """Empty tech_stack raises ValueError."""
        from src.infra.recipes import load_recipe

        bad_yaml = textwrap.dedent("""\
            name: auth
            version: v1
            description: "test"
            requires:
              tech_stack: []
            post_hooks:
              - imports_check
            templates:
              backend: backend.template.py
        """)
        yaml_file = tmp_path / "recipe.yaml"
        yaml_file.write_text(bad_yaml, encoding="utf-8")

        with pytest.raises(ValueError, match="tech_stack"):
            load_recipe(str(yaml_file))


# ---------------------------------------------------------------------------
# list_recipes
# ---------------------------------------------------------------------------

class TestListRecipes:
    def test_list_recipes_finds_auth_v1(self):
        """list_recipes scans recipes/ and returns auth/v1."""
        from src.infra.recipes import list_recipes

        recipes = list_recipes(str(WORKTREE_ROOT / "recipes"))
        assert len(recipes) >= 1
        names = {r.name for r in recipes}
        assert "auth" in names
        auth = next(r for r in recipes if r.name == "auth")
        assert auth.version == "v1"

    def test_list_recipes_skips_missing_yaml(self, tmp_path):
        """list_recipes silently skips version dirs without recipe.yaml."""
        from src.infra.recipes import list_recipes

        # Create a version dir with no recipe.yaml
        (tmp_path / "mything" / "v1").mkdir(parents=True)
        recipes = list_recipes(str(tmp_path))
        assert recipes == []

    def test_list_recipes_skips_invalid_yaml(self, tmp_path):
        """list_recipes silently skips invalid recipe.yaml files."""
        from src.infra.recipes import list_recipes

        ver_dir = tmp_path / "broken" / "v1"
        ver_dir.mkdir(parents=True)
        (ver_dir / "recipe.yaml").write_text("name: broken\n# missing required fields\n")
        recipes = list_recipes(str(tmp_path))
        assert recipes == []


# ---------------------------------------------------------------------------
# match_recipe fit scoring
# ---------------------------------------------------------------------------

class TestMatchRecipe:
    def _get_auth_recipe(self):
        from src.infra.recipes import list_recipes
        recipes = list_recipes(str(WORKTREE_ROOT / "recipes"))
        return recipes

    def test_exact_stack_match_score_1(self):
        """Exact stack match → fit_score 1.0."""
        from src.infra.recipes import match_recipe
        recipes = self._get_auth_recipe()
        results = match_recipe("fastapi+postgres+nextjs", recipes)
        assert results, "Expected at least one match"
        auth_results = [(r, s) for r, s in results if r.name == "auth"]
        assert auth_results
        assert auth_results[0][1] == 1.0

    def test_superset_stack_score_07(self):
        """Extra component (redis) → fit_score 0.7 (superset match)."""
        from src.infra.recipes import match_recipe
        recipes = self._get_auth_recipe()
        results = match_recipe("fastapi+postgres+nextjs+redis", recipes)
        auth_results = [(r, s) for r, s in results if r.name == "auth"]
        assert auth_results
        assert auth_results[0][1] == pytest.approx(0.7)

    def test_no_match_filtered(self):
        """Completely different stack → no match returned."""
        from src.infra.recipes import match_recipe
        recipes = self._get_auth_recipe()
        results = match_recipe("django+postgres", recipes)
        auth_results = [(r, s) for r, s in results if r.name == "auth"]
        assert auth_results == []


# ---------------------------------------------------------------------------
# pick_recipe verb
# ---------------------------------------------------------------------------

class TestPickRecipeVerb:
    def test_matching_feature_and_stack_returns_pick(self):
        """feature_decl='auth', matching stack → picked.name=='auth'."""
        async def _run():
            from mr_roboto.pick_recipe import pick_recipe
            result = await pick_recipe(
                mission_id=1,
                feature_decl="auth",
                stack="fastapi+postgres+nextjs",
                recipes_dir=str(WORKTREE_ROOT / "recipes"),
                min_fit=0.7,
            )
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result["ok"] is True
        assert result["picked"] is not None
        assert result["picked"]["name"] == "auth"
        assert result["picked"]["fit_score"] >= 0.7

    def test_non_matching_stack_returns_no_pick(self):
        """Non-matching stack → picked is None, reason mentions threshold."""
        async def _run():
            from mr_roboto.pick_recipe import pick_recipe
            result = await pick_recipe(
                mission_id=2,
                feature_decl="random feature xyz",
                stack="django+postgres",
                recipes_dir=str(WORKTREE_ROOT / "recipes"),
                min_fit=0.7,
            )
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result["ok"] is True
        assert result["picked"] is None
        assert "threshold" in result["reason"] or "0 candidates" in result["reason"]


# ---------------------------------------------------------------------------
# DB helpers: recipe_pin_log
# ---------------------------------------------------------------------------

class TestRecipePinLog:
    @pytest.fixture(autouse=True)
    def _patch_db(self, tmp_path, monkeypatch):
        """Use a temp DB so tests don't touch production."""
        db_path = str(tmp_path / "test.db")
        import src.app.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "DB_PATH", db_path)
        import src.infra.db as db_mod
        monkeypatch.setattr(db_mod, "DB_PATH", db_path)
        # Force connection reset
        db_mod._db_connection = None
        db_mod._db_connection_path = None
        yield
        # Teardown: close conn
        import asyncio as _aio
        try:
            _aio.get_event_loop().run_until_complete(db_mod.close_db(checkpoint=False))
        except Exception:
            pass

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def _init_db(self):
        from src.infra.db import init_db
        self._run(init_db())

    def test_pin_recipe_inserts_row(self):
        """pin_recipe inserts a row into recipe_pin_log."""
        from src.infra.recipes import pin_recipe, get_pinned_recipes

        self._init_db()
        self._run(pin_recipe(mission_id=10, recipe_name="auth", version="v1", fit_score=1.0))
        rows = self._run(get_pinned_recipes(10))
        assert len(rows) == 1
        assert rows[0]["recipe_name"] == "auth"
        assert rows[0]["version"] == "v1"
        assert rows[0]["fit_score"] == pytest.approx(1.0)

    def test_pin_recipe_is_idempotent(self):
        """Second pin_recipe call with same (mission_id, recipe_name) is no-op."""
        from src.infra.recipes import pin_recipe, get_pinned_recipes

        self._init_db()
        self._run(pin_recipe(mission_id=11, recipe_name="auth", version="v1", fit_score=1.0))
        # Second call should not raise and should not insert a duplicate.
        self._run(pin_recipe(mission_id=11, recipe_name="auth", version="v1", fit_score=0.9))
        rows = self._run(get_pinned_recipes(11))
        assert len(rows) == 1

    def test_get_pinned_recipes_empty(self):
        """get_pinned_recipes returns [] for mission with no pins."""
        from src.infra.recipes import get_pinned_recipes

        self._init_db()
        rows = self._run(get_pinned_recipes(999))
        assert rows == []
