"""Z2 T5C — recipe integration tests.

Tests cover:
1. pin_recipes_from_artifact: writes pins from JSON artifact; idempotent.
2. instantiate_recipe: default params, custom params (<<KEY>> substitution),
   output dir + files.
3. End-to-end pipeline: pick_recipe → pin_recipes_from_artifact →
   instantiate_recipe.
4. i2p_v3.json round-trip: step 8.0a is JSON-valid with expected shape.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

WORKTREE_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine synchronously, compatible with Python 3.10."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_db(tmp_path: Path):
    """Create a minimal in-memory-backed test DB and return the db module."""
    import importlib
    os.environ.setdefault("DB_PATH", str(tmp_path / "test.db"))
    import src.infra.db as db_mod
    _run(db_mod.init_db())
    return db_mod


# ---------------------------------------------------------------------------
# 1. pin_recipes_from_artifact
# ---------------------------------------------------------------------------

class TestPinRecipesFromArtifact:
    def test_two_picks_two_rows(self, tmp_path):
        """2 non-null picks → 2 rows in recipe_pin_log."""
        db_mod = _make_db(tmp_path)
        from src.infra.recipes import pin_recipes_from_artifact, get_pinned_recipes

        picks = {
            "auth": {"name": "auth", "version": "v1", "fit_score": 1.0},
            "uploads": {"name": "file_upload", "version": "v1", "fit_score": 0.8},
        }
        artifact = tmp_path / "recipe_picks.json"
        artifact.write_text(json.dumps(picks), encoding="utf-8")

        count = _run(pin_recipes_from_artifact(
            mission_id=1,
            recipe_picks_path=str(artifact),
        ))
        assert count == 2

        pinned = _run(get_pinned_recipes(1))
        names = {r["recipe_name"] for r in pinned}
        assert "auth" in names
        assert "file_upload" in names

        _run(db_mod.close_db(checkpoint=False))

    def test_idempotent(self, tmp_path):
        """Running twice with the same picks still yields 2 rows."""
        db_mod = _make_db(tmp_path)
        from src.infra.recipes import pin_recipes_from_artifact, get_pinned_recipes

        picks = {
            "auth": {"name": "auth", "version": "v1", "fit_score": 1.0},
            "search": {"name": "search", "version": "v1", "fit_score": 0.9},
        }
        artifact = tmp_path / "recipe_picks.json"
        artifact.write_text(json.dumps(picks), encoding="utf-8")

        _run(pin_recipes_from_artifact(mission_id=2, recipe_picks_path=str(artifact)))
        _run(pin_recipes_from_artifact(mission_id=2, recipe_picks_path=str(artifact)))

        pinned = _run(get_pinned_recipes(2))
        assert len(pinned) == 2  # not 4 — INSERT OR IGNORE

        _run(db_mod.close_db(checkpoint=False))

    def test_null_picks_skipped(self, tmp_path):
        """Null picks are skipped; only non-null entries are pinned."""
        db_mod = _make_db(tmp_path)
        from src.infra.recipes import pin_recipes_from_artifact, get_pinned_recipes

        picks = {
            "auth": {"name": "auth", "version": "v1", "fit_score": 1.0},
            "unmatched_feature": None,
        }
        artifact = tmp_path / "recipe_picks.json"
        artifact.write_text(json.dumps(picks), encoding="utf-8")

        count = _run(pin_recipes_from_artifact(
            mission_id=3,
            recipe_picks_path=str(artifact),
        ))
        assert count == 1

        pinned = _run(get_pinned_recipes(3))
        assert len(pinned) == 1
        assert pinned[0]["recipe_name"] == "auth"

        _run(db_mod.close_db(checkpoint=False))

    def test_missing_file_raises(self, tmp_path):
        """Missing artifact raises ValueError."""
        from src.infra.recipes import pin_recipes_from_artifact

        db_mod = _make_db(tmp_path)
        with pytest.raises(ValueError, match="Cannot read"):
            _run(pin_recipes_from_artifact(
                mission_id=99,
                recipe_picks_path=str(tmp_path / "nonexistent.json"),
            ))
        _run(db_mod.close_db(checkpoint=False))

    def test_invalid_json_raises(self, tmp_path):
        """Non-JSON file raises ValueError."""
        from src.infra.recipes import pin_recipes_from_artifact

        db_mod = _make_db(tmp_path)
        artifact = tmp_path / "recipe_picks.json"
        artifact.write_text("not json!", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON parse error"):
            _run(pin_recipes_from_artifact(
                mission_id=99,
                recipe_picks_path=str(artifact),
            ))
        _run(db_mod.close_db(checkpoint=False))


# ---------------------------------------------------------------------------
# 2. instantiate_recipe
# ---------------------------------------------------------------------------

class TestInstantiateRecipe:
    def _load_auth_recipe(self):
        from src.infra.recipes import load_recipe
        return load_recipe(
            str(WORKTREE_ROOT / "recipes" / "auth" / "v1" / "recipe.yaml")
        )

    def test_default_params_uses_recipe_defaults(self, tmp_path):
        """With empty params, defaults from RECIPE_PARAM markers are used."""
        from src.infra.recipes import instantiate_recipe

        recipe = self._load_auth_recipe()
        result = instantiate_recipe(recipe=recipe, target_dir=str(tmp_path), params={})

        assert result["ok"] is True
        assert len(result["files_written"]) >= 1
        assert result["error"] is None

        # Defaults should be present in written files
        backend_candidates = list(tmp_path.rglob("backend.template.py"))
        assert backend_candidates, "backend.template.py not written"
        content = backend_candidates[0].read_text(encoding="utf-8")
        # Default JWT_ALGO=HS256 should appear in JWT_ALGO = "HS256"
        assert "HS256" in content

    def test_custom_jwt_algo_substituted(self, tmp_path):
        """With params={JWT_ALGO: RS256}, output contains RS256."""
        from src.infra.recipes import instantiate_recipe

        recipe = self._load_auth_recipe()
        result = instantiate_recipe(
            recipe=recipe,
            target_dir=str(tmp_path),
            params={"JWT_ALGO": "RS256"},
        )

        assert result["ok"] is True
        assert "JWT_ALGO" in result["params_used"]
        assert result["params_used"]["JWT_ALGO"] == "RS256"

        backend_candidates = list(tmp_path.rglob("backend.template.py"))
        assert backend_candidates, "backend.template.py not written"
        content = backend_candidates[0].read_text(encoding="utf-8")
        # <<JWT_ALGO>> should have been replaced with RS256
        assert "RS256" in content
        assert "<<JWT_ALGO>>" not in content

    def test_custom_bcrypt_cost_substituted(self, tmp_path):
        """With params={BCRYPT_COST: 14}, output contains 14 in bcrypt config."""
        from src.infra.recipes import instantiate_recipe

        recipe = self._load_auth_recipe()
        result = instantiate_recipe(
            recipe=recipe,
            target_dir=str(tmp_path),
            params={"BCRYPT_COST": "14"},
        )

        assert result["ok"] is True
        backend_candidates = list(tmp_path.rglob("backend.template.py"))
        assert backend_candidates
        content = backend_candidates[0].read_text(encoding="utf-8")
        # <<BCRYPT_COST>> should have been replaced with 14
        assert "14" in content

    def test_custom_jwt_secret_env_substituted(self, tmp_path):
        """With params={JWT_SECRET_ENV: MY_SECRET}, env var name is replaced."""
        from src.infra.recipes import instantiate_recipe

        recipe = self._load_auth_recipe()
        result = instantiate_recipe(
            recipe=recipe,
            target_dir=str(tmp_path),
            params={"JWT_SECRET_ENV": "MY_APP_JWT_SECRET"},
        )

        assert result["ok"] is True
        backend_candidates = list(tmp_path.rglob("backend.template.py"))
        assert backend_candidates
        content = backend_candidates[0].read_text(encoding="utf-8")
        assert "MY_APP_JWT_SECRET" in content
        assert "<<JWT_SECRET_ENV>>" not in content

    def test_output_dir_created(self, tmp_path):
        """Target dir is created if it doesn't exist."""
        from src.infra.recipes import instantiate_recipe

        recipe = self._load_auth_recipe()
        nested = tmp_path / "nested" / "output"
        assert not nested.exists()

        result = instantiate_recipe(recipe=recipe, target_dir=str(nested), params={})

        assert result["ok"] is True
        assert nested.exists()

    def test_at_least_one_file_written(self, tmp_path):
        """At least one file (backend template) must be written."""
        from src.infra.recipes import instantiate_recipe

        recipe = self._load_auth_recipe()
        result = instantiate_recipe(recipe=recipe, target_dir=str(tmp_path), params={})

        assert result["ok"] is True
        assert len(result["files_written"]) >= 1

    def test_no_recipe_dir_returns_error(self, tmp_path):
        """Recipe with no _recipe_dir returns ok=False."""
        from src.infra.recipes import Recipe, instantiate_recipe

        recipe = Recipe(
            name="test", version="v1", description="test",
            requires={"tech_stack": ["fastapi"]},
            post_hooks=[],
            templates={"backend": "x.py"},
        )
        # _recipe_dir is None by default
        result = instantiate_recipe(recipe=recipe, target_dir=str(tmp_path), params={})
        assert result["ok"] is False
        assert result["error"] is not None

    def test_original_recipe_param_comments_preserved(self, tmp_path):
        """RECIPE_PARAM comment lines remain in output (not stripped)."""
        from src.infra.recipes import instantiate_recipe

        recipe = self._load_auth_recipe()
        result = instantiate_recipe(recipe=recipe, target_dir=str(tmp_path), params={})

        assert result["ok"] is True
        backend_candidates = list(tmp_path.rglob("backend.template.py"))
        assert backend_candidates
        content = backend_candidates[0].read_text(encoding="utf-8")
        # Marker comments must survive substitution
        assert "RECIPE_PARAM:JWT_ALGO" in content


# ---------------------------------------------------------------------------
# 3. End-to-end pipeline: pick → pin → instantiate
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    def test_pick_pin_instantiate(self, tmp_path):
        """pick_recipe → pin_recipes_from_artifact → instantiate_recipe pipeline."""
        db_mod = _make_db(tmp_path)
        from src.infra.recipes import (
            pin_recipes_from_artifact,
            get_pinned_recipes,
            load_recipe,
            instantiate_recipe,
        )

        # 1. Simulate pick_recipe output for a fastapi+postgres+nextjs mission
        picks = {
            "user_auth": {
                "name": "auth",
                "version": "v1",
                "fit_score": 1.0,
            }
        }
        recipe_picks_path = tmp_path / "recipe_picks.json"
        recipe_picks_path.write_text(json.dumps(picks), encoding="utf-8")

        # 2. Pin recipes
        count = _run(pin_recipes_from_artifact(
            mission_id=100,
            recipe_picks_path=str(recipe_picks_path),
        ))
        assert count == 1

        # 3. Verify pin in DB
        pinned = _run(get_pinned_recipes(100))
        assert len(pinned) == 1
        assert pinned[0]["recipe_name"] == "auth"
        assert pinned[0]["version"] == "v1"

        # 4. Instantiate
        recipe = load_recipe(
            str(WORKTREE_ROOT / "recipes" / "auth" / "v1" / "recipe.yaml")
        )
        out_dir = tmp_path / "workspace" / "auth"
        result = instantiate_recipe(
            recipe=recipe,
            target_dir=str(out_dir),
            params={"JWT_ALGO": "RS256"},
        )
        assert result["ok"] is True
        assert len(result["files_written"]) >= 1
        assert out_dir.exists()

        # 5. Check substitution happened
        backend_files = list(out_dir.rglob("backend.template.py"))
        assert backend_files
        content = backend_files[0].read_text(encoding="utf-8")
        assert "RS256" in content

        _run(db_mod.close_db(checkpoint=False))

    def test_pick_recipe_verb_fastapi_postgres_nextjs(self, tmp_path):
        """pick_recipe verb returns fit_score=1.0 for fastapi+postgres+nextjs."""
        async def _run_pick():
            from mr_roboto.pick_recipe import pick_recipe
            return await pick_recipe(
                mission_id=None,
                feature_decl="auth",
                stack="fastapi+postgres+nextjs",
                recipes_dir=str(WORKTREE_ROOT / "recipes"),
            )

        result = _run(_run_pick())
        assert result["ok"] is True
        assert result["picked"] is not None
        assert result["picked"]["name"] == "auth"
        assert result["picked"]["fit_score"] >= 0.9


# ---------------------------------------------------------------------------
# 4. i2p_v3.json round-trip
# ---------------------------------------------------------------------------

class TestI2pV3RecipeStep:
    def _load_workflow(self):
        with open(
            WORKTREE_ROOT / "src" / "workflows" / "i2p" / "i2p_v3.json",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    def test_json_valid(self):
        """i2p_v3.json is valid JSON after adding 8.0a."""
        wf = self._load_workflow()
        assert isinstance(wf, dict)
        assert "steps" in wf

    def test_step_8_0a_exists(self):
        """Step 8.0a is present in the workflow."""
        wf = self._load_workflow()
        step = next((s for s in wf["steps"] if s["id"] == "8.0a"), None)
        assert step is not None, "Step 8.0a not found in i2p_v3.json"

    def test_step_8_0a_agent_mechanical(self):
        """Step 8.0a has agent=mechanical, executor=mechanical."""
        wf = self._load_workflow()
        step = next(s for s in wf["steps"] if s["id"] == "8.0a")
        assert step["agent"] == "mechanical"
        assert step["executor"] == "mechanical"

    def test_step_8_0a_payload_action(self):
        """Step 8.0a payload.action == pick_recipe."""
        wf = self._load_workflow()
        step = next(s for s in wf["steps"] if s["id"] == "8.0a")
        assert step["payload"]["action"] == "pick_recipe"

    def test_step_8_0a_produces_recipe_picks(self):
        """Step 8.0a produces mission/recipe_picks.json."""
        wf = self._load_workflow()
        step = next(s for s in wf["steps"] if s["id"] == "8.0a")
        assert "mission/recipe_picks.json" in step.get("produces", [])

    def test_step_8_0a_depends_on_8_0(self):
        """Step 8.0a depends on 8.0 (implementation_backlog_initialization)."""
        wf = self._load_workflow()
        step = next(s for s in wf["steps"] if s["id"] == "8.0a")
        assert "8.0" in step.get("depends_on", [])

    def test_step_8_0a_pin_recipes_context(self):
        """Step 8.0a context has pin_recipes=true."""
        wf = self._load_workflow()
        step = next(s for s in wf["steps"] if s["id"] == "8.0a")
        assert step.get("context", {}).get("pin_recipes") is True

    def test_step_8_0a_phase_8(self):
        """Step 8.0a is in phase_8."""
        wf = self._load_workflow()
        step = next(s for s in wf["steps"] if s["id"] == "8.0a")
        assert step["phase"] == "phase_8"

    def test_existing_step_8_0_unchanged(self):
        """Existing step 8.0 (implementation_backlog_initialization) is unchanged."""
        wf = self._load_workflow()
        step = next(s for s in wf["steps"] if s["id"] == "8.0")
        assert step["name"] == "implementation_backlog_initialization"
        assert step["agent"] == "analyst"
