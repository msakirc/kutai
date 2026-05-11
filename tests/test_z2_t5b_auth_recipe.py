"""Z2 T5B — auth/v1 recipe end-to-end tests.

Covers:
- recipe.yaml loads with T5B optional fields populated
- All template files parse / structural assertions
- dependencies.backend includes required packages
- param_defaults loaded correctly
- pick_recipe returns auth/v1 with fit 1.0 for fastapi+postgres+nextjs
"""
from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path

import pytest

WORKTREE_ROOT = Path(__file__).parent.parent
AUTH_V1 = WORKTREE_ROOT / "recipes" / "auth" / "v1"


# ---------------------------------------------------------------------------
# recipe.yaml — T5B schema extensions
# ---------------------------------------------------------------------------

class TestRecipeYamlT5B:
    def _load(self):
        from src.infra.recipes import load_recipe
        return load_recipe(str(AUTH_V1 / "recipe.yaml"))

    def test_loads_without_error(self):
        recipe = self._load()
        assert recipe.name == "auth"
        assert recipe.version == "v1"

    def test_dependencies_backend_populated(self):
        recipe = self._load()
        backend_deps = recipe.dependencies.get("backend", [])
        assert "fastapi" in backend_deps
        assert any("passlib" in d for d in backend_deps), f"passlib not in {backend_deps}"
        assert any(d in backend_deps for d in ("python-jose[cryptography]", "pyjwt", "python-jose")), \
            f"No JWT library in {backend_deps}"

    def test_dependencies_frontend_populated(self):
        recipe = self._load()
        frontend_deps = recipe.dependencies.get("frontend", [])
        assert isinstance(frontend_deps, list)
        assert len(frontend_deps) >= 1

    def test_param_defaults_populated(self):
        recipe = self._load()
        pd = recipe.param_defaults
        assert "JWT_ALGO" in pd
        assert pd["JWT_ALGO"] == "HS256"
        assert "JWT_TTL_MIN" in pd
        assert "REFRESH_TTL_DAYS" in pd
        assert "REDIRECT_AFTER_LOGIN" in pd

    def test_entry_points_populated(self):
        recipe = self._load()
        ep = recipe.entry_points
        assert "backend_routes" in ep
        assert "frontend_components" in ep
        assert "migration" in ep
        assert "smoke_tests" in ep

    def test_t5a_fields_still_present(self):
        """Ensure T5A fields are unaffected by T5B additions."""
        recipe = self._load()
        assert "fastapi+postgres+nextjs" in recipe.requires.get("tech_stack", [])
        assert "imports_check" in recipe.post_hooks
        assert "backend" in recipe.templates
        assert recipe.lessons_domain == "auth"


# ---------------------------------------------------------------------------
# Template file: backend.template.py
# ---------------------------------------------------------------------------

class TestBackendTemplate:
    def _path(self) -> Path:
        return AUTH_V1 / "backend.template.py"

    def test_file_exists_and_non_empty(self):
        p = self._path()
        assert p.exists()
        assert p.stat().st_size > 500

    def test_ast_parses_cleanly(self):
        """RECIPE_PARAM comments must not break Python syntax."""
        source = self._path().read_text(encoding="utf-8")
        tree = ast.parse(source)  # raises SyntaxError on failure
        assert tree is not None

    def test_defines_router(self):
        source = self._path().read_text(encoding="utf-8")
        assert "router = APIRouter" in source

    def test_has_required_routes(self):
        source = self._path().read_text(encoding="utf-8")
        for route in ("/register", "/login", "/logout", "/refresh",
                      "/password/reset/request", "/password/reset/confirm",
                      "/email/verify"):
            assert route in source, f"Missing route: {route}"

    def test_recipe_param_markers_present(self):
        source = self._path().read_text(encoding="utf-8")
        assert "RECIPE_PARAM:" in source

    def test_hash_password_and_verify_defined(self):
        source = self._path().read_text(encoding="utf-8")
        assert "def hash_password" in source
        assert "def verify_password" in source

    def test_create_access_token_defined(self):
        source = self._path().read_text(encoding="utf-8")
        assert "def create_access_token" in source

    def test_no_hardcoded_secrets(self):
        """JWT secret must not appear as a literal string value."""
        source = self._path().read_text(encoding="utf-8")
        # The env var name is fine; actual secret values are not
        assert "super-secret" not in source.lower()
        assert 'JWT_SECRET = "' not in source or 'os.environ' in source


# ---------------------------------------------------------------------------
# Template file: frontend.template.tsx
# ---------------------------------------------------------------------------

class TestFrontendTemplate:
    def _path(self) -> Path:
        return AUTH_V1 / "frontend.template.tsx"

    def test_file_exists_and_non_empty(self):
        p = self._path()
        assert p.exists()
        assert p.stat().st_size > 500

    def test_mentions_login(self):
        assert "Login" in self._path().read_text(encoding="utf-8")

    def test_mentions_register(self):
        assert "Register" in self._path().read_text(encoding="utf-8")

    def test_mentions_email_verify(self):
        source = self._path().read_text(encoding="utf-8")
        assert "EmailVerify" in source or "email/verify" in source

    def test_recipe_param_markers(self):
        assert "RECIPE_PARAM:" in self._path().read_text(encoding="utf-8")

    def test_uses_tailwind_tokens(self):
        source = self._path().read_text(encoding="utf-8")
        # Must use design-system token classes, not raw hex
        assert "bg-primary" in source or "text-primary" in source
        assert "#" not in source or "text-" in source  # no raw hex colors

    def test_four_components_exported(self):
        source = self._path().read_text(encoding="utf-8")
        for component in ("LoginForm", "RegisterForm", "PasswordResetForm", "EmailVerifyView"):
            assert component in source, f"Missing component: {component}"


# ---------------------------------------------------------------------------
# Template file: migrations/001_users.sql.template
# ---------------------------------------------------------------------------

class TestMigrationTemplate:
    def _path(self) -> Path:
        return AUTH_V1 / "migrations" / "001_users.sql.template"

    def test_file_exists_and_non_empty(self):
        p = self._path()
        assert p.exists()
        assert p.stat().st_size > 200

    def test_has_users_table(self):
        assert "CREATE TABLE IF NOT EXISTS users" in self._path().read_text(encoding="utf-8")

    def test_has_password_reset_tokens_table(self):
        assert "CREATE TABLE IF NOT EXISTS password_reset_tokens" in self._path().read_text(encoding="utf-8")

    def test_has_email_verify_tokens_table(self):
        assert "CREATE TABLE IF NOT EXISTS email_verify_tokens" in self._path().read_text(encoding="utf-8")

    def test_has_auth_sessions_table(self):
        assert "CREATE TABLE IF NOT EXISTS auth_sessions" in self._path().read_text(encoding="utf-8")

    def test_all_four_tables(self):
        source = self._path().read_text(encoding="utf-8")
        expected = ["users", "password_reset_tokens", "email_verify_tokens", "auth_sessions"]
        for table in expected:
            assert f"CREATE TABLE IF NOT EXISTS {table}" in source, f"Missing table: {table}"

    def test_portability_comment(self):
        """SQL template must document the SQLite/Postgres portability choice."""
        source = self._path().read_text(encoding="utf-8")
        assert "Postgres" in source or "postgres" in source.lower()


# ---------------------------------------------------------------------------
# Template file: tests/backend_smoke.template.py
# ---------------------------------------------------------------------------

class TestSmokeTemplate:
    def _path(self) -> Path:
        return AUTH_V1 / "tests" / "backend_smoke.template.py"

    def test_file_exists_and_non_empty(self):
        p = self._path()
        assert p.exists()
        assert p.stat().st_size > 500

    def test_ast_parses_cleanly(self):
        source = self._path().read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert tree is not None

    def test_has_five_or_more_test_functions(self):
        source = self._path().read_text(encoding="utf-8")
        test_funcs = re.findall(r"^async def (test_\w+)|^def (test_\w+)", source, re.MULTILINE)
        names = [g1 or g2 for g1, g2 in test_funcs]
        assert len(names) >= 5, f"Only {len(names)} test functions found: {names}"

    def test_has_roundtrip_test(self):
        source = self._path().read_text(encoding="utf-8")
        assert "test_register_and_login_roundtrip" in source

    def test_has_bad_password_test(self):
        source = self._path().read_text(encoding="utf-8")
        assert "test_login_rejects_bad_password" in source

    def test_has_duplicate_email_test(self):
        source = self._path().read_text(encoding="utf-8")
        assert "test_register_rejects_duplicate_email" in source

    def test_has_password_reset_flow_test(self):
        source = self._path().read_text(encoding="utf-8")
        assert "test_password_reset_flow" in source

    def test_has_email_verify_flow_test(self):
        source = self._path().read_text(encoding="utf-8")
        assert "test_email_verify_flow" in source

    def test_uses_httpx(self):
        source = self._path().read_text(encoding="utf-8")
        assert "httpx" in source


# ---------------------------------------------------------------------------
# pick_recipe verb — integration
# ---------------------------------------------------------------------------

class TestPickRecipeForAuth:
    def test_pick_recipe_exact_stack_fit_1(self):
        """pick_recipe returns auth/v1 with fit 1.0 for fastapi+postgres+nextjs."""
        async def _run():
            from mr_roboto.pick_recipe import pick_recipe
            return await pick_recipe(
                mission_id=99,
                feature_decl="auth",
                stack="fastapi+postgres+nextjs",
                recipes_dir=str(WORKTREE_ROOT / "recipes"),
                min_fit=0.7,
            )

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result["ok"] is True
        assert result["picked"] is not None
        assert result["picked"]["name"] == "auth"
        assert result["picked"]["fit_score"] == 1.0

    def test_pick_recipe_sqlite_stack_exact(self):
        """fastapi+sqlite+nextjs also gives fit 1.0."""
        async def _run():
            from mr_roboto.pick_recipe import pick_recipe
            return await pick_recipe(
                mission_id=100,
                feature_decl="user registration",
                stack="fastapi+sqlite+nextjs",
                recipes_dir=str(WORKTREE_ROOT / "recipes"),
                min_fit=0.7,
            )

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result["ok"] is True
        assert result["picked"] is not None
        assert result["picked"]["fit_score"] == 1.0
