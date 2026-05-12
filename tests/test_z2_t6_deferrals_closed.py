"""Z2 T6 — deferrals closed tests.

Covers:
- All 5 recipes loadable via list_recipes
- pagination + search + file_upload match fastapi+postgres+nextjs
- audit_log backend.template.py is fully filled (no T6 WILL FILL, has real fns)
- audit_log smoke template has 4+ async tests and no T6 markers
- Severity blocker promotion: pattern_lint + design_system_check are "blocker"
- _apply_semgrep_blocker_verdict exists and is callable
- Routing test: blocker severity dispatches to blocker fn, warning to warning fn
"""
from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

WORKTREE_ROOT = Path(__file__).parent.parent
RECIPES_DIR = WORKTREE_ROOT / "recipes"
AUDIT_LOG_V1 = RECIPES_DIR / "audit_log" / "v1"


# ---------------------------------------------------------------------------
# A — 5 recipes loadable
# ---------------------------------------------------------------------------

class TestAllRecipesLoadable:
    def test_five_recipes_listed(self):
        from src.infra.recipes import list_recipes
        recipes = list_recipes(str(RECIPES_DIR))
        names = {r.name for r in recipes}
        assert "auth" in names, f"auth missing from {names}"
        assert "audit_log" in names, f"audit_log missing from {names}"
        assert "pagination" in names, f"pagination missing from {names}"
        assert "search" in names, f"search missing from {names}"
        assert "file_upload" in names, f"file_upload missing from {names}"
        assert len(recipes) >= 5

    def test_all_five_load_without_error(self):
        from src.infra.recipes import load_recipe
        for name in ("auth", "audit_log", "pagination", "search", "file_upload"):
            r = load_recipe(str(RECIPES_DIR / name / "v1" / "recipe.yaml"))
            assert r.name == name
            assert r.version == "v1"


# ---------------------------------------------------------------------------
# A — new recipes match fastapi+postgres+nextjs
# ---------------------------------------------------------------------------

class TestNewRecipesMatchStack:
    def _all(self):
        from src.infra.recipes import list_recipes
        return list_recipes(str(RECIPES_DIR))

    def _match_names(self, stack: str) -> set[str]:
        from src.infra.recipes import match_recipe
        results = match_recipe(stack, self._all())
        return {r.name for r, _ in results}

    def test_pagination_matches_fastapi_postgres_nextjs(self):
        names = self._match_names("fastapi+postgres+nextjs")
        assert "pagination" in names

    def test_search_matches_fastapi_postgres_nextjs(self):
        names = self._match_names("fastapi+postgres+nextjs")
        assert "search" in names

    def test_file_upload_matches_fastapi_postgres_nextjs(self):
        names = self._match_names("fastapi+postgres+nextjs")
        assert "file_upload" in names


# ---------------------------------------------------------------------------
# B — audit_log backend.template.py fully filled
# ---------------------------------------------------------------------------

class TestAuditLogTemplateFilled:
    def _src(self) -> str:
        return (AUDIT_LOG_V1 / "backend.template.py").read_text(encoding="utf-8")

    def test_no_t6_will_fill_markers(self):
        src = self._src()
        assert "# T6 WILL FILL" not in src, "Found leftover T6 WILL FILL marker"

    def test_record_event_defined(self):
        assert "record_event" in self._src()

    def test_sweep_retention_defined(self):
        assert "sweep_retention" in self._src()

    def test_list_events_for_resource_defined(self):
        assert "list_events_for_resource" in self._src()

    def test_ast_parses(self):
        src = self._src()
        tree = ast.parse(src)
        assert tree is not None

    def test_has_fastapi_router(self):
        assert "APIRouter" in self._src()

    def test_has_pydantic_models(self):
        src = self._src()
        assert "AuditEventCreate" in src
        assert "AuditEventOut" in src

    def test_recipe_param_markers_preserved(self):
        assert "RECIPE_PARAM:" in self._src()


# ---------------------------------------------------------------------------
# B — audit_log smoke template fully filled
# ---------------------------------------------------------------------------

class TestAuditLogSmokeTemplateFilled:
    def _src(self) -> str:
        return (AUDIT_LOG_V1 / "tests" / "backend_smoke.template.py").read_text(encoding="utf-8")

    def test_no_t6_will_fill_markers(self):
        src = self._src()
        assert "# T6 WILL FILL" not in src, "Found leftover T6 WILL FILL in smoke template"

    def test_four_or_more_async_test_functions(self):
        src = self._src()
        fns = re.findall(r"^async def (test_\w+)", src, re.MULTILINE)
        assert len(fns) >= 4, f"Only {len(fns)} async test functions: {fns}"

    def test_ast_parses(self):
        tree = ast.parse(self._src())
        assert tree is not None


# ---------------------------------------------------------------------------
# C — severity blocker promotion
# ---------------------------------------------------------------------------

class TestSeverityBlockerPromotion:
    def test_pattern_lint_is_blocker(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        spec = POST_HOOK_REGISTRY["pattern_lint"]
        assert spec.default_severity == "blocker", (
            f"pattern_lint.default_severity={spec.default_severity!r}, expected 'blocker'"
        )

    def test_design_system_check_is_blocker(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        spec = POST_HOOK_REGISTRY["design_system_check"]
        assert spec.default_severity == "blocker", (
            f"design_system_check.default_severity={spec.default_severity!r}, expected 'blocker'"
        )

    def test_apply_semgrep_blocker_verdict_exists_and_callable(self):
        from general_beckman.apply import _apply_semgrep_blocker_verdict
        assert callable(_apply_semgrep_blocker_verdict)


# ---------------------------------------------------------------------------
# C — routing test: blocker/warning dispatch
# ---------------------------------------------------------------------------

class TestSemgrepVerdictRouting:
    """Verify _apply_pattern_lint_verdict dispatches to the right fn based on severity."""

    def _make_source(self) -> dict:
        return {
            "id": 1,
            "status": "ungraded",
            "worker_attempts": 0,
            "max_worker_attempts": 15,
            "context": "{}",
            "result": "",
            "title": "test",
        }

    def _make_verdict(self, kind: str, passed: bool = False) -> object:
        from general_beckman.result_router import PostHookVerdict
        return PostHookVerdict(
            source_task_id=1,
            kind=kind,
            passed=passed,
            raw={"skipped": True},  # use skipped=True so no DB writes needed
        )

    def test_blocker_severity_routes_to_blocker_fn(self):
        """When default_severity=blocker, _apply_semgrep_blocker_verdict is called."""
        from general_beckman import apply as _apply_mod

        called_with: list[str] = []

        async def mock_blocker(kind, findings_ctx_key, dlq_reason_ctx_key,
                               blocker_threshold, source, ctx, pending, verdict):
            called_with.append("blocker")

        async def mock_warning(kind, findings_ctx_key, dlq_reason_ctx_key,
                               source, ctx, pending, verdict):
            called_with.append("warning")

        async def run():
            with (
                patch.object(_apply_mod, "_apply_semgrep_blocker_verdict", new=mock_blocker),
                patch.object(_apply_mod, "_apply_semgrep_warning_verdict", new=mock_warning),
            ):
                await _apply_mod._apply_pattern_lint_verdict(
                    source=self._make_source(),
                    ctx={},
                    pending=["pattern_lint"],
                    verdict=self._make_verdict("pattern_lint"),
                )

        asyncio.new_event_loop().run_until_complete(run())
        assert "blocker" in called_with, f"Expected blocker fn called, got: {called_with}"
        assert "warning" not in called_with

    def test_warning_severity_routes_to_warning_fn_when_overridden(self):
        """When default_severity is forced to warning, _apply_semgrep_warning_verdict is called."""
        from general_beckman import apply as _apply_mod
        from general_beckman.posthooks import POST_HOOK_REGISTRY, PostHookSpec

        called_with: list[str] = []

        async def mock_blocker(kind, findings_ctx_key, dlq_reason_ctx_key,
                               blocker_threshold, source, ctx, pending, verdict):
            called_with.append("blocker")

        async def mock_warning(kind, findings_ctx_key, dlq_reason_ctx_key,
                               source, ctx, pending, verdict):
            called_with.append("warning")

        # Temporarily override spec to warning for this test
        original_spec = POST_HOOK_REGISTRY["pattern_lint"]
        warning_spec = PostHookSpec(
            kind=original_spec.kind,
            verb=original_spec.verb,
            default_severity="warning",
            auto_wire_triggers=original_spec.auto_wire_triggers,
            description=original_spec.description,
        )

        async def run():
            POST_HOOK_REGISTRY["pattern_lint"] = warning_spec
            try:
                with (
                    patch.object(_apply_mod, "_apply_semgrep_blocker_verdict", new=mock_blocker),
                    patch.object(_apply_mod, "_apply_semgrep_warning_verdict", new=mock_warning),
                ):
                    await _apply_mod._apply_pattern_lint_verdict(
                        source=self._make_source(),
                        ctx={},
                        pending=["pattern_lint"],
                        verdict=self._make_verdict("pattern_lint"),
                    )
            finally:
                POST_HOOK_REGISTRY["pattern_lint"] = original_spec

        asyncio.new_event_loop().run_until_complete(run())
        assert "warning" in called_with, f"Expected warning fn called, got: {called_with}"
        assert "blocker" not in called_with
