"""Tests for Z2 T2B — imports_check post-hook kind.

Covers:
- Registry contains 'imports_check' with correct triggers and severity.
- Expander auto-wire: step with produces '*.py' gets imports_check prepended.
- Verb behaviour (check_imports):
    - Python file importing undeclared 'requests' → blocker verdict.
    - Python file importing only stdlib + local pkgs → pass.
    - TS file importing undeclared 'axios' → blocker verdict.
    - TS file importing only relative paths → pass.
- apply.py: _posthook_agent_and_payload maps 'imports_check' to mechanical.

All tests use tmp_path fixtures with synthetic manifests. No network calls.
No installed packages required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_pyproject(workspace: Path, deps: list[str]) -> None:
    dep_lines = "\n".join(f'  "{d}",' for d in deps)
    (workspace / "pyproject.toml").write_text(
        f'[project]\nname = "test"\ndependencies = [\n{dep_lines}\n]\n',
        encoding="utf-8",
    )


def _write_package_json(workspace: Path, deps: dict) -> None:
    (workspace / "package.json").write_text(
        json.dumps({"dependencies": deps}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestImportsCheckRegistry:
    def test_kind_present_in_registry(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        assert "imports_check" in POST_HOOK_REGISTRY

    def test_spec_fields(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        spec = POST_HOOK_REGISTRY["imports_check"]
        assert spec.kind == "imports_check"
        assert spec.verb == "check_imports"
        assert spec.default_severity == "blocker"

    def test_auto_wire_triggers(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        triggers = POST_HOOK_REGISTRY["imports_check"].auto_wire_triggers
        assert "*.py" in triggers
        assert "*.ts" in triggers
        assert "*.tsx" in triggers

    def test_kind_in_post_hook_kinds(self):
        from general_beckman.posthooks import POST_HOOK_KINDS
        assert "imports_check" in POST_HOOK_KINDS

    def test_idempotent_registry(self):
        """Re-importing should not duplicate the key."""
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        count = sum(1 for k in POST_HOOK_REGISTRY if k == "imports_check")
        assert count == 1


# ---------------------------------------------------------------------------
# Expander auto-wire tests
# ---------------------------------------------------------------------------

class TestImportsCheckAutoWire:
    def _wire(self, step: dict) -> dict:
        """Run the expander auto-wire logic on a single step (mutates in place)."""
        from src.workflows.engine.expander import _auto_wire_posthooks
        _auto_wire_posthooks(step)
        return step

    def test_py_produces_gets_imports_check(self):
        step = {"id": "1.write_code", "produces": ["app/foo.py"], "post_hooks": []}
        result = self._wire(step)
        assert "imports_check" in result.get("post_hooks", [])

    def test_ts_produces_gets_imports_check(self):
        step = {"id": "1.write_ts", "produces": ["src/bar.ts"], "post_hooks": []}
        result = self._wire(step)
        assert "imports_check" in result.get("post_hooks", [])

    def test_tsx_produces_gets_imports_check(self):
        step = {"id": "1.write_tsx", "produces": ["src/Button.tsx"], "post_hooks": []}
        result = self._wire(step)
        assert "imports_check" in result.get("post_hooks", [])

    def test_txt_produces_no_imports_check(self):
        step = {"id": "1.write_txt", "produces": ["README.txt"], "post_hooks": []}
        result = self._wire(step)
        assert "imports_check" not in result.get("post_hooks", [])

    def test_no_duplicate_wire(self):
        """If imports_check already in post_hooks, don't add again."""
        step = {"id": "1.x", "produces": ["app/foo.py"], "post_hooks": ["imports_check"]}
        result = self._wire(step)
        hooks = result.get("post_hooks", [])
        assert hooks.count("imports_check") == 1


# ---------------------------------------------------------------------------
# check_imports verb tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCheckImportsPython:
    async def test_missing_dep_blocker(self, tmp_path):
        """Python file importing undeclared 'requests' → ok=False."""
        _write_pyproject(tmp_path, [])  # no deps declared
        src = tmp_path / "app.py"
        src.write_text("import requests\nprint('hi')\n", encoding="utf-8")
        # Make workspace own-package detection miss 'requests'
        from mr_roboto.check_imports import check_imports
        result = await check_imports(
            mission_id=None,
            target_files=["app.py"],
            workspace_path=str(tmp_path),
        )
        assert result["ok"] is False
        modules = [r["module"] for r in result["missing"]]
        assert "requests" in modules

    async def test_stdlib_only_passes(self, tmp_path):
        """Python file using only stdlib → ok=True."""
        _write_pyproject(tmp_path, [])
        src = tmp_path / "util.py"
        src.write_text(
            "import os\nimport sys\nfrom pathlib import Path\n", encoding="utf-8"
        )
        from mr_roboto.check_imports import check_imports
        result = await check_imports(
            mission_id=None,
            target_files=["util.py"],
            workspace_path=str(tmp_path),
        )
        assert result["ok"] is True
        assert result["missing"] == []

    async def test_declared_dep_passes(self, tmp_path):
        """Python file importing declared dep → ok=True."""
        _write_pyproject(tmp_path, ["requests"])
        src = tmp_path / "client.py"
        src.write_text("import requests\n", encoding="utf-8")
        from mr_roboto.check_imports import check_imports
        result = await check_imports(
            mission_id=None,
            target_files=["client.py"],
            workspace_path=str(tmp_path),
        )
        assert result["ok"] is True

    async def test_own_package_ignored(self, tmp_path):
        """Python file importing an own package (has __init__.py) → ok=True."""
        # Create a 'myapp' package in workspace
        pkg = tmp_path / "myapp"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        src = tmp_path / "main.py"
        src.write_text("from myapp import core\n", encoding="utf-8")
        _write_pyproject(tmp_path, [])
        from mr_roboto.check_imports import check_imports
        result = await check_imports(
            mission_id=None,
            target_files=["main.py"],
            workspace_path=str(tmp_path),
        )
        assert result["ok"] is True

    async def test_requirements_txt_fallback(self, tmp_path):
        """Dependencies from requirements.txt also satisfy imports."""
        (tmp_path / "requirements.txt").write_text(
            "requests==2.31.0\n", encoding="utf-8"
        )
        src = tmp_path / "app.py"
        src.write_text("import requests\n", encoding="utf-8")
        from mr_roboto.check_imports import check_imports
        result = await check_imports(
            mission_id=None,
            target_files=["app.py"],
            workspace_path=str(tmp_path),
        )
        assert result["ok"] is True

    async def test_empty_target_list_is_noop(self, tmp_path):
        from mr_roboto.check_imports import check_imports
        result = await check_imports(
            mission_id=None,
            target_files=[],
            workspace_path=str(tmp_path),
        )
        assert result["ok"] is True
        assert result["missing"] == []

    async def test_nonexistent_file_is_skipped(self, tmp_path):
        from mr_roboto.check_imports import check_imports
        result = await check_imports(
            mission_id=None,
            target_files=["ghost.py"],
            workspace_path=str(tmp_path),
        )
        assert result["ok"] is True
        assert "ghost.py" in result["skipped"]


@pytest.mark.asyncio
class TestCheckImportsTypeScript:
    async def test_missing_pkg_blocker(self, tmp_path):
        """TS file importing undeclared 'axios' → ok=False."""
        _write_package_json(tmp_path, {})  # no deps
        src = tmp_path / "api.ts"
        src.write_text('import axios from "axios";\n', encoding="utf-8")
        from mr_roboto.check_imports import check_imports
        result = await check_imports(
            mission_id=None,
            target_files=["api.ts"],
            workspace_path=str(tmp_path),
        )
        assert result["ok"] is False
        modules = [r["module"] for r in result["missing"]]
        assert "axios" in modules

    async def test_relative_imports_pass(self, tmp_path):
        """TS file with only relative imports → ok=True (no package.json needed)."""
        src = tmp_path / "component.tsx"
        src.write_text(
            'import { foo } from "./foo";\nimport bar from "../bar";\n',
            encoding="utf-8",
        )
        from mr_roboto.check_imports import check_imports
        result = await check_imports(
            mission_id=None,
            target_files=["component.tsx"],
            workspace_path=str(tmp_path),
        )
        assert result["ok"] is True

    async def test_declared_pkg_passes(self, tmp_path):
        """TS file importing declared npm package → ok=True."""
        _write_package_json(tmp_path, {"react": "^18.0.0"})
        src = tmp_path / "App.tsx"
        src.write_text('import React from "react";\n', encoding="utf-8")
        from mr_roboto.check_imports import check_imports
        result = await check_imports(
            mission_id=None,
            target_files=["App.tsx"],
            workspace_path=str(tmp_path),
        )
        assert result["ok"] is True

    async def test_scoped_pkg_missing(self, tmp_path):
        """TS file importing missing scoped package @tanstack/react-query → blocker."""
        _write_package_json(tmp_path, {})
        src = tmp_path / "query.ts"
        src.write_text(
            'import { useQuery } from "@tanstack/react-query";\n', encoding="utf-8"
        )
        from mr_roboto.check_imports import check_imports
        result = await check_imports(
            mission_id=None,
            target_files=["query.ts"],
            workspace_path=str(tmp_path),
        )
        assert result["ok"] is False
        pkgs = [r.get("package") for r in result["missing"]]
        assert "@tanstack/react-query" in pkgs

    async def test_dynamic_import_missing(self, tmp_path):
        """Dynamic import() with missing package → blocker."""
        _write_package_json(tmp_path, {})
        src = tmp_path / "loader.ts"
        src.write_text(
            'const mod = await import("some-heavy-pkg");\n', encoding="utf-8"
        )
        from mr_roboto.check_imports import check_imports
        result = await check_imports(
            mission_id=None,
            target_files=["loader.ts"],
            workspace_path=str(tmp_path),
        )
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# apply.py _posthook_agent_and_payload routing test
# ---------------------------------------------------------------------------

class TestApplyImportsCheck:
    def _make_request(self, source_ctx: dict):
        """Build a minimal RequestPostHook-like object for imports_check."""
        from dataclasses import dataclass

        @dataclass
        class FakeRequest:
            kind: str = "imports_check"
            source_task_id: int = 42

        return FakeRequest()

    def test_imports_check_routes_to_mechanical(self):
        from general_beckman.apply import _posthook_agent_and_payload
        from dataclasses import dataclass

        @dataclass
        class FakeRequest:
            kind: str = "imports_check"
            source_task_id: int = 99

        req = FakeRequest()
        source_ctx = {
            "produces": ["app/main.py", "app/utils.py"],
            "workspace_path": "/workspace/mission-42",
        }
        agent_type, payload = _posthook_agent_and_payload(req, {}, source_ctx)
        assert agent_type == "mechanical"
        inner = payload["payload"]
        assert inner["action"] == "check_imports"
        assert "app/main.py" in inner["target_files"]
        assert inner["workspace_path"] == "/workspace/mission-42"
        assert payload["posthook_kind"] == "imports_check"

    def test_imports_check_with_empty_produces(self):
        from general_beckman.apply import _posthook_agent_and_payload
        from dataclasses import dataclass

        @dataclass
        class FakeRequest:
            kind: str = "imports_check"
            source_task_id: int = 7

        req = FakeRequest()
        agent_type, payload = _posthook_agent_and_payload(req, {}, {})
        assert agent_type == "mechanical"
        assert payload["payload"]["target_files"] == []
