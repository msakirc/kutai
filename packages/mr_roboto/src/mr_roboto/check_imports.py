"""Static import checker — verify that imports in produced files resolve against
the project manifest.

Mechanical executor. No LLM. Pure static analysis via ``ast`` (Python) and
regex (TypeScript/TSX/JS).  No runtime import resolution; no pip/npm network
calls.

**Python strategy** (A from the v2 doc):
- ``ast.parse`` each ``*.py`` file.
- Collect top-level ``Import`` + ``ImportFrom`` nodes.
- Resolve declared deps from ``pyproject.toml`` (``[project.dependencies]``
  + ``[project.optional-dependencies.*]``) and ``requirements*.txt`` fallback.
- Stdlib modules (``sys.stdlib_module_names``) → ignored.
- Top-level dirs in workspace → treated as "own packages" → ignored.
- Missing → blocker.
- Declared-but-unused: **deferred** (heuristic too noisy at file level; a
  package may be consumed by peer files, not imported directly).

**TypeScript strategy**:
- Regex-extract ``import ... from "X"`` and ``import("X")`` (handles static +
  dynamic forms).
- Resolve against ``package.json`` ``dependencies`` + ``devDependencies``.
- Relative imports (starting with ``./`` or ``../``) → ignored.

Return shape mirrors other mr_roboto verbs: ``ok``, ``missing``,
``file_results``, ``duration_s``.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-reuse]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.check_imports")

# ---------------------------------------------------------------------------
# Manifest readers
# ---------------------------------------------------------------------------

def _norm_pkg(name: str) -> str:
    """Normalise package name: lower-case, hyphens→underscores."""
    return name.lower().replace("-", "_").replace(".", "_")


def _read_pyproject_deps(workspace: Path) -> set[str]:
    """Return normalised declared dependencies from pyproject.toml."""
    p = workspace / "pyproject.toml"
    if not p.exists():
        return set()
    if tomllib is None:
        logger.warning("tomllib not available; pyproject.toml not parsed")
        return set()
    try:
        with p.open("rb") as f:
            data = tomllib.load(f)
    except Exception as exc:
        logger.warning("pyproject.toml parse error", error=str(exc))
        return set()
    deps: set[str] = set()
    project = data.get("project") or {}
    for raw in project.get("dependencies") or []:
        # PEP 508: package name ends before any whitespace or specifier
        m = re.match(r"^([A-Za-z0-9_\-\.]+)", raw.strip())
        if m:
            deps.add(_norm_pkg(m.group(1)))
    for extras in (project.get("optional-dependencies") or {}).values():
        for raw in extras or []:
            m = re.match(r"^([A-Za-z0-9_\-\.]+)", raw.strip())
            if m:
                deps.add(_norm_pkg(m.group(1)))
    return deps


def _read_requirements_deps(workspace: Path) -> set[str]:
    """Return normalised declared dependencies from requirements*.txt."""
    deps: set[str] = set()
    for req_file in sorted(workspace.glob("requirements*.txt")):
        try:
            for line in req_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                # Strip version specifiers and extras
                m = re.match(r"^([A-Za-z0-9_\-\.]+)", line)
                if m:
                    deps.add(_norm_pkg(m.group(1)))
        except Exception as exc:
            logger.warning("requirements file parse error",
                           file=str(req_file), error=str(exc))
    return deps


def _own_packages(workspace: Path) -> set[str]:
    """Top-level Python package dirs (have __init__.py) in the workspace."""
    own: set[str] = set()
    try:
        for item in workspace.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                own.add(_norm_pkg(item.name))
        # Also scan packages/ sub-dirs (monorepo layout)
        packages_dir = workspace / "packages"
        if packages_dir.is_dir():
            for pkg in packages_dir.iterdir():
                if pkg.is_dir():
                    src_dir = pkg / "src"
                    if src_dir.is_dir():
                        for sub in src_dir.iterdir():
                            if sub.is_dir() and (sub / "__init__.py").exists():
                                own.add(_norm_pkg(sub.name))
    except Exception:
        pass
    return own


# ---------------------------------------------------------------------------
# Python checker
# ---------------------------------------------------------------------------

_STDLIB: frozenset[str] = frozenset(
    _norm_pkg(m) for m in getattr(sys, "stdlib_module_names", frozenset())
)


def _py_top_level_imports(source: str) -> list[tuple[str, int]]:
    """Return (top_level_module, lineno) for all imports in source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    results: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                results.append((top, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                top = node.module.split(".")[0]
                results.append((top, node.lineno))
            # relative imports (level > 0) are project-internal → skip
    return results


def _check_py_file(
    path: Path,
    declared_py: set[str],
    own_pkgs: set[str],
) -> list[dict]:
    """Return list of missing-import records for one Python file."""
    missing: list[dict] = []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("cannot read py file", path=str(path), error=str(exc))
        return missing
    for (mod, lineno) in _py_top_level_imports(source):
        norm = _norm_pkg(mod)
        if norm in _STDLIB:
            continue
        if norm in own_pkgs:
            continue
        if norm in declared_py:
            continue
        missing.append({"module": mod, "line": lineno})
    return missing


# ---------------------------------------------------------------------------
# TypeScript / TSX checker
# ---------------------------------------------------------------------------

# Matches: import ... from "pkg" / import ... from 'pkg'
#          import("pkg") / import('pkg')
_TS_IMPORT_RE = re.compile(
    r"""(?:import\s.*?\bfrom\s+|import\s*\()['"]([^'"]+)['"]""",
    re.DOTALL,
)


def _read_package_json_deps(workspace: Path) -> set[str]:
    p = workspace / "package.json"
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("package.json parse error", error=str(exc))
        return set()
    deps: set[str] = set()
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        for raw in (data.get(key) or {}):
            # Scoped package: @scope/name → normalise as @scope/name
            deps.add(raw.lower())
    return deps


def _ts_specifier_to_pkg(specifier: str) -> str:
    """Convert an import specifier to its npm package name."""
    if specifier.startswith("@"):
        # @scope/name/sub → @scope/name
        parts = specifier.split("/")
        return "/".join(parts[:2]).lower()
    return specifier.split("/")[0].lower()


def _check_ts_file(
    path: Path,
    declared_ts: set[str],
) -> list[dict]:
    """Return list of missing-import records for one TS/TSX file."""
    missing: list[dict] = []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("cannot read ts file", path=str(path), error=str(exc))
        return missing
    for lineno, line in enumerate(source.splitlines(), start=1):
        for m in _TS_IMPORT_RE.finditer(line):
            specifier = m.group(1)
            # Skip relative imports
            if specifier.startswith("./") or specifier.startswith("../"):
                continue
            pkg = _ts_specifier_to_pkg(specifier)
            if pkg and pkg not in declared_ts:
                missing.append({"module": specifier, "package": pkg, "line": lineno})
    return missing


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def check_imports(
    mission_id: int | None,
    target_files: list[str] | None = None,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Check that all imports in *target_files* resolve against the project manifest.

    Parameters
    ----------
    target_files:
        List of workspace-relative file paths to check.  Globs are not
        expanded here — the caller (apply.py) passes concrete paths from
        the step's ``produces`` list.  If None or empty, nothing is checked
        and ``ok=True`` is returned immediately (no-op, idempotent).
    workspace_path:
        Absolute path to the mission workspace root.  Required for manifest
        resolution.

    Returns
    -------
    dict with keys:
        ``ok``          – True iff no missing imports were found.
        ``missing``     – list of {file, module, line} records (blockers).
        ``file_results``– per-file breakdown for debugging.
        ``skipped``     – files skipped (unsupported extension or not found).
        ``duration_s``  – wall time.
    """
    t0 = time.monotonic()

    if not target_files:
        return {
            "ok": True, "missing": [], "file_results": {}, "skipped": [],
            "duration_s": 0.0,
        }

    workspace = Path(workspace_path) if workspace_path else Path.cwd()

    # Resolve manifests once per invocation
    declared_py: set[str] = _read_pyproject_deps(workspace) | _read_requirements_deps(workspace)
    own_pkgs: set[str] = _own_packages(workspace)
    declared_ts: set[str] | None = None  # lazy: only if TS files present

    all_missing: list[dict] = []
    file_results: dict[str, Any] = {}
    skipped: list[str] = []

    for rel in target_files:
        path = workspace / rel if not Path(rel).is_absolute() else Path(rel)
        if not path.exists():
            skipped.append(rel)
            continue

        suffix = path.suffix.lower()

        if suffix == ".py":
            file_missing = _check_py_file(path, declared_py, own_pkgs)
            file_results[rel] = {"missing": file_missing}
            for rec in file_missing:
                all_missing.append({"file": rel, **rec})

        elif suffix in (".ts", ".tsx", ".js", ".jsx"):
            if declared_ts is None:
                declared_ts = _read_package_json_deps(workspace)
            file_missing = _check_ts_file(path, declared_ts)
            file_results[rel] = {"missing": file_missing}
            for rec in file_missing:
                all_missing.append({"file": rel, **rec})

        else:
            skipped.append(rel)

    duration = time.monotonic() - t0
    ok = len(all_missing) == 0

    logger.info(
        "check_imports complete",
        ok=ok, missing_count=len(all_missing), skipped=len(skipped),
        duration_s=round(duration, 3),
    )

    return {
        "ok": ok,
        "missing": all_missing,
        "file_results": file_results,
        "skipped": skipped,
        "duration_s": round(duration, 3),
    }
