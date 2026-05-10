"""Inventory walker — discovers scaffolding layers under packages/ + src/.

A "layer" is one of:
- a top-level package under `packages/` (e.g. `mr_roboto`, `fatih_hoca`)
- a top-level subdirectory under `src/` (e.g. `src/agents`, `src/core`)

Per layer we collect: LOC (Python source lines), public-symbol count
(top-level def/class without leading underscore), declared rationale (the
package's `__init__.py` module-docstring or top-of-package README first
non-blank line), test count (`test_*.py` files in the layer's tests/ dir),
dependency count (rough — distinct top-level imports), and last-touched
commit ISO (best-effort via `git log -1` on the layer path; None when git
isn't available or the layer is untracked).
"""
from __future__ import annotations

import ast
import dataclasses
import os
import subprocess
from pathlib import Path
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class LayerReport:
    name: str
    kind: str  # "package" | "src_module"
    path: str  # repo-relative
    loc: int
    public_symbols: int
    test_count: int
    dependency_count: int
    rationale: str
    last_touched_iso: str | None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def walk_layers(root: Path) -> list[LayerReport]:
    """Discover every package and src module under ``root``."""
    root = Path(root)
    out: list[LayerReport] = []

    pkg_root = root / "packages"
    if pkg_root.is_dir():
        for entry in sorted(pkg_root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            out.append(_inspect_package(root, entry))

    src_root = root / "src"
    if src_root.is_dir():
        for entry in sorted(src_root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if entry.name == "__pycache__":
                continue
            out.append(_inspect_src_module(root, entry))

    return out


# ---- internals -------------------------------------------------------------


def _inspect_package(repo_root: Path, pkg_dir: Path) -> LayerReport:
    name = pkg_dir.name
    src_dir = pkg_dir / "src" / name
    code_dir = src_dir if src_dir.is_dir() else pkg_dir
    loc = sum(_count_loc(p) for p in _iter_py(code_dir))
    pubs = sum(_public_symbols(p) for p in _iter_py(code_dir))

    tests_dir = pkg_dir / "tests"
    test_count = (
        sum(1 for p in tests_dir.glob("test_*.py")) if tests_dir.is_dir() else 0
    )

    deps = _dependency_count(code_dir, self_name=name)
    rationale = _rationale_for_package(pkg_dir)
    last = _last_touched(repo_root, pkg_dir)
    return LayerReport(
        name=name,
        kind="package",
        path=str(pkg_dir.relative_to(repo_root)).replace("\\", "/"),
        loc=loc,
        public_symbols=pubs,
        test_count=test_count,
        dependency_count=deps,
        rationale=rationale,
        last_touched_iso=last,
    )


def _inspect_src_module(repo_root: Path, mod_dir: Path) -> LayerReport:
    name = f"src/{mod_dir.name}"
    loc = sum(_count_loc(p) for p in _iter_py(mod_dir))
    pubs = sum(_public_symbols(p) for p in _iter_py(mod_dir))

    # Tests for src/foo live under tests/foo or tests/test_foo*.py — count
    # both for an approximate signal, never zero out unfairly.
    tests_root = repo_root / "tests"
    test_count = 0
    if tests_root.is_dir():
        candidate = tests_root / mod_dir.name
        if candidate.is_dir():
            test_count += sum(1 for _ in candidate.rglob("test_*.py"))
        # Loose-prefix match for flat tests files
        test_count += sum(
            1 for _ in tests_root.glob(f"test_{mod_dir.name}*.py")
        )

    deps = _dependency_count(mod_dir, self_name=mod_dir.name)
    rationale = _rationale_for_src(mod_dir)
    last = _last_touched(repo_root, mod_dir)
    return LayerReport(
        name=name,
        kind="src_module",
        path=str(mod_dir.relative_to(repo_root)).replace("\\", "/"),
        loc=loc,
        public_symbols=pubs,
        test_count=test_count,
        dependency_count=deps,
        rationale=rationale,
        last_touched_iso=last,
    )


def _iter_py(d: Path) -> Iterable[Path]:
    if not d.is_dir():
        return []
    return (
        p for p in d.rglob("*.py")
        if "__pycache__" not in p.parts and not p.name.startswith(".")
    )


def _count_loc(path: Path) -> int:
    """Source lines: non-blank, non-comment-only."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0
    n = 0
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        n += 1
    return n


def _public_symbols(path: Path) -> int:
    """Top-level def/class without leading underscore."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return 0
    count = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                count += 1
    return count


def _dependency_count(code_dir: Path, self_name: str) -> int:
    """Rough import fan-out: distinct top-level module names imported,
    excluding stdlib-ish + self.
    """
    seen: set[str] = set()
    for path in _iter_py(code_dir):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            mod: str | None = None
            if isinstance(node, ast.Import):
                for n in node.names:
                    seen.add(n.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mod = node.module.split(".", 1)[0]
                    if mod:
                        seen.add(mod)
    seen.discard(self_name)
    seen.discard("__future__")
    return len(seen)


def _rationale_for_package(pkg_dir: Path) -> str:
    name = pkg_dir.name
    init = pkg_dir / "src" / name / "__init__.py"
    if init.is_file():
        doc = _module_docstring(init)
        if doc:
            return doc
    readme = pkg_dir / "README.md"
    if readme.is_file():
        return _first_para(readme)
    return ""


def _rationale_for_src(mod_dir: Path) -> str:
    init = mod_dir / "__init__.py"
    if init.is_file():
        doc = _module_docstring(init)
        if doc:
            return doc
    # Fall back to the first .py file's docstring
    for p in sorted(mod_dir.rglob("*.py")):
        doc = _module_docstring(p)
        if doc:
            return doc
    return ""


def _module_docstring(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return ""
    doc = ast.get_docstring(tree) or ""
    return doc.strip().splitlines()[0].strip() if doc else ""


def _first_para(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return s
    return ""


def _last_touched(repo_root: Path, layer_path: Path) -> str | None:
    """Best-effort `git log -1 --format=%cI <path>`. Returns None on failure."""
    try:
        rel = layer_path.relative_to(repo_root)
    except ValueError:
        return None
    try:
        cp = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", str(rel)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        out = (cp.stdout or "").strip()
        return out or None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
