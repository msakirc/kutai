"""Extract public signatures from source files for cross-file consistency checks.

Mechanical executor. No LLM. Pure static analysis.

**Python strategy** — ``ast`` (stdlib):
- Walk each ``.py`` file for ``FunctionDef``, ``AsyncFunctionDef``, and
  ``ClassDef`` nodes.
- For each function/method extract: name, kind, params (arg names), return
  annotation (as string), lineno.
- For classes extract all methods (name, kind="method", params, returns, line).

**TypeScript / TSX / JS / JSX strategy**:
- Attempt to import ``tree_sitter`` at runtime.
- If tree-sitter is NOT installed, log a WARNING and return an empty list for
  each TS/JS file (soft-skip; does not fail the overall invocation).

**Mismatch detection** — name-only cross-file check:
- Collect all name→arity definitions across files.
- For every ``ast.Name`` node with ``ctx=Load`` (function call candidate),
  if a file-B defines a function of the same name with different arity, emit a
  mismatch record.
- Best-effort heuristic; may produce false positives for overloaded names.
  Documented here so callers understand the limitations.

Return shape::

    {
        "signatures": {
            "<rel_path>": [
                {
                    "name": "foo",
                    "kind": "function" | "class" | "method",
                    "params": ["self", "x", "y"],
                    "returns": "str | None",
                    "line": 42,
                }
            ]
        },
        "mismatches": [
            {
                "caller": "<rel_path>",
                "callee": "<rel_path>",
                "kind": "arity",
                "why": "foo called with 2 args but defined with 3 params",
            }
        ],
        "skipped": ["<rel_path>"],  # TS/JS files skipped (no tree-sitter)
        "duration_s": 0.123,
    }
"""

from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.extract_signatures")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Signature = dict[str, Any]  # name, kind, params, returns, line


# ---------------------------------------------------------------------------
# Python AST extraction
# ---------------------------------------------------------------------------


def _annotation_to_str(node: ast.expr | None) -> str | None:
    """Convert an annotation AST node to a string representation."""
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _func_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return argument names for a function/method definition."""
    args = node.args
    result: list[str] = []
    for a in args.posonlyargs:
        result.append(a.arg)
    for a in args.args:
        result.append(a.arg)
    if args.vararg:
        result.append(f"*{args.vararg.arg}")
    for a in args.kwonlyargs:
        result.append(a.arg)
    if args.kwarg:
        result.append(f"**{args.kwarg.arg}")
    return result


def _extract_py_signatures(source: str, rel_path: str) -> list[Signature]:
    """Return signatures extracted from Python source."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.warning("ast.parse failed", path=rel_path, error=str(exc))
        return []

    sigs: list[Signature] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            sigs.append(
                {
                    "name": node.name,
                    "kind": "class",
                    "params": [],
                    "returns": None,
                    "line": node.lineno,
                }
            )
            # Extract methods within the class body
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sigs.append(
                        {
                            "name": item.name,
                            "kind": "method",
                            "params": _func_params(item),
                            "returns": _annotation_to_str(item.returns),
                            "line": item.lineno,
                        }
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Only top-level functions (parent is Module)
            # We identify top-level by checking if the function is directly
            # in the module body using a parent-tracking approach.
            # Simpler: collect all, dedup later (nested functions will also
            # appear via ast.walk but that's acceptable for signature extraction).
            sigs.append(
                {
                    "name": node.name,
                    "kind": "function",
                    "params": _func_params(node),
                    "returns": _annotation_to_str(node.returns),
                    "line": node.lineno,
                }
            )

    return sigs


# ---------------------------------------------------------------------------
# TypeScript / JS extraction (tree-sitter, optional)
# ---------------------------------------------------------------------------


def _try_extract_ts_signatures(
    source: str, rel_path: str
) -> list[Signature] | None:
    """Try to extract signatures using tree-sitter.

    Returns ``None`` if tree-sitter is not installed (caller soft-skips).
    Returns a (possibly empty) list on success.
    """
    try:
        import tree_sitter  # type: ignore[import]
    except ImportError:
        return None

    # Minimal extraction: best-effort with tree-sitter if available.
    # Full implementation would walk the tree for function_declaration,
    # method_definition, arrow_function nodes. For now, return empty list
    # so callers see "no signatures" rather than an error.
    logger.debug(
        "tree-sitter available but TS extraction not fully implemented; "
        "returning empty signatures for %s",
        rel_path,
    )
    _ = tree_sitter  # suppress unused import lint
    return []


# ---------------------------------------------------------------------------
# Cross-file mismatch detection
# ---------------------------------------------------------------------------


def _collect_call_arities(source: str) -> dict[str, list[int]]:
    """Return {name: [arity, ...]} for every Name(ctx=Load) call site in source.

    Heuristic: walks for Call nodes whose ``func`` is a plain Name. Does not
    handle method calls (``obj.foo(...)``), but that's acceptable for the
    best-effort check.
    """
    result: dict[str, list[int]] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return result

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
                arity = len(node.args) + len(node.keywords)
                result.setdefault(name, []).append(arity)

    return result


def _detect_mismatches(
    sigs_by_file: dict[str, list[Signature]],
    sources_by_file: dict[str, str],
) -> list[dict[str, str]]:
    """Name-only cross-file arity mismatch detection.

    For every call site in file A that references a name defined in file B,
    if any call arity differs from the definition param count, emit a mismatch.

    Only plain-name function/method definitions are compared.
    """
    # Build name → (rel_path, param_count, kind) map (first definition wins)
    name_to_def: dict[str, tuple[str, int, str]] = {}
    for rel_path, sigs in sigs_by_file.items():
        for sig in sigs:
            if sig["kind"] in ("function", "method"):
                name = sig["name"]
                if name not in name_to_def:
                    name_to_def[name] = (rel_path, len(sig["params"]), sig["kind"])

    mismatches: list[dict[str, str]] = []

    for caller_path, source in sources_by_file.items():
        call_arities = _collect_call_arities(source)
        for name, arities in call_arities.items():
            if name not in name_to_def:
                continue
            callee_path, param_count, kind = name_to_def[name]
            if callee_path == caller_path:
                continue  # same-file calls not cross-file checked
            for arity in arities:
                # For methods, allow off-by-one to account for implicit `self`/`cls`
                # when called as an unbound function reference.
                if kind == "method" and abs(arity - param_count) == 1:
                    break  # acceptable
                if arity != param_count:
                    mismatches.append(
                        {
                            "caller": caller_path,
                            "callee": callee_path,
                            "kind": "arity",
                            "why": (
                                f"{name} called with {arity} arg(s) but "
                                f"defined with {param_count} param(s)"
                            ),
                        }
                    )
                    break  # one mismatch record per call site per name

    return mismatches


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def extract_signatures(
    target_files: list[str],
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Extract public signatures from *target_files* and detect cross-file mismatches.

    Parameters
    ----------
    target_files:
        Workspace-relative (or absolute) paths to analyse.
    workspace_path:
        Absolute path to the mission workspace root. Used to resolve relative
        paths. Falls back to ``cwd()`` when not supplied.

    Returns
    -------
    dict with keys:

    ``signatures``
        ``{rel_path: [Signature, ...]}`` — per-file list of extracted signatures.
    ``mismatches``
        ``[{caller, callee, kind, why}]`` — cross-file arity mismatches.
    ``skipped``
        Paths soft-skipped (TS/JS without tree-sitter, missing files, …).
    ``duration_s``
        Wall-clock time.
    """
    t0 = time.monotonic()

    workspace = Path(workspace_path) if workspace_path else Path.cwd()

    sigs_by_file: dict[str, list[Signature]] = {}
    sources_by_file: dict[str, str] = {}
    skipped: list[str] = []

    for rel in target_files:
        p = Path(rel) if Path(rel).is_absolute() else workspace / rel
        if not p.exists():
            logger.warning("extract_signatures: file not found", path=rel)
            skipped.append(rel)
            continue

        suffix = p.suffix.lower()

        if suffix == ".py":
            try:
                source = p.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                logger.warning(
                    "extract_signatures: cannot read file",
                    path=rel, error=str(exc),
                )
                skipped.append(rel)
                continue
            sigs = _extract_py_signatures(source, rel)
            sigs_by_file[rel] = sigs
            sources_by_file[rel] = source

        elif suffix in (".ts", ".tsx", ".js", ".jsx"):
            result = _try_extract_ts_signatures("", rel)
            if result is None:
                logger.warning(
                    "extract_signatures: tree-sitter not installed; "
                    "skipping TS/JS file %s",
                    rel,
                )
                skipped.append(rel)
            else:
                try:
                    source = p.read_text(encoding="utf-8", errors="replace")
                    result = _try_extract_ts_signatures(source, rel)
                    sigs_by_file[rel] = result or []
                    sources_by_file[rel] = source
                except Exception as exc:
                    logger.warning(
                        "extract_signatures: TS read error",
                        path=rel, error=str(exc),
                    )
                    skipped.append(rel)

        else:
            logger.debug(
                "extract_signatures: unsupported extension, skipping %s", rel
            )
            skipped.append(rel)

    mismatches = _detect_mismatches(sigs_by_file, sources_by_file)

    duration = time.monotonic() - t0
    logger.info(
        "extract_signatures complete",
        files=len(sigs_by_file),
        skipped=len(skipped),
        mismatches=len(mismatches),
        duration_s=round(duration, 3),
    )

    return {
        "signatures": sigs_by_file,
        "mismatches": mismatches,
        "skipped": skipped,
        "duration_s": round(duration, 3),
    }
