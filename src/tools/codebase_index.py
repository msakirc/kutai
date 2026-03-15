# tools/codebase_index.py
"""
Codebase indexing — builds a structural index of source files.

Phase 12 upgrade: now uses tree-sitter multi-language parser (with
ast/regex fallback) to support Python, JS, TS, Go, Rust, Java, C/C++.

Provides:
  - build_index: Scan a directory and index all source files
  - query_index: Search the index by function/class/module name
  - get_codebase_map: Generate a high-level codebase map for agents
  - detect_conventions: Detect coding conventions from existing code
"""

import ast
import os
from typing import Optional

from ..parsing.tree_sitter_parser import (
    detect_language,
    get_parseable_extensions,
    parse_file as ts_parse_file,
)

# In-memory index store: { workspace_path: { filepath: FileIndex } }
_INDEX_CACHE: dict[str, dict[str, dict]] = {}


def _extract_module_name(import_text: str) -> str:
    """Extract the module name from an import statement text.

    'import os'                       -> 'os'
    'from datetime import datetime'   -> 'datetime'
    'import os.path'                  -> 'os.path'
    """
    text = import_text.strip()
    if text.startswith("from "):
        # "from X import Y" -> X
        parts = text.split()
        if len(parts) >= 2:
            return parts[1]
    elif text.startswith("import "):
        parts = text.split()
        if len(parts) >= 2:
            return parts[1].rstrip(",")
    return text


def _convert_ts_result(ts_result: dict) -> dict:
    """Convert tree-sitter parser output to the canonical index format."""
    # Convert imports: ts format is [{type, text, line}] -> [str (module name)]
    imports = [
        _extract_module_name(imp.get("text", ""))
        for imp in ts_result.get("imports", [])
    ]

    # Convert functions: ts format has {name, signature, docstring,
    # line_start, line_end, body_preview, decorators}
    # -> {name, args, line, end_line, is_async, docstring, decorators}
    functions = []
    for fn in ts_result.get("functions", []):
        # Extract args from signature if available (best-effort)
        sig = fn.get("signature", "")
        args: list[str] = []
        if "(" in sig and ")" in sig:
            params_str = sig[sig.index("(") + 1 : sig.rindex(")")]
            if params_str.strip():
                args = [p.strip().split(":")[0].split("=")[0].strip()
                        for p in params_str.split(",") if p.strip()]
        functions.append({
            "name": fn.get("name", ""),
            "args": args,
            "line": fn.get("line_start", 0),
            "end_line": fn.get("line_end", 0),
            "is_async": (fn.get("name", "").startswith("async ") or
                        "async " in fn.get("signature", "") or
                        fn.get("body_preview", "").lstrip().startswith("async ")),
            "docstring": (fn.get("docstring") or "")[:100],
            "decorators": fn.get("decorators", []),
        })

    # Convert classes: ts format has {name, bases, docstring, line_start,
    # line_end, methods: [{name, line}]}
    # -> {name, bases, line, end_line, methods: [{name, args, line, is_async}],
    #     docstring}
    classes = []
    for cls in ts_result.get("classes", []):
        methods = []
        for m in cls.get("methods", []):
            methods.append({
                "name": m.get("name", ""),
                "args": [],
                "line": m.get("line", 0),
                "is_async": False,
            })
        classes.append({
            "name": cls.get("name", ""),
            "bases": cls.get("bases", []),
            "line": cls.get("line_start", 0),
            "end_line": cls.get("line_end", 0),
            "methods": methods,
            "docstring": (cls.get("docstring") or "")[:100],
        })

    return {
        "imports": imports,
        "functions": functions,
        "classes": classes,
        "docstring": (ts_result.get("module_docstring") or "")[:200],
        "line_count": ts_result.get("line_count", 0),
    }


def _parse_file(filepath: str) -> Optional[dict]:
    """Parse a source file and extract structural info.

    Tries the tree-sitter multi-language parser first.  Falls back to the
    Python ``ast`` parser for ``.py`` files when tree-sitter is unavailable
    or when it fails.
    """
    try:
        ts_result = ts_parse_file(filepath)
        if ts_result is not None:
            return _convert_ts_result(ts_result)
    except Exception:
        pass  # fall through to ast fallback

    # Fallback: only .py files can be parsed with the ast module
    if filepath.endswith(".py"):
        return _parse_file_python_ast(filepath)

    return None


def _parse_file_python_ast(filepath: str) -> Optional[dict]:
    """Parse a single Python file using the stdlib ``ast`` module."""
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
    except (OSError, UnicodeDecodeError):
        return None

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return None

    imports = []
    functions = []
    classes = []
    module_docstring = ast.get_docstring(tree)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append(module)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            docstring = ast.get_docstring(node)
            functions.append({
                "name": node.name,
                "args": args,
                "line": node.lineno,
                "end_line": node.end_lineno,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "docstring": (docstring or "")[:100],
                "decorators": [
                    (d.id if isinstance(d, ast.Name) else
                     ast.dump(d) if not isinstance(d, ast.Attribute) else
                     f"{d.value.id if isinstance(d.value, ast.Name) else '?'}.{d.attr}")
                    for d in node.decorator_list
                ],
            })
        elif isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(
                        f"{base.value.id if isinstance(base.value, ast.Name) else '?'}.{base.attr}"
                    )
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_args = [a.arg for a in item.args.args if a.arg != "self" and a.arg != "cls"]
                    methods.append({
                        "name": item.name,
                        "args": m_args,
                        "line": item.lineno,
                        "is_async": isinstance(item, ast.AsyncFunctionDef),
                    })
            classes.append({
                "name": node.name,
                "bases": bases,
                "line": node.lineno,
                "end_line": node.end_lineno,
                "methods": methods,
                "docstring": (ast.get_docstring(node) or "")[:100],
            })

    return {
        "imports": imports,
        "functions": functions,
        "classes": classes,
        "docstring": (module_docstring or "")[:200],
        "line_count": len(source.splitlines()),
    }


def _default_extensions() -> tuple:
    """Return parseable extensions — multi-language if tree-sitter is available."""
    return tuple(get_parseable_extensions())


def build_index(root_path: str, extensions: tuple | None = None) -> dict[str, dict]:
    """
    Scan directory recursively and build a structural index.

    Returns dict of { relative_filepath: file_info }.
    Also caches the result in memory.
    """
    if extensions is None:
        extensions = _default_extensions()
    root = os.path.normpath(root_path)
    index = {}

    skip_dirs = {
        "__pycache__", ".git", "node_modules", ".venv", "venv",
        ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
        ".eggs", "*.egg-info",
    }

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden/cache directories
        dirnames[:] = [
            d for d in dirnames
            if d not in skip_dirs and not d.startswith(".")
        ]

        for fname in filenames:
            if not any(fname.endswith(ext) for ext in extensions):
                continue
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, root)
            # Normalize to forward slashes
            rel_path = rel_path.replace("\\", "/")
            info = _parse_file(full_path)
            if info:
                index[rel_path] = info

    _INDEX_CACHE[root] = index
    return index


def get_cached_index(root_path: str) -> Optional[dict[str, dict]]:
    """Return cached index if available."""
    root = os.path.normpath(root_path)
    return _INDEX_CACHE.get(root)


def clear_index(root_path: str) -> None:
    """Clear cached index for a path."""
    root = os.path.normpath(root_path)
    _INDEX_CACHE.pop(root, None)


def query_index(
    index: dict[str, dict],
    query: str,
    search_type: str = "all",
) -> list[dict]:
    """
    Search the index for functions, classes, or modules matching a query.

    search_type: "all", "function", "class", "import", "file"
    """
    query_lower = query.lower()
    results = []

    for filepath, info in index.items():
        if search_type in ("all", "file"):
            if query_lower in filepath.lower():
                results.append({
                    "type": "file",
                    "filepath": filepath,
                    "line_count": info["line_count"],
                    "docstring": info["docstring"],
                })

        if search_type in ("all", "function"):
            for func in info["functions"]:
                if query_lower in func["name"].lower():
                    results.append({
                        "type": "function",
                        "name": func["name"],
                        "filepath": filepath,
                        "line": func["line"],
                        "args": func["args"],
                        "docstring": func["docstring"],
                    })

            for cls in info["classes"]:
                for method in cls["methods"]:
                    if query_lower in method["name"].lower():
                        results.append({
                            "type": "method",
                            "name": f"{cls['name']}.{method['name']}",
                            "filepath": filepath,
                            "line": method["line"],
                            "args": method["args"],
                        })

        if search_type in ("all", "class"):
            for cls in info["classes"]:
                if query_lower in cls["name"].lower():
                    results.append({
                        "type": "class",
                        "name": cls["name"],
                        "filepath": filepath,
                        "line": cls["line"],
                        "bases": cls["bases"],
                        "methods": [m["name"] for m in cls["methods"]],
                        "docstring": cls["docstring"],
                    })

        if search_type in ("all", "import"):
            for imp in info["imports"]:
                if query_lower in imp.lower():
                    results.append({
                        "type": "import",
                        "module": imp,
                        "filepath": filepath,
                    })

    return results


def get_codebase_map(
    index: dict[str, dict],
    max_depth: int = 3,
) -> str:
    """
    Generate a high-level codebase map from the index.

    Groups files by directory and shows modules, key classes, and functions.
    Suitable for injection into agent context for large codebases.
    """
    if not index:
        return "📁 Empty index — no files found."

    # Group by directory
    dir_groups: dict[str, list[str]] = {}
    for filepath in sorted(index.keys()):
        parts = filepath.replace("\\", "/").split("/")
        if len(parts) > max_depth:
            dir_key = "/".join(parts[:max_depth])
        else:
            dir_key = "/".join(parts[:-1]) if len(parts) > 1 else "."
        dir_groups.setdefault(dir_key, []).append(filepath)

    lines = ["📁 Codebase Map\n"]

    for dir_path, files in sorted(dir_groups.items()):
        lines.append(f"📂 {dir_path}/")
        for fp in files:
            info = index[fp]
            fname = fp.split("/")[-1]
            parts = []
            if info["classes"]:
                cls_names = [c["name"] for c in info["classes"][:3]]
                parts.append(f"classes: {', '.join(cls_names)}")
            if info["functions"]:
                fn_names = [f["name"] for f in info["functions"][:5]]
                parts.append(f"funcs: {', '.join(fn_names)}")
            desc = info["docstring"].split("\n")[0][:60] if info["docstring"] else ""
            summary = f" — {'; '.join(parts)}" if parts else ""
            doc = f" | {desc}" if desc else ""
            lines.append(f"  📄 {fname} ({info['line_count']} lines){summary}{doc}")
        lines.append("")

    return "\n".join(lines)


def detect_conventions(index: dict[str, dict]) -> dict:
    """
    Detect coding conventions from the indexed codebase.

    Returns a dict with detected patterns:
      - naming_style: snake_case, camelCase, etc.
      - has_type_hints: bool
      - has_docstrings: bool
      - async_style: bool (uses async/await)
      - common_imports: list of frequently imported modules
      - avg_function_length: float
    """
    if not index:
        return {"error": "Empty index"}

    all_func_names = []
    all_class_names = []
    has_docstrings = 0
    total_functions = 0
    async_count = 0
    total_lines = 0
    import_counts: dict[str, int] = {}

    for info in index.values():
        for func in info["functions"]:
            all_func_names.append(func["name"])
            total_functions += 1
            if func.get("docstring"):
                has_docstrings += 1
            if func.get("is_async"):
                async_count += 1
            if func.get("end_line") and func.get("line"):
                total_lines += func["end_line"] - func["line"] + 1

        for cls in info["classes"]:
            all_class_names.append(cls["name"])
            for method in cls["methods"]:
                total_functions += 1
                if method.get("is_async"):
                    async_count += 1

        for imp in info["imports"]:
            top = imp.split(".")[0]
            import_counts[top] = import_counts.get(top, 0) + 1

    # Detect naming style
    snake_count = sum(1 for n in all_func_names if "_" in n and n == n.lower())
    camel_count = sum(1 for n in all_func_names if n[0].islower() and any(c.isupper() for c in n)) if all_func_names else 0
    naming_style = "snake_case" if snake_count >= camel_count else "camelCase"

    # Common imports (top 10)
    common_imports = sorted(
        import_counts.items(), key=lambda x: -x[1]
    )[:10]

    avg_func_len = total_lines / max(total_functions, 1)

    return {
        "naming_style": naming_style,
        "has_docstrings": has_docstrings > total_functions * 0.3,
        "docstring_ratio": round(has_docstrings / max(total_functions, 1), 2),
        "async_style": async_count > total_functions * 0.2,
        "async_ratio": round(async_count / max(total_functions, 1), 2),
        "common_imports": [f"{name} ({count})" for name, count in common_imports],
        "avg_function_length": round(avg_func_len, 1),
        "total_files": len(index),
        "total_functions": total_functions,
        "total_classes": len(all_class_names),
    }


async def index_workspace(path: str = ".") -> str:
    """
    Tool-compatible wrapper: index a workspace and return summary.
    """
    from .workspace import WORKSPACE_DIR
    full_path = os.path.join(WORKSPACE_DIR, path) if path != "." else WORKSPACE_DIR
    index = build_index(full_path)

    total_funcs = sum(len(i["functions"]) for i in index.values())
    total_classes = sum(len(i["classes"]) for i in index.values())

    return (
        f"✅ Indexed {len(index)} files: "
        f"{total_funcs} functions, {total_classes} classes\n"
        f"Use query_codebase to search the index."
    )


async def query_codebase(query: str, search_type: str = "all") -> str:
    """
    Tool-compatible wrapper: search the codebase index.
    """
    from .workspace import WORKSPACE_DIR

    index = get_cached_index(WORKSPACE_DIR)
    if not index:
        index = build_index(WORKSPACE_DIR)

    results = query_index(index, query, search_type)

    if not results:
        return f"❌ No results for '{query}' (type: {search_type})"

    lines = [f"🔍 Found {len(results)} result(s) for '{query}':\n"]
    for r in results[:20]:  # Limit to 20 results
        if r["type"] == "file":
            lines.append(f"  📄 {r['filepath']} ({r['line_count']} lines)")
        elif r["type"] == "function":
            lines.append(
                f"  🔧 {r['name']}({', '.join(r['args'])}) "
                f"in {r['filepath']}:{r['line']}"
            )
        elif r["type"] == "method":
            lines.append(
                f"  🔧 {r['name']}({', '.join(r['args'])}) "
                f"in {r['filepath']}:{r['line']}"
            )
        elif r["type"] == "class":
            methods = ", ".join(r["methods"][:5])
            lines.append(
                f"  📦 class {r['name']} in {r['filepath']}:{r['line']} "
                f"[methods: {methods}]"
            )
        elif r["type"] == "import":
            lines.append(f"  📥 {r['module']} imported in {r['filepath']}")

    if len(results) > 20:
        lines.append(f"\n  ... and {len(results) - 20} more results")

    return "\n".join(lines)


async def codebase_map(path: str = ".") -> str:
    """
    Tool-compatible wrapper: generate and return codebase map.
    """
    from .workspace import WORKSPACE_DIR
    full_path = os.path.join(WORKSPACE_DIR, path) if path != "." else WORKSPACE_DIR
    index = get_cached_index(full_path)
    if not index:
        index = build_index(full_path)

    return get_codebase_map(index)
