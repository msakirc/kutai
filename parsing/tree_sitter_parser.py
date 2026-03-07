# parsing/tree_sitter_parser.py
"""
Phase 12.1 — Tree-sitter Multi-Language Parsing

Unified parsing interface for Python, JavaScript, TypeScript, Go,
Rust, Java, C, and C++.

Tries three parsing strategies in order:
  1. tree-sitter (if installed + language grammar available)
  2. Python ast module (for .py files only)
  3. Regex-based fallback (works for any C-family language)

Public API:
    result = parse_file(filepath)
    result = parse_source(source, language)
    lang   = detect_language(filepath)
    ok     = tree_sitter_available()
"""
import ast as python_ast
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Language Detection ──────────────────────────────────────────────────────

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
}

# Languages supported by each parsing backend
TREE_SITTER_LANGUAGES = {
    "python", "javascript", "typescript", "go", "rust", "java", "c", "cpp",
}


def detect_language(filepath: str) -> str:
    """Detect programming language from file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    return EXTENSION_MAP.get(ext, "unknown")


# ─── Tree-sitter Backend ────────────────────────────────────────────────────

_ts_parsers: dict[str, object] = {}
_ts_available: Optional[bool] = None

# Map language name to tree-sitter-<lang> package
_TS_PACKAGES = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
    "java": "tree_sitter_java",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
}


def tree_sitter_available() -> bool:
    """Check if tree-sitter base package is installed."""
    global _ts_available
    if _ts_available is None:
        try:
            import tree_sitter  # noqa: F401
            _ts_available = True
        except ImportError:
            _ts_available = False
    return _ts_available


def _get_ts_parser(language: str):
    """Get or create a tree-sitter parser for a language."""
    if language in _ts_parsers:
        return _ts_parsers[language]

    if not tree_sitter_available():
        return None

    pkg_name = _TS_PACKAGES.get(language)
    if not pkg_name:
        return None

    try:
        import importlib
        from tree_sitter import Language, Parser

        lang_module = importlib.import_module(pkg_name)

        # Handle TypeScript special case (has typescript() and tsx())
        if language == "typescript" and hasattr(lang_module, "language_typescript"):
            lang_obj = Language(lang_module.language_typescript())
        elif hasattr(lang_module, "language"):
            lang_obj = Language(lang_module.language())
        else:
            logger.debug(f"tree-sitter {pkg_name} has no language() function")
            return None

        parser = Parser(lang_obj)
        _ts_parsers[language] = parser
        logger.debug(f"tree-sitter parser loaded for {language}")
        return parser

    except ImportError:
        logger.debug(f"tree-sitter grammar not installed: {pkg_name}")
        return None
    except Exception as e:
        logger.debug(f"tree-sitter {language} init failed: {e}")
        return None


def _parse_ts(source: str, language: str) -> Optional[dict]:
    """Parse source using tree-sitter, extract structural info."""
    parser = _get_ts_parser(language)
    if parser is None:
        return None

    try:
        tree = parser.parse(source.encode("utf-8"))
    except Exception as e:
        logger.debug(f"tree-sitter parse failed: {e}")
        return None

    root = tree.root_node
    functions = []
    classes = []
    imports = []
    exports = []

    def _node_text(node) -> str:
        return source[node.start_byte:node.end_byte]

    def _extract_children(node, depth=0):
        """Walk tree and extract structural elements."""
        if depth > 10:
            return

        ntype = node.type

        # ── Functions ──
        if ntype in (
            "function_definition",       # Python, Go
            "function_declaration",      # JS, TS, C, C++, Java
            "method_definition",         # JS, TS
            "function_item",             # Rust
            "method_declaration",        # Java, Go
            "arrow_function",            # JS, TS (only named via assignment)
        ):
            name = ""
            signature = ""
            docstring = ""
            decorators = []
            body_preview = ""

            for child in node.children:
                if child.type in ("identifier", "name", "property_identifier"):
                    name = _node_text(child)
                elif child.type in ("parameters", "formal_parameters",
                                     "parameter_list", "type_parameters"):
                    signature = _node_text(child)
                elif child.type == "decorator":
                    decorators.append(_node_text(child).strip())

            # Get body preview (first 5 lines)
            body_lines = source[node.start_byte:node.end_byte].split("\n")
            body_preview = "\n".join(body_lines[:5])

            # Extract docstring (first string in body for Python)
            if language == "python":
                for child in node.children:
                    if child.type == "block":
                        for stmt in child.children:
                            if stmt.type == "expression_statement":
                                for expr in stmt.children:
                                    if expr.type == "string":
                                        docstring = _node_text(expr).strip("'\"")[:200]
                                break

            if name:
                functions.append({
                    "name": name,
                    "signature": signature,
                    "docstring": docstring[:200],
                    "line_start": node.start_point[0] + 1,
                    "line_end": node.end_point[0] + 1,
                    "body_preview": body_preview[:500],
                    "decorators": decorators,
                })

        # ── Classes / Structs ──
        elif ntype in (
            "class_definition",          # Python
            "class_declaration",         # JS, TS, Java
            "struct_item",               # Rust
            "struct_specifier",          # C, C++
            "impl_item",                 # Rust (impl block)
            "interface_declaration",     # TS, Java
        ):
            name = ""
            bases = []
            docstring = ""
            methods = []

            for child in node.children:
                if child.type in ("identifier", "name", "type_identifier"):
                    if not name:
                        name = _node_text(child)
                elif child.type in ("argument_list", "superclass",
                                     "extends_clause", "implements_clause"):
                    bases.append(_node_text(child).strip("()"))
                elif child.type in ("class_body", "body", "block",
                                     "declaration_list", "field_declaration_list"):
                    # Extract methods from class body
                    for item in child.children:
                        if item.type in (
                            "function_definition", "method_definition",
                            "method_declaration", "function_item",
                            "function_declaration",
                        ):
                            mname = ""
                            for mc in item.children:
                                if mc.type in ("identifier", "name",
                                               "property_identifier"):
                                    mname = _node_text(mc)
                                    break
                            if mname:
                                methods.append({
                                    "name": mname,
                                    "line": item.start_point[0] + 1,
                                })

            if name:
                classes.append({
                    "name": name,
                    "bases": bases,
                    "docstring": docstring[:200],
                    "line_start": node.start_point[0] + 1,
                    "line_end": node.end_point[0] + 1,
                    "methods": methods,
                })

        # ── Imports ──
        elif ntype in (
            "import_statement",          # Python
            "import_from_statement",     # Python
            "import_declaration",        # Java, Go
            "import_statement",          # Various
            "use_declaration",           # Rust
            "include_directive",         # C, C++
        ):
            imports.append({
                "type": ntype,
                "text": _node_text(node).strip(),
                "line": node.start_point[0] + 1,
            })

        # ── Exports (JS/TS) ──
        elif ntype in ("export_statement", "export_declaration"):
            exports.append({
                "name": _node_text(node)[:100],
                "line": node.start_point[0] + 1,
            })

        # Recurse into children (but not into function/class bodies
        # since we handle them above)
        if ntype not in (
            "function_definition", "function_declaration",
            "method_definition", "function_item",
        ):
            for child in node.children:
                _extract_children(child, depth + 1)

    # Also check for require() calls in JS
    if language in ("javascript", "typescript"):
        for match in re.finditer(
            r'(?:const|let|var)\s+\w+\s*=\s*require\s*\(["\']([^"\']+)["\']\)',
            source,
        ):
            line = source[:match.start()].count("\n") + 1
            imports.append({
                "type": "require",
                "text": match.group(0),
                "line": line,
            })

    _extract_children(root)

    return {
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "exports": exports,
        "line_count": source.count("\n") + 1,
    }


# ─── Python AST Backend ─────────────────────────────────────────────────────

def _parse_python_ast(source: str) -> Optional[dict]:
    """Parse Python using the built-in ast module (high-quality fallback)."""
    try:
        tree = python_ast.parse(source)
    except SyntaxError:
        return None

    functions = []
    classes = []
    imports = []
    exports = []
    module_docstring = python_ast.get_docstring(tree) or ""
    lines = source.splitlines()

    for node in python_ast.iter_child_nodes(tree):
        if isinstance(node, (python_ast.FunctionDef, python_ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            docstring = python_ast.get_docstring(node) or ""
            body_preview = "\n".join(
                lines[node.lineno - 1:node.lineno + 4]
            )
            decorators = []
            for d in node.decorator_list:
                if isinstance(d, python_ast.Name):
                    decorators.append(f"@{d.id}")
                elif isinstance(d, python_ast.Attribute):
                    decorators.append(f"@{python_ast.dump(d)}")

            functions.append({
                "name": node.name,
                "signature": f"({', '.join(args)})",
                "docstring": docstring[:200],
                "line_start": node.lineno,
                "line_end": node.end_lineno or node.lineno,
                "body_preview": body_preview[:500],
                "decorators": decorators,
            })

        elif isinstance(node, python_ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, python_ast.Name):
                    bases.append(base.id)
                elif isinstance(base, python_ast.Attribute):
                    bases.append(
                        f"{base.value.id if isinstance(base.value, python_ast.Name) else '?'}.{base.attr}"
                    )

            methods = []
            for item in node.body:
                if isinstance(item, (python_ast.FunctionDef, python_ast.AsyncFunctionDef)):
                    methods.append({
                        "name": item.name,
                        "line": item.lineno,
                    })

            docstring = python_ast.get_docstring(node) or ""

            classes.append({
                "name": node.name,
                "bases": bases,
                "docstring": docstring[:200],
                "line_start": node.lineno,
                "line_end": node.end_lineno or node.lineno,
                "methods": methods,
            })

        elif isinstance(node, python_ast.Import):
            for alias in node.names:
                imports.append({
                    "type": "import",
                    "text": f"import {alias.name}",
                    "line": node.lineno,
                })
        elif isinstance(node, python_ast.ImportFrom):
            module = node.module or ""
            names = ", ".join(a.name for a in node.names)
            imports.append({
                "type": "from_import",
                "text": f"from {module} import {names}",
                "line": node.lineno,
            })

    return {
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "exports": exports,
        "line_count": len(lines),
        "module_docstring": module_docstring[:200],
    }


# ─── Regex Fallback ──────────────────────────────────────────────────────────

# Patterns for C-family languages (JS, TS, Java, Go, C, C++, Rust)
_FUNC_PATTERNS = {
    "javascript": re.compile(
        r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)',
        re.MULTILINE,
    ),
    "typescript": re.compile(
        r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)',
        re.MULTILINE,
    ),
    "go": re.compile(
        r'^func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(([^)]*)\)',
        re.MULTILINE,
    ),
    "rust": re.compile(
        r'^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)',
        re.MULTILINE,
    ),
    "java": re.compile(
        r'^\s*(?:public|private|protected|static|final|abstract|synchronized|native)*\s*'
        r'(?:\w+(?:<[^>]*>)?)\s+(\w+)\s*\(([^)]*)\)',
        re.MULTILINE,
    ),
    "c": re.compile(
        r'^(?:\w+\s+)*(\w+)\s*\(([^)]*)\)\s*\{',
        re.MULTILINE,
    ),
    "cpp": re.compile(
        r'^(?:\w+\s+)*(\w+)\s*(?:::\w+)?\s*\(([^)]*)\)\s*(?:const\s*)?\{',
        re.MULTILINE,
    ),
}

_CLASS_PATTERNS = {
    "javascript": re.compile(
        r'^(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?',
        re.MULTILINE,
    ),
    "typescript": re.compile(
        r'^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:<[^>]*>)?'
        r'(?:\s+extends\s+(\w+))?(?:\s+implements\s+(\w+))?',
        re.MULTILINE,
    ),
    "java": re.compile(
        r'^\s*(?:public|private|protected|abstract|final)?\s*class\s+(\w+)'
        r'(?:\s+extends\s+(\w+))?(?:\s+implements\s+[\w,\s]+)?',
        re.MULTILINE,
    ),
    "rust": re.compile(
        r'^(?:pub\s+)?struct\s+(\w+)',
        re.MULTILINE,
    ),
    "cpp": re.compile(
        r'^(?:class|struct)\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+(\w+))?',
        re.MULTILINE,
    ),
}

_IMPORT_PATTERNS = {
    "javascript": re.compile(
        r'^(?:import\s+.*?from\s+["\']([^"\']+)["\']|'
        r'(?:const|let|var)\s+\w+\s*=\s*require\s*\(["\']([^"\']+)["\']\))',
        re.MULTILINE,
    ),
    "typescript": re.compile(
        r'^import\s+.*?from\s+["\']([^"\']+)["\']',
        re.MULTILINE,
    ),
    "go": re.compile(
        r'^\s*(?:import\s+)?["\']([^"\']+)["\']',
        re.MULTILINE,
    ),
    "rust": re.compile(
        r'^use\s+([\w:]+)',
        re.MULTILINE,
    ),
    "java": re.compile(
        r'^import\s+([\w.]+(?:\.\*)?)\s*;',
        re.MULTILINE,
    ),
    "c": re.compile(
        r'^#include\s+[<"]([^>"]+)[>"]',
        re.MULTILINE,
    ),
    "cpp": re.compile(
        r'^#include\s+[<"]([^>"]+)[>"]',
        re.MULTILINE,
    ),
}


def _parse_regex(source: str, language: str) -> Optional[dict]:
    """Fallback regex-based parser for any C-family language."""
    lines = source.splitlines()
    functions = []
    classes = []
    imports = []
    exports = []

    # Extract functions
    func_pat = _FUNC_PATTERNS.get(language)
    if func_pat:
        for m in func_pat.finditer(source):
            name = m.group(1)
            sig = m.group(2) if m.lastindex >= 2 else ""
            line_start = source[:m.start()].count("\n") + 1
            # Estimate end line (find matching brace or next function)
            line_end = min(line_start + 20, len(lines))
            body_preview = "\n".join(lines[line_start - 1:line_start + 4])

            functions.append({
                "name": name,
                "signature": f"({sig})",
                "docstring": "",
                "line_start": line_start,
                "line_end": line_end,
                "body_preview": body_preview[:500],
                "decorators": [],
            })

    # Extract classes
    class_pat = _CLASS_PATTERNS.get(language)
    if class_pat:
        for m in class_pat.finditer(source):
            name = m.group(1)
            base = m.group(2) if m.lastindex >= 2 and m.group(2) else ""
            line_start = source[:m.start()].count("\n") + 1
            line_end = min(line_start + 50, len(lines))

            classes.append({
                "name": name,
                "bases": [base] if base else [],
                "docstring": "",
                "line_start": line_start,
                "line_end": line_end,
                "methods": [],
            })

    # Extract imports
    import_pat = _IMPORT_PATTERNS.get(language)
    if import_pat:
        for m in import_pat.finditer(source):
            module = m.group(1) or (m.group(2) if m.lastindex >= 2 else "")
            if module:
                line = source[:m.start()].count("\n") + 1
                imports.append({
                    "type": "import",
                    "text": m.group(0).strip(),
                    "line": line,
                })

    return {
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "exports": exports,
        "line_count": len(lines),
    }


# ─── Public API ──────────────────────────────────────────────────────────────

def parse_source(source: str, language: str) -> dict:
    """
    Parse source code and extract structural information.

    Tries in order:
      1. tree-sitter (if available for this language)
      2. Python ast (if language is python)
      3. Regex fallback

    Args:
        source:   Source code as a string.
        language: Language identifier (e.g., "python", "javascript").

    Returns:
        Dict with keys: functions, classes, imports, exports, line_count.
    """
    if not source or not source.strip():
        return {
            "functions": [], "classes": [], "imports": [],
            "exports": [], "line_count": 0,
        }

    # Strategy 1: tree-sitter
    if language in TREE_SITTER_LANGUAGES:
        result = _parse_ts(source, language)
        if result:
            result["_parser"] = "tree_sitter"
            return result

    # Strategy 2: Python ast (high-quality fallback for .py)
    if language == "python":
        result = _parse_python_ast(source)
        if result:
            result["_parser"] = "python_ast"
            return result

    # Strategy 3: Regex fallback
    result = _parse_regex(source, language)
    if result:
        result["_parser"] = "regex"
        return result

    # Nothing worked
    return {
        "functions": [], "classes": [], "imports": [],
        "exports": [], "line_count": source.count("\n") + 1,
        "_parser": "none",
    }


def parse_file(filepath: str) -> dict:
    """
    Parse a source file and extract structural information.

    Auto-detects language from file extension.

    Args:
        filepath: Path to the source file.

    Returns:
        Dict with keys: filepath, language, functions, classes,
        imports, exports, line_count, _parser.
    """
    language = detect_language(filepath)

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except OSError as e:
        logger.debug(f"Cannot read {filepath}: {e}")
        return {
            "filepath": filepath,
            "language": language,
            "functions": [], "classes": [], "imports": [],
            "exports": [], "line_count": 0,
            "_parser": "error",
        }

    result = parse_source(source, language)
    result["filepath"] = filepath
    result["language"] = language
    return result


def validate_syntax(source: str, language: str) -> tuple[bool, str]:
    """
    Check if source code is syntactically valid.

    Args:
        source:   Source code string.
        language: Language identifier.

    Returns:
        (is_valid, error_message)
    """
    # Python: use ast.parse for definitive check
    if language == "python":
        try:
            python_ast.parse(source)
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"

    # tree-sitter: check for ERROR nodes
    if tree_sitter_available():
        parser = _get_ts_parser(language)
        if parser:
            try:
                tree = parser.parse(source.encode("utf-8"))
                root = tree.root_node
                if root.has_error:
                    # Find first error node
                    def _find_error(node):
                        if node.type == "ERROR":
                            return node
                        for child in node.children:
                            err = _find_error(child)
                            if err:
                                return err
                        return None

                    err = _find_error(root)
                    if err:
                        return False, (
                            f"Syntax error at line {err.start_point[0] + 1}, "
                            f"col {err.start_point[1]}"
                        )
                    return False, "Syntax error detected"
                return True, ""
            except Exception as e:
                return False, f"Parse error: {e}"

    # No validation possible — assume valid
    return True, ""


# ─── Supported Languages List ───────────────────────────────────────────────

def get_supported_languages() -> list[str]:
    """Return list of languages we can parse (at least via regex)."""
    return sorted(set(EXTENSION_MAP.values()) - {"unknown"})


def get_parseable_extensions() -> tuple[str, ...]:
    """Return tuple of file extensions we can parse."""
    return tuple(EXTENSION_MAP.keys())
