# tools/ast_tools.py
"""
AST-aware code tools for Python.

Provides structural code operations:
  - get_function: Extract a function/method by name
  - replace_function: Replace a function body
  - list_classes: List all classes with their methods
  - list_functions: List all top-level functions
  - get_imports: Extract all import statements
"""

import ast
import os
import textwrap
from typing import Optional

from .workspace import WORKSPACE_DIR


def _workspace_dir() -> str:
    """Return WORKSPACE_DIR at call time (allows test-time patching)."""
    return WORKSPACE_DIR


def _safe_path(filepath: str) -> Optional[str]:
    """Resolve filepath within workspace, block escapes."""
    ws = _workspace_dir()
    full = os.path.normpath(os.path.join(ws, filepath))
    if not full.startswith(os.path.normpath(ws)):
        return None
    return full


def _read_source(filepath: str) -> tuple[Optional[str], Optional[str]]:
    """Read source file, return (content, error)."""
    full = _safe_path(filepath)
    if not full:
        return None, "❌ Path escapes workspace"
    if not os.path.isfile(full):
        return None, f"❌ File not found: {filepath}"
    try:
        with open(full, encoding="utf-8") as f:
            return f.read(), None
    except Exception as e:
        return None, f"❌ Read error: {e}"


def _write_source(filepath: str, content: str) -> Optional[str]:
    """Write source file, return error or None."""
    full = _safe_path(filepath)
    if not full:
        return "❌ Path escapes workspace"
    try:
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return None
    except Exception as e:
        return f"❌ Write error: {e}"


def _get_node_source(source: str, node: ast.AST) -> str:
    """Extract source text for an AST node using line numbers."""
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1  # 0-indexed
    end = node.end_lineno  # already 1-indexed, exclusive
    return "".join(lines[start:end])


async def get_function(filepath: str, function_name: str) -> str:
    """
    Extract a function or method by name from a Python file.

    Returns the full function source code including decorators.
    For methods, use 'ClassName.method_name' format.
    """
    source, err = _read_source(filepath)
    if err:
        return err

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"❌ Syntax error in {filepath}: {e}"

    # Check for ClassName.method format
    if "." in function_name:
        class_name, method_name = function_name.split(".", 1)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == method_name:
                            src = _get_node_source(source, item)
                            return (
                                f"✅ Found `{function_name}` in {filepath} "
                                f"(line {item.lineno}-{item.end_lineno}):\n\n```python\n{src}```"
                            )
                return f"❌ Method `{method_name}` not found in class `{class_name}`"
        return f"❌ Class `{class_name}` not found in {filepath}"

    # Top-level function search
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                src = _get_node_source(source, node)
                return (
                    f"✅ Found `{function_name}` in {filepath} "
                    f"(line {node.lineno}-{node.end_lineno}):\n\n```python\n{src}```"
                )

    # Also search inside classes
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == function_name:
                        src = _get_node_source(source, item)
                        return (
                            f"✅ Found `{node.name}.{function_name}` in {filepath} "
                            f"(line {item.lineno}-{item.end_lineno}):\n\n```python\n{src}```"
                        )

    return f"❌ Function `{function_name}` not found in {filepath}"


async def replace_function(
    filepath: str, function_name: str, new_code: str
) -> str:
    """
    Replace an entire function/method body with new code.

    The new_code should be the complete function definition including
    the def line, decorators, and body.
    For methods, use 'ClassName.method_name' format.
    """
    source, err = _read_source(filepath)
    if err:
        return err

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"❌ Syntax error in {filepath}: {e}"

    lines = source.splitlines(keepends=True)
    target_node = None

    if "." in function_name:
        class_name, method_name = function_name.split(".", 1)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == method_name:
                            target_node = item
                            break
                break
    else:
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name:
                    target_node = node
                    break

    if target_node is None:
        return f"❌ Function `{function_name}` not found in {filepath}"

    start = target_node.lineno - 1  # 0-indexed
    end = target_node.end_lineno  # 1-indexed, exclusive

    # Ensure new_code ends with newline
    if not new_code.endswith("\n"):
        new_code += "\n"

    new_lines = lines[:start] + [new_code] + lines[end:]
    new_source = "".join(new_lines)

    # Validate the result parses
    try:
        ast.parse(new_source)
    except SyntaxError as e:
        return f"❌ Replacement produces invalid syntax: {e}"

    err = _write_source(filepath, new_source)
    if err:
        return err

    return (
        f"✅ Replaced `{function_name}` in {filepath} "
        f"(was lines {start + 1}-{end}, now updated)"
    )


async def list_classes(filepath: str) -> str:
    """
    List all classes in a Python file with their methods and base classes.
    """
    source, err = _read_source(filepath)
    if err:
        return err

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"❌ Syntax error in {filepath}: {e}"

    classes = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(ast.dump(base))
                else:
                    bases.append("?")
            methods = []
            attrs = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    prefix = "async " if isinstance(item, ast.AsyncFunctionDef) else ""
                    args = [
                        a.arg for a in item.args.args
                        if a.arg != "self" and a.arg != "cls"
                    ]
                    methods.append(
                        f"  {'async ' if prefix else ''}def {item.name}({', '.join(args)}) "
                        f"[line {item.lineno}]"
                    )
                elif isinstance(item, ast.Assign):
                    for t in item.targets:
                        if isinstance(t, ast.Name):
                            attrs.append(t.id)

            base_str = f"({', '.join(bases)})" if bases else ""
            classes.append({
                "name": node.name,
                "bases": base_str,
                "line": node.lineno,
                "methods": methods,
                "attrs": attrs,
            })

    if not classes:
        return f"📄 No classes found in {filepath}"

    lines = [f"📄 Classes in {filepath}:\n"]
    for cls in classes:
        lines.append(f"class {cls['name']}{cls['bases']} [line {cls['line']}]")
        if cls["attrs"]:
            lines.append(f"  attrs: {', '.join(cls['attrs'])}")
        for m in cls["methods"]:
            lines.append(m)
        lines.append("")

    return "\n".join(lines)


async def list_functions(filepath: str) -> str:
    """
    List all top-level functions in a Python file.
    """
    source, err = _read_source(filepath)
    if err:
        return err

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"❌ Syntax error in {filepath}: {e}"

    funcs = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            args = [a.arg for a in node.args.args]
            funcs.append(
                f"  {prefix}def {node.name}({', '.join(args)}) "
                f"[line {node.lineno}-{node.end_lineno}]"
            )

    if not funcs:
        return f"📄 No top-level functions in {filepath}"

    return f"📄 Functions in {filepath}:\n" + "\n".join(funcs)


async def get_imports(filepath: str) -> str:
    """
    Extract all import statements from a Python file.
    """
    source, err = _read_source(filepath)
    if err:
        return err

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"❌ Syntax error in {filepath}: {e}"

    imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ", ".join(
                a.name + (f" as {a.asname}" if a.asname else "")
                for a in node.names
            )
            imports.append(f"from {module} import {names}")

    if not imports:
        return f"📄 No imports in {filepath}"

    return f"📄 Imports in {filepath}:\n" + "\n".join(f"  {i}" for i in imports)
