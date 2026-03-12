# tools/edit_file.py
"""
Diff-based file editing — replace a line range instead of rewriting the whole file.
Saves tokens on large files compared to write_file.
"""

import os
import logging
from typing import Optional

from src.tools.workspace import _safe_resolve

logger = logging.getLogger(__name__)

# Re-use workspace safety from workspace.py


async def edit_file(
    filepath: str,
    start_line: int,
    end_line: int,
    new_content: str,
) -> str:
    """
    Replace lines [start_line, end_line] (1-indexed, inclusive) in a
    workspace file with *new_content*.

    Args:
        filepath:    Path relative to workspace root.
        start_line:  First line to replace (1-indexed).
        end_line:    Last line to replace (1-indexed, inclusive).
        new_content: Replacement text (may contain newlines).

    Returns:
        Confirmation with before/after line count, or error message.
    """
    full_path = _safe_resolve(filepath)
    if full_path is None:
        return "❌ Access denied: path is outside workspace."

    if not os.path.isfile(full_path):
        return f"❌ File not found: {filepath}"

    # Validate line numbers
    if start_line < 1:
        return "❌ start_line must be >= 1."
    if end_line < start_line:
        return "❌ end_line must be >= start_line."

    try:
        with open(full_path, "r", errors="replace") as f:
            lines = f.readlines()

        total = len(lines)

        if start_line > total:
            return (
                f"❌ start_line ({start_line}) exceeds file length "
                f"({total} lines). Use write_file to append instead."
            )

        # Clamp end_line to file length
        actual_end = min(end_line, total)

        # Build new file content
        before = lines[: start_line - 1]          # lines before edit
        after = lines[actual_end:]                 # lines after edit

        # Ensure new_content ends with newline
        replacement = new_content
        if replacement and not replacement.endswith("\n"):
            replacement += "\n"

        new_lines = before + [replacement] + after

        with open(full_path, "w") as f:
            f.writelines(new_lines)

        removed = actual_end - start_line + 1
        added = replacement.count("\n")
        new_total = len(before) + added + len(after)

        res = (
            f"✅ Edited {filepath}: replaced lines {start_line}-{actual_end} "
            f"({removed} lines removed, ~{added} lines added). "
            f"File now has {new_total} lines."
        )

        # Auto-lint Python files
        if full_path.endswith(".py"):
            from linting import auto_lint
            lint_res = await auto_lint(filepath)
            res += f"\n\n--- Auto-Linting ---\n{lint_res}"
            
        return res

    except Exception as exc:
        logger.error(f"Error editing {filepath}: {exc}", exc_info=True)
        return f"❌ Error editing {filepath}: {exc}"
