# tools/apply_diff.py
"""
Phase 12.5 — Diff-First Editing

Apply unified diff format patches to files.  Uses Python's difflib
for platform-independent patching (no external `patch` command needed).

Public API:
    result = await apply_diff(filepath, unified_diff)

After applying, validates syntax with tree-sitter (if available)
and runs auto_lint on Python files.
"""
import os
import re
from typing import Optional

from src.infra.logging_config import get_logger
from .linting import auto_lint
from .workspace import _safe_resolve

logger = get_logger("tools.apply_diff")


def _parse_unified_diff(diff_text: str) -> list[dict]:
    """
    Parse a unified diff into a list of hunks.

    Each hunk is a dict:
      {
        "old_start": int,    # 1-indexed start line in original
        "old_count": int,    # number of lines in original
        "new_start": int,    # 1-indexed start line in new
        "new_count": int,    # number of lines in new
        "lines": list[str],  # diff lines (prefixed with +, -, or space)
      }
    """
    hunks = []
    current_hunk = None

    # Also accept diff without file headers (just @@ hunks)
    for line in diff_text.splitlines():
        # Hunk header
        match = re.match(
            r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',
            line,
        )
        if match:
            if current_hunk:
                hunks.append(current_hunk)
            current_hunk = {
                "old_start": int(match.group(1)),
                "old_count": int(match.group(2) or "1"),
                "new_start": int(match.group(3)),
                "new_count": int(match.group(4) or "1"),
                "lines": [],
            }
            continue

        # Skip file headers
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("diff "):
            continue

        # Hunk content
        if current_hunk is not None:
            if line.startswith("+") or line.startswith("-") or line.startswith(" "):
                current_hunk["lines"].append(line)
            elif line == "":
                # Treat empty lines in diff as context (space prefix)
                current_hunk["lines"].append(" ")

    if current_hunk:
        hunks.append(current_hunk)

    return hunks


def _apply_hunks(original: str, hunks: list[dict]) -> tuple[Optional[str], str]:
    """
    Apply parsed hunks to original text.

    Returns (new_text, error_message).
    If error_message is non-empty, new_text is None.
    """
    if not hunks:
        return None, "No hunks found in diff"

    orig_lines = original.splitlines(keepends=True)

    # Apply hunks in reverse order (so line numbers don't shift)
    sorted_hunks = sorted(hunks, key=lambda h: -h["old_start"])

    result_lines = list(orig_lines)

    for hunk in sorted_hunks:
        old_start = hunk["old_start"] - 1  # Convert to 0-indexed
        old_count = hunk["old_count"]

        # Build the new lines for this hunk
        new_lines: list[str] = []
        removed_lines: list[str] = []

        for diff_line in hunk["lines"]:
            if diff_line.startswith("+"):
                content = diff_line[1:]
                if not content.endswith("\n"):
                    content += "\n"
                new_lines.append(content)
            elif diff_line.startswith("-"):
                removed_lines.append(diff_line[1:])
            elif diff_line.startswith(" "):
                content = diff_line[1:]
                if not content.endswith("\n"):
                    content += "\n"
                new_lines.append(content)

        # Verify context (first few lines of the hunk should match)
        # This catches off-by-one or wrong-file errors
        if old_start < len(result_lines):
            context_ok = True
            check_idx = 0
            for diff_line in hunk["lines"]:
                if diff_line.startswith(" ") or diff_line.startswith("-"):
                    expected = diff_line[1:]
                    actual_idx = old_start + check_idx
                    if actual_idx < len(result_lines):
                        actual = result_lines[actual_idx].rstrip("\n")
                        if actual != expected.rstrip("\n"):
                            # Fuzzy match: strip whitespace
                            if actual.strip() != expected.strip():
                                context_ok = False
                                break
                    check_idx += 1

            if not context_ok:
                return None, (
                    f"Context mismatch at line {old_start + 1}. "
                    f"The diff doesn't match the current file content."
                )

        # Replace the old lines with new lines
        result_lines[old_start:old_start + old_count] = new_lines

    return "".join(result_lines), ""


async def apply_diff(filepath: str, unified_diff: str) -> str:
    """
    Apply a unified diff to a file.

    The diff should be in standard unified diff format:
        @@ -start,count +start,count @@
        -removed line
        +added line
         context line

    After applying, validates syntax (if tree-sitter available)
    and runs auto_lint on Python files.

    Args:
        filepath:     Path to the file (relative to workspace).
        unified_diff: The unified diff text.

    Returns:
        Success/error message string.
    """
    full_path = _safe_resolve(filepath)
    if not full_path:
        return "❌ Path escapes workspace"

    if not os.path.isfile(full_path):
        return f"❌ File not found: {filepath}"

    # Read original file
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            original = f.read()
    except Exception as e:
        return f"❌ Read error: {e}"

    # Parse the diff
    hunks = _parse_unified_diff(unified_diff)
    if not hunks:
        return "❌ No valid diff hunks found. Use unified diff format (@@ -N,M +N,M @@)"

    # Apply the diff
    new_content, error = _apply_hunks(original, hunks)
    if error:
        return f"❌ Diff apply failed: {error}"

    # Validate syntax post-edit
    from ..parsing.tree_sitter_parser import detect_language, validate_syntax
    language = detect_language(filepath)
    is_valid, syntax_error = validate_syntax(new_content, language)

    if not is_valid:
        return (
            f"❌ Diff would produce invalid {language} syntax: {syntax_error}\n"
            f"The diff was NOT applied. Fix the diff and try again."
        )

    # Write the result
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        return f"❌ Write error: {e}"

    # Auto-lint Python files
    lint_result = ""
    if language == "python":
        try:
            lint_out = await auto_lint(filepath)
            if lint_out and "error" in lint_out.lower():
                lint_result = f"\nLint: {lint_out}"
        except Exception:
            pass

    old_lines = original.count("\n") + 1
    new_lines = new_content.count("\n") + 1
    diff_summary = f"+{new_lines - old_lines}" if new_lines >= old_lines else str(new_lines - old_lines)

    return (
        f"✅ Applied diff to {filepath} "
        f"({len(hunks)} hunk(s), {old_lines}→{new_lines} lines, {diff_summary})"
        f"{lint_result}"
    )
