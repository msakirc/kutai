# tools/patch_file.py
"""
Search-and-replace file editing tool.

Models are far better at specifying text blocks than line numbers.
This tool replaces the line-number-based edit_file with a more
reliable search-and-replace approach.
"""

import os

from src.infra.logging_config import get_logger
from .workspace import _safe_resolve

logger = get_logger("tools.patch_file")

async def patch_file(
    filepath: str,
    search_block: str,
    replace_block: str,
) -> str:
    """
    Find *search_block* in a workspace file and replace it with
    *replace_block*.

    Args:
        filepath:      Path relative to workspace root.
        search_block:  Exact text to find (multi-line OK).
        replace_block: Text to replace it with (may be empty for deletion).

    Returns:
        Confirmation or error message.
    """
    full_path = _safe_resolve(filepath)
    if full_path is None:
        return "❌ Access denied: path is outside workspace."

    if not os.path.isfile(full_path):
        return f"❌ File not found: {filepath}"

    if not search_block:
        return "❌ search_block cannot be empty."

    try:
        with open(full_path, "r", errors="replace") as f:
            content = f.read()

        # Count occurrences
        count = content.count(search_block)
        if count == 0:
            # Try with normalized line endings
            normalized = content.replace("\r\n", "\n")
            search_normalized = search_block.replace("\r\n", "\n")
            count = normalized.count(search_normalized)
            if count == 0:
                # Show nearby context to help debug
                lines = content.split("\n")
                search_first_line = search_block.strip().split("\n")[0].strip()
                candidates = [
                    (i + 1, line.strip())
                    for i, line in enumerate(lines)
                    if search_first_line[:30] in line
                ]
                hint = ""
                if candidates:
                    hints = [f"  Line {n}: {l[:80]}" for n, l in candidates[:3]]
                    hint = (
                        f"\n\nSimilar lines found:\n" + "\n".join(hints) +
                        "\n\nCheck whitespace and exact text."
                    )
                return f"❌ search_block not found in {filepath}.{hint}"
            content = normalized
            search_block = search_normalized

        if count > 1:
            return (
                f"❌ search_block found {count} times in {filepath}. "
                f"Provide more context to make the match unique."
            )

        # Perform replacement
        new_content = content.replace(search_block, replace_block, 1)

        with open(full_path, "w", newline="") as f:
            f.write(new_content)

        old_lines = search_block.count("\n") + 1
        new_lines = replace_block.count("\n") + 1 if replace_block else 0
        total_lines = new_content.count("\n") + 1

        res = (
            f"✅ Patched {filepath}: replaced {old_lines} lines "
            f"with {new_lines} lines. File now has {total_lines} lines."
        )

        # Auto-lint Python files
        if full_path.endswith(".py"):
            from .linting import auto_lint
            lint_res = await auto_lint(filepath)
            res += f"\n\n--- Auto-Linting ---\n{lint_res}"

        return res

    except Exception as exc:
        logger.error(f"Error patching {filepath}: {exc}", exc_info=True)
        return f"❌ Error patching {filepath}: {exc}"
