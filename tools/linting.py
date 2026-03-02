# tools/linting.py
"""
Auto-linting post-step — run ruff check --fix and ruff format on Python files
inside the Docker sandbox. Non-blocking: auto-fixes what it can.
"""

import logging
from tools.shell import run_shell

logger = logging.getLogger(__name__)

# Extensions that can be linted
_LINTABLE = {".py"}


def _is_lintable(filepath: str) -> bool:
    """Check if the file extension is supported for linting."""
    import os
    _, ext = os.path.splitext(filepath)
    return ext.lower() in _LINTABLE


async def auto_lint(filepath: str) -> str:
    """
    Run ruff check + format on a file inside the Docker sandbox.

    Args:
        filepath: Path relative to workspace root.

    Returns:
        Combined lint/format output, or skip message for non-Python files.
    """
    if not _is_lintable(filepath):
        return f"⏭️ Skipping lint for {filepath} (not a Python file)."

    results = []

    # Step 1: ruff check --fix (auto-fix what's fixable)
    try:
        fix_output = await run_shell(
            f"ruff check --fix --quiet {filepath} 2>&1 || true",
            timeout=30,
        )
        if fix_output and "error" not in fix_output.lower():
            results.append(f"🔧 ruff check --fix: {fix_output.strip()}")
        elif fix_output:
            results.append(f"ruff check: {fix_output.strip()}")
    except Exception as e:
        results.append(f"ruff check skipped: {e}")

    # Step 2: ruff format (code formatting)
    try:
        fmt_output = await run_shell(
            f"ruff format --quiet {filepath} 2>&1 || true",
            timeout=30,
        )
        if fmt_output and "error" not in fmt_output.lower():
            results.append(f"✨ ruff format: {fmt_output.strip()}")
        elif fmt_output:
            results.append(f"ruff format: {fmt_output.strip()}")
    except Exception as e:
        results.append(f"ruff format skipped: {e}")

    if not results:
        return f"✅ {filepath}: lint clean."

    return "\n".join(results) or f"✅ {filepath}: lint clean."
