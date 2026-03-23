# tools/linting.py
"""
Auto-linting post-step — delegates to language-specific toolkits (Phase 10.1).
Runs lint + format commands inside the Docker sandbox. Non-blocking: auto-fixes
what it can.
"""
import os

from src.infra.logging_config import get_logger
from .shell import run_shell

logger = get_logger("tools.linting")

# Extension → language mapping for toolkit dispatch
_EXT_LANG = {
    ".py": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
}


def _detect_lang(filepath: str) -> str | None:
    """Detect language from file extension."""
    _, ext = os.path.splitext(filepath)
    return _EXT_LANG.get(ext.lower())


async def auto_lint(filepath: str) -> str:
    """
    Run language-appropriate lint + format on a file.

    Delegates to the matching LanguageToolkit. Falls back to ruff for Python
    if the toolkit import fails.

    Args:
        filepath: Path relative to workspace root.

    Returns:
        Combined lint/format output, or skip message for unsupported files.
    """
    lang = _detect_lang(filepath)
    if not lang:
        return f"⏭️ Skipping lint for {filepath} (unsupported extension)."

    # Try language toolkit first
    try:
        from src.languages import get_toolkit
        toolkit = get_toolkit(lang)
        if toolkit:
            lint_cmd = toolkit.lint_command(filepath)
            if lint_cmd:
                output = await run_shell(
                    f"{lint_cmd} 2>&1 || true",
                    timeout=60,
                )
                if output and output.strip():
                    return f"🔧 {lang} lint: {output.strip()}"
                return f"✅ {filepath}: lint clean."
    except Exception as e:
        logger.debug(f"Toolkit lint failed for {filepath}, falling back: {e}")

    # Fallback: Python-specific ruff commands
    if lang == "python":
        return await _lint_python_fallback(filepath)

    return f"⏭️ Skipping lint for {filepath} ({lang} toolkit unavailable)."


async def _lint_python_fallback(filepath: str) -> str:
    """Fallback Python linting with ruff."""
    results = []

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

    return "\n".join(results) if results else f"✅ {filepath}: lint clean."
