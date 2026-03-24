# pipeline_utils.py
"""
Pure utility functions for the coding pipeline.

Separated so tests can import them without triggering the
agents → router → litellm dependency chain.
"""

import json
import os
from typing import Optional

from src.infra.logging_config import get_logger
import src.tools.workspace as _ws
from src.tools import git_diff
from src.tools.codebase_index import get_cached_index, build_index, \
    get_codebase_map

logger = get_logger("workflows.pipeline.pipeline_utils")


# ─── Complexity Classification ──────────────────────────────────────────────

_ONELINER_KEYWORDS = [
    "fix typo", "rename", "change string", "update constant",
    "change value", "fix import", "add comment",
]
_BUGFIX_KEYWORDS = [
    "fix bug", "bugfix", "fix error", "fix crash", "fix issue",
    "patch", "hotfix", "resolve issue",
]
_REFACTOR_KEYWORDS = [
    "refactor", "restructure", "reorganize", "clean up",
    "simplify", "extract", "move to",
]

_STAGES_MAP = {
    "oneliner":  ["implement", "review", "commit"],
    "bugfix":    ["fix", "deps", "review", "commit"],
    "refactor":  ["architect", "implement", "deps", "review_fix", "commit"],
    "feature":   ["architect", "implement", "deps", "test", "review_fix", "commit"],
    "tdd":       ["architect", "test", "implement", "deps", "review_fix", "commit"],
}


def classify_complexity(title: str, description: str) -> str:
    """
    Classify task complexity to select pipeline stages.

    Returns: "oneliner" | "bugfix" | "refactor" | "tdd" | "feature"
    """
    text = f"{title} {description}".lower()

    for kw in _ONELINER_KEYWORDS:
        if kw in text:
            return "oneliner"

    for kw in _BUGFIX_KEYWORDS:
        if kw in text:
            return "bugfix"

    for kw in _REFACTOR_KEYWORDS:
        if kw in text:
            return "refactor"

    if "test-driven" in text or "tdd" in text or "tests first" in text:
        return "tdd"

    return "feature"


def get_stages_for_complexity(complexity: str) -> list[str]:
    """Return the list of pipeline stage names for a given complexity."""
    return list(_STAGES_MAP.get(complexity, _STAGES_MAP["feature"]))


# ─── Incremental Progress ────────────────────────────────────────────────────

def _load_progress(mission_id: Optional[int]) -> dict:
    """Load incremental progress from workspace."""
    if not mission_id:
        return {}
    try:
        import tools.workspace as _ws
        progress_file = os.path.join(
            _ws.WORKSPACE_DIR, f".pipeline_progress_{mission_id}.json"
        )
        if os.path.isfile(progress_file):
            with open(progress_file, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_progress(mission_id: Optional[int], progress: dict) -> None:
    """Save incremental progress to workspace."""
    if not mission_id:
        return
    try:
        import tools.workspace as _ws
        progress_file = os.path.join(
            _ws.WORKSPACE_DIR, f".pipeline_progress_{mission_id}.json"
        )
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
    except Exception:
        pass


def _cleanup_progress(mission_id: Optional[int]) -> None:
    """Remove progress file on successful completion."""
    if not mission_id:
        return
    try:
        import tools.workspace as _ws
        progress_file = os.path.join(
            _ws.WORKSPACE_DIR, f".pipeline_progress_{mission_id}.json"
        )
        if os.path.isfile(progress_file):
            os.remove(progress_file)
    except Exception:
        pass


# ─── PR Summary ─────────────────────────────────────────────────────────────

async def generate_pr_summary(
    title: str,
    files_changed: list[str],
    stages_run: list[str],
    review_iterations: int,
    complexity: str,
) -> str:
    """Generate a PR-style summary of pipeline results."""
    diff_output = ""
    try:
        diff_output = await git_diff(stat_only=True)
    except Exception:
        diff_output = "(diff unavailable)"

    lines = [
        f"## Pipeline Complete: {title}",
        "",
        f"**Complexity:** {complexity}",
        f"**Stages run:** {', '.join(stages_run)}",
        f"**Review iterations:** {review_iterations}",
        "",
        "### Files Changed",
    ]

    if files_changed:
        for f in files_changed:
            lines.append(f"  - {f}")
    else:
        lines.append("  (no specific files tracked)")

    if diff_output and diff_output.strip():
        lines.append("")
        lines.append("### Diff Summary")
        lines.append(f"```\n{diff_output[:2000]}\n```")

    return "\n".join(lines)


# ─── Convention Context ──────────────────────────────────────────────────────

def _get_convention_context() -> str:
    """Detect and format codebase conventions for agent prompt injection."""
    try:
        import src.tools.workspace as _ws
        from src.tools.codebase_index import build_index, detect_conventions, get_cached_index

        index = get_cached_index(_ws.WORKSPACE_DIR)
        if not index:
            index = build_index(_ws.WORKSPACE_DIR)

        if not index:
            return ""

        conventions = detect_conventions(index)
        if not conventions or "error" in conventions:
            return ""

        lines = ["\n--- Project Conventions ---"]
        lines.append(f"Naming: {conventions.get('naming_style', 'unknown')}")
        if conventions.get("has_docstrings"):
            lines.append("Style: Include docstrings")
        if conventions.get("async_style"):
            lines.append("Pattern: async/await is the dominant pattern")
        if conventions.get("common_imports"):
            lines.append(f"Common imports: {', '.join(conventions['common_imports'][:5])}")
        lines.append(f"Avg function length: {conventions.get('avg_function_length', '?')} lines")
        lines.append("--- End Conventions ---\n")
        return "\n".join(lines)
    except Exception:
        return ""


def _get_codebase_map_context() -> str:
    """Generate codebase map for large codebases."""
    try:
        index = get_cached_index(_ws.WORKSPACE_DIR)
        if not index:
            index = build_index(_ws.WORKSPACE_DIR)

        if not index or len(index) < 5:
            return ""

        cmap = get_codebase_map(index)
        if len(cmap) > 3000:
            cmap = cmap[:3000] + "\n... (truncated)"

        return f"\n--- Codebase Map ---\n{cmap}\n--- End Map ---\n"
    except Exception:
        return ""
