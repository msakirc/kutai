# tools/coverage.py
"""
Phase 10.4 — Code Coverage

Run pytest --cov or jest --coverage inside the Docker sandbox,
parse coverage reports, identify undertested paths.
"""
from __future__ import annotations
import re

from src.infra.logging_config import get_logger
from .shell import run_shell

logger = get_logger("tools.coverage")


async def run_python_coverage(
    project_path: str = ".",
    min_coverage: int = 0,
) -> str:
    """
    Run pytest --cov inside the sandbox.

    Args:
        project_path: Path to the Python project (relative to workspace).
        min_coverage: Minimum acceptable coverage %. 0 = no threshold.

    Returns:
        Coverage report string.
    """
    cmd = (
        f"cd {project_path} && "
        f"pytest --cov=. --cov-report=term-missing --tb=no -q 2>&1 | head -60"
    )
    output = await run_shell(cmd, timeout=120)

    # Parse overall coverage percentage
    coverage_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', output)
    if coverage_match:
        pct = int(coverage_match.group(1))
        if min_coverage > 0 and pct < min_coverage:
            output += f"\n⚠️ Coverage {pct}% is below minimum {min_coverage}%"
        else:
            output += f"\n✅ Coverage: {pct}%"

    return output


async def run_js_coverage(
    project_path: str = ".",
) -> str:
    """
    Run jest --coverage inside the sandbox.

    Args:
        project_path: Path to the JS project (relative to workspace).

    Returns:
        Coverage report string.
    """
    cmd = (
        f"cd {project_path} && "
        f"npm test -- --coverage --coverageReporters=text 2>&1 | tail -40"
    )
    return await run_shell(cmd, timeout=120)


async def get_coverage_summary(
    project_path: str = ".",
    language: str = "python",
) -> str:
    """
    Get a coverage summary for a project.

    Args:
        project_path: Path relative to workspace.
        language: 'python' or 'javascript'/'typescript'.

    Returns:
        Coverage summary string.
    """
    if language in ("javascript", "typescript"):
        return await run_js_coverage(project_path)
    return await run_python_coverage(project_path)
