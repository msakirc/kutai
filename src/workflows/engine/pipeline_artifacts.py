"""Extract structured artifacts from CodingPipeline execution results.

After CodingPipeline runs, this module analyzes the result and workspace
to produce structured artifacts: files changed, test results, dependencies
added, and architectural decisions made.
"""
import json
import os
import re
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("workflows.engine.pipeline_artifacts")


async def extract_pipeline_artifacts(
    task: dict,
    result: dict,
    workspace_path: Optional[str] = None,
) -> dict[str, str]:
    """Extract structured artifacts from a completed pipeline task.

    Returns a dict of artifact_name -> artifact_content to be stored.
    """
    artifacts = {}
    task_ctx = _parse_context(task)
    step_id = task_ctx.get("workflow_step_id") or task_ctx.get("step_id", "unknown")

    # 1. Files changed summary
    files_changed = await _extract_files_changed(result, workspace_path)
    if files_changed:
        artifacts[f"{step_id}_files_changed"] = json.dumps(files_changed)

    # 2. Test results (if the pipeline ran tests)
    test_results = _extract_test_results(result)
    if test_results:
        artifacts[f"{step_id}_test_results"] = json.dumps(test_results)

    # 3. Implementation summary (structured from result text)
    summary = _build_implementation_summary(task, result, files_changed)
    artifacts[f"{step_id}_implementation_summary"] = summary

    return artifacts


def _parse_context(task: dict) -> dict:
    ctx = task.get("context", {})
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    return ctx if isinstance(ctx, dict) else {}


async def _extract_files_changed(
    result: dict, workspace_path: Optional[str]
) -> list[dict]:
    """Extract list of files created/modified from result or workspace git diff."""
    files = []

    # Try to extract from result metadata
    result_text = result.get("result", "")

    # Parse file references from result text (common patterns)
    # Match patterns like "Created src/foo.py", "Modified bar.js", "Updated test_x.py"
    for match in re.finditer(
        r"(?:Created|Modified|Updated|Added|Wrote|Edited)\s+[`\"]?([^\s`\"]+\.\w+)",
        result_text,
    ):
        filepath = match.group(1)
        files.append({"path": filepath, "action": "modified"})

    # Try git diff if workspace available
    if workspace_path and os.path.isdir(workspace_path):
        try:
            import subprocess

            git_result = subprocess.run(
                ["git", "diff", "--name-status", "HEAD~1"],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if git_result.returncode == 0:
                for line in git_result.stdout.strip().split("\n"):
                    if line:
                        parts = line.split("\t", 1)
                        if len(parts) == 2:
                            status_map = {
                                "A": "added",
                                "M": "modified",
                                "D": "deleted",
                            }
                            action = status_map.get(parts[0], "modified")
                            files.append({"path": parts[1], "action": action})
        except Exception:
            pass

    # Deduplicate by path
    seen: set[str] = set()
    unique: list[dict] = []
    for f in files:
        if f["path"] not in seen:
            seen.add(f["path"])
            unique.append(f)

    return unique


def _extract_test_results(result: dict) -> Optional[dict]:
    """Extract test pass/fail counts from result text."""
    result_text = result.get("result", "")

    # Pattern: "X passed, Y failed" or "X tests passed"
    m = re.search(r"(\d+)\s*(?:tests?\s+)?passed", result_text, re.IGNORECASE)
    if m:
        passed = int(m.group(1))
        failed_m = re.search(
            r"(\d+)\s*(?:tests?\s+)?failed", result_text, re.IGNORECASE
        )
        failed = int(failed_m.group(1)) if failed_m else 0
        return {"passed": passed, "failed": failed, "total": passed + failed}

    return None


def _build_implementation_summary(
    task: dict, result: dict, files: list[dict]
) -> str:
    """Build a structured implementation summary."""
    ctx = _parse_context(task)
    feature_name = ctx.get("workflow_context", {}).get("feature_name", "")
    result_text = result.get("result", "")

    lines = [
        f"## Implementation: {task.get('title', 'Unknown')}",
    ]
    if feature_name:
        lines.append(f"**Feature:** {feature_name}")

    if files:
        lines.append(f"**Files touched:** {len(files)}")
        for f in files[:10]:  # Cap at 10
            lines.append(f"  - {f['action']}: {f['path']}")

    # Include first 300 chars of result as excerpt
    excerpt = result_text[:300]
    if len(result_text) > 300:
        excerpt += "..."
    lines.append(f"\n**Result excerpt:**\n{excerpt}")

    return "\n".join(lines)
