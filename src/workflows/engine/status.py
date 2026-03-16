"""Workflow progress tracking and status reporting."""

import json

PHASE_NAMES: dict[str, str] = {
    "phase_-1": "Existing Project Onboarding",
    "phase_0": "Idea Capture & Clarification",
    "phase_1": "Market & Competitive Research",
    "phase_2": "Product Strategy & Definition",
    "phase_3": "Requirements Engineering",
    "phase_4": "Architecture & Technical Design",
    "phase_5": "UX/UI Design Specification",
    "phase_6": "Project Planning & Sprint Setup",
    "phase_7": "Development Environment Setup",
    "phase_8": "Core Implementation",
    "phase_9": "Comprehensive Testing",
    "phase_10": "Security Hardening",
    "phase_11": "Documentation",
    "phase_12": "Legal & Compliance",
    "phase_13": "Pre-Launch Preparation",
    "phase_14": "Launch",
    "phase_15": "Post-Launch Operations",
}


def compute_phase_progress(tasks: list[dict]) -> dict:
    """Compute completed/total counts per workflow phase.

    Args:
        tasks: List of task dicts, each with a ``context`` field that is
               either a dict or a JSON string containing ``workflow_phase``
               and ``status``.

    Returns:
        Dict keyed by phase_id. Each value has ``completed``, ``total``,
        and ``name`` fields.
    """
    progress: dict[str, dict] = {}

    for task in tasks:
        ctx = task.get("context")
        if ctx is None:
            continue

        # Parse JSON string context if needed.
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except (json.JSONDecodeError, TypeError):
                continue

        if not isinstance(ctx, dict):
            continue

        phase = ctx.get("workflow_phase")
        if not phase:
            continue

        if phase not in progress:
            progress[phase] = {
                "completed": 0,
                "total": 0,
                "name": PHASE_NAMES.get(phase, phase),
            }

        progress[phase]["total"] += 1
        if ctx.get("status") == "completed":
            progress[phase]["completed"] += 1

    return progress


def format_status_message(workflow_id: str, goal_id: int, progress: dict) -> str:
    """Format a Telegram-friendly status message for a workflow.

    Args:
        workflow_id: Unique workflow identifier.
        goal_id: Associated goal number.
        progress: Output of :func:`compute_phase_progress`.

    Returns:
        Multi-line status string with icons, progress bars, and counts.
    """
    lines: list[str] = [
        f"\U0001f4ca Workflow Status: {workflow_id} (goal #{goal_id})",
        "",
    ]

    def _phase_sort_key(phase_id: str) -> int:
        # "phase_-1" -> -1, "phase_3" -> 3
        return int(phase_id.split("_", 1)[1])

    sorted_phases = sorted(progress.keys(), key=_phase_sort_key)

    for phase_id in sorted_phases:
        info = progress[phase_id]
        completed = info["completed"]
        total = info["total"]
        name = info["name"]

        # Choose icon.
        if completed == total and total > 0:
            icon = "\u2705"  # ✅
        elif completed > 0:
            icon = "\U0001f504"  # 🔄
        else:
            icon = "\u2b1c"  # ⬜

        # Build progress bar (10 chars).
        if total > 0:
            filled = round(10 * completed / total)
        else:
            filled = 0
        bar = "\u2588" * filled + "\u2591" * (10 - filled)

        lines.append(f"{icon} {name}  [{bar}] {completed}/{total}")

    return "\n".join(lines)
