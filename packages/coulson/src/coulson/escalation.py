"""Escalation helpers — message trimming for model escalation.

Pure functions. No agent state, no side effects.

Public API:
    trim_for_escalation(messages, iteration, max_iterations) -> list[dict]
    escalate_requirements(reqs)  -- forward-compat shim

Extracted from ``src/agents/base.py`` (Phase A.7 of runtime extraction).
"""
from __future__ import annotations


def trim_for_escalation(
    messages: list[dict], iteration: int, max_iterations: int,
) -> list[dict]:
    """Trim message history on model escalation.

    Keeps: system prompt, task description, successful tool results,
    most recent error. Strips: old model's reasoning, failed retries,
    format corrections, guard rejections.
    """
    trimmed: list[dict] = []
    last_error: dict | None = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Always keep system prompt
        if role == "system":
            trimmed.append(msg)
            continue

        # Keep original task (first user message after system)
        if role == "user" and len(trimmed) <= 1 and "## Tool Result" not in content:
            trimmed.append(msg)
            continue

        # Keep successful tool results
        if (
            role == "user"
            and "## Tool Result" in content
            and not content.lstrip().startswith("❌")
            and not content.lstrip().startswith("\U0001f6ab")
        ):
            trimmed.append(msg)
            continue

        # Track last error for context
        if role == "user" and (
            content.lstrip().startswith("❌")
            or content.lstrip().startswith("\U0001f6ab")
        ):
            last_error = msg

        # Everything else (assistant reasoning, guard corrections,
        # format retries) is stripped

    # Include last error if found and not already in trimmed
    if last_error and last_error not in trimmed:
        trimmed.append(last_error)

    # Inject escalation context
    remaining = max_iterations - iteration - 1
    trimmed.append({
        "role": "user",
        "content": (
            "A previous attempt at this task encountered difficulties. "
            "The tool results above are from that attempt — they contain valid data. "
            "You have a fresh start with better capabilities. "
            f"Iterations remaining: {remaining}."
        ),
    })

    return trimmed


def escalate_requirements(reqs):
    """Wrapper around reqs.escalate(). Forward-compat shim — runtime should NOT
    call this directly per the new architecture (failure-signal interface
    replaces it). Kept for BaseAgent delegate while phase C is in flight."""
    return reqs.escalate()
