"""Z6 T7C — extra system-prompt blocks injected by the runtime.

When the runtime decides a task needs a special instruction beyond the
profile's normal system prompt (e.g. ``needs_real_tools=true`` tasks
where prerequisites are satisfied), it appends one of these blocks
through the prompt-builder pipeline.

Blocks are plain strings — no formatting — so they survive verbatim
through prompt composition.
"""
from __future__ import annotations


def real_world_side_effects_warning(reversibility: str | None) -> str:
    """Block injected when a task has ``needs_real_tools=true`` AND its
    adapter+credentials are satisfied.

    Tells the LLM to use the ``vendor_call`` tool rather than fabricate
    API responses and to surface a clarify action when it can't complete
    the task with the available tools.
    """
    rev_txt = reversibility or "unknown"
    return (
        "⚠ This task has real-world side effects "
        f"(needs_real_tools=true, reversibility={rev_txt}). "
        "Use the vendor_call tool — DO NOT fabricate API responses. "
        "If you cannot complete the task with available tools, emit a "
        "clarify action."
    )


REAL_WORLD_BLOCK_MARKER = "⚠ This task has real-world side effects"
"""Substring callers can grep for to confirm the block was injected."""


__all__ = [
    "real_world_side_effects_warning",
    "REAL_WORLD_BLOCK_MARKER",
]
