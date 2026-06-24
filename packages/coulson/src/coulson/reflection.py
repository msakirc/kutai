"""Self-reflection prompt blocks — re-export surface.

The inline ``self_reflect`` LLM call was deleted in SP5 (2026-06-10); it had
no live callers after SP3b Task 7 moved self-reflection to a Beckman post-hook
child (``coulson/posthooks/reflection_posthook.py``). This module survives only
to re-export the per-agent reflection blocks + prompt builders that the Z2/Z3
stack/layer tests and the post-hook child import from ``coulson.reflection``.

Public API
----------
build_reflection_prompt(agent_name, iteration) -> str
    Return a role-specific self-check checklist injected into the reviewer's
    system message. Falls back to a generic prompt for unknown agents.
"""
from __future__ import annotations

from yazbunu import get_logger

logger = get_logger("coulson.reflection")


# ────────────────────────────────────────────────────────────────────────────
# Per-agent reflection checklists — live in coulson/posthooks/reflection_posthook
# (moved from src/core 2026-06-07, P3). These names are re-exported here for
# back-compat: the Z2/Z3 stack/layer tests + the reflection post-hook child
# import them from coulson.reflection.
# ────────────────────────────────────────────────────────────────────────────

from .posthooks.reflection_posthook import (  # noqa: F401
    STACK_BLOCKS,
    LAYER_BLOCKS,
    REFLECTION_BLOCKS,
    _GENERIC_REFLECTION_BLOCK,
    build_reflection_prompt,
    build_reflect_messages,
)
