"""Shared LLM helper for shopping intelligence modules — ADMITTED-PRODUCER SEAM.

INTENTIONALLY INERT. This helper used to route shopping intelligence LLM
calls through the ``LLMDispatcher`` ``request`` shim — a deprecated,
non-Beckman-admitted shopping-only path. It is being retired (SP5).
``_llm_call`` now returns
``""`` unconditionally; every caller already degrades to its rule-based path
on an empty response (see the module-level guards in review_synthesizer,
timing, alternatives, delivery_compare, query_analyzer, etc.).

The LLM home for shopping intelligence is an **admitted v3 producer step**
(prep handler -> producer agent -> apply handler), e.g. ``shopping_synthesizer``
wired as the ``synth_dispatch`` triad in ``src/workflows/shopping/shopping_v3.json``.
Review synthesis already runs admitted that way.

DO NOT reintroduce the dispatcher ``request`` call / ``await_inline`` / direct
``husam.run`` here. To make a capability use an LLM *live*, wire it as a
producer triad — never as a mid-ReAct call through this helper.

The signature is preserved so the 13 intelligence modules import and call it
unchanged (they keep working via their rule-based fallbacks).

See ``docs/superpowers/specs/2026-06-09-shopping-intelligence-cps-migration-design.md``.
"""

from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence._llm")


async def _llm_call(
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    task_id: int | None = None,
    mission_id: int | None = None,
) -> str:
    """Inert seam — always returns ``""`` so callers use their rule fallback.

    Kept for signature compatibility with the 13 intelligence modules. The
    former dispatcher ``request`` body was removed to retire the deprecated
    ``LLMDispatcher`` shopping shim (SP5). Wire live LLM use as an
    admitted producer triad instead (see module docstring).

    Args:
        prompt: User prompt text (ignored).
        system: Optional system prompt (ignored).
        temperature: Sampling temperature (ignored).
        task_id: Optional task ID (ignored).
        mission_id: Optional mission ID (ignored).

    Returns:
        Always ``""``.
    """
    return ""
