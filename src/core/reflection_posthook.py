"""Back-compat shim.

The reflection / constrained-emit prompt-builder impl moved to
``packages/coulson/src/coulson/posthooks/reflection_posthook.py`` on
2026-06-07 (P3 of docs/2026-05-31-modularization-finish-plan.md).
Architecture rule: LLM-prompt logic lives in packages, not core. This
re-export keeps existing ``from src.core.reflection_posthook import ...``
call sites (general_beckman/apply.py, posthooks.py, tests) working.
coulson/reflection.py imports the impl directly (it needs a private name).
"""

from coulson.posthooks.reflection_posthook import *  # noqa: F401,F403
from coulson.posthooks.reflection_posthook import (  # noqa: F401  explicit re-exports
    STACK_BLOCKS,
    LAYER_BLOCKS,
    REFLECTION_BLOCKS,
    build_reflection_prompt,
    build_reflect_messages,
    build_emit_messages,
    schema_response_format,
    should_skip_emit,
)
