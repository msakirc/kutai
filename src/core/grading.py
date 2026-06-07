"""Back-compat shim.

The grading prompt-builder impl moved to
``packages/coulson/src/coulson/posthooks/grading.py`` on 2026-06-07 (P3 of
docs/2026-05-31-modularization-finish-plan.md). Architecture rule: LLM-prompt
logic lives in packages, not core. This re-export keeps existing
``from src.core.grading import ...`` call sites (general_beckman/apply.py,
posthook_continuations.py, tests) working.
"""

from coulson.posthooks.grading import *  # noqa: F401,F403
from coulson.posthooks.grading import (  # noqa: F401  explicit re-exports
    GRADING_SYSTEM,
    GRADING_PROMPT,
    GradeResult,
    parse_grade_response,
    build_grading_spec,
)
