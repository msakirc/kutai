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
    GradeResult,
    parse_grade_response,
    build_grading_spec,
)
# NOTE: GRADING_SYSTEM / GRADING_PROMPT were removed when the grading prompt
# migrated to Foundry rubrics (build_messages("grading", ...)). They are no
# longer exported here. No production caller imported them; the spec is now
# produced by build_grading_spec().
