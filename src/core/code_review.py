"""Back-compat shim.

The code-review prompt-builder impl moved to
``packages/coulson/src/coulson/posthooks/code_review.py`` on 2026-06-07 (P3 of
docs/2026-05-31-modularization-finish-plan.md). Architecture rule: LLM-prompt
logic lives in packages, not core. This re-export keeps existing
``from src.core.code_review import ...`` call sites (general_beckman/apply.py,
posthook_continuations.py, tests) working.
"""

from coulson.posthooks.code_review import *  # noqa: F401,F403
from coulson.posthooks.code_review import (  # noqa: F401  explicit re-exports
    CodeReviewResult,
    parse_code_review_response,
    build_code_review_spec,
)
