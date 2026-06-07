"""LLM-child post-hook prompt builders.

Pure builders for the raw_dispatch LLM children spawned by
general_beckman/apply.py: the grader, the code-reviewer, and the
self-reflection / constrained-emit children. No DB, no LLM dispatch —
they only build Beckman spec dicts + parse child responses.

Relocated 2026-06-07 from src/core/ (grading.py, code_review.py,
reflection_posthook.py). Architecture rule: LLM-prompt logic lives in
packages, not core. src/core/ keeps thin re-export shims for back-compat.
"""
