"""Z7 T1.0 posthook handler stubs.

Each Z7 posthook kind has its own module here so 4 independent feature agents
can each edit exactly one file without merge conflicts:

  briefing_compose          -> briefing_compose.py       (agent A0)
  brand_voice_lint          -> brand_voice_lint.py       (agent A5)
  copy_compliance_review    -> copy_compliance_review.py (agent A6)
  audit_completeness_check  -> audit_completeness_check.py (agents B5/B9)

This package is imported by posthooks.py to wire the stubs into POST_HOOK_REGISTRY.
"""
from __future__ import annotations
