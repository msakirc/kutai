"""c21_paraflow_diff — Z1 Tier 7B (C21).

Bundle-quality regression harness: compare a KutAI mission workspace to
a Paraflow golden bundle for the same archetype. Returns coverage,
coherence, design fitness, gaps, and an overall verdict.

Public surface:

- :func:`diff_bundle`           — main entry point (rule-based)
- :func:`load_golden`           — load a golden archetype's artifacts
- :data:`KNOWN_ARCHETYPES`      — archetypes available under
                                  ``tests/goldens/paraflow/``
"""
from __future__ import annotations

from c21_paraflow_diff.diff_bundle import (
    diff_bundle,
    load_golden,
    GoldenNotFoundError,
    KNOWN_ARCHETYPES,
    DEFAULT_GOLDENS_ROOT,
)

__all__ = [
    "diff_bundle",
    "load_golden",
    "GoldenNotFoundError",
    "KNOWN_ARCHETYPES",
    "DEFAULT_GOLDENS_ROOT",
]
