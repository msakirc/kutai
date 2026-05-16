"""Intersect — thin per-task match+expose layer over the yalayut catalog.

One public function: ``flash(task)``. Invoked once per task by the
orchestrator pump, before dispatch. Imports yalayut (in-process catalog
read) plus db/embeddings only. Never imports LLMDispatcher — Phase 2 has
no LLM-bind (locked KutAI rule: only Beckman calls the dispatcher).
"""
from __future__ import annotations

from intersect.exposure import (
    THETA_PREEMPT, THETA_INJECT, THETA_TOOL, THETA_MIN,
)
from intersect.flash import flash

__all__ = [
    "flash",
    "THETA_PREEMPT", "THETA_INJECT", "THETA_TOOL", "THETA_MIN",
]
