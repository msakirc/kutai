"""Sade Kalsin (Turkish: "stay simple") — quarterly bash-audit harness.

Walks `packages/` + `src/` and emits a per-layer report asking "what does
this layer do that bash + Claude can't?" — the Mini-SWE-agent (65% SWE-bench
in 100 LOC + bash) framing for spotting over-engineered scaffolding.

Featherweight: stdlib only, no runtime deps. Audit cadence is quarterly
(first of Jan/Apr/Jul/Oct, 9am) wired through general_beckman scheduled_jobs
+ mr_roboto's `run_bash_audit` mechanical action.
"""
from __future__ import annotations

from sade_kalsin.inventory import LayerReport, walk_layers
from sade_kalsin.audit_questions import audit_questions_for, AUDIT_QUESTIONS
from sade_kalsin.audit_report import (
    rank_hot_spots,
    render_report,
    write_report,
    quarter_for_date,
    run_audit,
)

__all__ = [
    "LayerReport",
    "walk_layers",
    "audit_questions_for",
    "AUDIT_QUESTIONS",
    "rank_hot_spots",
    "render_report",
    "write_report",
    "quarter_for_date",
    "run_audit",
]
