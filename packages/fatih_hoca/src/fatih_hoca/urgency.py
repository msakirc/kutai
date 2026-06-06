"""Mid-task urgency policy (single source of truth).

A task admitted under a given pool-pressure urgency must not be re-judged
*stricter* mid-flight than it was at admission — otherwise a started task
gets vetoed off the alive band (scalar in -0.75..-1.0) that admitted it and
raises "No model candidates available". The mid-task urgency is therefore
the admission urgency plus a small finish-bias ("a little higher than
pre-dispatch", user design 2026-05-03), with an extra bump while a retry
failure is being adapted around.

Does NOT change the pool-pressure gate formula
(``selector.py``: threshold = -0.5 - 0.5*urgency) or the -1.0 hard veto.
"""
from __future__ import annotations

FINISH_BIAS = 0.1   # mid-task urgency sits a little above pre-dispatch urgency
FAILURE_BUMP = 0.1  # extra escalation while adapting around a transport failure


def mid_task_urgency(admission_urgency: float | None, *, has_failures: bool) -> float:
    """Urgency for a mid-task (re-)selection.

    ``admission_urgency`` is the value Beckman computed at admission
    (``compute_urgency``), stamped on the task as ``_admission_urgency``.
    Falls back to 0.5 when unknown.
    """
    base = 0.5 if admission_urgency is None else float(admission_urgency)
    urgency = min(1.0, base + FINISH_BIAS)
    if has_failures:
        urgency = min(1.0, urgency + FAILURE_BUMP)
    return urgency
