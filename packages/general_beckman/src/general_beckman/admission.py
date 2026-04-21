"""Admission policy: urgency composition and threshold function.

Task is a dict; urgency derives from three terms:
  priority / 10          — baseline in [0.1, 1.0]
  age / 24h * 0.05       — small bump for aged tasks
  unblocks / 5 * 0.05    — small bump for tasks that unblock many others

Threshold is linear: 0.5 intercept, slope -1.0 — an idle task (urgency=0)
requires clearly abundant pool pressure (>= +0.5), a max-urgency task
(urgency=1) accepts mild depletion (>= -0.5).
"""
from __future__ import annotations

from general_beckman.types import Task, task_age_seconds, task_unblocks_count

AGE_SCALE_S = 86400.0
AGE_WEIGHT = 0.05
BLOCKER_CAP = 5
BLOCKER_WEIGHT = 0.05


def compute_urgency(task: Task) -> float:
    # priority==0 means "lowest urgency" — keep it at 0.0 rather than
    # silently boosting to the default. Only fall back when the key is
    # missing or the value is None.
    priority = task.get("priority")
    if priority is None:
        priority = 5
    priority_term = float(priority) / 10.0
    age_term = min(1.0, task_age_seconds(task) / AGE_SCALE_S) * AGE_WEIGHT
    unblocks = task_unblocks_count(task)
    blocker_term = min(1.0, unblocks / BLOCKER_CAP) * BLOCKER_WEIGHT
    return max(0.0, min(1.0, priority_term + age_term + blocker_term))


def threshold(urgency: float) -> float:
    return max(-1.0, min(1.0, 0.5 - urgency))
