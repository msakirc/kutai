"""Queue look-ahead: hold back cloud-heavy tasks when quota headroom is low.

Reinstates the quota-look-ahead that was lost during the earlier
``quota_planner`` extraction. Consumes the ``SystemSnapshot`` produced by
nerd_herd to project whether the currently queued cloud-heavy demand can
fit into the remaining request budget.
"""
from __future__ import annotations

from typing import Any

_CLOUD_AGENT_TYPES = {"researcher", "planner", "architect"}
_HEADROOM_FACTOR = 1.5  # keep requests_remaining > queue_depth * factor


def _total_requests_remaining(snapshot: Any) -> int:
    """Sum ``limits.rpm.remaining`` across cloud providers. ``None``/missing
    values count as zero. Returns -1 when there's no cloud info at all (so
    the caller can distinguish "no quota signal" from "zero quota").
    """
    if snapshot is None:
        return -1
    cloud = getattr(snapshot, "cloud", None) or {}
    if not cloud:
        return -1
    total = 0
    saw_any = False
    for state in cloud.values():
        limits = getattr(state, "limits", None)
        if limits is None:
            continue
        rpm = getattr(limits, "rpm", None)
        if rpm is None:
            continue
        rem = getattr(rpm, "remaining", None)
        if rem is None:
            continue
        total += int(rem)
        saw_any = True
    return total if saw_any else -1


def should_hold_back(candidate_task: dict, snapshot: Any,
                     cloud_queue_depth: int) -> bool:
    """Return True if ``candidate_task`` should be held back because cloud
    quota headroom is insufficient for the pending cloud-heavy queue."""
    agent = candidate_task.get("agent_type", "")
    if agent == "mechanical":
        return False
    if agent not in _CLOUD_AGENT_TYPES:
        return False
    total_remaining = _total_requests_remaining(snapshot)
    if total_remaining < 0:
        # No signal — don't gate.
        return False
    required = max(1, int(cloud_queue_depth * _HEADROOM_FACTOR))
    return total_remaining < required
