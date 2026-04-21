"""Core types: Task, AgentResult, Lane.

Task stays a `dict[str, Any]` (not a dataclass) to keep row-shape
compatibility with existing callers. Urgency and admission extensions
are exposed as helper functions + well-known keys:

    - `preselected_pick`          — Fatih Hoca Pick attached by admission.
    - `downstream_unblocks_count` — count of tasks this one unblocks.

Helpers:
    - `task_age_seconds(task)`
    - `task_unblocks_count(task)`
    - `task_preselected_pick(task)`
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any

Task = dict[str, Any]
AgentResult = dict[str, Any]


class Lane(str, Enum):
    LOCAL_LLM = "local_llm"
    CLOUD_LLM = "cloud_llm"
    MECHANICAL = "mechanical"


def task_age_seconds(task: Task) -> float:
    """Seconds since `created_at`. 0 if field missing or malformed."""
    created = task.get("created_at")
    if created is None:
        return 0.0
    try:
        return max(0.0, time.time() - float(created))
    except (TypeError, ValueError):
        return 0.0


def task_unblocks_count(task: Task) -> int:
    """Downstream-unblock count for this task. 0 when unpopulated."""
    return int(task.get("downstream_unblocks_count", 0) or 0)


def task_preselected_pick(task: Task) -> Any:
    """Fatih Hoca Pick attached by admission (iteration-0 reuse). None by default."""
    return task.get("preselected_pick")
