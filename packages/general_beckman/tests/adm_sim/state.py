"""Admission-sim state — used by run_admission_scenarios.py and pytest harness.

Shares concepts with packages/fatih_hoca/tests/sim/state.py but carries
admission-specific fixtures: a task queue, in-flight counts per provider,
and a virtual-clock hook for task ages.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SimPool:
    """A cloud pool counter (time-bucketed or per-call)."""
    remaining: int
    limit: int
    reset_at: float = 0.0  # wall-clock seconds for time-bucketed; 0 for per-call
    in_flight: int = 0


@dataclass
class SimTask:
    id: int
    agent_type: str = "coder"
    difficulty: int = 5
    priority: int = 5
    created_at: float = 0.0
    downstream_unblocks_count: int = 0
    claimed: bool = False
    completed: bool = False
    # Pick model hint — maps to scenario provider. Selector uses this
    # rather than running the real Fatih Hoca.
    intended_model: str = "claude-sonnet-4-6"
    intended_provider: str = "anthropic"
    intended_is_free: bool = False


@dataclass
class SimState:
    wall_anchor: float = field(default_factory=time.time)
    tick: int = 0
    per_call: dict[str, SimPool] = field(default_factory=dict)
    time_bucketed: dict[str, SimPool] = field(default_factory=dict)
    tasks: list[SimTask] = field(default_factory=list)
    admits: list[int] = field(default_factory=list)  # task ids admitted, in order

    def unclaimed_tasks(self) -> list[SimTask]:
        return [t for t in self.tasks if not t.claimed]

    def claimed_unfinished(self) -> int:
        return sum(1 for t in self.tasks if t.claimed and not t.completed)

    def pool_for(self, provider: str, model: str, is_free: bool) -> SimPool | None:
        if is_free:
            return self.time_bucketed.get(model)
        return self.per_call.get(model)
