"""Stateful simulator state (Phase 2d test infrastructure)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimPoolCounter:
    remaining: int
    limit: int
    reset_at: float  # virtual-clock seconds


@dataclass
class SimLocalModel:
    is_loaded: bool = False
    idle_seconds: float = 0.0
    tokens_per_second: float = 20.0


@dataclass
class SimState:
    virtual_clock: float = 0.0
    time_bucketed: dict[str, SimPoolCounter] = field(default_factory=dict)
    per_call: dict[str, SimPoolCounter] = field(default_factory=dict)
    locals: dict[str, SimLocalModel] = field(default_factory=dict)

    def advance_clock(self, delta_seconds: float) -> None:
        self.virtual_clock += delta_seconds

    def maybe_reset_buckets(self) -> None:
        """Reset any bucket whose reset_at has elapsed; roll reset_at forward by 24h."""
        for counter in self.time_bucketed.values():
            while counter.reset_at <= self.virtual_clock:
                counter.remaining = counter.limit
                counter.reset_at += 86400.0

    def tick_locals(self, delta_seconds: float, used_local_name: str | None) -> None:
        """Increment idle for all loaded locals; zero the one that was used."""
        for name, local in self.locals.items():
            if not local.is_loaded:
                continue
            if name == used_local_name:
                local.idle_seconds = 0.0
            else:
                local.idle_seconds += delta_seconds
