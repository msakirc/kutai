"""Stateful simulator runner (Phase 2d test infrastructure).

Evolves a SimState through a sequence of SimTasks by calling a
caller-provided select function + snapshot factory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sim.state import SimState


@dataclass
class SimTask:
    idx: int
    difficulty: int
    estimated_output_tokens: int = 1000
    task_name: str = "generic"


@dataclass
class SimPick:
    task_idx: int
    task_difficulty: int
    model_name: str
    pool: str
    cap_score_100: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass
class SimRun:
    picks: list[SimPick] = field(default_factory=list)
    final_state: SimState = field(default_factory=SimState)


def run_simulation(
    tasks: list[SimTask],
    initial_state: SimState,
    select_fn: Callable[[SimState, SimTask], Any],
    snapshot_factory: Callable[[SimState], Any],
) -> SimRun:
    """Run the simulator and return SimRun with per-task picks + final state.

    `select_fn(state, task)` must return an object with:
        .model_name, .pool, .estimated_output_tokens, .tokens_per_second
        (optionally .cap_score_100 for reporting)

    `snapshot_factory(state)` builds the SystemSnapshot-like object passed
    into the selector. This is scenario-specific because each scenario wires
    its own pool state into a SystemSnapshot shape.
    """
    state = initial_state
    picks: list[SimPick] = []

    for task in tasks:
        state.maybe_reset_buckets()
        pick = select_fn(state, task)

        elapsed = (
            task.estimated_output_tokens / pick.tokens_per_second
            if pick.tokens_per_second > 0 else 0.0
        )

        used_local = pick.model_name if pick.pool == "local" else None
        state.tick_locals(delta_seconds=elapsed, used_local_name=used_local)
        state.advance_clock(elapsed)

        if pick.pool == "time_bucketed":
            counter = state.time_bucketed.get(pick.model_name)
            if counter is not None and counter.remaining > 0:
                counter.remaining -= 1
        elif pick.pool == "per_call":
            counter = state.per_call.get(pick.model_name)
            if counter is not None and counter.remaining > 0:
                counter.remaining -= 1

        picks.append(SimPick(
            task_idx=task.idx,
            task_difficulty=task.difficulty,
            model_name=pick.model_name,
            pool=pick.pool,
            cap_score_100=getattr(pick, "cap_score_100", 0.0),
            elapsed_seconds=elapsed,
        ))

    return SimRun(picks=picks, final_state=state)
