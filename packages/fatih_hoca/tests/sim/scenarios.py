"""Scenario factories for Phase 2d simulator.

Each scenario returns a `Scenario` with: `tasks`, `initial_state`,
`snapshot_factory`, `select_fn`. Wire through the real fatih_hoca.select() in
`select_fn` via `selector.select_for_simulation()` (implemented in Task 12).

Keep scenarios focused on the state + demand profile; shared helpers
`_build_select_fn` and `_build_snapshot_factory` translate SimState into
the shapes the selector expects.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from sim.runner import SimTask
from sim.state import (
    SimLocalModel,
    SimPoolCounter,
    SimState,
)


RNG = random.Random(42)


@dataclass
class Scenario:
    name: str
    tasks: list[SimTask]
    initial_state: SimState
    snapshot_factory: Callable[[SimState], Any]
    select_fn: Callable[[SimState, SimTask], Any]


def _standard_i2p_task_mix(count: int = 182, seed: int = 42) -> list[SimTask]:
    """Approximate i2p v3's task-difficulty distribution.

    Roughly 50% d<=3, 30% d=4-6, 15% d=7-8, 5% d=9-10.
    Uses a fixed-seed RNG for determinism.
    """
    rng = random.Random(seed)
    tasks = []
    for i in range(count):
        r = rng.random()
        if r < 0.50:
            d = rng.choice([1, 2, 3])
        elif r < 0.80:
            d = rng.choice([4, 5, 6])
        elif r < 0.95:
            d = rng.choice([7, 8])
        else:
            d = rng.choice([9, 10])
        est_out = 1000 if d <= 5 else (2500 if d <= 7 else 5000)
        tasks.append(SimTask(idx=i, difficulty=d, estimated_output_tokens=est_out))
    return tasks


def _build_snapshot_factory(scenario_providers: dict[str, Any]):
    """Builds a closure that turns SimState -> SystemSnapshot-like object.

    Uses real nerd_herd types so `snapshot.pressure_for(model)` is available
    to the scarcity layer (supply side relocated to nerd_herd).
    """
    import time as _time

    from nerd_herd.types import (
        CloudModelState,
        CloudProviderState,
        LocalModelState,
        RateLimit,
        RateLimitMatrix,
        SystemSnapshot,
    )

    # Pin a wall-clock anchor so virtual-clock -> wall-clock mapping is stable
    # within a sim run. We project `counter.reset_at` (virtual seconds) onto the
    # same wall clock for rate-limit reset time calculations.
    wall_anchor = _time.time()

    def factory(state: SimState) -> Any:
        if state.locals:
            loaded_name = next(
                (n for n, l in state.locals.items() if l.is_loaded), ""
            )
            loaded = state.locals.get(loaded_name)
            local_snap = LocalModelState(
                model_name=loaded_name,
                idle_seconds=loaded.idle_seconds if loaded else 0.0,
                measured_tps=loaded.tokens_per_second if loaded else 0.0,
                thinking_enabled=False,
            )
        else:
            local_snap = LocalModelState(
                model_name=None, idle_seconds=0.0, measured_tps=0.0,
                thinking_enabled=False,
            )

        cloud: dict[str, CloudProviderState] = {}
        for provider, prov_cfg in scenario_providers.items():
            models: dict[str, CloudModelState] = {}
            for model_id, _model_cfg in prov_cfg.get("models", {}).items():
                if prov_cfg.get("is_free"):
                    counter = state.time_bucketed.get(model_id)
                else:
                    counter = state.per_call.get(model_id)
                if counter is None:
                    models[model_id] = CloudModelState(
                        model_id=model_id,
                        utilization_pct=0.0,
                        limits=RateLimitMatrix(),
                    )
                    continue
                reset_in_secs = max(0.0, counter.reset_at - state.virtual_clock)
                rpd = RateLimit(
                    limit=counter.limit,
                    remaining=counter.remaining,
                    reset_at=int(wall_anchor + reset_in_secs),
                )
                util = 100.0 * (1.0 - counter.remaining / max(1, counter.limit))
                models[model_id] = CloudModelState(
                    model_id=model_id,
                    utilization_pct=util,
                    limits=RateLimitMatrix(rpd=rpd),
                )
            cloud[provider] = CloudProviderState(
                provider=provider,
                utilization_pct=0.0,
                consecutive_failures=0,
                limits=RateLimitMatrix(),
                models=models,
            )
        return SystemSnapshot(local=local_snap, cloud=cloud)

    return factory


def _build_select_fn(scenario_providers: dict[str, Any], tasks: list[SimTask] | None = None):
    """Wires through the real fatih_hoca.select() against the SimState.

    If ``tasks`` is provided, a live ``QueueProfile`` is built from the
    remaining tail (``tasks[task.idx:]``) at each tick and threaded into
    ``select_for_simulation``. This fixes Phase 2d bug #2 — scenarios
    previously fed a fresh empty QueueProfile, making the queue-pressure
    arm of per_call scarcity permanently dormant.
    """
    from types import SimpleNamespace
    from fatih_hoca import selector as _selector
    from fatih_hoca.requirements import QueueProfile
    snapshot_factory = _build_snapshot_factory(scenario_providers)

    def select(state: SimState, task: SimTask) -> Any:
        queue_profile = None
        if tasks is not None:
            # Remaining slice starting at this task. Use positional index,
            # since task.idx is dense sequential (0..N-1) in all scenarios.
            remaining = tasks[task.idx:]
            total = len(remaining)
            hard = sum(1 for t in remaining if t.difficulty >= 7)
            max_d = max((t.difficulty for t in remaining), default=0)
            queue_profile = QueueProfile(
                total_tasks=total,
                hard_tasks_count=hard,
                max_difficulty=max_d,
            )

        picked = _selector.select_for_simulation(
            task_name=task.task_name,
            difficulty=task.difficulty,
            estimated_output_tokens=task.estimated_output_tokens,
            snapshot=snapshot_factory(state),
            providers_cfg=scenario_providers,
            queue_profile=queue_profile,
        )
        return SimpleNamespace(
            model_name=picked.model_name,
            pool=picked.pool,
            cap_score_100=picked.cap_score_100,
            estimated_output_tokens=task.estimated_output_tokens,
            tokens_per_second=picked.tokens_per_second,
        )

    return select


# -- Scenario factories -------------------------------------------------------

def baseline() -> Scenario:
    providers = {
        "groq": {
            "is_free": True,
            "models": {"groq/llama-3.1-70b": {"cap_score_100": 72}},
        },
        "anthropic": {
            "is_free": False,
            "models": {"anthropic/claude-sonnet": {"cap_score_100": 93}},
        },
    }
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0, tokens_per_second=20.0)
    state.time_bucketed["groq/llama-3.1-70b"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    state.per_call["anthropic/claude-sonnet"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    tasks = _standard_i2p_task_mix()
    return Scenario(
        name="baseline",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers, tasks=tasks),
    )


def claude_constrained() -> Scenario:
    providers = {
        "anthropic": {
            "is_free": False,
            "models": {"anthropic/claude-sonnet": {"cap_score_100": 93}},
        },
        "groq": {
            "is_free": True,
            "models": {"groq/llama-3.1-70b": {"cap_score_100": 72}},
        },
    }
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0)
    # Limit sized above the scenario's hard-task count (~47 d≥7 in the
    # standard 182-task mix) so the "constrained" test exercises scarcity
    # depletion behavior without being structurally impossible — a pool
    # smaller than hard demand can never reach 90% hard_task_satisfaction
    # no matter how well the equation balances.
    state.per_call["anthropic/claude-sonnet"] = SimPoolCounter(remaining=60, limit=60, reset_at=86400.0)
    state.time_bucketed["groq/llama-3.1-70b"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    tasks = _standard_i2p_task_mix()
    return Scenario(
        name="claude_constrained",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers, tasks=tasks),
    )


def groq_near_reset() -> Scenario:
    providers = {
        "groq": {"is_free": True, "models": {"groq/llama-3.1-70b": {"cap_score_100": 72}}},
        "anthropic": {"is_free": False, "models": {"anthropic/claude-sonnet": {"cap_score_100": 93}}},
    }
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0)
    state.time_bucketed["groq/llama-3.1-70b"] = SimPoolCounter(remaining=850, limit=1000, reset_at=1800.0)
    state.per_call["anthropic/claude-sonnet"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    tasks = _standard_i2p_task_mix()
    return Scenario(
        name="groq_near_reset",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers, tasks=tasks),
    )


def diverse_pool() -> Scenario:
    providers = {
        "groq": {"is_free": True, "models": {"groq/llama-3.1-70b": {"cap_score_100": 72}}},
        "gemini": {"is_free": True, "models": {"gemini/gemini-1.5-flash": {"cap_score_100": 68}}},
        "openrouter": {"is_free": True, "models": {"openrouter/free-mistral": {"cap_score_100": 70}}},
        "anthropic": {"is_free": False, "models": {"anthropic/claude-sonnet": {"cap_score_100": 93}}},
    }
    # Scenario intent: one workflow (~182 tasks) should consume a meaningful
    # fraction of each free pool's daily budget. Realistic per-workflow limits
    # (~100-150) scale the 70% utilization target to what's reachable in 182
    # tasks. Per-day limits (1000+) assume many workflows/day — not modeled here.
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0)
    # Staggered reset times — realistic diverse state, not all-fresh. Groq
    # mid-cycle (6h to reset), gemini closer (3h), openrouter near reset
    # (1.5h). This exercises "waste avoidance" — pools approaching reset
    # with unused quota should get consumed, not discarded.
    state.time_bucketed["groq/llama-3.1-70b"] = SimPoolCounter(remaining=15, limit=15, reset_at=21600.0)
    state.time_bucketed["gemini/gemini-1.5-flash"] = SimPoolCounter(remaining=10, limit=10, reset_at=10800.0)
    state.time_bucketed["openrouter/free-mistral"] = SimPoolCounter(remaining=8, limit=8, reset_at=5400.0)
    state.per_call["anthropic/claude-sonnet"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    tasks = _standard_i2p_task_mix()
    return Scenario(
        name="diverse_pool",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers, tasks=tasks),
    )


def exhaustion_sequence() -> Scenario:
    providers = {
        "groq": {"is_free": True, "models": {"groq/llama-3.1-70b": {"cap_score_100": 72}}},
        "gemini": {"is_free": True, "models": {"gemini/gemini-1.5-flash": {"cap_score_100": 68}}},
        "anthropic": {"is_free": False, "models": {"anthropic/claude-sonnet": {"cap_score_100": 93}}},
    }
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0)
    state.time_bucketed["groq/llama-3.1-70b"] = SimPoolCounter(remaining=40, limit=40, reset_at=86400.0)
    state.time_bucketed["gemini/gemini-1.5-flash"] = SimPoolCounter(remaining=40, limit=40, reset_at=86400.0)
    state.per_call["anthropic/claude-sonnet"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    tasks = _standard_i2p_task_mix(count=182)
    return Scenario(
        name="exhaustion_sequence",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers, tasks=tasks),
    )


def back_to_back_i2p() -> Scenario:
    providers_cfg = {
        "groq": {"is_free": True, "models": {"groq/llama-3.1-70b": {"cap_score_100": 72}}},
        "anthropic": {"is_free": False, "models": {"anthropic/claude-sonnet": {"cap_score_100": 93}}},
    }
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0)
    state.time_bucketed["groq/llama-3.1-70b"] = SimPoolCounter(remaining=1500, limit=1500, reset_at=86400.0)
    state.per_call["anthropic/claude-sonnet"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    tasks: list[SimTask] = []
    for seed in (42, 43, 44):
        batch = _standard_i2p_task_mix(count=182, seed=seed)
        for t in batch:
            tasks.append(SimTask(
                idx=len(tasks), difficulty=t.difficulty,
                estimated_output_tokens=t.estimated_output_tokens,
            ))
    return Scenario(
        name="back_to_back_i2p",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers_cfg),
        select_fn=_build_select_fn(providers_cfg, tasks=tasks),
    )


def staggered_i2p() -> Scenario:
    providers_cfg = {
        "groq": {"is_free": True, "models": {"groq/llama-3.1-70b": {"cap_score_100": 72}}},
        "anthropic": {"is_free": False, "models": {"anthropic/claude-sonnet": {"cap_score_100": 93}}},
    }
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0)
    state.time_bucketed["groq/llama-3.1-70b"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    state.per_call["anthropic/claude-sonnet"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)

    first = _standard_i2p_task_mix(count=91, seed=100)
    second = _standard_i2p_task_mix(count=182, seed=101)
    tasks: list[SimTask] = []
    for t in first + second:
        tasks.append(SimTask(
            idx=len(tasks), difficulty=t.difficulty,
            estimated_output_tokens=t.estimated_output_tokens,
        ))
    return Scenario(
        name="staggered_i2p",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers_cfg),
        select_fn=_build_select_fn(providers_cfg, tasks=tasks),
    )
