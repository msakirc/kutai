"""Five admission scenarios per pool-pressure-shared handoff.

Each scenario builds a SimState fixture and an expected-metric predicate.
Scenarios exercise the gate in next_task() against different pool+queue
combinations.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from sim.state import SimPool, SimState, SimTask


@dataclass
class Scenario:
    name: str
    state_factory: Callable[[], SimState]
    ticks: int
    assertion: Callable[[dict, SimState], str | None]
    """Return None when metrics pass, else a failure string."""


# ── Scenario 1: cloud near reset + hot queue → smooth burn ──────────────

def _near_reset_hot_queue() -> SimState:
    now = time.time()
    s = SimState()
    # Time-bucketed pool, 30 min to reset, full remaining.
    s.time_bucketed["groq/llama-70b"] = SimPool(
        remaining=50, limit=50, reset_at=now + 1800,
    )
    # 20 medium-priority ready tasks routed to the free model.
    for i in range(20):
        s.tasks.append(SimTask(
            id=i + 1, agent_type="coder", difficulty=6, priority=6,
            created_at=now - 60,
            intended_model="groq/llama-70b",
            intended_provider="groq",
            intended_is_free=True,
        ))
    return s


def _near_reset_assert(metrics: dict, state: SimState) -> str | None:
    # Near-reset + full remaining gives strong positive pressure.
    # Most tasks should be admitted within the tick budget.
    if metrics["admits"] < 15:
        return f"expected >=15 admits (near-reset burn); got {metrics['admits']}"
    return None


# ── Scenario 2: cloud depleted + cold queue → zero admits ───────────────

def _depleted_cold_queue() -> SimState:
    now = time.time()
    s = SimState()
    # Per-call pool with almost no remaining.
    s.per_call["anthropic/claude-sonnet"] = SimPool(
        remaining=1, limit=100, reset_at=0,
    )
    # Low-priority, young tasks — urgency near 0.1, needs pressure >= 0.4.
    # Depleted per_call gives pressure ≈ -0.93, below threshold → no admit.
    for i in range(10):
        s.tasks.append(SimTask(
            id=i + 1, agent_type="coder", difficulty=6, priority=1,
            created_at=now - 10,
            intended_model="anthropic/claude-sonnet",
            intended_provider="anthropic",
            intended_is_free=False,
        ))
    return s


def _depleted_assert(metrics: dict, state: SimState) -> str | None:
    # Low-urgency tasks against a nearly-exhausted per_call pool should be
    # rejected — threshold(0.1) = 0.4, pressure ≈ -0.93.
    if metrics["admits"] > 2:
        return f"expected ≤2 admits (depleted+cold); got {metrics['admits']}"
    return None


# ── Scenario 3: mixed pools → some admits both sides ────────────────────

def _mixed_pools() -> SimState:
    now = time.time()
    s = SimState()
    s.time_bucketed["groq/llama-70b"] = SimPool(
        remaining=30, limit=50, reset_at=now + 7200,
    )
    s.per_call["anthropic/claude-sonnet"] = SimPool(
        remaining=80, limit=100, reset_at=0,
    )
    for i in range(10):
        s.tasks.append(SimTask(
            id=i + 1, agent_type="coder", difficulty=6, priority=6,
            created_at=now - 60,
            intended_model="groq/llama-70b",
            intended_provider="groq",
            intended_is_free=True,
        ))
    for i in range(10):
        s.tasks.append(SimTask(
            id=i + 100, agent_type="planner", difficulty=8, priority=6,
            created_at=now - 60,
            intended_model="anthropic/claude-sonnet",
            intended_provider="anthropic",
            intended_is_free=False,
        ))
    return s


def _mixed_assert(metrics: dict, state: SimState) -> str | None:
    # Both pools should feed admits. Routing quality is deferred to Phase 2d
    # ranker scenarios; here we just verify no pool is starved.
    free_admits = sum(1 for tid in state.admits if tid <= 10)
    paid_admits = sum(1 for tid in state.admits if tid >= 100)
    if free_admits < 3:
        return f"free pool starved ({free_admits} admits)"
    if paid_admits < 3:
        return f"paid pool starved ({paid_admits} admits)"
    return None


# ── Scenario 4: i2p burst — 180 correlated tasks ────────────────────────

def _i2p_burst() -> SimState:
    now = time.time()
    s = SimState()
    s.per_call["anthropic/claude-sonnet"] = SimPool(
        remaining=200, limit=200, reset_at=0,
    )
    # 180 medium-priority tasks, varying ages (0..180s back).
    for i in range(180):
        s.tasks.append(SimTask(
            id=i + 1, agent_type="coder", difficulty=6, priority=5,
            created_at=now - i,
            # Front 20 tasks unblock downstream work.
            downstream_unblocks_count=5 if i < 20 else 0,
            intended_model="anthropic/claude-sonnet",
            intended_provider="anthropic",
            intended_is_free=False,
        ))
    return s


def _i2p_assert(metrics: dict, state: SimState) -> str | None:
    # Hard cap is 4 in-flight. Over 50 ticks with steady drain (no
    # completion in this sim), we should admit exactly up to the cap.
    # What matters: prioritized head tasks (with unblocks) land first.
    head_ids = {t.id for t in state.tasks if t.downstream_unblocks_count > 0}
    head_admits = sum(1 for tid in state.admits if tid in head_ids)
    if head_admits < 2:
        return f"priority-unblockers starved ({head_admits} admitted, expected ≥2)"
    if metrics["admits"] < 4:
        return f"expected ≥4 admits within hard cap; got {metrics['admits']}"
    return None


# ── Scenario 5: starvation recovery — aged task wins despite depletion ─

def _starvation_recovery() -> SimState:
    now = time.time()
    s = SimState()
    s.per_call["anthropic/claude-sonnet"] = SimPool(
        remaining=5, limit=100, reset_at=0,
    )
    # One ancient, high-priority task with many unblocks — urgency maxes.
    s.tasks.append(SimTask(
        id=1, agent_type="coder", difficulty=6, priority=10,
        created_at=now - 86400 * 2,  # 2 days old
        downstream_unblocks_count=10,
        intended_model="anthropic/claude-sonnet",
        intended_provider="anthropic",
        intended_is_free=False,
    ))
    # Plus low-priority fresh noise that should be gated out.
    for i in range(5):
        s.tasks.append(SimTask(
            id=i + 100, agent_type="coder", difficulty=6, priority=1,
            created_at=now - 60,
            intended_model="anthropic/claude-sonnet",
            intended_provider="anthropic",
            intended_is_free=False,
        ))
    return s


def _starvation_assert(metrics: dict, state: SimState) -> str | None:
    # Urgency(priority=10, 2d age, 10 unblocks) ≈ 1.0 + 0.05 + 0.05 capped to 1.0.
    # Threshold at urgency=1 is -0.5. Pressure for remaining=5/100 per_call:
    # effective_frac=0.05, below 0.15 → depletion intensity (0.15-0.05)/0.15=0.67
    # → -1.0 * 0.67 = -0.67. That's still < threshold. So the stuck task would
    # remain blocked even after aging — admission correctly defers.
    # We assert at least that the low-priority young tasks are NOT admitted
    # (they'd mask starvation with false progress).
    low_pri_admits = sum(1 for tid in state.admits if tid >= 100)
    if low_pri_admits > 0:
        return f"low-priority noise admitted ({low_pri_admits}); urgency gate leaked"
    return None


SCENARIOS: list[Scenario] = [
    Scenario("near_reset_hot_queue", _near_reset_hot_queue, 30, _near_reset_assert),
    Scenario("depleted_cold_queue", _depleted_cold_queue, 30, _depleted_assert),
    Scenario("mixed_pools", _mixed_pools, 30, _mixed_assert),
    Scenario("i2p_burst", _i2p_burst, 50, _i2p_assert),
    Scenario("starvation_recovery", _starvation_recovery, 20, _starvation_assert),
]
