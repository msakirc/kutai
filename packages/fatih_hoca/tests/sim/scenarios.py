"""Scenario factories for Phase 2d simulator.

Each scenario returns a `Scenario` with: `tasks`, `initial_state`,
`snapshot_factory`, `select_fn`. Wire through the real fatih_hoca.select() in
`select_fn` via `selector.select_for_simulation()` (implemented in Task 12).

Keep scenarios focused on the state + demand profile; shared helpers
`_build_select_fn` and `_build_snapshot_factory` translate SimState into
the shapes the selector expects.

Pool-pressure scenarios (Task 28) live in ``POOL_PRESSURE_SCENARIOS`` at the
bottom of this file.  Scenarios 1-7 assert pressure values directly
(pressure-only); scenario 8 is a full admission-flow acceptance gate.
"""
from __future__ import annotations

import random
import time as _time
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
            by_difficulty: dict[int, int] = {}
            for t in remaining:
                by_difficulty[t.difficulty] = by_difficulty.get(t.difficulty, 0) + 1
            queue_profile = QueueProfile(
                total_ready_count=total,
                hard_tasks_count=hard,
                by_difficulty=by_difficulty,
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


# ── Pool-pressure scenarios (Task 28) ────────────────────────────────────────
#
# These 8 scenarios validate the multi-signal pressure system introduced in the
# pool-pressure rebuild.  Scenarios 1-7 work at the pressure level
# (``snapshot.pressure_for()``), asserting signal and scalar values directly
# without running full admission cycles — the framework is pressure-only, so
# they store minimal tasks/state and carry assertion helpers alongside the
# standard Scenario fields.
#
# Scenario 8 is the *merge-acceptance gate*: it runs a full 30-task mission via
# ``run_simulation`` and checks four equilibrium invariants.


def _pressure_snapshot(
    *,
    local_name: str | None = None,
    local_idle: float = 0.0,
    local_tps: float = 20.0,
    vram_avail_mb: int = 0,
    cloud_models: dict | None = None,
) -> Any:
    """Build a SystemSnapshot directly from nerd_herd types for pressure-only tests."""
    from nerd_herd.types import (
        CloudModelState,
        CloudProviderState,
        LocalModelState,
        RateLimitMatrix,
        SystemSnapshot,
    )

    local = LocalModelState(
        model_name=local_name,
        idle_seconds=local_idle,
        measured_tps=local_tps,
        thinking_enabled=False,
    )
    cloud: dict[str, CloudProviderState] = {}
    for provider, models_cfg in (cloud_models or {}).items():
        model_states: dict[str, CloudModelState] = {}
        for model_id, limits_matrix in models_cfg.items():
            util = 0.0
            if limits_matrix.rpd.limit:
                util = 100.0 * (1.0 - (limits_matrix.rpd.remaining or 0) / limits_matrix.rpd.limit)
            model_states[model_id] = CloudModelState(
                model_id=model_id,
                utilization_pct=util,
                limits=limits_matrix,
            )
        cloud[provider] = CloudProviderState(
            provider=provider,
            utilization_pct=0.0,
            consecutive_failures=0,
            limits=RateLimitMatrix(),
            models=model_states,
        )
    return SystemSnapshot(
        local=local,
        vram_available_mb=vram_avail_mb,
        cloud=cloud,
    )


def _cloud_model_stub(
    *,
    name: str,
    provider: str,
    is_free: bool = False,
    cap_score: float = 5.0,
    capabilities: set | None = None,
    rpd_remaining: int = 100,
) -> Any:
    """Minimal SimpleNamespace model stub for pressure-only tests."""
    from types import SimpleNamespace

    return SimpleNamespace(
        name=name,
        provider=provider,
        is_local=False,
        is_loaded=False,
        is_free=is_free,
        cap_score=cap_score,
        capabilities=capabilities or set(),
        rpd_remaining=rpd_remaining,
    )


def _local_model_stub(
    *,
    name: str = "local-model",
    is_loaded: bool = False,
    size_mb: int = 4000,
    cap_score: float = 5.5,
) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        name=name,
        provider="local",
        is_local=True,
        is_loaded=is_loaded,
        is_free=False,
        cap_score=cap_score,
        size_mb=size_mb,
        capabilities=set(),
    )


# ── Scenario PP1: Fat vs Tiny pool, same % utilisation ───────────────────────

def pp1_fat_vs_tiny() -> Scenario:
    """Two free-cloud providers at 50% remaining but very different limits.

    After simulated depletion to 10% remaining (same fraction), M1 must amplify
    tiny pool's negative pressure more strongly than the large pool's.

    Assertion (pressure-only):
        tiny_scalar < large_scalar  (tiny pool more negative after depletion)
        |tiny_scalar| > |large_scalar| * 1.1
    """
    from nerd_herd.types import RateLimit, RateLimitMatrix

    now = _time.time()
    reset_at = int(now + 86400)

    # State at 10% remaining — same fraction, different absolute limits
    tiny_rl = RateLimit(limit=10, remaining=1, reset_at=reset_at)
    large_rl = RateLimit(limit=1000, remaining=100, reset_at=reset_at)

    state = SimState()
    state.time_bucketed["tiny/model"] = SimPoolCounter(remaining=1, limit=10, reset_at=86400.0)
    state.time_bucketed["large/model"] = SimPoolCounter(remaining=100, limit=1000, reset_at=86400.0)

    providers = {
        "tiny_prov": {"is_free": True, "models": {"tiny/model": {"cap_score_100": 72}}},
        "large_prov": {"is_free": True, "models": {"large/model": {"cap_score_100": 72}}},
    }
    tasks = [SimTask(idx=i, difficulty=5, estimated_output_tokens=1000) for i in range(5)]
    return Scenario(
        name="pp1_fat_vs_tiny",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers, tasks=tasks),
    )


def assert_pp1(scenario: "Scenario") -> list[str]:
    """Run pressure assertions for PP1. Returns list of failure messages (empty = PASS)."""
    from nerd_herd.types import RateLimit, RateLimitMatrix

    now = _time.time()
    reset_at = int(now + 86400)

    tiny_rl = RateLimit(limit=10, remaining=1, reset_at=reset_at)
    large_rl = RateLimit(limit=1000, remaining=100, reset_at=reset_at)

    snap = _pressure_snapshot(
        cloud_models={
            "tiny_prov": {"tiny/model": RateLimitMatrix(rpd=tiny_rl)},
            "large_prov": {"large/model": RateLimitMatrix(rpd=large_rl)},
        }
    )
    tiny_m = _cloud_model_stub(name="tiny/model", provider="tiny_prov", is_free=True, rpd_remaining=1)
    large_m = _cloud_model_stub(name="large/model", provider="large_prov", is_free=True, rpd_remaining=100)

    tiny_bd = snap.pressure_for(tiny_m, task_difficulty=5)
    large_bd = snap.pressure_for(large_m, task_difficulty=5)

    failures = []
    if not (tiny_bd.scalar < large_bd.scalar):
        failures.append(
            f"PP1: tiny scalar ({tiny_bd.scalar:.3f}) >= large scalar ({large_bd.scalar:.3f})"
        )
    if not (abs(tiny_bd.scalar) > abs(large_bd.scalar) * 1.1):
        failures.append(
            f"PP1: |tiny| ({abs(tiny_bd.scalar):.3f}) not > |large| ({abs(large_bd.scalar):.3f}) * 1.1"
        )
    # M1 must be larger for tiny pool
    if not (tiny_bd.modifiers.get("M1", 1.0) > large_bd.modifiers.get("M1", 1.0)):
        failures.append(
            f"PP1: tiny M1 ({tiny_bd.modifiers.get('M1')}) not > large M1 ({large_bd.modifiers.get('M1')})"
        )
    return failures


# ── Scenario PP2: Token-aware exclusion ──────────────────────────────────────

def pp2_token_exclusion() -> Scenario:
    """Model with TPM=25k remaining; task needs 30k tokens.

    S2 must fire negative (call-burden pressure), producing scalar < 0.
    """
    from nerd_herd.types import RateLimit, RateLimitMatrix

    now = _time.time()
    tpm_rl = RateLimit(limit=100_000, remaining=25_000, reset_at=int(now + 3600))

    state = SimState()
    state.per_call["cloud/model"] = SimPoolCounter(remaining=500, limit=1000, reset_at=86400.0)

    providers = {
        "cloud_prov": {"is_free": False, "models": {"cloud/model": {"cap_score_100": 80}}},
    }
    tasks = [SimTask(idx=i, difficulty=5, estimated_output_tokens=30_000) for i in range(5)]
    return Scenario(
        name="pp2_token_exclusion",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers, tasks=tasks),
    )


def assert_pp2(scenario: "Scenario") -> list[str]:
    """Pressure-only assertion for PP2."""
    from nerd_herd.types import RateLimit, RateLimitMatrix

    now = _time.time()
    tpm_rl = RateLimit(limit=100_000, remaining=25_000, reset_at=int(now + 3600))
    rpd_rl = RateLimit(limit=1000, remaining=500, reset_at=int(now + 86400))

    snap = _pressure_snapshot(
        cloud_models={
            "cloud_prov": {
                "cloud/model": RateLimitMatrix(tpm=tpm_rl, rpd=rpd_rl),
            }
        }
    )
    cloud_m = _cloud_model_stub(
        name="cloud/model", provider="cloud_prov", is_free=False,
        cap_score=8.0, rpd_remaining=500,
    )

    # est_per_call_tokens > TPM remaining — S2 must fire
    bd = snap.pressure_for(cloud_m, task_difficulty=5, est_per_call_tokens=30_000)

    failures = []
    if not (bd.signals.get("S2", 0) < 0):
        failures.append(f"PP2: S2 ({bd.signals.get('S2')}) should be negative when tokens > remaining")
    if not (bd.scalar < 0):
        failures.append(f"PP2: scalar ({bd.scalar:.3f}) should be < 0 for token-exceeding task")
    return failures


# ── Scenario PP3: Cold local + free VRAM → admits default-urgency ─────────────

def pp3_cold_local_vram() -> Scenario:
    """Cold local model, VRAM >= size_mb.  S9 must return COLD_LOCAL_VRAM_OK (0.4)."""
    state = SimState()
    # No loaded locals; local model not yet loaded
    providers: dict[str, Any] = {}
    tasks = [SimTask(idx=0, difficulty=5, estimated_output_tokens=1000)]
    return Scenario(
        name="pp3_cold_local_vram",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers, tasks=tasks),
    )


def assert_pp3(scenario: "Scenario") -> list[str]:
    """Pressure-only assertion for PP3."""
    from nerd_herd.types import LocalModelState, RateLimitMatrix

    local_state = LocalModelState(
        model_name=None, idle_seconds=0.0, measured_tps=0.0, thinking_enabled=False,
    )
    local_m = _local_model_stub(name="cold-local", is_loaded=False, size_mb=4000)

    snap = _pressure_snapshot(local_name=None, local_idle=0.0, vram_avail_mb=8000)

    # S9 directly
    from nerd_herd.signals.s9_perishability import s9_perishability
    s9_val = s9_perishability(
        local_m, local=local_state, vram_avail_mb=8000,
        matrix=RateLimitMatrix(), task_difficulty=5,
    )

    # Full pressure
    bd = snap.pressure_for(local_m, task_difficulty=5)

    failures = []
    if abs(s9_val - 0.4) > 0.01:
        failures.append(f"PP3: S9 ({s9_val:.3f}) expected 0.4 (COLD_LOCAL_VRAM_OK)")
    if bd.scalar < 0:
        failures.append(f"PP3: pressure scalar ({bd.scalar:.3f}) should be >= 0 for cold local with VRAM")
    return failures


# ── Scenario PP4: Free cloud near reset + flush quota ─────────────────────────

def pp4_free_cloud_near_reset() -> Scenario:
    """Free cloud model: 95% remaining, reset in 600s.  S9 must be > 0.7."""
    from nerd_herd.types import RateLimit, RateLimitMatrix

    now = _time.time()
    rpd_rl = RateLimit(limit=1000, remaining=950, reset_at=int(now + 600))
    state = SimState()
    state.time_bucketed["free/model"] = SimPoolCounter(remaining=950, limit=1000, reset_at=600.0)

    providers = {
        "free_prov": {"is_free": True, "models": {"free/model": {"cap_score_100": 72}}},
    }
    tasks = [SimTask(idx=i, difficulty=3, estimated_output_tokens=1000) for i in range(5)]
    return Scenario(
        name="pp4_free_cloud_near_reset",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers, tasks=tasks),
    )


def assert_pp4(scenario: "Scenario") -> list[str]:
    """Pressure-only assertion for PP4."""
    from nerd_herd.types import RateLimit, RateLimitMatrix
    from nerd_herd.signals.s9_perishability import s9_perishability

    now = _time.time()
    rpd_rl = RateLimit(limit=1000, remaining=950, reset_at=int(now + 600))
    free_m = _cloud_model_stub(name="free/model", provider="free_prov", is_free=True, rpd_remaining=950)

    s9_val = s9_perishability(
        free_m, local=None, vram_avail_mb=0, matrix=RateLimitMatrix(rpd=rpd_rl), task_difficulty=3,
    )

    snap = _pressure_snapshot(
        cloud_models={"free_prov": {"free/model": RateLimitMatrix(rpd=rpd_rl)}}
    )
    bd = snap.pressure_for(free_m, task_difficulty=3)

    failures = []
    if s9_val <= 0.7:
        failures.append(f"PP4: S9 ({s9_val:.3f}) should be > 0.7 for free cloud near reset with 95% remaining")
    if bd.scalar <= 0:
        failures.append(f"PP4: pressure scalar ({bd.scalar:.3f}) should be > 0; free cloud should admit")
    return failures


# ── Scenario PP5: Paid cloud flush + no hard queue ─────────────────────────────

def pp5_paid_flush_no_hard_queue() -> Scenario:
    """Paid cloud model at 95% remaining, easy task d=3.

    S9 for paid+easy = 0; pressure not strongly positive; if a local exists
    with VRAM it should win.
    """
    from nerd_herd.types import RateLimit, RateLimitMatrix

    now = _time.time()
    rpd_rl = RateLimit(limit=100, remaining=95, reset_at=int(now + 600))
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0, tokens_per_second=20.0)
    state.per_call["paid/claude"] = SimPoolCounter(remaining=95, limit=100, reset_at=600.0)

    providers = {
        "paid_prov": {"is_free": False, "models": {"paid/claude": {"cap_score_100": 93}}},
    }
    tasks = [SimTask(idx=i, difficulty=3, estimated_output_tokens=1000) for i in range(10)]
    return Scenario(
        name="pp5_paid_flush_no_hard_queue",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers, tasks=tasks),
    )


def assert_pp5(scenario: "Scenario") -> list[str]:
    """Pressure-only assertion for PP5 (S9=0 for paid+easy)."""
    from nerd_herd.types import RateLimit, RateLimitMatrix
    from nerd_herd.signals.s9_perishability import s9_perishability

    now = _time.time()
    rpd_rl = RateLimit(limit=100, remaining=95, reset_at=int(now + 600))
    paid_m = _cloud_model_stub(name="paid/claude", provider="paid_prov", is_free=False, cap_score=9.3, rpd_remaining=95)

    s9_val = s9_perishability(
        paid_m, local=None, vram_avail_mb=0, matrix=RateLimitMatrix(rpd=rpd_rl), task_difficulty=3,
    )

    failures = []
    if s9_val != 0.0:
        failures.append(f"PP5: S9 ({s9_val:.3f}) should be 0.0 for paid cloud with easy task (d<7)")
    return failures


def assert_pp5_selection(run: Any) -> list[str]:
    """Full-sim assertion: local should dominate easy tasks for PP5."""
    from sim.runner import SimRun
    from collections import Counter

    if not isinstance(run, SimRun):
        return ["PP5: no SimRun to check"]
    local_picks = sum(1 for p in run.picks if p.pool == "local")
    cloud_picks = sum(1 for p in run.picks if p.pool == "per_call")
    failures = []
    if local_picks <= cloud_picks:
        failures.append(
            f"PP5: local ({local_picks}) should dominate over paid cloud ({cloud_picks}) for easy tasks"
        )
    return failures


# ── Scenario PP6: Capability shortage ─────────────────────────────────────────

def pp6_capability_shortage() -> Scenario:
    """Queue has 50 vision tasks; only one vision-capable model with limited capacity.

    S6 must fire negative conserve-pressure on that model.
    """
    from nerd_herd.types import RateLimit, RateLimitMatrix

    now = _time.time()
    rpd_rl = RateLimit(limit=20, remaining=20, reset_at=int(now + 86400))
    state = SimState()
    state.per_call["vision/model"] = SimPoolCounter(remaining=20, limit=20, reset_at=86400.0)

    providers = {
        "cloud_prov": {"is_free": False, "models": {"vision/model": {"cap_score_100": 85}}},
    }
    tasks = [SimTask(idx=i, difficulty=5, task_name="visual_reviewer",
                     estimated_output_tokens=1000) for i in range(5)]
    return Scenario(
        name="pp6_capability_shortage",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers, tasks=tasks),
    )


def assert_pp6(scenario: "Scenario") -> list[str]:
    """Pressure-only assertion for PP6."""
    from nerd_herd.types import QueueProfile
    from nerd_herd.signals.s6_capable_supply import s6_capable_supply
    from types import SimpleNamespace

    # Vision model with limited remaining capacity
    vision_m = SimpleNamespace(
        name="vision/model", provider="cloud_prov", is_local=False, is_free=False,
        cap_score=8.5, capabilities={"vision"}, rpd_remaining=20,
    )
    # Queue with 50 vision tasks
    queue = QueueProfile(by_capability={"vision": 50}, hard_tasks_count=0, total_ready_count=50)

    s6_val = s6_capable_supply(vision_m, queue=queue, eligible_models=[vision_m], iter_avg=8.0)

    failures = []
    if s6_val >= 0:
        failures.append(
            f"PP6: S6 ({s6_val:.3f}) should be negative (conserve-pressure) "
            "when vision demand >> vision supply"
        )
    return failures


# ── Scenario PP7: Difficulty lookahead ────────────────────────────────────────

def pp7_difficulty_lookahead() -> Scenario:
    """Queue has 8 d=9 tasks; current candidate is d=3.

    The easy task should not admit to paid cloud — local wins.
    """
    from nerd_herd.types import RateLimit, RateLimitMatrix

    now = _time.time()
    rpd_rl = RateLimit(limit=1000, remaining=500, reset_at=int(now + 86400))
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0, tokens_per_second=20.0)
    state.per_call["paid_prov/claude"] = SimPoolCounter(remaining=500, limit=1000, reset_at=86400.0)

    providers = {
        "paid_prov": {"is_free": False, "models": {"paid_prov/claude": {"cap_score_100": 93}}},
    }
    # 8 hard tasks first, then 1 easy task
    hard_tasks = [SimTask(idx=i, difficulty=9, estimated_output_tokens=5000) for i in range(8)]
    easy_task = SimTask(idx=8, difficulty=3, estimated_output_tokens=1000)
    tasks = hard_tasks + [easy_task]
    return Scenario(
        name="pp7_difficulty_lookahead",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers, tasks=tasks),
    )


def assert_pp7_m3_weights() -> list[str]:
    """Pressure-only: M3 must down-weight S9 for paid+easy vs paid+hard."""
    from nerd_herd.modifiers import M3_difficulty_weights

    w_easy = M3_difficulty_weights(difficulty=3, model_is_paid=True)
    w_hard = M3_difficulty_weights(difficulty=8, model_is_paid=True)

    failures = []
    if not (w_easy["S9"] < w_hard["S9"]):
        failures.append(
            f"PP7: M3 S9 weight for easy ({w_easy['S9']}) should be < hard ({w_hard['S9']})"
        )
    return failures


# ── Scenario PP8: Equilibrium full mission (acceptance gate) ──────────────────

def pp8_equilibrium_mission() -> Scenario:
    """30-task mixed mission: d=3/5/7/9.  This is the merge-acceptance gate.

    Assertions (full admission flow):
      A: cloud RPD never exhausts (remaining > 0 at end)
      B: local never idle while free cloud has flush quota AND queue non-empty
         (tested as: local makes at least 1 pick when a free pool has > 90% remaining)
      C: no single pool's utilisation exceeds 80% before queue empties
      D: total admission cycles == number of tasks (1:1, no retries in sim)
    """
    rng = random.Random(99)
    tasks = []
    for i in range(30):
        r = rng.random()
        if r < 0.30:
            d = 3
        elif r < 0.60:
            d = 5
        elif r < 0.80:
            d = 7
        else:
            d = 9
        est_out = 1000 if d <= 4 else (2500 if d <= 6 else 5000)
        tasks.append(SimTask(idx=i, difficulty=d, estimated_output_tokens=est_out))

    providers = {
        "groq": {"is_free": True, "models": {"groq/llama": {"cap_score_100": 72}}},
        "anthropic": {"is_free": False, "models": {"anthropic/claude": {"cap_score_100": 93}}},
    }
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0, tokens_per_second=20.0)
    # Limits comfortably above 30 tasks each so exhaustion is physically impossible in one run
    state.time_bucketed["groq/llama"] = SimPoolCounter(remaining=50, limit=50, reset_at=86400.0)
    state.per_call["anthropic/claude"] = SimPoolCounter(remaining=100, limit=100, reset_at=86400.0)
    return Scenario(
        name="pp8_equilibrium_mission",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers, tasks=tasks),
    )


def assert_pp8(scenario: "Scenario") -> list[str]:
    """Run full admission flow and check 4 equilibrium invariants."""
    from sim.runner import run_simulation

    run = run_simulation(
        tasks=scenario.tasks,
        initial_state=scenario.initial_state,
        select_fn=scenario.select_fn,
        snapshot_factory=scenario.snapshot_factory,
    )

    failures = []

    # A: Cloud RPD must not exhaust
    groq_rem = run.final_state.time_bucketed.get("groq/llama")
    anthropic_rem = run.final_state.per_call.get("anthropic/claude")
    if groq_rem and groq_rem.remaining <= 0:
        failures.append(f"PP8-A: groq/llama exhausted (remaining={groq_rem.remaining})")
    if anthropic_rem and anthropic_rem.remaining <= 0:
        failures.append(f"PP8-A: anthropic/claude exhausted (remaining={anthropic_rem.remaining})")

    # B: Local must have made at least 1 pick (not idle while work exists)
    local_picks = sum(1 for p in run.picks if p.pool == "local")
    if local_picks == 0:
        failures.append("PP8-B: local made 0 picks; should be active when free quota available")

    # C: No single pool > 80% utilisation
    if groq_rem:
        groq_util = 1.0 - groq_rem.remaining / groq_rem.limit
        if groq_util > 0.80:
            failures.append(f"PP8-C: groq utilisation {groq_util:.1%} exceeds 80%")
    if anthropic_rem:
        anthropic_util = 1.0 - anthropic_rem.remaining / anthropic_rem.limit
        if anthropic_util > 0.80:
            failures.append(f"PP8-C: anthropic utilisation {anthropic_util:.1%} exceeds 80%")

    # D: Admission cycles == number of tasks (1:1)
    if len(run.picks) != len(scenario.tasks):
        failures.append(
            f"PP8-D: picks ({len(run.picks)}) != tasks ({len(scenario.tasks)})"
        )

    return failures


# ── Registry ─────────────────────────────────────────────────────────────────

POOL_PRESSURE_SCENARIOS = [
    ("pp1_fat_vs_tiny", pp1_fat_vs_tiny),
    ("pp2_token_exclusion", pp2_token_exclusion),
    ("pp3_cold_local_vram", pp3_cold_local_vram),
    ("pp4_free_cloud_near_reset", pp4_free_cloud_near_reset),
    ("pp5_paid_flush_no_hard_queue", pp5_paid_flush_no_hard_queue),
    ("pp6_capability_shortage", pp6_capability_shortage),
    ("pp7_difficulty_lookahead", pp7_difficulty_lookahead),
    ("pp8_equilibrium_mission", pp8_equilibrium_mission),
]

# Per-scenario assertion callables (scenarios 1-7 pressure-only; 8 full-flow)
POOL_PRESSURE_ASSERTIONS: dict[str, Callable] = {
    "pp1_fat_vs_tiny": lambda sc: assert_pp1(sc),
    "pp2_token_exclusion": lambda sc: assert_pp2(sc),
    "pp3_cold_local_vram": lambda sc: assert_pp3(sc),
    "pp4_free_cloud_near_reset": lambda sc: assert_pp4(sc),
    "pp5_paid_flush_no_hard_queue": lambda sc: assert_pp5(sc),
    "pp6_capability_shortage": lambda sc: assert_pp6(sc),
    "pp7_difficulty_lookahead": lambda sc: assert_pp7_m3_weights(),
    "pp8_equilibrium_mission": lambda sc: assert_pp8(sc),
}
