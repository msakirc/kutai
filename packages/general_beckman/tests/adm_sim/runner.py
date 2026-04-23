"""Admission-sim runner: wires SimState into general_beckman.next_task()
with minimal dependency mocking. Tick loop drains admissions, records which
tasks get admitted, and returns metrics for per-scenario assertions.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from adm_sim.state import SimState, SimTask


def _build_snapshot(state: SimState) -> Any:
    from nerd_herd.types import (
        CloudModelState,
        CloudProviderState,
        LocalModelState,
        RateLimit,
        RateLimits,
        SystemSnapshot,
    )
    cloud: dict[str, CloudProviderState] = {}
    # Group per-model pools by provider via task fixtures.
    provider_to_models: dict[str, dict[str, tuple[str, bool]]] = {}
    for t in state.tasks:
        provider_to_models.setdefault(t.intended_provider, {})[t.intended_model] = (
            t.intended_model, t.intended_is_free,
        )
    for provider, models in provider_to_models.items():
        mdict: dict[str, CloudModelState] = {}
        for model_id, (_mid, is_free) in models.items():
            pool = state.pool_for(provider, model_id, is_free)
            if pool is None:
                mdict[model_id] = CloudModelState(model_id=model_id, limits=RateLimits())
                continue
            rpd = RateLimit(
                limit=pool.limit,
                remaining=pool.remaining,
                reset_at=int(pool.reset_at) if pool.reset_at else 0,
                in_flight=pool.in_flight,
            )
            mdict[model_id] = CloudModelState(
                model_id=model_id, limits=RateLimits(rpd=rpd),
            )
        cloud[provider] = CloudProviderState(
            provider=provider, limits=RateLimits(), models=mdict,
        )
    return SystemSnapshot(local=LocalModelState(), cloud=cloud)


def _pick_for_task(task: SimTask) -> Any:
    model = SimpleNamespace(
        name=task.intended_model,
        provider=task.intended_provider,
        is_local=False,
        is_free=task.intended_is_free,
    )
    return SimpleNamespace(
        model=model,
        composite=0.7,
        score=0.7,
        min_time_seconds=8.0,
        estimated_load_seconds=0.0,
    )


def _task_dict(task: SimTask) -> dict:
    return {
        "id": task.id,
        "agent_type": task.agent_type,
        "difficulty": task.difficulty,
        "priority": task.priority,
        "created_at": task.created_at,
        "downstream_unblocks_count": task.downstream_unblocks_count,
    }


async def run_ticks(state: SimState, ticks: int = 50) -> dict:
    """Drive next_task() for N ticks. Returns metrics dict.

    Each tick:
      - builds snapshot from current pool state
      - lets admission loop pick a candidate (or None)
      - on admit: decrements pool remaining and bumps in_flight; marks claimed
      - advances virtual clock by 1 second

    No post-completion logic — this sim tests admission gate only.
    """
    import general_beckman as gb

    async def fake_top_k(k: int = 5):
        return [_task_dict(t) for t in state.unclaimed_tasks()[:k]]

    async def fake_claim(task_id: int) -> bool:
        for t in state.tasks:
            if t.id == task_id and not t.claimed:
                t.claimed = True
                state.admits.append(task_id)
                # Charge the pool
                pool = state.pool_for(t.intended_provider, t.intended_model, t.intended_is_free)
                if pool is not None:
                    pool.remaining = max(0, pool.remaining - 1)
                    pool.in_flight += 1
                return True
        return False

    async def fake_fire_due():
        return None

    async def fake_posthook_run():
        return None

    async def fake_snapshot():
        return _build_snapshot(state)

    def fake_select(**kwargs):
        # Match next candidate. The admission loop calls select() once per
        # candidate in priority order — we match by the head of unclaimed.
        # Scenarios configure per-task intended_model so we map by agent_type.
        agent = kwargs.get("agent_type") or kwargs.get("task")
        for t in state.unclaimed_tasks():
            if t.agent_type == agent:
                return _pick_for_task(t)
        return None

    for _ in range(ticks):
        with patch("general_beckman.queue.pick_ready_top_k", new=fake_top_k), \
             patch("general_beckman._claim_task", new=fake_claim), \
             patch("general_beckman.cron.fire_due", new=fake_fire_due), \
             patch("general_beckman.posthook_migration.run", new=fake_posthook_run), \
             patch("nerd_herd.refresh_snapshot", new=fake_snapshot, create=True), \
             patch("fatih_hoca.select", side_effect=fake_select, create=True):
            task = await gb.next_task()
        # Simulate instant completion — admission test only; don't pile up
        # fake in-flight against the hard cap. Real dispatch would take ticks
        # per task, but that orthogonal concern belongs to a ranker sim.
        if task is not None:
            for t in state.tasks:
                if t.id == task["id"]:
                    t.completed = True
                    pool = state.pool_for(t.intended_provider, t.intended_model, t.intended_is_free)
                    if pool is not None and pool.in_flight > 0:
                        pool.in_flight -= 1
                    break
        state.tick += 1

    return {
        "ticks": ticks,
        "admits": len(state.admits),
        "admit_ids": list(state.admits),
        "unclaimed_at_end": len(state.unclaimed_tasks()),
    }
