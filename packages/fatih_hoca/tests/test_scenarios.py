"""Scenario validation for Phase 2d (spec §6, §7)."""
import pytest

from sim.report import compute_metrics
from sim.runner import run_simulation
from sim.scenarios import (
    baseline, claude_constrained, groq_near_reset, diverse_pool,
    exhaustion_sequence, back_to_back_i2p, staggered_i2p,
)


SCENARIOS = [
    ("baseline", baseline),
    ("claude_constrained", claude_constrained),
    ("groq_near_reset", groq_near_reset),
    ("diverse_pool", diverse_pool),
    ("exhaustion_sequence", exhaustion_sequence),
    ("back_to_back_i2p", back_to_back_i2p),
    ("staggered_i2p", staggered_i2p),
]


def _run(scenario):
    return run_simulation(
        tasks=scenario.tasks,
        initial_state=scenario.initial_state,
        select_fn=scenario.select_fn,
        snapshot_factory=scenario.snapshot_factory,
    )


@pytest.mark.parametrize("name,factory", SCENARIOS)
def test_scenario_hard_task_satisfaction(name, factory):
    scenario = factory()
    run = _run(scenario)
    m = compute_metrics(run)
    assert m.hard_task_satisfaction >= 0.90, (
        f"{name}: hard-task satisfaction {m.hard_task_satisfaction:.2%} < 90%"
    )


@pytest.mark.parametrize("name,factory", SCENARIOS)
def test_scenario_easy_task_waste(name, factory):
    scenario = factory()
    run = _run(scenario)
    m = compute_metrics(run)
    assert m.easy_task_waste < 0.10, (
        f"{name}: easy-task waste {m.easy_task_waste:.2%} >= 10%"
    )


def test_diverse_pool_free_quota_utilization():
    scenario = diverse_pool()
    run = _run(scenario)
    m = compute_metrics(run)
    assert m.free_quota_utilization > 0.70, (
        f"diverse_pool: free_quota_utilization {m.free_quota_utilization:.2%} <= 70%"
    )


def test_exhaustion_sequence_no_crashes():
    scenario = exhaustion_sequence()
    run = _run(scenario)
    assert len(run.picks) == len(scenario.tasks)
