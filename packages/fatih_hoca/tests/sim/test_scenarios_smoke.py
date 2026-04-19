"""Smoke tests for Phase 2d scenario factories -- verify they construct cleanly.

Task 12 adds the full pass-criteria scenario tests once `select_for_simulation`
is implemented.
"""
from sim.scenarios import (
    Scenario, baseline, claude_constrained, groq_near_reset,
    diverse_pool, exhaustion_sequence, back_to_back_i2p, staggered_i2p,
)


SCENARIOS = [baseline, claude_constrained, groq_near_reset, diverse_pool,
             exhaustion_sequence, back_to_back_i2p, staggered_i2p]


def test_all_scenarios_construct():
    for factory in SCENARIOS:
        scen = factory()
        assert isinstance(scen, Scenario)
        assert scen.name
        assert scen.tasks  # non-empty


def test_back_to_back_has_three_i2p_worth_of_tasks():
    scen = back_to_back_i2p()
    assert len(scen.tasks) == 546  # 3 x 182


def test_staggered_has_91_plus_182():
    scen = staggered_i2p()
    assert len(scen.tasks) == 273


def test_baseline_task_mix_reasonable():
    scen = baseline()
    counts = {d: 0 for d in range(1, 11)}
    for t in scen.tasks:
        counts[t.difficulty] += 1
    # At least some easy and some hard
    assert sum(counts[d] for d in (1, 2, 3)) > 0
    assert sum(counts[d] for d in (7, 8, 9, 10)) > 0
