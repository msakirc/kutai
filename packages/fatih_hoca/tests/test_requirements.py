# test_requirements.py
"""Tests for fatih_hoca.requirements module."""

from fatih_hoca.requirements import (
    ModelRequirements,
    AGENT_REQUIREMENTS,
    CAPABILITY_TO_TASK,
    QuotaPlanner,
)
from fatih_hoca.capabilities import TASK_PROFILES


def test_model_requirements_defaults():
    reqs = ModelRequirements(task="coder")
    assert reqs.task == "coder"
    assert reqs.difficulty == 5
    assert reqs.needs_function_calling is False
    assert reqs.priority == 5


def test_model_requirements_effective_task():
    reqs = ModelRequirements(task="coder", primary_capability="code_generation")
    et = reqs.effective_task
    assert isinstance(et, str)


def test_model_requirements_task_profile():
    reqs = ModelRequirements(task="coder")
    profile = reqs.task_profile
    assert isinstance(profile, dict)
    assert len(profile) > 0


def test_model_requirements_escalate():
    reqs = ModelRequirements(task="coder", difficulty=5)
    escalated = reqs.escalate()
    assert escalated.difficulty > reqs.difficulty
    assert escalated.prefer_quality is True


def test_agent_requirements_has_shopping_advisor():
    assert "shopping_advisor" in AGENT_REQUIREMENTS
    reqs = AGENT_REQUIREMENTS["shopping_advisor"]
    assert isinstance(reqs, ModelRequirements)


def test_agent_requirements_has_coder():
    assert "coder" in AGENT_REQUIREMENTS


def test_capability_to_task_mapping():
    assert isinstance(CAPABILITY_TO_TASK, dict)
    assert len(CAPABILITY_TO_TASK) > 0
    for cap, task in CAPABILITY_TO_TASK.items():
        assert isinstance(cap, str)
        assert isinstance(task, str)


def test_quota_planner_initial_threshold():
    qp = QuotaPlanner()
    assert 1 <= qp.expensive_threshold <= 10


def test_quota_planner_recalculate_low_util():
    qp = QuotaPlanner()
    qp.update_paid_utilization("anthropic", 20.0, reset_in=3600)
    threshold = qp.recalculate()
    assert threshold <= 5


def test_quota_planner_recalculate_high_util():
    qp = QuotaPlanner()
    qp.update_paid_utilization("anthropic", 90.0, reset_in=300)
    threshold = qp.recalculate()
    assert threshold >= 7


def test_queue_profile_property_returns_default_on_init():
    from fatih_hoca.requirements import QuotaPlanner, QueueProfile
    planner = QuotaPlanner()
    profile = planner.queue_profile
    assert isinstance(profile, QueueProfile)
    assert profile.total_ready_count == 0
    assert profile.hard_tasks_count == 0


def test_queue_profile_property_reflects_set_value():
    from fatih_hoca.requirements import QuotaPlanner, QueueProfile
    planner = QuotaPlanner()
    profile = QueueProfile(
        total_ready_count=182,
        hard_tasks_count=18,
        by_difficulty={9: 18, 5: 100, 3: 64},
    )
    planner.set_queue_profile(profile)
    assert planner.queue_profile.total_ready_count == 182
    assert planner.queue_profile.hard_tasks_count == 18
    # max difficulty derived from by_difficulty.keys()
    assert planner._max_upcoming_difficulty == 9


def test_queue_profile_unified_type_with_nerd_herd():
    """fatih_hoca.requirements.QueueProfile must be the same type as
    nerd_herd.types.QueueProfile after the 2026-04-29 collapse."""
    from fatih_hoca.requirements import QueueProfile as FHQP
    from nerd_herd.types import QueueProfile as NHQP
    assert FHQP is NHQP, "QueueProfile collapse incomplete — still two types"


def test_quota_planner_reads_cloud_only_from_by_capability():
    """Recalculate must read cloud_only / thinking counts via by_capability,
    not from removed flat needs_*_count fields."""
    from fatih_hoca.requirements import QuotaPlanner, QueueProfile
    planner = QuotaPlanner()
    planner.update_paid_utilization("anthropic", 60.0, reset_in=3600)
    profile = QueueProfile(
        total_ready_count=10,
        by_capability={"cloud_only": 5, "thinking": 0},
    )
    planner.set_queue_profile(profile)
    threshold = planner.recalculate()
    # cloud_only >= 3 forces threshold >= 6
    assert threshold >= 6
