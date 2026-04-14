"""Tests for fatih_hoca.capabilities — Cap enum, task profiles, and scoring."""

import pytest
from fatih_hoca.capabilities import (
    ALL_CAPABILITIES,
    Cap,
    TASK_PROFILES,
    TaskRequirements,
    rank_models_for_task,
    score_model_for_task,
)


# ─── Cap enum ────────────────────────────────────────────────────────────────

def test_cap_enum_exists():
    assert Cap is not None


def test_cap_reasoning_value():
    assert Cap.REASONING == "reasoning"
    assert Cap.REASONING.value == "reasoning"


def test_cap_is_str_enum():
    # Cap inherits from str — enum values usable as dict keys directly
    assert isinstance(Cap.REASONING, str)


def test_all_capabilities_length():
    assert len(ALL_CAPABILITIES) >= 14


def test_all_capabilities_contains_reasoning():
    assert "reasoning" in ALL_CAPABILITIES


def test_all_capabilities_strings():
    for cap in ALL_CAPABILITIES:
        assert isinstance(cap, str)


# ─── Task profiles ───────────────────────────────────────────────────────────

def test_task_profiles_coder_exists():
    assert "coder" in TASK_PROFILES


def test_task_profiles_coder_code_generation():
    profile = TASK_PROFILES["coder"]
    # code_generation key may be a Cap enum or string
    val = profile.get(Cap.CODE_GENERATION) or profile.get("code_generation", 0)
    assert val > 0.5


def test_task_profiles_shopping_advisor_exists():
    assert "shopping_advisor" in TASK_PROFILES


def test_task_profiles_shopping_advisor_has_turkish():
    profile = TASK_PROFILES["shopping_advisor"]
    val = profile.get(Cap.TURKISH) or profile.get("turkish", 0)
    assert val > 0.5


def test_task_profiles_all_weights_in_range():
    for task_name, profile in TASK_PROFILES.items():
        for cap, weight in profile.items():
            assert 0.0 <= weight <= 1.0, (
                f"Task '{task_name}', cap '{cap}': weight {weight} out of [0,1]"
            )


def test_task_profiles_known_tasks():
    expected = {"planner", "coder", "researcher", "shopping_advisor", "analyst"}
    assert expected.issubset(set(TASK_PROFILES.keys()))


# ─── TaskRequirements dataclass ───────────────────────────────────────────────

def test_task_requirements_defaults():
    req = TaskRequirements(task_name="coder")
    assert req.task_name == "coder"
    assert req.min_context == 0
    assert req.needs_function_calling is False
    assert req.needs_vision is False
    assert req.needs_thinking is False
    assert req.prefer_local is False
    assert req.prefer_fast is False
    assert req.latency_sensitive is False
    assert req.min_capability is None


def test_task_requirements_custom():
    req = TaskRequirements(
        task_name="executor",
        min_context=4096,
        needs_function_calling=True,
        needs_json_mode=True,
    )
    assert req.min_context == 4096
    assert req.needs_function_calling is True
    assert req.needs_json_mode is True


# ─── score_model_for_task ─────────────────────────────────────────────────────

def _strong_coder_caps() -> dict:
    """Capability vector for a strong coder model (0-10 scale)."""
    return {
        "reasoning": 8.0,
        "planning": 7.0,
        "analysis": 7.0,
        "code_generation": 9.5,
        "code_reasoning": 8.5,
        "system_design": 7.0,
        "prose_quality": 7.0,
        "instruction_adherence": 8.5,
        "domain_knowledge": 8.0,
        "context_utilization": 8.0,
        "structured_output": 8.0,
        "tool_use": 8.0,
        "vision": 0.0,
        "conversation": 7.0,
        "turkish": 3.0,
    }


def _basic_ops() -> dict:
    return {
        "supports_function_calling": True,
        "supports_json_mode": True,
        "context_length": 32768,
        "cost_per_1k_output": 0.0,
        "location": "local",
        "tokens_per_second": 60,
    }


def test_score_good_match_returns_positive():
    caps = _strong_coder_caps()
    ops = _basic_ops()
    req = TaskRequirements(task_name="coder")
    score = score_model_for_task(caps, ops, req)
    assert score > 0
    assert score <= 10.0


def test_score_strong_coder_for_coder_task():
    caps = _strong_coder_caps()
    ops = _basic_ops()
    req = TaskRequirements(task_name="coder")
    score = score_model_for_task(caps, ops, req)
    # A model with 9.5 code_generation should score well above midpoint
    assert score > 5.0


def test_score_hard_reject_no_function_calling():
    caps = _strong_coder_caps()
    ops = {**_basic_ops(), "supports_function_calling": False}
    req = TaskRequirements(task_name="executor", needs_function_calling=True)
    score = score_model_for_task(caps, ops, req)
    assert score == -1.0


def test_score_hard_reject_context_too_small():
    caps = _strong_coder_caps()
    ops = {**_basic_ops(), "context_length": 2048}
    req = TaskRequirements(task_name="coder", min_context=8192)
    score = score_model_for_task(caps, ops, req)
    assert score == -1.0


def test_score_hard_reject_vision_cap_too_low():
    caps = {**_strong_coder_caps(), "vision": 0.0}
    ops = _basic_ops()
    req = TaskRequirements(task_name="visual_reviewer", needs_vision=True)
    score = score_model_for_task(caps, ops, req)
    assert score == -1.0


def test_score_hard_reject_min_capability():
    caps = {**_strong_coder_caps(), "code_generation": 3.0}
    ops = _basic_ops()
    req = TaskRequirements(task_name="coder", min_capability={"code_generation": 8.0})
    score = score_model_for_task(caps, ops, req)
    assert score == -1.0


def test_score_vision_variant_excluded_from_non_vision_task():
    caps = _strong_coder_caps()
    ops = {**_basic_ops(), "variant_flags": {"vision"}}
    req = TaskRequirements(task_name="coder", needs_vision=False)
    score = score_model_for_task(caps, ops, req)
    assert score == -1.0


def test_score_unknown_task_uses_flat_weights():
    caps = _strong_coder_caps()
    ops = _basic_ops()
    req = TaskRequirements(task_name="nonexistent_task_xyz")
    score = score_model_for_task(caps, ops, req)
    assert score >= 0


def test_score_thinking_model_bonus():
    caps = _strong_coder_caps()
    ops_thinking = {**_basic_ops(), "thinking_model": True}
    ops_plain = {**_basic_ops(), "thinking_model": False}
    req = TaskRequirements(task_name="planner", needs_thinking=True)
    score_thinking = score_model_for_task(caps, ops_thinking, req)
    score_plain = score_model_for_task(caps, ops_plain, req)
    assert score_thinking > score_plain


def test_score_prefer_local_bonus():
    caps = _strong_coder_caps()
    ops_local = {**_basic_ops(), "location": "local"}
    ops_cloud = {**_basic_ops(), "location": "cloud"}
    req = TaskRequirements(task_name="coder", prefer_local=True)
    score_local = score_model_for_task(caps, ops_local, req)
    score_cloud = score_model_for_task(caps, ops_cloud, req)
    assert score_local > score_cloud


def test_score_cost_rejection():
    caps = _strong_coder_caps()
    ops = {**_basic_ops(), "cost_per_1k_output": 0.20}
    req = TaskRequirements(task_name="coder", max_cost_per_1k_output=0.10)
    score = score_model_for_task(caps, ops, req)
    assert score == -1.0


# ─── rank_models_for_task ─────────────────────────────────────────────────────

def test_rank_models_returns_sorted():
    weak_caps = {k: 2.0 for k in ALL_CAPABILITIES}
    strong_caps = {k: 9.0 for k in ALL_CAPABILITIES}
    models = {
        "weak": (weak_caps, _basic_ops()),
        "strong": (strong_caps, _basic_ops()),
    }
    req = TaskRequirements(task_name="coder")
    ranked = rank_models_for_task(models, req)
    assert ranked[0][0] == "strong"
    assert ranked[0][1] > ranked[1][1]


def test_rank_models_excludes_rejects():
    caps = _strong_coder_caps()
    ops_ok = _basic_ops()
    ops_bad = {**_basic_ops(), "supports_function_calling": False}
    models = {
        "good": (caps, ops_ok),
        "bad": (caps, ops_bad),
    }
    req = TaskRequirements(task_name="executor", needs_function_calling=True)
    ranked = rank_models_for_task(models, req)
    names = [name for name, _ in ranked]
    assert "good" in names
    assert "bad" not in names


def test_rank_models_top_k():
    models = {f"model_{i}": ({k: float(i) for k in ALL_CAPABILITIES}, _basic_ops()) for i in range(10)}
    req = TaskRequirements(task_name="coder")
    ranked = rank_models_for_task(models, req, top_k=3)
    assert len(ranked) <= 3
