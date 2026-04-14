"""Verify shims preserve all existing import paths."""
import pytest


def test_capabilities_imports():
    """All symbols that callers import from src.models.capabilities must work."""
    from src.models.capabilities import (
        Cap,
        ALL_CAPABILITIES,
        KNOWLEDGE_DIMENSIONS,
        REASONING_DIMENSIONS,
        EXECUTION_DIMENSIONS,
        TASK_PROFILES,
        TaskRequirements,
        score_model_for_task,
        rank_models_for_task,
    )
    assert Cap.REASONING is not None
    assert len(ALL_CAPABILITIES) >= 14
    assert "shopping_advisor" in TASK_PROFILES
    assert KNOWLEDGE_DIMENSIONS is not None
    assert REASONING_DIMENSIONS is not None
    assert EXECUTION_DIMENSIONS is not None
    assert callable(score_model_for_task)
    assert callable(rank_models_for_task)


def test_quota_planner_imports():
    """All symbols that callers import from src.models.quota_planner must work."""
    from src.models.quota_planner import QuotaPlanner, QueueProfile, get_quota_planner
    qp = get_quota_planner()
    assert isinstance(qp, QuotaPlanner)
    profile = QueueProfile(total_tasks=5)
    assert profile.total_tasks == 5


def test_registry_imports():
    """All symbols that callers import from src.models.model_registry must work."""
    from src.models.model_registry import (
        ModelInfo,
        ModelRegistry,
        get_registry,
        reload_registry,
        scan_model_directory,
        calculate_dynamic_context,
        calculate_gpu_layers,
        detect_vision_support,
        find_mmproj_path,
        detect_function_calling,
        detect_thinking_model,
        estimate_capabilities,
        read_gguf_metadata,
        KNOWN_PROVIDERS,
        PROVIDER_PREFIXES,
        _FREE_TIER_DEFAULTS,
        _TOOL_CALL_FAMILIES,
        _THINKING_FAMILIES,
        _apply_thinking_deltas,
        _create_model_variants,
        detect_cloud_model,
    )
    assert ModelInfo is not None
    assert ModelRegistry is not None
    assert callable(get_registry)
    assert callable(reload_registry)
    assert isinstance(KNOWN_PROVIDERS, (dict, set))
    assert isinstance(PROVIDER_PREFIXES, dict)
    assert isinstance(_FREE_TIER_DEFAULTS, dict)
    assert isinstance(_TOOL_CALL_FAMILIES, (dict, set))
    assert _THINKING_FAMILIES is not None


def test_registry_singleton():
    """get_registry() returns a usable ModelRegistry instance."""
    from src.models.model_registry import get_registry, ModelRegistry
    reg = get_registry()
    assert isinstance(reg, ModelRegistry)
    # Should have .models dict
    assert hasattr(reg, "models")
    assert isinstance(reg.models, dict)
    # Should have is_demoted()
    assert callable(reg.is_demoted)
    # Should have cloud_models()
    assert callable(reg.cloud_models)


def test_model_profiles_imports():
    """All symbols that callers import from src.models.model_profiles must work."""
    from src.models.model_profiles import (
        FamilyProfile,
        get_sampling_params,
        detect_family,
        get_default_profile,
        get_task_params,
        SAMPLING_KEYS,
    )
    assert FamilyProfile is not None
    assert callable(get_sampling_params)
    assert callable(detect_family)
    assert SAMPLING_KEYS is not None


def test_router_imports():
    """All symbols that callers import from src.core.router must work."""
    from src.core.router import (
        ModelRequirements,
        ModelCallFailed,
        ScoredModel,
        select_model,
        select_for_task,
        call_model,
        get_kdv,
        check_cost_budget,
        CAPABILITY_TO_TASK,
        AGENT_REQUIREMENTS,
        _make_adhoc_profile,
    )
    assert ModelRequirements is not None
    assert ModelCallFailed is not None
    assert ScoredModel is not None
    assert callable(select_model)
    assert callable(select_for_task)
    assert callable(call_model)
    assert callable(get_kdv)
    assert callable(check_cost_budget)
    assert isinstance(CAPABILITY_TO_TASK, dict)
    assert isinstance(AGENT_REQUIREMENTS, dict)
    assert "shopping_advisor" in AGENT_REQUIREMENTS
    assert callable(_make_adhoc_profile)


def test_model_requirements_interface():
    """ModelRequirements from router has the expected interface."""
    from src.core.router import ModelRequirements
    reqs = ModelRequirements(
        task="coder",
        difficulty=6,
        needs_function_calling=True,
        estimated_output_tokens=2000,
    )
    assert reqs.task == "coder"
    assert reqs.difficulty == 6
    assert reqs.needs_function_calling is True
    assert reqs.effective_min_score >= 0
    assert reqs.effective_context_needed > 0


def test_capability_to_task_mapping():
    """CAPABILITY_TO_TASK has all expected entries."""
    from src.core.router import CAPABILITY_TO_TASK
    assert CAPABILITY_TO_TASK["shopping"] == "shopping_advisor"
    assert CAPABILITY_TO_TASK["reasoning"] == "planner"
    assert CAPABILITY_TO_TASK["code_generation"] == "coder"


def test_model_call_failed_exception():
    """ModelCallFailed can be raised and caught correctly."""
    from src.core.router import ModelCallFailed
    exc = ModelCallFailed("test-id", "timeout", "timeout")
    assert exc.call_id == "test-id"
    assert exc.last_error == "timeout"
    assert exc.error_category == "timeout"
    assert isinstance(exc, RuntimeError)
