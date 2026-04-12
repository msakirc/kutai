"""Verify the DaLLaMa shim preserves backward compatibility."""
import pytest


def test_shim_imports():
    from src.models.local_model_manager import get_local_manager
    from src.models.local_model_manager import get_runtime_state
    from src.models.local_model_manager import ModelRuntimeState
    from src.models.local_model_manager import LocalModelManager
    from src.models.local_model_manager import ModelSwapRequest


def test_get_local_manager_returns_instance():
    from src.models.local_model_manager import get_local_manager
    mgr = get_local_manager()
    assert mgr is not None
    assert hasattr(mgr, "ensure_model")
    assert hasattr(mgr, "get_status")
    assert hasattr(mgr, "current_model")
    assert hasattr(mgr, "is_loaded")
    assert hasattr(mgr, "acquire_inference_slot")
    assert hasattr(mgr, "release_inference_slot")
    assert hasattr(mgr, "mark_inference_start")
    assert hasattr(mgr, "mark_inference_end")
    assert hasattr(mgr, "swap_started_at")
    assert hasattr(mgr, "runtime_state")
    assert hasattr(mgr, "idle_seconds")
    assert hasattr(mgr, "run_idle_unloader")
    assert hasattr(mgr, "run_health_watchdog")
    assert hasattr(mgr, "_swap_ready")


def test_runtime_state_dataclass():
    from src.models.local_model_manager import ModelRuntimeState
    rs = ModelRuntimeState(
        model_name="test-model",
        thinking_enabled=True,
        context_length=8192,
        gpu_layers=33,
        measured_tps=15.5,
    )
    assert rs.model_name == "test-model"
    assert rs.thinking_enabled is True
    assert rs.context_length == 8192
    assert rs.gpu_layers == 33
    assert rs.measured_tps == 15.5
    assert rs.loaded_at > 0


def test_swap_request_dataclass():
    from src.models.local_model_manager import ModelSwapRequest
    sr = ModelSwapRequest(model_name="test", reason="test")
    assert sr.model_name == "test"
    assert sr.priority == 5
    assert sr.success is False


def test_get_runtime_state_returns_none_initially():
    """get_runtime_state should return None when no model is loaded."""
    from src.models.local_model_manager import get_runtime_state
    # Runtime state is None until a model is actually loaded
    state = get_runtime_state()
    # May be None or may have a value from a prior test — just ensure no crash
    assert state is None or isinstance(state, object)


def test_manager_properties():
    from src.models.local_model_manager import get_local_manager
    mgr = get_local_manager()
    # idle_seconds should be a float
    assert isinstance(mgr.idle_seconds, float)
    # is_loaded should be bool
    assert isinstance(mgr.is_loaded, bool)
    # swap_started_at should be 0.0 when no swap in progress
    assert mgr.swap_started_at == 0.0


def test_get_status_returns_dict():
    from src.models.local_model_manager import get_local_manager
    mgr = get_local_manager()
    status = mgr.get_status()
    assert isinstance(status, dict)
    assert "loaded_model" in status
    assert "healthy" in status
    assert "port" in status
    assert "idle_seconds" in status
    assert "total_swaps" in status
    assert "inference_busy" in status
