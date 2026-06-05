"""Unit tests for src.models.introspection (Phase 4 eviction)."""

import types

from src.models import introspection


def test_get_loaded_litellm_name_returns_name(monkeypatch):
    mgr = types.SimpleNamespace(current_model="qwen3.5-9b")
    reg = types.SimpleNamespace(
        get=lambda name: types.SimpleNamespace(litellm_name="openai/qwen3.5-9b")
    )
    monkeypatch.setattr(
        "src.models.local_model_manager.get_local_manager", lambda: mgr
    )
    monkeypatch.setattr("src.models.model_registry.get_registry", lambda: reg)

    assert introspection.get_loaded_litellm_name() == "openai/qwen3.5-9b"


def test_get_loaded_litellm_name_none_when_unloaded(monkeypatch):
    mgr = types.SimpleNamespace(current_model=None)
    monkeypatch.setattr(
        "src.models.local_model_manager.get_local_manager", lambda: mgr
    )
    assert introspection.get_loaded_litellm_name() is None


def test_get_loaded_litellm_name_swallows_errors(monkeypatch):
    def _boom():
        raise RuntimeError("no manager")

    monkeypatch.setattr(
        "src.models.local_model_manager.get_local_manager", _boom
    )
    assert introspection.get_loaded_litellm_name() is None
