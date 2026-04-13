# tests/test_load.py
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from nerd_herd.load import LoadManager
from nerd_herd.types import GPUState, ExternalGPUUsage


@pytest.fixture
def gpu_collector():
    mock = MagicMock()
    mock.gpu_state.return_value = GPUState(available=True, vram_free_mb=8000)
    mock.detect_external_gpu_usage.return_value = ExternalGPUUsage()
    return mock


def test_default_mode():
    lm = LoadManager(gpu_collector=MagicMock())
    assert lm.get_load_mode() == "full"


def test_set_mode():
    lm = LoadManager(gpu_collector=MagicMock())
    lm.set_load_mode("shared", source="user")
    assert lm.get_load_mode() == "shared"
    assert not lm.is_auto_managed()


def test_invalid_mode():
    lm = LoadManager(gpu_collector=MagicMock())
    result = lm.set_load_mode("turbo")
    assert "Unknown" in result


def test_vram_budget_fraction():
    lm = LoadManager(gpu_collector=MagicMock())
    assert lm.get_vram_budget_fraction() == 1.0
    lm.set_load_mode("shared")
    assert lm.get_vram_budget_fraction() == 0.5
    lm.set_load_mode("minimal")
    assert lm.get_vram_budget_fraction() == 0.0


def test_vram_budget_mb(gpu_collector):
    lm = LoadManager(gpu_collector=gpu_collector)
    assert lm.get_vram_budget_mb() == 8000  # full mode, 100%
    lm.set_load_mode("shared")
    assert lm.get_vram_budget_mb() == 4000  # 50% of 8000


def test_local_inference_allowed():
    lm = LoadManager(gpu_collector=MagicMock())
    assert lm.is_local_inference_allowed()
    lm.set_load_mode("minimal")
    assert not lm.is_local_inference_allowed()


def test_enable_auto_management():
    lm = LoadManager(gpu_collector=MagicMock())
    lm.set_load_mode("shared", source="user")
    assert not lm.is_auto_managed()
    lm.enable_auto_management()
    assert lm.is_auto_managed()


def test_on_mode_change_callback():
    lm = LoadManager(gpu_collector=MagicMock())
    calls = []
    lm.on_mode_change(lambda old, new, src: calls.append((old, new, src)))
    lm.set_load_mode("heavy", source="auto")
    assert calls == [("full", "heavy", "auto")]


def test_suggest_mode():
    lm = LoadManager(gpu_collector=MagicMock())
    assert lm.suggest_mode_for_external_usage(0.05) == "full"
    assert lm.suggest_mode_for_external_usage(0.20) == "heavy"
    assert lm.suggest_mode_for_external_usage(0.45) == "shared"
    assert lm.suggest_mode_for_external_usage(0.70) == "minimal"


def test_collect_dict():
    lm = LoadManager(gpu_collector=MagicMock())
    result = lm.collect()
    assert result["load_mode"] == "full"
    assert result["vram_budget_fraction"] == 1.0
    assert result["auto_managed"] == 1


def test_no_callback_on_same_mode():
    lm = LoadManager(gpu_collector=MagicMock())
    calls = []
    lm.on_mode_change(lambda old, new, src: calls.append((old, new, src)))
    lm.set_load_mode("full")  # already full
    assert calls == []
