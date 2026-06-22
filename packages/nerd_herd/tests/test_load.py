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
    lm.set_load_mode("balanced", source="user")
    assert lm.get_load_mode() == "balanced"
    assert not lm.is_auto_managed()


def test_invalid_mode():
    lm = LoadManager(gpu_collector=MagicMock())
    result = lm.set_load_mode("turbo")
    assert "Unknown" in result


def test_vram_budget_fraction():
    lm = LoadManager(gpu_collector=MagicMock())
    assert lm.get_vram_budget_fraction() == 1.0
    lm.set_load_mode("balanced")
    assert lm.get_vram_budget_fraction() == 0.5
    lm.set_load_mode("minimal")
    assert lm.get_vram_budget_fraction() == 0.0


def test_vram_budget_mb(gpu_collector):
    lm = LoadManager(gpu_collector=gpu_collector)
    assert lm.get_vram_budget_mb() == 8000
    lm.set_load_mode("balanced")
    assert lm.get_vram_budget_mb() == 8000


def test_local_inference_allowed():
    lm = LoadManager(gpu_collector=MagicMock())
    assert lm.is_local_inference_allowed()
    lm.set_load_mode("minimal")
    assert not lm.is_local_inference_allowed()


def test_enable_auto_management():
    lm = LoadManager(gpu_collector=MagicMock())
    lm.set_load_mode("balanced", source="user")
    assert not lm.is_auto_managed()
    lm.enable_auto_management()
    assert lm.is_auto_managed()


def test_on_mode_change_callback():
    lm = LoadManager(gpu_collector=MagicMock())
    calls = []
    lm.on_mode_change(lambda old, new, src: calls.append((old, new, src)))
    lm.set_load_mode("balanced", source="auto")
    assert calls == [("full", "balanced", "auto")]


def test_suggest_mode():
    lm = LoadManager(gpu_collector=MagicMock())
    assert lm.suggest_mode_for_external_usage(0.05) == "full"
    assert lm.suggest_mode_for_external_usage(0.30) == "balanced"
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


def test_vram_budget_mb_is_raw_free_regardless_of_mode():
    class _G:
        def gpu_state(self):
            from nerd_herd.types import GPUState
            return GPUState(available=True, vram_total_mb=8000, vram_free_mb=8000)
    lm = LoadManager(gpu_collector=_G())
    lm.set_load_mode("balanced", source="user")
    assert lm.get_vram_budget_mb() == 8000


def test_normalize_legacy_modes():
    from nerd_herd.load import _normalize_mode
    assert _normalize_mode("heavy") == "balanced"
    assert _normalize_mode("shared") == "balanced"
    assert _normalize_mode("full") == "full"
    assert _normalize_mode("balanced") == "balanced"
    assert _normalize_mode("minimal") == "minimal"
    assert _normalize_mode("turbo") == "full"


def test_set_legacy_mode_normalizes():
    lm = LoadManager(gpu_collector=MagicMock())
    msg = lm.set_load_mode("shared", source="user")
    assert lm.get_load_mode() == "balanced"
    assert "Unknown" not in msg


def test_init_normalizes_legacy_initial_mode():
    lm = LoadManager(gpu_collector=MagicMock(), initial_mode="heavy")
    assert lm.get_load_mode() == "balanced"


def test_load_modes_set():
    from nerd_herd.load import LOAD_MODES
    assert LOAD_MODES == ("full", "balanced", "minimal")


class _Presence:
    def __init__(self, idle_s, fullscreen):
        self._d = {"user_idle_s": idle_s, "foreground_fullscreen": fullscreen}
    def collect(self):
        return self._d


def test_suggest_mode_fullscreen_forces_minimal():
    lm = LoadManager(gpu_collector=MagicMock(),
                     presence_collector=_Presence(idle_s=1.0, fullscreen=True))
    assert lm._suggest_mode(0.0, lm._presence.collect()) == "minimal"


def test_suggest_mode_high_external_minimal():
    lm = LoadManager(gpu_collector=MagicMock(),
                     presence_collector=_Presence(idle_s=1e9, fullscreen=False))
    assert lm._suggest_mode(0.70, lm._presence.collect()) == "minimal"


def test_suggest_mode_present_balanced():
    lm = LoadManager(gpu_collector=MagicMock(),
                     presence_collector=_Presence(idle_s=5.0, fullscreen=False))
    assert lm._suggest_mode(0.0, lm._presence.collect()) == "balanced"


def test_suggest_mode_away_idle_full():
    lm = LoadManager(gpu_collector=MagicMock(),
                     presence_collector=_Presence(idle_s=1e9, fullscreen=False))
    assert lm._suggest_mode(0.05, lm._presence.collect()) == "full"


def test_suggest_mode_mid_external_balanced_even_if_away():
    lm = LoadManager(gpu_collector=MagicMock(),
                     presence_collector=_Presence(idle_s=1e9, fullscreen=False))
    assert lm._suggest_mode(0.20, lm._presence.collect()) == "balanced"


def test_suggest_mode_no_presence_degrades_to_external_only():
    lm = LoadManager(gpu_collector=MagicMock())
    assert lm._presence is None
    assert lm._suggest_mode(0.05, None) == "full"
    assert lm._suggest_mode(0.30, None) == "balanced"
    assert lm._suggest_mode(0.70, None) == "minimal"
