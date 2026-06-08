import pytest

from fatih_hoca.image_select import select_image
from fatih_hoca.types import Pick


def _snap(*, llm_in_flight=0, llm_loaded=False, llm_queue=0,
          image_resident=False, vram_mb=6000):
    class _Local:
        model_name = "qwen2.5" if llm_loaded else None
        requests_processing = llm_in_flight
    class _QP:
        total_ready_count = llm_queue
    class _S:
        local = _Local()
        queue_profile = _QP()
        in_flight_calls = []
        image_server_resident = image_resident
        image_server_vram_mb = 4500 if image_resident else 0
        vram_available_mb = vram_mb
    return _S()


def test_huge_when_llm_in_flight(monkeypatch):
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(llm_in_flight=1))
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", "/fake/exe")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert isinstance(pick, Pick)
    assert pick.model.is_local is False


def test_high_when_llm_loaded(monkeypatch):
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(llm_loaded=True))
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", "/fake/exe")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.is_local is False


def test_low_when_idle_first_call_still_cloud(monkeypatch):
    """8.0 (HF) > 7.5 − 2.0 = 5.5 (local LOW) → HF wins cold start."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot", lambda: _snap())
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", "/fake/exe")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.name == "huggingface/flux-schnell"


def test_resident_with_warm_bonus_picks_local(monkeypatch):
    """Image-server warm → local score = 7.5 + 1.0 = 8.5 > HF 8.0."""
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(image_resident=True))
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", "/fake/exe")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.provider == "clair_obscur"


def test_vram_too_low_filters_local(monkeypatch):
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(vram_mb=2000))
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("CLAIR_OBSCUR_EXE", "/fake/exe")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.is_local is False


def test_local_unavailable_filters_local(monkeypatch):
    monkeypatch.setattr("fatih_hoca.image_select._snapshot",
                        lambda: _snap(vram_mb=8000))
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.delenv("CLAIR_OBSCUR_EXE", raising=False)
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert pick.model.is_local is False
