from fatih_hoca.image_select import select_image
from fatih_hoca.types import Pick, SelectionFailure


def test_picks_highest_quality_available(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert isinstance(pick, Pick)
    assert pick.model.name == "huggingface/flux-schnell"


def test_falls_back_to_pollinations_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    pick = select_image(quality_tier="fast", failures=[], hf_available=False)
    assert pick.model.name == "pollinations/flux"


def test_excludes_failed_name_string():
    pick = select_image(quality_tier="fast",
                        failures=["pollinations/flux"], hf_available=False)
    assert isinstance(pick, SelectionFailure)
    assert pick.reason == "availability"


def test_failed_pollinations_falls_to_hf():
    pick = select_image(quality_tier="fast",
                        failures=["pollinations/flux"], hf_available=True)
    assert pick.model.name == "huggingface/flux-schnell"


def test_failures_with_unknown_names_does_not_crash():
    pick = select_image(quality_tier="fast",
                        failures=["something/unrelated"], hf_available=True)
    assert pick.model.name == "huggingface/flux-schnell"


def test_pick_carries_top_summary():
    pick = select_image(quality_tier="quality", failures=[], hf_available=True)
    assert isinstance(pick, Pick)
    assert "huggingface/flux-schnell" in pick.top_summary
