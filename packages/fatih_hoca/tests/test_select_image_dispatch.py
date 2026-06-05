import fatih_hoca
from fatih_hoca.types import Pick
from fatih_hoca.registry import ImageModelInfo, ModelInfo


def test_select_needs_image_returns_image_pick(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    pick = fatih_hoca.select(needs_image=True, quality_tier="fast")
    assert isinstance(pick, Pick)
    assert isinstance(pick.model, ImageModelInfo)
    assert pick.model.name == "pollinations/flux"


def test_select_needs_image_accepts_failure_objects(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    class _F:
        class model:
            name = "pollinations/flux"
    pick = fatih_hoca.select(needs_image=True, failures=[_F()])
    from fatih_hoca.types import SelectionFailure
    assert isinstance(pick, SelectionFailure)


def test_select_needs_image_accepts_plain_name_strings(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from fatih_hoca.types import SelectionFailure
    pick = fatih_hoca.select(needs_image=True, failures=["pollinations/flux"])
    assert isinstance(pick, SelectionFailure)


def test_text_select_unchanged(monkeypatch):
    res = fatih_hoca.select(task="router", agent_type="router", difficulty=3)
    if res is not None and hasattr(res, "model"):
        assert not isinstance(res.model, ImageModelInfo)


def test_benchmark_enrichment_skips_image_entries():
    from fatih_hoca.image_providers import image_catalog
    from fatih_hoca import _is_image_entry
    for m in image_catalog():
        assert _is_image_entry(m) is True
