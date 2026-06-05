from fatih_hoca.image_providers import image_catalog
from fatih_hoca.registry import ImageModelInfo


def test_catalog_has_pollinations_and_hf():
    cat = image_catalog()
    names = {m.name for m in cat}
    assert "pollinations/flux" in names
    assert "huggingface/flux-schnell" in names
    assert all(isinstance(m, ImageModelInfo) for m in cat)


def test_catalog_entries_are_cloud_free():
    cat = {m.name: m for m in image_catalog()}
    p = cat["pollinations/flux"]
    assert p.provider == "pollinations" and not p.is_local
    assert p.cost_per_image == 0.0 and p.tier == "free"
    assert p.supports_image_generation is True
