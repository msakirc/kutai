from fatih_hoca.image_providers import image_catalog
from fatih_hoca.registry import ImageModelInfo


def test_catalog_has_clair_obscur_entry():
    cat = {m.name: m for m in image_catalog()}
    assert "clair_obscur/sdxl-turbo" in cat
    co = cat["clair_obscur/sdxl-turbo"]
    assert isinstance(co, ImageModelInfo)
    assert co.provider == "clair_obscur"
    assert co.is_local is True
    assert 3000 <= co.vram_mb <= 6000  # fits 8GB after llama unload
    assert co.cost_per_image == 0.0


def test_cloud_entries_still_present():
    """Plan 2 must NOT remove or reorder Plan 1 v2's cloud entries."""
    names = {m.name for m in image_catalog()}
    assert "pollinations/flux" in names
    assert "huggingface/flux-schnell" in names
