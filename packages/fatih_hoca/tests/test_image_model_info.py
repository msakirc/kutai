from fatih_hoca.registry import ModelInfo, ImageModelInfo


def test_image_model_info_basic_fields():
    m = ImageModelInfo(
        name="pollinations/flux", provider="pollinations", location="cloud",
        endpoint="https://image.pollinations.ai/prompt/",
        quality_rank=6.0, cost_per_image=0.0, vram_mb=0, supports_seed=True,
    )
    assert m.name == "pollinations/flux"
    assert m.is_local is False
    assert m.supports_image_generation is True
    assert m.tier == "free"


def test_image_model_info_local_flag():
    m = ImageModelInfo(name="x", provider="clair_obscur", location="local",
                       endpoint="http://127.0.0.1:7860", quality_rank=7.0,
                       cost_per_image=0.0, vram_mb=4000, supports_seed=True)
    assert m.is_local is True


def test_image_and_text_branch_by_isinstance():
    im = ImageModelInfo(name="im", provider="p", location="cloud",
                       endpoint="", quality_rank=5.0)
    tm = ModelInfo(name="tm", location="cloud", provider="p", litellm_name="p/tm")
    assert isinstance(im, ImageModelInfo)
    assert not isinstance(tm, ImageModelInfo)
    assert isinstance(tm, ModelInfo)
    assert not isinstance(im, ModelInfo)


def test_modelinfo_still_constructs_unchanged():
    # Regression — ModelInfo is NOT touched by this task.
    m = ModelInfo(name="x", location="cloud", provider="p", litellm_name="p/x")
    assert m.supports_function_calling is False  # default preserved
