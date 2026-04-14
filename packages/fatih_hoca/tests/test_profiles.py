from fatih_hoca.profiles import (
    FamilyProfile,
    FAMILY_PATTERNS,
    CLOUD_PROFILES,
    detect_family,
    get_default_profile,
    get_quant_retention,
    interpolate_size_multiplier,
)


def test_detect_family_qwen():
    # detect_family expects name_lower (already lowercased)
    assert detect_family("qwen3-30b-a3b-q4_k_m.gguf") == "qwen3"


def test_detect_family_llama():
    assert detect_family("meta-llama-3.1-8b-instruct-q5_k_m.gguf") == "llama31"


def test_detect_family_unknown():
    result = detect_family("completely-unknown-model.gguf")
    assert result == "unknown" or result is None


def test_cloud_profiles_has_claude():
    assert any("claude" in k for k in CLOUD_PROFILES)


def test_family_profile_fields():
    fp = get_default_profile()
    assert isinstance(fp, FamilyProfile)
    assert hasattr(fp, "base_capabilities")
    assert isinstance(fp.base_capabilities, dict)


def test_quant_retention():
    q8 = get_quant_retention("Q8_0")
    q4 = get_quant_retention("Q4_K_M")
    assert q8 > q4


def test_interpolate_size_multiplier():
    small = interpolate_size_multiplier(1.0)
    large = interpolate_size_multiplier(30.0)
    assert large > small


def test_family_patterns_is_list():
    assert isinstance(FAMILY_PATTERNS, list)
    assert len(FAMILY_PATTERNS) > 0
    # Each entry is (list[str], str)
    for patterns, key in FAMILY_PATTERNS:
        assert isinstance(patterns, list)
        assert isinstance(key, str)


def test_detect_family_coder_variant():
    # qwen3_coder should match before qwen3 generic
    assert detect_family("qwen3-coder-32b-q4_k_m.gguf") == "qwen3_coder"


def test_detect_family_deepseek_r1():
    assert detect_family("deepseek-r1-distill-qwen-32b.gguf") == "deepseek_r1"
