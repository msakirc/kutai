"""Tests for model variant registration (thinking/vision)."""

import pytest
from dataclasses import replace as dc_replace

from src.models.model_registry import (
    ModelInfo,
    _apply_thinking_deltas,
    _create_model_variants,
)
from src.models.model_profiles import FamilyProfile


def _make_base_model(name="test-model", mmproj_path=None, **kwargs):
    """Create a minimal ModelInfo for testing."""
    caps = {
        "reasoning": 5.0,
        "planning": 5.0,
        "analysis": 5.0,
        "code_reasoning": 5.0,
        "code_generation": 5.0,
        "instruction_adherence": 5.0,
        "structured_output": 5.0,
        "conversation": 5.0,
        "knowledge": 5.0,
    }
    return ModelInfo(
        name=name,
        location="local",
        provider="llama_cpp",
        litellm_name=f"openai/{name}",
        capabilities=caps,
        context_length=32768,
        max_tokens=8192,
        thinking_model=True,   # raw detection says thinking
        has_vision=True,       # raw detection says vision
        mmproj_path=mmproj_path,
        path=f"/models/{name}.gguf",
        family="test_family",
        **kwargs,
    )


# ─── _apply_thinking_deltas ────────────────────────────────────────────────


class TestApplyThinkingDeltas:
    def test_boosts_reasoning_dimensions(self):
        caps = {"reasoning": 5.0, "planning": 5.0, "analysis": 5.0, "code_reasoning": 5.0}
        result = _apply_thinking_deltas(caps)
        assert result["reasoning"] == 6.0
        assert result["planning"] == 6.0
        assert result["analysis"] == 6.0
        assert result["code_reasoning"] == 6.0

    def test_boosts_code_generation(self):
        caps = {"code_generation": 5.0}
        result = _apply_thinking_deltas(caps)
        assert result["code_generation"] == pytest.approx(5.3)

    def test_penalizes_instruction_structured_conversation(self):
        caps = {"instruction_adherence": 5.0, "structured_output": 5.0, "conversation": 5.0}
        result = _apply_thinking_deltas(caps)
        assert result["instruction_adherence"] == pytest.approx(4.3)
        assert result["structured_output"] == pytest.approx(4.3)
        assert result["conversation"] == pytest.approx(4.5)

    def test_caps_at_10(self):
        caps = {"reasoning": 9.5, "planning": 10.0}
        result = _apply_thinking_deltas(caps)
        assert result["reasoning"] == 10.0
        assert result["planning"] == 10.0

    def test_floors_at_0(self):
        caps = {"instruction_adherence": 0.3, "conversation": 0.2}
        result = _apply_thinking_deltas(caps)
        assert result["instruction_adherence"] == 0.0
        assert result["conversation"] == 0.0

    def test_does_not_modify_original(self):
        caps = {"reasoning": 5.0}
        _apply_thinking_deltas(caps)
        assert caps["reasoning"] == 5.0

    def test_ignores_unknown_keys(self):
        caps = {"knowledge": 7.0, "reasoning": 5.0}
        result = _apply_thinking_deltas(caps)
        assert result["knowledge"] == 7.0
        assert result["reasoning"] == 6.0


# ─── _create_model_variants ────────────────────────────────────────────────


class TestCreateModelVariants:
    def test_no_family_profile_returns_base_only(self):
        base = _make_base_model()
        variants = _create_model_variants(base, None)
        assert len(variants) == 1
        assert variants[0].name == "test-model"
        assert variants[0].thinking_model is False
        assert variants[0].has_vision is False
        assert variants[0].is_variant is False

    def test_non_thinking_non_vision_family(self):
        base = _make_base_model()
        profile = FamilyProfile(
            base_capabilities={},
            thinking_capable=False,
            has_vision=False,
        )
        variants = _create_model_variants(base, profile)
        assert len(variants) == 1
        assert variants[0].is_variant is False

    def test_thinking_only_family(self):
        base = _make_base_model()
        profile = FamilyProfile(
            base_capabilities={},
            thinking_capable=True,
            has_vision=False,
        )
        variants = _create_model_variants(base, profile)
        assert len(variants) == 2

        names = {v.name for v in variants}
        assert names == {"test-model", "test-model-thinking"}

        thinking = [v for v in variants if v.name == "test-model-thinking"][0]
        assert thinking.thinking_model is True
        assert thinking.has_vision is False
        assert thinking.is_variant is True
        assert thinking.base_model_name == "test-model"
        assert thinking.variant_flags == {"thinking"}
        assert thinking.path == base.path  # same GGUF

    def test_vision_only_family_with_mmproj(self):
        base = _make_base_model(mmproj_path="/models/mmproj.gguf")
        profile = FamilyProfile(
            base_capabilities={},
            thinking_capable=False,
            has_vision=True,
        )
        variants = _create_model_variants(base, profile)
        assert len(variants) == 2

        vision = [v for v in variants if v.name == "test-model-vision"][0]
        assert vision.has_vision is True
        assert vision.thinking_model is False
        assert vision.is_variant is True
        assert vision.variant_flags == {"vision"}

    def test_vision_family_without_mmproj_no_variant(self):
        base = _make_base_model(mmproj_path=None)
        profile = FamilyProfile(
            base_capabilities={},
            thinking_capable=False,
            has_vision=True,
        )
        variants = _create_model_variants(base, profile)
        assert len(variants) == 1  # no vision variant without mmproj

    def test_thinking_and_vision_four_entries(self):
        base = _make_base_model(mmproj_path="/models/mmproj.gguf")
        profile = FamilyProfile(
            base_capabilities={},
            thinking_capable=True,
            has_vision=True,
        )
        variants = _create_model_variants(base, profile)
        assert len(variants) == 4

        names = {v.name for v in variants}
        assert names == {
            "test-model",
            "test-model-thinking",
            "test-model-vision",
            "test-model-thinking-vision",
        }

        tv = [v for v in variants if v.name == "test-model-thinking-vision"][0]
        assert tv.thinking_model is True
        assert tv.has_vision is True
        assert tv.is_variant is True
        assert tv.variant_flags == {"thinking", "vision"}
        assert tv.base_model_name == "test-model"

    def test_base_entry_strips_thinking_and_vision(self):
        """Base entry must have thinking_model=False, has_vision=False regardless of raw detection."""
        base = _make_base_model(mmproj_path="/models/mmproj.gguf")
        # Raw model has thinking_model=True, has_vision=True
        assert base.thinking_model is True
        assert base.has_vision is True

        profile = FamilyProfile(
            base_capabilities={},
            thinking_capable=True,
            has_vision=True,
        )
        variants = _create_model_variants(base, profile)
        base_entry = [v for v in variants if v.name == "test-model"][0]
        assert base_entry.thinking_model is False
        assert base_entry.has_vision is False

    def test_thinking_variant_has_same_base_capabilities(self):
        """Thinking variant starts with same capabilities as base.

        Real benchmark data is applied later by enrichment, not hardcoded deltas.
        """
        base = _make_base_model()
        profile = FamilyProfile(
            base_capabilities={},
            thinking_capable=True,
            has_vision=False,
        )
        variants = _create_model_variants(base, profile)
        thinking = [v for v in variants if v.name == "test-model-thinking"][0]

        # Thinking variant has same scores as base (no hardcoded deltas)
        assert thinking.capabilities["reasoning"] == 5.0
        assert thinking.capabilities["instruction_adherence"] == 5.0

        # Base unchanged
        base_entry = [v for v in variants if v.name == "test-model"][0]
        assert base_entry.capabilities["reasoning"] == 5.0

    def test_all_variants_share_same_path(self):
        base = _make_base_model(mmproj_path="/models/mmproj.gguf")
        profile = FamilyProfile(
            base_capabilities={},
            thinking_capable=True,
            has_vision=True,
        )
        variants = _create_model_variants(base, profile)
        for v in variants:
            assert v.path == base.path

    def test_litellm_names_match_variant_names(self):
        base = _make_base_model(mmproj_path="/models/mmproj.gguf")
        profile = FamilyProfile(
            base_capabilities={},
            thinking_capable=True,
            has_vision=True,
        )
        variants = _create_model_variants(base, profile)
        for v in variants:
            assert v.litellm_name == f"openai/{v.name}"


# ─── Variant Swap Detection ──────────────────────────────────────────────


class TestVariantSwapDetection:
    """Test that variant swaps are detected via shared GGUF path."""

    def test_same_path_is_variant_swap(self):
        """Two ModelInfo entries with same path = variant swap."""
        base = ModelInfo(
            name="test", location="local", provider="llama_cpp",
            litellm_name="openai/test",
            capabilities={"reasoning": 7.0},
            context_length=8192, max_tokens=2048,
            path="/models/test.gguf",
        )
        thinking = ModelInfo(
            name="test-thinking", location="local", provider="llama_cpp",
            litellm_name="openai/test-thinking",
            capabilities={"reasoning": 8.0},
            context_length=8192, max_tokens=2048,
            path="/models/test.gguf",
            thinking_model=True,
            is_variant=True, base_model_name="test",
            variant_flags={"thinking"},
        )

        # Same path means variant swap
        assert base.path == thinking.path
        assert thinking.is_variant is True
        assert thinking.base_model_name == "test"

    def test_different_path_is_not_variant_swap(self):
        """Two ModelInfo entries with different paths = full swap."""
        model_a = ModelInfo(
            name="model-a", location="local", provider="llama_cpp",
            litellm_name="openai/model-a",
            capabilities={"reasoning": 7.0},
            context_length=8192, max_tokens=2048,
            path="/models/a.gguf",
        )
        model_b = ModelInfo(
            name="model-b", location="local", provider="llama_cpp",
            litellm_name="openai/model-b",
            capabilities={"reasoning": 8.0},
            context_length=8192, max_tokens=2048,
            path="/models/b.gguf",
        )

        assert model_a.path != model_b.path
