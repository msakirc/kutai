"""Integration test for the benchmark enrichment pipeline."""
import pytest
from src.models.benchmark.benchmark_fetcher import enrich_registry_with_benchmarks
from src.models.model_registry import ModelInfo, ModelRegistry


class TestEnrichmentPipeline:
    def test_enrichment_does_not_crash_on_local_model(self, tmp_path):
        registry = ModelRegistry()
        registry.models = {
            "test-model": ModelInfo(
                name="test-model", location="local", provider="llama_cpp",
                litellm_name="openai/test-model",
                capabilities={
                    "reasoning": 5.0, "code_generation": 5.0, "turkish": 3.0,
                    "tool_use": 4.0, "conversation": 5.0, "domain_knowledge": 5.0,
                    "instruction_adherence": 5.0,
                },
                context_length=8192, max_tokens=2048, family="qwen35",
            ),
        }
        enriched = enrich_registry_with_benchmarks(
            registry, cache_dir=tmp_path / "cache", min_confidence_sources=1,
        )
        assert isinstance(enriched, dict)

    def test_enrichment_does_not_crash_on_cloud_model(self, tmp_path):
        registry = ModelRegistry()
        registry.models = {
            "claude-sonnet": ModelInfo(
                name="claude-sonnet", location="cloud", provider="anthropic",
                litellm_name="anthropic/claude-sonnet-4-20250514",
                capabilities={"reasoning": 9.5, "turkish": 8.0},
                context_length=200000, max_tokens=8192,
            ),
        }
        enriched = enrich_registry_with_benchmarks(
            registry, cache_dir=tmp_path / "cache", min_confidence_sources=1,
        )
        assert isinstance(enriched, dict)

    def test_enrichment_handles_variant_models(self, tmp_path):
        registry = ModelRegistry()
        registry.models = {
            "test": ModelInfo(
                name="test", location="local", provider="llama_cpp",
                litellm_name="openai/test",
                capabilities={"reasoning": 7.0},
                context_length=8192, max_tokens=2048,
                thinking_model=False, is_variant=False,
            ),
            "test-thinking": ModelInfo(
                name="test-thinking", location="local", provider="llama_cpp",
                litellm_name="openai/test-thinking",
                capabilities={"reasoning": 8.0},
                context_length=8192, max_tokens=2048,
                thinking_model=True, is_variant=True,
                base_model_name="test", variant_flags={"thinking"},
            ),
        }
        enriched = enrich_registry_with_benchmarks(
            registry, cache_dir=tmp_path / "cache", min_confidence_sources=1,
        )
        assert isinstance(enriched, dict)

    def test_enrichment_preserves_unenriched_capabilities(self, tmp_path):
        """Capabilities not covered by benchmarks should remain unchanged."""
        registry = ModelRegistry()
        registry.models = {
            "test": ModelInfo(
                name="test", location="local", provider="llama_cpp",
                litellm_name="openai/test",
                capabilities={"reasoning": 5.0, "turkish": 3.0, "vision": 0.0},
                context_length=8192, max_tokens=2048,
            ),
        }
        original_turkish = registry.models["test"].capabilities["turkish"]
        original_vision = registry.models["test"].capabilities["vision"]

        enrich_registry_with_benchmarks(
            registry, cache_dir=tmp_path / "cache", min_confidence_sources=1,
        )

        # Vision should remain unchanged (no benchmark source covers it for unknown models)
        assert registry.models["test"].capabilities.get("vision", 0) == original_vision
