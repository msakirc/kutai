"""Qwen3 trio must never cross-match. Each local GGUF → its own AA key."""
from src.models.benchmark.benchmark_fetcher import _fuzzy_match_model


# Realistic AA key list (subset of _bulk_artificial_analysis.json as of 2026-04-17)
AA_KEYS = [
    "qwen3-30b-a3b-instruct::thinking",
    "qwen3-32b-instruct",
    "qwen3-coder-30b-a3b-instruct",   # NEW — real AA entry
    "qwen3-coder-480b-a35b-instruct",
    "qwen3-8b-instruct",
    "qwen3-14b-instruct",
    "qwen3-235b-a22b-instruct-2507",
    "llama-3-3-instruct-70b",          # updated to match live cache
    "apriel-v1-6-15b-thinker",         # updated
]


class TestQwenTrioDisambiguation:
    def test_base_a3b_matches_a3b(self):
        """Qwen3-30B-A3B-Instruct GGUF → qwen3-30b-a3b-instruct::thinking, NOT qwen3-32b."""
        m = _fuzzy_match_model("Qwen3-30B-A3B-Instruct-Q4_K_M", AA_KEYS)
        assert m == "qwen3-30b-a3b-instruct::thinking"

    def test_base_a3b_does_not_match_coder(self):
        m = _fuzzy_match_model("Qwen3-30B-A3B-Instruct-Q4_K_M", AA_KEYS)
        assert m != "qwen3-coder-480b-a35b-instruct"

    def test_base_a3b_does_not_match_dense_32b(self):
        m = _fuzzy_match_model("Qwen3-30B-A3B-Instruct-Q4_K_M", AA_KEYS)
        assert m != "qwen3-32b-instruct"

    def test_coder_30b_matches_coder_30b_not_480b_or_base(self):
        """Coder 30B GGUF → coder 30B AA entry, NOT 480B coder, NOT base 30B A3B, NOT 32B dense."""
        m = _fuzzy_match_model("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M", AA_KEYS)
        assert m == "qwen3-coder-30b-a3b-instruct", \
            f"expected qwen3-coder-30b-a3b-instruct, got {m!r}"

    def test_dense_32b_matches_itself(self):
        m = _fuzzy_match_model("Qwen3-32B-Instruct-Q4_K_M", AA_KEYS)
        assert m == "qwen3-32b-instruct"

    def test_dense_32b_does_not_match_a3b(self):
        m = _fuzzy_match_model("Qwen3-32B-Instruct-Q4_K_M", AA_KEYS)
        assert m != "qwen3-30b-a3b-instruct::thinking"


class TestOtherLocals:
    def test_apriel_matches(self):
        m = _fuzzy_match_model("ServiceNow-AI_Apriel-1.6-15b-Thinker-Q4_K_L", AA_KEYS)
        assert m == "apriel-v1-6-15b-thinker"

    def test_llama_3_3_matches(self):
        m = _fuzzy_match_model("Llama-3.3-70B-Instruct-Q4_K_M", AA_KEYS)
        assert m == "llama-3-3-instruct-70b"
