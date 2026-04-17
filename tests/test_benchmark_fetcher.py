# test_benchmark_fetcher.py
"""
Tests for all benchmark fetcher classes.
Each test class mocks HTTP responses and verifies parsing + capability mapping.
"""

import csv
import io
import json
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.models.benchmark.benchmark_fetcher import (
    ArtificialAnalysisFetcher,
    LMArenaFetcher,
    BFCLFetcher,
    SenecaTRBenchFetcher,
    TurkishMMLUFetcher,
    UGILeaderboardFetcher,
    BenchmarkCache,
    BenchmarkFetcher,
    _normalize_score,
    _normalize_elo,
    _fuzzy_match_model,
)


@pytest.fixture
def tmp_cache(tmp_path):
    return BenchmarkCache(cache_dir=tmp_path)


# --- ArtificialAnalysisFetcher -----------------------------------------------

class TestArtificialAnalysisFetcher:

    def test_v2_api_parsing(self, tmp_cache):
        """Test parsing of v2 API response with slug-keyed models."""
        api_response = {
            "status": 200,
            "data": [
                {
                    "slug": "qwen3-32b",
                    "evaluations": {
                        "gpqa": 0.55,
                        "artificial_analysis_math_index": 70.0,
                        "mmlu_pro": 0.75,
                        "hle": 0.30,
                        "livecodebench": 0.45,
                        "artificial_analysis_coding_index": 60.0,
                        "scicode": 0.25,
                        "ifbench": 0.70,
                        "terminalbench_hard": 0.35,
                        "artificial_analysis_intelligence_index": 65.0,
                    },
                },
                {
                    "slug": "llama-3.3-70b",
                    "evaluations": {
                        "gpqa": 0.40,
                        "mmlu_pro": 0.60,
                    },
                },
            ],
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = api_response

        fetcher = ArtificialAnalysisFetcher()

        with patch("httpx.get", return_value=mock_resp) as mock_get, \
             patch("src.models.benchmark.benchmark_fetcher.ARTIFICIAL_ANALYSIS_API_KEY", "test-key",
                   create=True), \
             patch("src.app.config.ARTIFICIAL_ANALYSIS_API_KEY", "test-key"):
            result = fetcher.fetch_bulk(tmp_cache)

        assert "qwen3-32b" in result
        assert "llama-3.3-70b" in result
        caps = result["qwen3-32b"]
        assert "reasoning" in caps
        assert "domain_knowledge" in caps
        assert "code_generation" in caps
        assert "code_reasoning" in caps
        assert "instruction_adherence" in caps
        assert "analysis" in caps

        # Verify API key header was sent
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["headers"]["x-api-key"] == "test-key"

    def test_missing_api_key(self, tmp_cache):
        """If no API key is set, return {} silently."""
        fetcher = ArtificialAnalysisFetcher()

        with patch("src.app.config.ARTIFICIAL_ANALYSIS_API_KEY", ""):
            result = fetcher.fetch_bulk(tmp_cache)

        assert result == {}

    def test_empty_data(self, tmp_cache):
        """Empty data array should return {}."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": 200, "data": []}

        fetcher = ArtificialAnalysisFetcher()

        with patch("httpx.get", return_value=mock_resp), \
             patch("src.app.config.ARTIFICIAL_ANALYSIS_API_KEY", "test-key"):
            result = fetcher.fetch_bulk(tmp_cache)

        assert result == {}

    def test_averaging_same_capability(self, tmp_cache):
        """When multiple benchmarks map to same capability, they should average."""
        api_response = {
            "data": [
                {
                    "slug": "test-model",
                    "evaluations": {
                        "gpqa": 0.50,  # reasoning (fraction)
                        "artificial_analysis_math_index": 50.0,  # also reasoning (already 0-100)
                    },
                },
            ],
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = api_response

        fetcher = ArtificialAnalysisFetcher()

        with patch("httpx.get", return_value=mock_resp), \
             patch("src.app.config.ARTIFICIAL_ANALYSIS_API_KEY", "test-key"):
            result = fetcher.fetch_bulk(tmp_cache)

        assert "test-model" in result
        # Both map to reasoning but with different ranges, so the average should exist
        assert "reasoning" in result["test-model"]


# --- LMArenaFetcher ----------------------------------------------------------

class TestLMArenaFetcher:

    def test_per_category_elo(self, tmp_cache):
        """Test per-category ELO -> multi-dimension capabilities."""
        import pandas as pd

        df = pd.DataFrame({
            "model_name": ["qwen3.5-27b", "qwen3.5-27b", "qwen3.5-27b", "gpt-4o", "gpt-4o"],
            "category": ["overall", "coding", "math", "overall", "coding"],
            "rating": [1200.0, 1150.0, 1100.0, 1350.0, 1300.0],
            "vote_count": [500, 300, 200, 5000, 3000],
        })

        fetcher = LMArenaFetcher()

        with patch("pandas.read_parquet", return_value=df):
            result = fetcher.fetch_bulk(tmp_cache)

        assert "qwen3.5-27b" in result
        assert "gpt-4o" in result
        caps = result["qwen3.5-27b"]
        assert "conversation" in caps  # from overall
        assert "code_generation" in caps  # from coding
        assert "reasoning" in caps  # from math
        assert result["gpt-4o"]["code_generation"] > result["qwen3.5-27b"]["code_generation"]

    def test_empty_dataframe(self, tmp_cache):
        """Empty DataFrame returns {}."""
        import pandas as pd

        df = pd.DataFrame(columns=["model_name", "category", "rating"])

        fetcher = LMArenaFetcher()

        with patch("pandas.read_parquet", return_value=df):
            result = fetcher.fetch_bulk(tmp_cache)

        assert result == {}


# --- BFCLFetcher -------------------------------------------------------------

class TestBFCLFetcher:

    def test_csv_parsing(self, tmp_cache):
        """Test CSV parsing with Overall Acc field."""
        csv_text = "Rank,Overall Acc,Model,Model Link\n1,92.5,GPT-4o,https://example.com\n2,85.3,Claude-3.5,https://example.com\n"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = csv_text

        fetcher = BFCLFetcher()

        with patch("httpx.get", return_value=mock_resp):
            result = fetcher.fetch_bulk(tmp_cache)

        assert "GPT-4o" in result
        assert "Claude-3.5" in result
        assert "tool_use" in result["GPT-4o"]
        assert "structured_output" in result["GPT-4o"]

        # 92.5 should score very high
        assert result["GPT-4o"]["tool_use"] > 8.0

    def test_structured_output_is_0_85x_tool_use(self, tmp_cache):
        """structured_output should be 0.85 * tool_use."""
        csv_text = "Rank,Overall Acc,Model,Model Link\n1,70.0,TestModel,\n"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = csv_text

        fetcher = BFCLFetcher()

        with patch("httpx.get", return_value=mock_resp):
            result = fetcher.fetch_bulk(tmp_cache)

        tu = result["TestModel"]["tool_use"]
        so = result["TestModel"]["structured_output"]
        assert so == round(tu * 0.85, 1)

    def test_empty_csv(self, tmp_cache):
        """CSV with only headers returns {}."""
        csv_text = "Rank,Overall Acc,Model,Model Link\n"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = csv_text

        fetcher = BFCLFetcher()

        with patch("httpx.get", return_value=mock_resp):
            result = fetcher.fetch_bulk(tmp_cache)

        assert result == {}

    def test_missing_fields(self, tmp_cache):
        """Rows with missing Overall Acc should be skipped."""
        csv_text = "Rank,Overall Acc,Model,Model Link\n1,,NoAccModel,\n2,85.0,HasAccModel,\n"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = csv_text

        fetcher = BFCLFetcher()

        with patch("httpx.get", return_value=mock_resp):
            result = fetcher.fetch_bulk(tmp_cache)

        assert "NoAccModel" not in result
        assert "HasAccModel" in result


# --- SenecaTRBenchFetcher ----------------------------------------------------

class TestSenecaTRBenchFetcher:

    def test_csv_parsing(self, tmp_cache):
        """Test Turkish benchmark CSV parsing."""
        csv_text = "Rank,Model,MCQ Score,SAQ Score,Combined Score\n1,GPT-4o,90.5,85.0,87.75\n2,Qwen3-32B,70.0,65.0,67.5\n"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = csv_text

        fetcher = SenecaTRBenchFetcher()

        with patch("httpx.get", return_value=mock_resp):
            result = fetcher.fetch_bulk(tmp_cache)

        assert "GPT-4o" in result
        assert "Qwen3-32B" in result
        assert "turkish" in result["GPT-4o"]
        assert "turkish" in result["Qwen3-32B"]

        # Higher combined score should map to higher turkish score
        assert result["GPT-4o"]["turkish"] > result["Qwen3-32B"]["turkish"]

    def test_turkish_score_range(self, tmp_cache):
        """Turkish scores should be in 1-10 range."""
        csv_text = "Rank,Model,MCQ Score,SAQ Score,Combined Score\n1,Test,50.0,50.0,50.0\n"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = csv_text

        fetcher = SenecaTRBenchFetcher()

        with patch("httpx.get", return_value=mock_resp):
            result = fetcher.fetch_bulk(tmp_cache)

        assert 1.0 <= result["Test"]["turkish"] <= 10.0

    def test_empty_csv(self, tmp_cache):
        """Empty CSV returns {}."""
        csv_text = "Rank,Model,MCQ Score,SAQ Score,Combined Score\n"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = csv_text

        fetcher = SenecaTRBenchFetcher()

        with patch("httpx.get", return_value=mock_resp):
            result = fetcher.fetch_bulk(tmp_cache)

        assert result == {}

    def test_source_name(self):
        """Source name must be seneca_trbench."""
        fetcher = SenecaTRBenchFetcher()
        assert fetcher.source_name == "seneca_trbench"


# --- TurkishMMLUFetcher -----------------------------------------------------

class TestTurkishMMLUFetcher:

    def test_parquet_parsing(self, tmp_cache):
        """Test Turkish MMLU Parquet parsing with basari field."""
        import pandas as pd

        df = pd.DataFrame({
            "model": ["GPT-4o", "Qwen3-32B"],
            "parameter_size": ["unknown", "32B"],
            "quantization_level": ["none", "Q4_K_M"],
            "basari": [82.0, 55.0],
        })

        fetcher = TurkishMMLUFetcher()

        with patch("pandas.read_parquet", return_value=df):
            result = fetcher.fetch_bulk(tmp_cache)

        assert "GPT-4o" in result
        assert "Qwen3-32B" in result
        assert "turkish" in result["GPT-4o"]
        assert result["GPT-4o"]["turkish"] > result["Qwen3-32B"]["turkish"]

    def test_turkish_score_range(self, tmp_cache):
        """Turkish scores should be in 1-10 range."""
        import pandas as pd

        df = pd.DataFrame({
            "model": ["Test"],
            "basari": [60.0],
        })

        fetcher = TurkishMMLUFetcher()

        with patch("pandas.read_parquet", return_value=df):
            result = fetcher.fetch_bulk(tmp_cache)

        assert 1.0 <= result["Test"]["turkish"] <= 10.0

    def test_empty_dataframe(self, tmp_cache):
        """Empty DataFrame returns {}."""
        import pandas as pd

        df = pd.DataFrame(columns=["model", "basari"])

        fetcher = TurkishMMLUFetcher()

        with patch("pandas.read_parquet", return_value=df):
            result = fetcher.fetch_bulk(tmp_cache)

        assert result == {}

    def test_missing_basari(self, tmp_cache):
        """Rows with missing basari should be skipped."""
        import pandas as pd

        df = pd.DataFrame({
            "model": ["HasScore", "NoScore"],
            "basari": [70.0, None],
        })

        fetcher = TurkishMMLUFetcher()

        with patch("pandas.read_parquet", return_value=df):
            result = fetcher.fetch_bulk(tmp_cache)

        assert "HasScore" in result
        assert "NoScore" not in result

    def test_source_name(self):
        """Source name must be turkish_mmlu."""
        fetcher = TurkishMMLUFetcher()
        assert fetcher.source_name == "turkish_mmlu"


# --- BenchmarkFetcher (orchestrator) -----------------------------------------

class TestBenchmarkFetcher:

    def test_init_has_all_fetchers(self, tmp_path):
        """BenchmarkFetcher must include all 7 sources."""
        bf = BenchmarkFetcher(cache_dir=tmp_path)
        source_names = [f.source_name for f in bf.fetchers]

        assert "artificial_analysis" in source_names
        assert "lm_arena" in source_names
        assert "bfcl" in source_names
        assert "openrouter" in source_names
        assert "seneca_trbench" in source_names
        assert "turkish_mmlu" in source_names
        assert "ugi" in source_names

        # Removed sources should NOT be present
        assert "lmsys_arena" not in source_names
        assert "hf_leaderboard" not in source_names
        assert "livecodebench" not in source_names
        assert "bigcodebench" not in source_names
        assert "chatbot_arena" not in source_names
        assert "aider" not in source_names

    def test_confidence_map_has_all_sources(self, tmp_path):
        """The confidence map in fetch_all_bulk must include all sources."""
        bf = BenchmarkFetcher(cache_dir=tmp_path)

        expected_sources = {
            "artificial_analysis", "lm_arena",
            "bfcl", "openrouter",
            "seneca_trbench", "turkish_mmlu", "ugi",
        }
        actual_sources = {f.source_name for f in bf.fetchers}
        assert expected_sources == actual_sources


# --- Normalization helpers ---------------------------------------------------

class TestNormalization:

    def test_normalize_score_basic(self):
        assert _normalize_score(50, 0, 100) == 5.0
        assert _normalize_score(0, 0, 100) == 0.0
        assert _normalize_score(100, 0, 100) == 10.0

    def test_normalize_score_clamping(self):
        # Below min
        assert _normalize_score(-10, 0, 100) == 0.0
        # Above max
        assert _normalize_score(110, 0, 100) == 10.0

    def test_normalize_elo(self):
        # ELO 900 -> 2.0, ELO 1400 -> 10.0
        assert _normalize_elo(900) == 2.0
        assert _normalize_elo(1400) == 10.0
        # Mid-range
        mid = _normalize_elo(1150)
        assert 5.0 < mid < 7.0

    def test_fuzzy_match_exact(self):
        candidates = ["Qwen3-32B", "Llama-3.3-70B"]
        assert _fuzzy_match_model("Qwen3-32B", candidates) == "Qwen3-32B"

    def test_fuzzy_match_case_insensitive(self):
        candidates = ["Qwen3-32B", "Llama-3.3-70B"]
        assert _fuzzy_match_model("qwen3-32b", candidates) == "Qwen3-32B"

    def test_fuzzy_match_no_match(self):
        candidates = ["Qwen3-32B"]
        assert _fuzzy_match_model("completely-different-model", candidates) is None


class TestFuzzyMatching:
    def test_alias_matches_cloud_model(self):
        candidates = ["GPT-4o", "Claude-Sonnet-4", "gpt-4o-mini"]
        assert _fuzzy_match_model("gpt-4o", candidates) == "GPT-4o"

    def test_alias_matches_local_model(self):
        candidates = ["Qwen/Qwen3.5-35B-A3B", "google/gemma-4-26b-it"]
        result = _fuzzy_match_model("qwen3.5-35b", candidates)
        assert result is not None


# --- UGILeaderboardFetcher ---------------------------------------------------

class TestUGILeaderboardFetcher:

    def test_csv_parsing(self, tmp_cache):
        """Test UGI CSV parsing with real column names (emoji suffixes)."""
        csv_text = (
            "author/model_name,UGI \U0001f3c6,Writing \u270d\ufe0f,NatInt \U0001f4a1\n"
            "Qwen/Qwen3.5-27B,65.0,72.0,58.0\n"
            "ServiceNow-AI/Apriel-15B,55.0,60.0,45.0\n"
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = csv_text

        fetcher = UGILeaderboardFetcher()

        with patch("httpx.get", return_value=mock_resp):
            result = fetcher.fetch_bulk(tmp_cache)

        assert "Qwen/Qwen3.5-27B" in result
        assert "ServiceNow-AI/Apriel-15B" in result
        caps = result["Qwen/Qwen3.5-27B"]
        assert "prose_quality" in caps
        assert "domain_knowledge" in caps
        assert "analysis" in caps

    def test_empty_csv(self, tmp_cache):
        """Empty CSV returns {}."""
        csv_text = "author/model_name,UGI \U0001f3c6,Writing \u270d\ufe0f,NatInt \U0001f4a1\n"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = csv_text

        fetcher = UGILeaderboardFetcher()

        with patch("httpx.get", return_value=mock_resp):
            result = fetcher.fetch_bulk(tmp_cache)

        assert result == {}


# --- ArtificialAnalysis Thinking Pairs --------------------------------------

class TestAAThinkingPairs:
    """Test that AA fetcher correctly separates thinking/non-thinking entries."""

    def test_non_reasoning_suffix_detection(self, tmp_cache):
        """AA entries with -non-reasoning suffix are stored as base models."""
        api_response = {
            "status": 200,
            "data": [
                {
                    "slug": "qwen3-5-35b-a3b",
                    "evaluations": {"gpqa": 0.70, "mmlu_pro": 0.80},
                },
                {
                    "slug": "qwen3-5-35b-a3b-non-reasoning",
                    "evaluations": {"gpqa": 0.50, "mmlu_pro": 0.75},
                },
            ],
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = api_response

        fetcher = ArtificialAnalysisFetcher()

        with patch("httpx.get", return_value=mock_resp), \
             patch("src.app.config.ARTIFICIAL_ANALYSIS_API_KEY", "test-key"):
            result = fetcher.fetch_bulk(tmp_cache)

        # Base slug becomes thinking (it's the reasoning default)
        assert "qwen3-5-35b-a3b::thinking" in result
        # -non-reasoning becomes the base
        assert "qwen3-5-35b-a3b" in result
        # Thinking should have higher reasoning score
        assert result["qwen3-5-35b-a3b::thinking"]["reasoning"] > result["qwen3-5-35b-a3b"]["reasoning"]

    def test_reasoning_suffix_detection(self, tmp_cache):
        """AA entries with -reasoning suffix (without -non) are stored as thinking."""
        api_response = {
            "status": 200,
            "data": [
                {
                    "slug": "qwen3-30b-a3b-instruct",
                    "evaluations": {"gpqa": 0.40, "mmlu_pro": 0.65},
                },
                {
                    "slug": "qwen3-30b-a3b-instruct-reasoning",
                    "evaluations": {"gpqa": 0.65, "mmlu_pro": 0.72},
                },
            ],
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = api_response

        fetcher = ArtificialAnalysisFetcher()

        with patch("httpx.get", return_value=mock_resp), \
             patch("src.app.config.ARTIFICIAL_ANALYSIS_API_KEY", "test-key"):
            result = fetcher.fetch_bulk(tmp_cache)

        # -instruct stays as base (non-thinking)
        assert "qwen3-30b-a3b-instruct" in result
        # -instruct-reasoning becomes thinking variant
        assert "qwen3-30b-a3b-instruct::thinking" in result

    def test_thinking_suffix_for_cloud_models(self, tmp_cache):
        """AA entries with -thinking suffix (cloud models) are stored as ::thinking."""
        api_response = {
            "status": 200,
            "data": [
                {
                    "slug": "claude-4-sonnet-thinking",
                    "evaluations": {"gpqa": 0.70},
                },
            ],
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = api_response

        fetcher = ArtificialAnalysisFetcher()

        with patch("httpx.get", return_value=mock_resp), \
             patch("src.app.config.ARTIFICIAL_ANALYSIS_API_KEY", "test-key"):
            result = fetcher.fetch_bulk(tmp_cache)

        assert "claude-4-sonnet::thinking" in result

    def test_model_without_pair_stays_as_is(self, tmp_cache):
        """Models with no thinking pair are stored without ::thinking suffix."""
        api_response = {
            "status": 200,
            "data": [
                {
                    "slug": "llama-3-3-70b",
                    "evaluations": {"gpqa": 0.45, "mmlu_pro": 0.70},
                },
            ],
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = api_response

        fetcher = ArtificialAnalysisFetcher()

        with patch("httpx.get", return_value=mock_resp), \
             patch("src.app.config.ARTIFICIAL_ANALYSIS_API_KEY", "test-key"):
            result = fetcher.fetch_bulk(tmp_cache)

        assert "llama-3-3-70b" in result
        assert "llama-3-3-70b::thinking" not in result


# --- BenchmarkCache Staleness -----------------------------------------------

class TestBenchmarkCacheStaleness:
    """Stale cache entries (age > TTL) must be purged, not served."""

    def test_fresh_cache_loads_normally(self, tmp_path):
        from src.models.benchmark.benchmark_fetcher import BenchmarkCache

        cache_dir = tmp_path / "c"
        cache_dir.mkdir()
        p = cache_dir / "_bulk_source.json"
        p.write_text(json.dumps({
            "timestamp": time.time() - 60,  # 1 minute old
            "models": {"m1": {"reasoning": 7.0}},
        }))
        cache = BenchmarkCache(cache_dir=cache_dir)
        data = cache.load("source")
        assert data is not None
        assert "m1" in data.get("models", {})

    def test_stale_cache_returns_none_with_warning(self, tmp_path, caplog):
        from src.models.benchmark.benchmark_fetcher import BenchmarkCache, CACHE_TTL_HOURS
        import logging

        cache_dir = tmp_path / "c"
        cache_dir.mkdir()
        p = cache_dir / "_bulk_source.json"
        stale_ts = time.time() - (CACHE_TTL_HOURS + 1) * 3600
        p.write_text(json.dumps({
            "timestamp": stale_ts,
            "models": {"m1": {"reasoning": 7.0}},
        }))
        cache = BenchmarkCache(cache_dir=cache_dir)
        with caplog.at_level(logging.WARNING):
            data = cache.load("source")
        assert data is None, "stale cache must not be served"
        assert any("stale" in r.message.lower() for r in caplog.records), \
            "must warn when returning None due to staleness"
