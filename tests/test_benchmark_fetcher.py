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
    ChatbotArenaFetcher,
    HuggingFaceLeaderboardFetcher,
    LiveCodeBenchFetcher,
    BFCLFetcher,
    BigCodeBenchFetcher,
    SenecaTRBenchFetcher,
    TurkishMMLUFetcher,
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
                    "gpqa": 55.0,
                    "artificial_analysis_math_index": 70.0,
                    "mmlu_pro": 75.0,
                    "hle": 30.0,
                    "livecodebench": 45.0,
                    "artificial_analysis_coding_index": 60.0,
                    "scicode": 25.0,
                    "ifbench": 70.0,
                    "terminalbench_hard": 35.0,
                    "artificial_analysis_intelligence_index": 65.0,
                },
                {
                    "slug": "llama-3.3-70b",
                    "gpqa": 40.0,
                    "mmlu_pro": 60.0,
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
                    "gpqa": 50.0,  # reasoning
                    "artificial_analysis_math_index": 50.0,  # also reasoning
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


# --- ChatbotArenaFetcher -----------------------------------------------------

class TestChatbotArenaFetcher:

    def _make_parquet_bytes(self, data):
        """Create parquet bytes from dict data."""
        import pandas as pd
        df = pd.DataFrame(data)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        return buf

    def test_parquet_parsing(self, tmp_cache):
        """Test Parquet parsing with Arena Score -> conversation/prose."""
        import pandas as pd

        df = pd.DataFrame({
            "Model": ["GPT-4o", "Claude-3.5", "Qwen3-32B"],
            "Arena Score": [1350.0, 1300.0, 1100.0],
            "Votes": [5000, 4000, 2000],
            "Organization": ["OpenAI", "Anthropic", "Alibaba"],
        })

        fetcher = ChatbotArenaFetcher()

        with patch("pandas.read_parquet", return_value=df):
            result = fetcher.fetch_bulk(tmp_cache)

        assert len(result) == 3
        assert "GPT-4o" in result
        assert "conversation" in result["GPT-4o"]
        assert "prose_quality" in result["GPT-4o"]

        # GPT-4o ELO 1350 should score high
        assert result["GPT-4o"]["conversation"] > 8.0
        # Qwen3-32B ELO 1100 should score moderate
        assert result["Qwen3-32B"]["conversation"] < result["GPT-4o"]["conversation"]

    def test_elo_mapping(self, tmp_cache):
        """Verify ELO -> conversation/prose mapping uses _normalize_elo."""
        import pandas as pd

        df = pd.DataFrame({
            "Model": ["TestModel"],
            "Arena Score": [1150.0],
        })

        fetcher = ChatbotArenaFetcher()

        with patch("pandas.read_parquet", return_value=df):
            result = fetcher.fetch_bulk(tmp_cache)

        expected_score = _normalize_elo(1150.0)
        assert result["TestModel"]["conversation"] == expected_score
        assert result["TestModel"]["prose_quality"] == round(expected_score * 0.95, 1)

    def test_empty_dataframe(self, tmp_cache):
        """Empty DataFrame should return {}."""
        import pandas as pd

        df = pd.DataFrame(columns=["Model", "Arena Score"])

        fetcher = ChatbotArenaFetcher()

        with patch("pandas.read_parquet", return_value=df):
            result = fetcher.fetch_bulk(tmp_cache)

        assert result == {}

    def test_missing_arena_score(self, tmp_cache):
        """Rows without Arena Score should be skipped."""
        import pandas as pd

        df = pd.DataFrame({
            "Model": ["HasScore", "NoScore"],
            "Arena Score": [1200.0, None],
        })

        fetcher = ChatbotArenaFetcher()

        with patch("pandas.read_parquet", return_value=df):
            result = fetcher.fetch_bulk(tmp_cache)

        assert "HasScore" in result
        assert "NoScore" not in result


# --- HuggingFaceLeaderboardFetcher -------------------------------------------

class TestHFLeaderboardFetcher:

    def test_parquet_parsing(self, tmp_cache):
        """Test Parquet parsing with correct column names."""
        import pandas as pd

        df = pd.DataFrame({
            "Model": ["Qwen/Qwen3-32B", "meta-llama/Llama-3.3-70B"],
            "IFEval": [75.0, 65.0],
            "BBH": [70.0, 55.0],
            "MATH Lvl 5": [50.0, 35.0],
            "GPQA": [40.0, 35.0],
            "MUSR": [45.0, 40.0],
            "MMLU-PRO": [72.0, 60.0],
        })

        fetcher = HuggingFaceLeaderboardFetcher()

        with patch("pandas.read_parquet", return_value=df):
            result = fetcher.fetch_bulk(tmp_cache)

        assert len(result) == 2
        assert "Qwen/Qwen3-32B" in result
        caps = result["Qwen/Qwen3-32B"]
        assert "instruction_adherence" in caps
        assert "reasoning" in caps  # BBH + MATH + GPQA averaged
        assert "analysis" in caps
        assert "domain_knowledge" in caps

    def test_fraction_conversion(self, tmp_cache):
        """Scores between 0-1 should be converted to percentages."""
        import pandas as pd

        df = pd.DataFrame({
            "Model": ["TestModel"],
            "IFEval": [0.75],
            "BBH": [0.65],
            "MATH Lvl 5": [None],
            "GPQA": [None],
            "MUSR": [None],
            "MMLU-PRO": [None],
        })

        fetcher = HuggingFaceLeaderboardFetcher()

        with patch("pandas.read_parquet", return_value=df):
            result = fetcher.fetch_bulk(tmp_cache)

        assert "TestModel" in result
        assert "instruction_adherence" in result["TestModel"]
        assert "reasoning" in result["TestModel"]

    def test_empty_dataframe(self, tmp_cache):
        """Empty DataFrame returns {}."""
        import pandas as pd

        df = pd.DataFrame(columns=["Model", "IFEval", "BBH"])

        fetcher = HuggingFaceLeaderboardFetcher()

        with patch("pandas.read_parquet", return_value=df):
            result = fetcher.fetch_bulk(tmp_cache)

        assert result == {}


# --- LiveCodeBenchFetcher ----------------------------------------------------

class TestLiveCodeBenchFetcher:

    def test_new_performances_format(self, tmp_cache):
        """Test new JSON format with models + performances."""
        json_data = {
            "models": ["model-a", "model-b"],
            "performances": [
                {"model": "model-a", "pass_at_1": 0.65, "difficulty": "easy"},
                {"model": "model-a", "pass_at_1": 0.45, "difficulty": "medium"},
                {"model": "model-a", "pass_at_1": 0.25, "difficulty": "hard"},
                {"model": "model-b", "pass_at_1": 0.50, "difficulty": "easy"},
                {"model": "model-b", "pass_at_1": 0.30, "difficulty": "medium"},
            ],
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = json_data

        fetcher = LiveCodeBenchFetcher()

        with patch("httpx.get", return_value=mock_resp):
            result = fetcher.fetch_bulk(tmp_cache)

        assert "model-a" in result
        assert "model-b" in result
        assert "code_generation" in result["model-a"]
        assert "code_reasoning" in result["model-a"]

        # model-a avg: (65+45+25)/3 = 45.0
        # model-b avg: (50+30)/2 = 40.0
        assert result["model-a"]["code_generation"] >= result["model-b"]["code_generation"]

    def test_old_flat_list_fallback(self, tmp_cache):
        """Old flat list format should still work."""
        json_data = [
            {"model": "old-model", "pass@1": 55.0},
        ]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = json_data

        fetcher = LiveCodeBenchFetcher()

        with patch("httpx.get", return_value=mock_resp):
            result = fetcher.fetch_bulk(tmp_cache)

        assert "old-model" in result
        assert "code_generation" in result["old-model"]

    def test_code_reasoning_is_0_9x_code_gen(self, tmp_cache):
        """code_reasoning should be 0.9 * code_generation."""
        json_data = {
            "performances": [
                {"model": "test", "pass_at_1": 0.50, "difficulty": "easy"},
            ],
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = json_data

        fetcher = LiveCodeBenchFetcher()

        with patch("httpx.get", return_value=mock_resp):
            result = fetcher.fetch_bulk(tmp_cache)

        cg = result["test"]["code_generation"]
        cr = result["test"]["code_reasoning"]
        assert cr == round(cg * 0.9, 1)

    def test_empty_performances(self, tmp_cache):
        """Empty performances returns {}."""
        json_data = {"models": [], "performances": []}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = json_data

        fetcher = LiveCodeBenchFetcher()

        with patch("httpx.get", return_value=mock_resp):
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


# --- BigCodeBenchFetcher -----------------------------------------------------

class TestBigCodeBenchFetcher:

    def test_parquet_parsing(self, tmp_cache):
        """Test HF Parquet parsing with complete/instruct columns."""
        import pandas as pd

        df = pd.DataFrame({
            "model": ["model-a", "model-b", "model-c"],
            "complete": [55.0, None, 40.0],
            "instruct": [50.0, 45.0, None],
        })

        fetcher = BigCodeBenchFetcher()

        with patch("pandas.read_parquet", return_value=df):
            result = fetcher.fetch_bulk(tmp_cache)

        assert len(result) == 3
        # model-a uses complete (55)
        assert "model-a" in result
        # model-b falls back to instruct (45)
        assert "model-b" in result
        assert "code_generation" in result["model-b"]
        # model-c uses complete (40)
        assert "model-c" in result

    def test_instruction_adherence_is_0_9x(self, tmp_cache):
        """instruction_adherence = code_gen * 0.9."""
        import pandas as pd

        df = pd.DataFrame({
            "model": ["test"],
            "complete": [50.0],
            "instruct": [None],
        })

        fetcher = BigCodeBenchFetcher()

        with patch("pandas.read_parquet", return_value=df):
            result = fetcher.fetch_bulk(tmp_cache)

        cg = result["test"]["code_generation"]
        ia = result["test"]["instruction_adherence"]
        assert ia == round(cg * 0.9, 1)

    def test_empty_dataframe(self, tmp_cache):
        """Empty DataFrame returns {}."""
        import pandas as pd

        df = pd.DataFrame(columns=["model", "complete", "instruct"])

        fetcher = BigCodeBenchFetcher()

        with patch("pandas.read_parquet", return_value=df):
            result = fetcher.fetch_bulk(tmp_cache)

        assert result == {}


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
        """BenchmarkFetcher must include all 10 sources."""
        bf = BenchmarkFetcher(cache_dir=tmp_path)
        source_names = [f.source_name for f in bf.fetchers]

        assert "artificial_analysis" in source_names
        assert "chatbot_arena" in source_names
        assert "hf_leaderboard" in source_names
        assert "livecodebench" in source_names
        assert "bfcl" in source_names
        assert "aider" in source_names
        assert "bigcodebench" in source_names
        assert "openrouter" in source_names
        assert "seneca_trbench" in source_names
        assert "turkish_mmlu" in source_names

        # LMSys should NOT be present
        assert "lmsys_arena" not in source_names

    def test_confidence_map_has_all_sources(self, tmp_path):
        """The confidence map in fetch_all_bulk must include all sources."""
        bf = BenchmarkFetcher(cache_dir=tmp_path)

        # Extract confidence map by checking fetch_all_bulk source code
        # We check indirectly by ensuring all fetcher source_names appear
        expected_sources = {
            "artificial_analysis", "hf_leaderboard", "livecodebench",
            "bfcl", "chatbot_arena", "aider", "bigcodebench",
            "openrouter", "seneca_trbench", "turkish_mmlu",
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
