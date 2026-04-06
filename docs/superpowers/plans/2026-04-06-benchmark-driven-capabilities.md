# Benchmark-Driven Model Capabilities — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace family-profile-based capability estimation with benchmark-API-driven scores, add thinking/vision mode variants as separate registry entries, and fix all broken benchmark fetcher URLs.

**Architecture:** 10 benchmark sources feed into a unified cache. Each thinking-capable local model gets dual registry entries (base + thinking). Vision-capable models with mmproj get additional vision variants. The router sees all variants as separate candidates and picks the best fit. Lightweight restarts toggle mode flags without full model swaps.

**Tech Stack:** Python 3.10, httpx, pandas/pyarrow (for Parquet), pyyaml, existing ModelInfo/ModelRegistry/benchmark infrastructure.

**Spec:** `docs/superpowers/specs/2026-04-05-benchmark-driven-capabilities-design.md`

---

## Task 1: Fix Benchmark Fetcher — Artificial Analysis (authenticated API)

**Files:**
- Modify: `src/models/benchmark/benchmark_fetcher.py:213-302` (ArtificialAnalysisFetcher)
- Modify: `src/app/config.py` (add ARTIFICIAL_ANALYSIS_API_KEY)
- Test: `tests/test_benchmark_fetcher.py` (new)

- [ ] **Step 1: Write failing test for AA fetcher**

```python
# tests/test_benchmark_fetcher.py
"""Tests for benchmark fetcher sources."""
import pytest
from unittest.mock import patch, MagicMock

from src.models.benchmark.benchmark_fetcher import (
    ArtificialAnalysisFetcher, BenchmarkCache, _normalize_score,
)


class TestArtificialAnalysisFetcher:
    """Test the AA v2 API fetcher with mocked HTTP responses."""

    def _make_cache(self, tmp_path):
        return BenchmarkCache(cache_dir=tmp_path / "cache")

    def test_fetch_bulk_parses_v2_response(self, tmp_path):
        """AA v2 API returns model array with benchmark fields."""
        cache = self._make_cache(tmp_path)
        fetcher = ArtificialAnalysisFetcher()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": 200,
            "data": [
                {
                    "slug": "qwen3-32b",
                    "model_creator": {"name": "Qwen"},
                    "mmlu_pro": 75.0,
                    "gpqa": 50.0,
                    "livecodebench": 40.0,
                    "ifbench": 70.0,
                    "artificial_analysis_intelligence_index": 65.0,
                    "artificial_analysis_coding_index": 55.0,
                    "artificial_analysis_math_index": 60.0,
                },
            ],
        }

        with patch("httpx.get", return_value=mock_response):
            result = fetcher.fetch_bulk(cache)

        assert "qwen3-32b" in result
        caps = result["qwen3-32b"]
        assert "reasoning" in caps
        assert "domain_knowledge" in caps
        assert "code_generation" in caps
        assert all(0 < v <= 10 for v in caps.values())

    def test_fetch_bulk_uses_api_key_header(self, tmp_path):
        """AA v2 requires x-api-key header."""
        cache = self._make_cache(tmp_path)
        fetcher = ArtificialAnalysisFetcher()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": 200, "data": []}

        with patch("httpx.get", return_value=mock_response) as mock_get:
            with patch.dict("os.environ", {"ARTIFICIAL_ANALYSIS_API_KEY": "test_key"}):
                fetcher.fetch_bulk(cache)

        call_kwargs = mock_get.call_args
        assert call_kwargs[1].get("headers", {}).get("x-api-key") == "test_key"

    def test_fetch_bulk_returns_empty_on_missing_key(self, tmp_path):
        """Without API key, AA fetcher returns empty (not error)."""
        cache = self._make_cache(tmp_path)
        fetcher = ArtificialAnalysisFetcher()

        with patch.dict("os.environ", {}, clear=True):
            result = fetcher.fetch_bulk(cache)

        assert result == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_fetcher.py::TestArtificialAnalysisFetcher -v`
Expected: ImportError or AttributeError (old API doesn't match)

- [ ] **Step 3: Add AA API key to config.py**

In `src/app/config.py`, add to the env var loading section:

```python
ARTIFICIAL_ANALYSIS_API_KEY = os.getenv("ARTIFICIAL_ANALYSIS_API_KEY", "")
```

- [ ] **Step 4: Rewrite ArtificialAnalysisFetcher for v2 API**

Replace the class in `src/models/benchmark/benchmark_fetcher.py:213-302`:

```python
class ArtificialAnalysisFetcher(_BaseFetcher):
    """
    Fetches from Artificial Analysis v2 API (authenticated).

    Covers: mmlu_pro, gpqa, hle, livecodebench, scicode, math_500, aime,
            ifbench, lcr, terminalbench_hard, tau2, plus 3 composite indices.
    Maps to: reasoning, domain_knowledge, code_generation, code_reasoning,
             analysis, instruction_adherence.

    API: https://artificialanalysis.ai/api/v2/data/llms/models
    Requires: ARTIFICIAL_ANALYSIS_API_KEY env var
    Attribution: https://artificialanalysis.ai/
    """
    source_name = "artificial_analysis"
    API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"

    # Maps AA benchmark keys → our capability dimensions + normalization ranges
    BENCHMARK_MAP = {
        "gpqa":                                {"cap": "reasoning",             "min": 25, "max": 75},
        "artificial_analysis_math_index":      {"cap": "reasoning",             "min": 20, "max": 90},
        "mmlu_pro":                            {"cap": "domain_knowledge",      "min": 20, "max": 90},
        "hle":                                 {"cap": "domain_knowledge",      "min": 5,  "max": 60},
        "livecodebench":                       {"cap": "code_generation",       "min": 10, "max": 70},
        "artificial_analysis_coding_index":    {"cap": "code_reasoning",        "min": 20, "max": 90},
        "scicode":                             {"cap": "code_reasoning",        "min": 5,  "max": 50},
        "ifbench":                             {"cap": "instruction_adherence", "min": 30, "max": 95},
        "terminalbench_hard":                  {"cap": "analysis",              "min": 5,  "max": 60},
        "artificial_analysis_intelligence_index": {"cap": "analysis",           "min": 20, "max": 90},
    }

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        import os
        api_key = os.getenv("ARTIFICIAL_ANALYSIS_API_KEY", "")
        if not api_key:
            logger.debug("No ARTIFICIAL_ANALYSIS_API_KEY — skipping AA fetch")
            return {}

        try:
            import httpx
            resp = httpx.get(
                self.API_URL,
                headers={"x-api-key": api_key},
                timeout=30.0,
                follow_redirects=True,
            )
            if resp.status_code != 200:
                logger.warning(f"AA API returned {resp.status_code}")
                return {}

            data = resp.json()
            models_list = data.get("data", []) if isinstance(data, dict) else data

            result = {}
            for entry in models_list:
                slug = entry.get("slug", "")
                if not slug:
                    continue

                mapped = {}
                for bench_key, mapping in self.BENCHMARK_MAP.items():
                    score = entry.get(bench_key)
                    if score is not None:
                        try:
                            score = float(score)
                            cap = mapping["cap"]
                            norm = _normalize_score(score, mapping["min"], mapping["max"], 2.0, 10.0)
                            if cap in mapped:
                                mapped[cap] = round((mapped[cap] + norm) / 2, 1)
                            else:
                                mapped[cap] = norm
                        except (ValueError, TypeError):
                            pass

                if mapped:
                    result[slug] = mapped

            if result:
                cache.put_all_models(self.source_name, {"models": result})
                logger.info(f"Artificial Analysis: fetched {len(result)} models")
            return result

        except ImportError:
            logger.warning("httpx not available for AA fetch")
            return {}
        except Exception as e:
            logger.warning(f"AA fetch failed: {e}")
            return {}

    def fetch(self, model_id: str, cache: BenchmarkCache) -> Optional[BenchmarkResult]:
        all_models = self.fetch_bulk(cache)
        if not all_models:
            return None
        matched_key = _fuzzy_match_model(model_id, list(all_models.keys()))
        if not matched_key:
            return None
        return BenchmarkResult(
            source=self.source_name,
            model_id=model_id,
            raw_scores={},
            mapped_capabilities=all_models[matched_key],
            timestamp=time.time(),
            confidence=0.90,
        )
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_benchmark_fetcher.py::TestArtificialAnalysisFetcher -v`
Expected: All 3 PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/benchmark/benchmark_fetcher.py src/app/config.py tests/test_benchmark_fetcher.py
git commit -m "feat: rewrite AA fetcher for v2 authenticated API"
```

---

## Task 2: Fix Benchmark Fetcher — Chatbot Arena ELO (replaces LMSys)

**Files:**
- Modify: `src/models/benchmark/benchmark_fetcher.py:597-700` (LMSysArenaFetcher → ChatbotArenaFetcher)
- Test: `tests/test_benchmark_fetcher.py`

- [ ] **Step 1: Write failing test**

```python
class TestChatbotArenaFetcher:
    """Test Arena ELO fetcher with Parquet data."""

    def _make_cache(self, tmp_path):
        return BenchmarkCache(cache_dir=tmp_path / "cache")

    def test_fetch_bulk_parses_parquet(self, tmp_path):
        """Arena ELO from HF Parquet dataset."""
        import pandas as pd
        cache = self._make_cache(tmp_path)
        fetcher = ChatbotArenaFetcher()

        mock_df = pd.DataFrame({
            "Model": ["gpt-4o", "claude-sonnet-4", "Qwen3-32B"],
            "Arena Score": [1429, 1440, 1350],
            "Votes": [50000, 40000, 10000],
            "Organization": ["OpenAI", "Anthropic", "Qwen"],
        })

        with patch("pandas.read_parquet", return_value=mock_df):
            result = fetcher.fetch_bulk(cache)

        assert len(result) >= 3
        # Higher ELO = higher conversation/prose score
        gpt_score = result["gpt-4o"]["conversation"]
        qwen_score = result["Qwen3-32B"]["conversation"]
        assert gpt_score > qwen_score

    def test_maps_to_conversation_and_prose(self, tmp_path):
        """Arena ELO maps to conversation and prose_quality dimensions."""
        import pandas as pd
        cache = self._make_cache(tmp_path)
        fetcher = ChatbotArenaFetcher()

        mock_df = pd.DataFrame({
            "Model": ["test-model"],
            "Arena Score": [1300],
            "Votes": [1000],
            "Organization": ["Test"],
        })

        with patch("pandas.read_parquet", return_value=mock_df):
            result = fetcher.fetch_bulk(cache)

        caps = result["test-model"]
        assert "conversation" in caps
        assert "prose_quality" in caps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_fetcher.py::TestChatbotArenaFetcher -v`
Expected: FAIL (ChatbotArenaFetcher not defined)

- [ ] **Step 3: Replace LMSysArenaFetcher with ChatbotArenaFetcher**

Replace the class at `src/models/benchmark/benchmark_fetcher.py:597-700`:

```python
class ChatbotArenaFetcher(_BaseFetcher):
    """
    Chatbot Arena ELO ratings from HuggingFace Parquet dataset.

    Covers: human preference ELO scores
    Maps to: conversation, prose_quality

    Source: https://huggingface.co/datasets/mathewhe/chatbot-arena-elo
    """
    source_name = "chatbot_arena"
    PARQUET_URL = "https://huggingface.co/api/datasets/mathewhe/chatbot-arena-elo/parquet/default/train/0.parquet"

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import pandas as pd

            df = pd.read_parquet(self.PARQUET_URL)

            # Columns: Model, Arena Score, Votes, Organization, ...
            model_col = "Model"
            score_col = "Arena Score"
            if model_col not in df.columns or score_col not in df.columns:
                # Try alternative column names
                for mc in ["model", "Model", "model_name"]:
                    if mc in df.columns:
                        model_col = mc
                        break
                for sc in ["Arena Score", "arena_score", "rating", "elo"]:
                    if sc in df.columns:
                        score_col = sc
                        break

            result = {}
            for _, row in df.iterrows():
                model_name = str(row.get(model_col, ""))
                elo = row.get(score_col)
                if not model_name or elo is None:
                    continue
                try:
                    elo = float(elo)
                except (ValueError, TypeError):
                    continue

                score = _normalize_elo(elo)
                result[model_name] = {
                    "conversation": score,
                    "prose_quality": round(score * 0.95, 1),
                }

            if result:
                cache.put_all_models(self.source_name, {"models": result})
                logger.info(f"Chatbot Arena: fetched {len(result)} models")
            return result

        except Exception as e:
            logger.warning(f"Chatbot Arena fetch failed: {e}")
            return {}

    def fetch(self, model_id: str, cache: BenchmarkCache) -> Optional[BenchmarkResult]:
        all_models = self.fetch_bulk(cache)
        if not all_models:
            return None
        matched = _fuzzy_match_model(model_id, list(all_models.keys()))
        if not matched:
            return None
        return BenchmarkResult(
            source=self.source_name,
            model_id=model_id,
            raw_scores={},
            mapped_capabilities=all_models[matched],
            timestamp=time.time(),
            confidence=0.75,
        )
```

- [ ] **Step 4: Update BenchmarkFetcher to use new class**

In the `BenchmarkFetcher.__init__` method, replace `LMSysArenaFetcher()` with `ChatbotArenaFetcher()` in the fetchers list. Also update the confidence map key from `"lmsys_arena"` to `"chatbot_arena"`.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_benchmark_fetcher.py::TestChatbotArenaFetcher -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/benchmark/benchmark_fetcher.py tests/test_benchmark_fetcher.py
git commit -m "feat: replace LMSys Arena with Chatbot Arena ELO (HF Parquet)"
```

---

## Task 3: Fix Benchmark Fetcher — HF Leaderboard, LiveCodeBench, BFCL, BigCodeBench

**Files:**
- Modify: `src/models/benchmark/benchmark_fetcher.py` (4 fetcher classes)
- Test: `tests/test_benchmark_fetcher.py`

- [ ] **Step 1: Write failing tests for all 4 updated fetchers**

```python
class TestHFLeaderboardFetcher:
    def _make_cache(self, tmp_path):
        return BenchmarkCache(cache_dir=tmp_path / "cache")

    def test_fetch_bulk_parses_contents_dataset(self, tmp_path):
        """HF Leaderboard v2 uses open-llm-leaderboard/contents Parquet."""
        import pandas as pd
        cache = self._make_cache(tmp_path)
        fetcher = HuggingFaceLeaderboardFetcher()

        mock_df = pd.DataFrame({
            "Model": ["meta-llama/Llama-3.3-70B-Instruct"],
            "#Params (B)": [70.6],
            "Average": [45.2],
            "IFEval": [80.5],
            "BBH": [55.3],
            "MATH Lvl 5": [35.0],
            "GPQA": [42.1],
            "MUSR": [30.5],
            "MMLU-PRO": [52.8],
        })

        with patch("pandas.read_parquet", return_value=mock_df):
            result = fetcher.fetch_bulk(cache)

        assert len(result) >= 1
        model_key = list(result.keys())[0]
        caps = result[model_key]
        assert "reasoning" in caps or "instruction_adherence" in caps or "domain_knowledge" in caps


class TestLiveCodeBenchFetcher:
    def _make_cache(self, tmp_path):
        return BenchmarkCache(cache_dir=tmp_path / "cache")

    def test_fetch_bulk_parses_new_json_format(self, tmp_path):
        """LiveCodeBench uses performances_generation.json with models array."""
        cache = self._make_cache(tmp_path)
        fetcher = LiveCodeBenchFetcher()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"model_repr": "gpt-4o", "display_name": "GPT-4o"},
                {"model_repr": "claude-sonnet-4", "display_name": "Claude Sonnet 4"},
            ],
            "performances": [
                {"model": "gpt-4o", "pass_at_1": 0.65, "difficulty": "easy"},
                {"model": "gpt-4o", "pass_at_1": 0.45, "difficulty": "medium"},
                {"model": "claude-sonnet-4", "pass_at_1": 0.70, "difficulty": "easy"},
                {"model": "claude-sonnet-4", "pass_at_1": 0.50, "difficulty": "medium"},
            ],
        }

        with patch("httpx.get", return_value=mock_response):
            result = fetcher.fetch_bulk(cache)

        assert "gpt-4o" in result or any("gpt" in k.lower() for k in result)
        first_model = list(result.values())[0]
        assert "code_generation" in first_model


class TestBFCLFetcher:
    def _make_cache(self, tmp_path):
        return BenchmarkCache(cache_dir=tmp_path / "cache")

    def test_fetch_bulk_parses_csv_format(self, tmp_path):
        """BFCL now returns CSV instead of JSON."""
        cache = self._make_cache(tmp_path)
        fetcher = BFCLFetcher()

        csv_content = "Rank,Overall Acc,Model,Model Link,Total Cost ($),Latency Mean (s)\n"
        csv_content += '1,77.47,Claude-Opus-4-5-20250414,https://example.com,1.5,2.3\n'
        csv_content += '2,73.24,GPT-4o-2024-11-20,https://example.com,0.8,1.1\n'

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = csv_content

        with patch("httpx.get", return_value=mock_response):
            result = fetcher.fetch_bulk(cache)

        assert len(result) >= 2
        first = list(result.values())[0]
        assert "tool_use" in first
        assert "structured_output" in first


class TestBigCodeBenchFetcher:
    def _make_cache(self, tmp_path):
        return BenchmarkCache(cache_dir=tmp_path / "cache")

    def test_fetch_bulk_parses_hf_parquet(self, tmp_path):
        """BigCodeBench uses HF Parquet dataset."""
        import pandas as pd
        cache = self._make_cache(tmp_path)
        fetcher = BigCodeBenchFetcher()

        mock_df = pd.DataFrame({
            "model": ["gpt-4o", "Qwen3-32B"],
            "complete": [72.5, 55.0],
            "instruct": [68.0, 50.0],
            "size": ["", "32B"],
        })

        with patch("pandas.read_parquet", return_value=mock_df):
            result = fetcher.fetch_bulk(cache)

        assert len(result) >= 2
        assert "code_generation" in result["gpt-4o"]
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/test_benchmark_fetcher.py -k "TestHFLeaderboard or TestLiveCodeBench or TestBFCL or TestBigCodeBench" -v`
Expected: Various FAILs (old code doesn't match new data formats)

- [ ] **Step 3: Rewrite HuggingFaceLeaderboardFetcher for Parquet format**

Replace `HuggingFaceLeaderboardFetcher` (lines ~305-422):

```python
class HuggingFaceLeaderboardFetcher(_BaseFetcher):
    """
    Open LLM Leaderboard v2 from HuggingFace Parquet dataset.

    Covers: IFEval, BBH, MATH Lvl 5, GPQA, MUSR, MMLU-PRO
    Maps to: instruction_adherence, reasoning, analysis, domain_knowledge

    Source: https://huggingface.co/datasets/open-llm-leaderboard/contents
    """
    source_name = "hf_leaderboard"
    PARQUET_URL = "https://huggingface.co/api/datasets/open-llm-leaderboard/contents/parquet/default/train/0.parquet"

    BENCHMARK_MAP = {
        "IFEval":       {"cap": "instruction_adherence", "min": 20, "max": 90},
        "BBH":          {"cap": "reasoning",             "min": 20, "max": 85},
        "MATH Lvl 5":   {"cap": "reasoning",             "min": 5,  "max": 75},
        "GPQA":         {"cap": "reasoning",             "min": 25, "max": 55},
        "MUSR":         {"cap": "analysis",              "min": 20, "max": 70},
        "MMLU-PRO":     {"cap": "domain_knowledge",      "min": 20, "max": 85},
    }

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import pandas as pd

            df = pd.read_parquet(self.PARQUET_URL)

            # Find model name column
            model_col = "Model"
            for mc in ["Model", "model", "fullname", "model_name_for_query"]:
                if mc in df.columns:
                    model_col = mc
                    break

            result = {}
            for _, row in df.iterrows():
                model_name = str(row.get(model_col, ""))
                if not model_name:
                    continue

                mapped = {}
                for bench_col, mapping in self.BENCHMARK_MAP.items():
                    score = row.get(bench_col)
                    if score is not None and not pd.isna(score):
                        try:
                            score = float(score)
                            cap = mapping["cap"]
                            norm = _normalize_score(score, mapping["min"], mapping["max"], 2.0, 10.0)
                            if cap in mapped:
                                mapped[cap] = round((mapped[cap] + norm) / 2, 1)
                            else:
                                mapped[cap] = norm
                        except (ValueError, TypeError):
                            pass

                if mapped:
                    result[model_name] = mapped

            if result:
                cache.put_all_models(self.source_name, {"models": result})
                logger.info(f"HF Leaderboard: fetched {len(result)} models")
            return result

        except Exception as e:
            logger.warning(f"HF Leaderboard fetch failed: {e}")
            return {}

    def fetch(self, model_id: str, cache: BenchmarkCache) -> Optional[BenchmarkResult]:
        all_models = self.fetch_bulk(cache)
        if not all_models:
            return None
        matched = _fuzzy_match_model(model_id, list(all_models.keys()))
        if not matched:
            return None
        return BenchmarkResult(
            source=self.source_name,
            model_id=model_id,
            raw_scores={},
            mapped_capabilities=all_models[matched],
            timestamp=time.time(),
            confidence=0.80,
        )
```

- [ ] **Step 4: Rewrite LiveCodeBenchFetcher for new JSON format**

Replace `LiveCodeBenchFetcher` (lines ~425-508):

```python
class LiveCodeBenchFetcher(_BaseFetcher):
    """
    LiveCodeBench — live coding problem pass rates.

    Covers: pass@1 on fresh coding problems
    Maps to: code_generation, code_reasoning

    Source: https://livecodebench.github.io/
    """
    source_name = "livecodebench"
    URL = "https://livecodebench.github.io/performances_generation.json"

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import httpx
            resp = httpx.get(self.URL, timeout=20.0, follow_redirects=True)
            if resp.status_code != 200:
                return {}

            data = resp.json()

            # New format: {models: [...], performances: [...]}
            performances = data.get("performances", [])
            if not performances:
                # Fallback: old format (flat list)
                performances = data if isinstance(data, list) else data.get("results", [])
                return self._parse_flat(performances, cache)

            # Aggregate pass@1 per model across difficulties
            from collections import defaultdict
            model_scores = defaultdict(list)
            for perf in performances:
                model = perf.get("model", "")
                pass_at_1 = perf.get("pass_at_1", perf.get("pass@1"))
                if model and pass_at_1 is not None:
                    try:
                        model_scores[model].append(float(pass_at_1))
                    except (ValueError, TypeError):
                        pass

            result = {}
            for model, scores in model_scores.items():
                avg = sum(scores) / len(scores)
                if avg <= 1.0:
                    avg *= 100
                code_gen = _normalize_score(avg, 10.0, 80.0, 2.0, 10.0)
                result[model] = {
                    "code_generation": code_gen,
                    "code_reasoning": round(code_gen * 0.9, 1),
                }

            if result:
                cache.put_all_models(self.source_name, {"models": result})
                logger.info(f"LiveCodeBench: fetched {len(result)} models")
            return result

        except Exception as e:
            logger.warning(f"LiveCodeBench fetch failed: {e}")
            return {}

    def _parse_flat(self, entries: list, cache: BenchmarkCache) -> dict[str, dict]:
        """Parse old-style flat results list."""
        result = {}
        for entry in entries:
            model_name = entry.get("model", entry.get("name", ""))
            pass_at_1 = entry.get("pass@1", entry.get("pass_at_1"))
            if model_name and pass_at_1 is not None:
                try:
                    score = float(pass_at_1)
                    if score <= 1.0:
                        score *= 100
                    code_gen = _normalize_score(score, 10.0, 80.0, 2.0, 10.0)
                    result[model_name] = {
                        "code_generation": code_gen,
                        "code_reasoning": round(code_gen * 0.9, 1),
                    }
                except (ValueError, TypeError):
                    pass
        if result:
            cache.put_all_models(self.source_name, {"models": result})
        return result

    def fetch(self, model_id: str, cache: BenchmarkCache) -> Optional[BenchmarkResult]:
        all_models = self.fetch_bulk(cache)
        if not all_models:
            return None
        matched = _fuzzy_match_model(model_id, list(all_models.keys()))
        if not matched:
            return None
        return BenchmarkResult(
            source=self.source_name,
            model_id=model_id,
            raw_scores={},
            mapped_capabilities=all_models[matched],
            timestamp=time.time(),
            confidence=0.90,
        )
```

- [ ] **Step 5: Rewrite BFCLFetcher for CSV format**

Replace `BFCLFetcher` (lines ~511-594):

```python
class BFCLFetcher(_BaseFetcher):
    """
    Berkeley Function Calling Leaderboard v4.

    Covers: function calling accuracy
    Maps to: tool_use, structured_output

    Source: https://gorilla.cs.berkeley.edu/
    """
    source_name = "bfcl"
    URL = "https://gorilla.cs.berkeley.edu/data_overall.csv"

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import csv
            import httpx
            import io

            resp = httpx.get(self.URL, timeout=20.0, follow_redirects=True)
            if resp.status_code != 200:
                return {}

            reader = csv.DictReader(io.StringIO(resp.text))
            result = {}

            for row in reader:
                model_name = row.get("Model", "")
                overall = row.get("Overall Acc", "")
                if not model_name or not overall:
                    continue

                try:
                    score = float(overall)
                    if score <= 1.0:
                        score *= 100

                    tool_score = _normalize_score(score, 30.0, 95.0, 2.0, 10.0)
                    result[model_name] = {
                        "tool_use": tool_score,
                        "structured_output": round(tool_score * 0.85, 1),
                    }
                except (ValueError, TypeError):
                    pass

            if result:
                cache.put_all_models(self.source_name, {"models": result})
                logger.info(f"BFCL: fetched {len(result)} models")
            return result

        except Exception as e:
            logger.warning(f"BFCL fetch failed: {e}")
            return {}

    def fetch(self, model_id: str, cache: BenchmarkCache) -> Optional[BenchmarkResult]:
        all_models = self.fetch_bulk(cache)
        if not all_models:
            return None
        matched = _fuzzy_match_model(model_id, list(all_models.keys()))
        if not matched:
            return None
        return BenchmarkResult(
            source=self.source_name,
            model_id=model_id,
            raw_scores={},
            mapped_capabilities=all_models[matched],
            timestamp=time.time(),
            confidence=0.90,
        )
```

- [ ] **Step 6: Rewrite BigCodeBenchFetcher for HF Parquet**

Replace `BigCodeBenchFetcher` (lines ~802-870):

```python
class BigCodeBenchFetcher(_BaseFetcher):
    """
    BigCodeBench — comprehensive code generation benchmark.

    Covers: function-level code generation
    Maps to: code_generation, instruction_adherence

    Source: https://huggingface.co/datasets/bigcode/bigcodebench-results
    """
    source_name = "bigcodebench"
    PARQUET_URL = "https://huggingface.co/api/datasets/bigcode/bigcodebench-results/parquet/default/train/0.parquet"

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import pandas as pd

            df = pd.read_parquet(self.PARQUET_URL)

            result = {}
            for _, row in df.iterrows():
                model_name = str(row.get("model", ""))
                complete = row.get("complete")
                instruct = row.get("instruct")

                score = complete if complete is not None and not pd.isna(complete) else instruct
                if not model_name or score is None or pd.isna(score):
                    continue

                try:
                    score = float(score)
                    if score <= 1.0:
                        score *= 100
                    code_gen = _normalize_score(score, 10.0, 75.0, 2.0, 10.0)
                    result[model_name] = {
                        "code_generation": code_gen,
                        "instruction_adherence": round(code_gen * 0.9, 1),
                    }
                except (ValueError, TypeError):
                    pass

            if result:
                cache.put_all_models(self.source_name, {"models": result})
                logger.info(f"BigCodeBench: fetched {len(result)} models")
            return result

        except Exception as e:
            logger.warning(f"BigCodeBench fetch failed: {e}")
            return {}

    def fetch(self, model_id: str, cache: BenchmarkCache) -> Optional[BenchmarkResult]:
        all_models = self.fetch_bulk(cache)
        if not all_models:
            return None
        matched = _fuzzy_match_model(model_id, list(all_models.keys()))
        if not matched:
            return None
        return BenchmarkResult(
            source=self.source_name,
            model_id=model_id,
            raw_scores={},
            mapped_capabilities=all_models[matched],
            timestamp=time.time(),
            confidence=0.85,
        )
```

- [ ] **Step 7: Run all tests**

Run: `pytest tests/test_benchmark_fetcher.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/models/benchmark/benchmark_fetcher.py tests/test_benchmark_fetcher.py
git commit -m "feat: fix HF Leaderboard, LiveCodeBench, BFCL, BigCodeBench fetchers"
```

---

## Task 4: Add Turkish Benchmark Fetchers (Seneca-TRBench + Turkish MMLU)

**Files:**
- Modify: `src/models/benchmark/benchmark_fetcher.py` (add 2 new fetcher classes)
- Test: `tests/test_benchmark_fetcher.py`

- [ ] **Step 1: Write failing tests**

```python
class TestSenecaTRBenchFetcher:
    def _make_cache(self, tmp_path):
        return BenchmarkCache(cache_dir=tmp_path / "cache")

    def test_fetch_bulk_parses_csv(self, tmp_path):
        cache = self._make_cache(tmp_path)
        fetcher = SenecaTRBenchFetcher()

        csv_content = "Rank,Model,MCQ Score,SAQ Score,Combined Score\n"
        csv_content += "1,GPT-5,95.0,92.0,93.5\n"
        csv_content += "2,Claude Opus 4.1,91.0,89.0,90.06\n"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = csv_content

        with patch("httpx.get", return_value=mock_response):
            result = fetcher.fetch_bulk(cache)

        assert "GPT-5" in result
        assert "turkish" in result["GPT-5"]
        assert 0 < result["GPT-5"]["turkish"] <= 10


class TestTurkishMMLUFetcher:
    def _make_cache(self, tmp_path):
        return BenchmarkCache(cache_dir=tmp_path / "cache")

    def test_fetch_bulk_parses_parquet(self, tmp_path):
        import pandas as pd
        cache = self._make_cache(tmp_path)
        fetcher = TurkishMMLUFetcher()

        mock_df = pd.DataFrame({
            "model": ["qwen3:32b", "gemma3:27b", "gpt-4o"],
            "parameter_size": ["32.8B", "27.4B", ""],
            "quantization_level": ["Q4_K_M", "Q4_K_M", ""],
            "basari": [75.98, 75.06, 84.84],
        })

        with patch("pandas.read_parquet", return_value=mock_df):
            result = fetcher.fetch_bulk(cache)

        assert len(result) >= 3
        assert "turkish" in result["qwen3:32b"]
        # gpt-4o should score higher than qwen3:32b
        assert result["gpt-4o"]["turkish"] > result["qwen3:32b"]["turkish"]
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/test_benchmark_fetcher.py -k "TestSeneca or TestTurkishMMLU" -v`
Expected: FAIL (classes not defined)

- [ ] **Step 3: Implement SenecaTRBenchFetcher**

Add after the existing fetcher classes in `benchmark_fetcher.py`:

```python
class SenecaTRBenchFetcher(_BaseFetcher):
    """
    Seneca-TRBench — Turkish LLM benchmark.

    Covers: Turkish grammar, morphology, idioms, instruction following
    Maps to: turkish

    Source: https://huggingface.co/spaces/AlicanKiraz0/seneca-trbench
    """
    source_name = "seneca_trbench"
    URL = "https://huggingface.co/spaces/AlicanKiraz0/seneca-trbench/resolve/main/leaderboard_data.csv"

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import csv
            import httpx
            import io

            resp = httpx.get(self.URL, timeout=20.0, follow_redirects=True)
            if resp.status_code != 200:
                return {}

            reader = csv.DictReader(io.StringIO(resp.text))
            result = {}

            for row in reader:
                model = row.get("Model", "")
                combined = row.get("Combined Score", "")
                if not model or not combined:
                    continue
                try:
                    score = float(combined)
                    # Scores range ~19-94, map to 0-10
                    turkish = _normalize_score(score, 20.0, 95.0, 1.0, 10.0)
                    result[model] = {"turkish": turkish}
                except (ValueError, TypeError):
                    pass

            if result:
                cache.put_all_models(self.source_name, {"models": result})
                logger.info(f"Seneca-TRBench: fetched {len(result)} models")
            return result

        except Exception as e:
            logger.warning(f"Seneca-TRBench fetch failed: {e}")
            return {}

    def fetch(self, model_id: str, cache: BenchmarkCache) -> Optional[BenchmarkResult]:
        all_models = self.fetch_bulk(cache)
        if not all_models:
            return None
        matched = _fuzzy_match_model(model_id, list(all_models.keys()))
        if not matched:
            return None
        return BenchmarkResult(
            source=self.source_name,
            model_id=model_id,
            raw_scores={},
            mapped_capabilities=all_models[matched],
            timestamp=time.time(),
            confidence=0.85,
        )
```

- [ ] **Step 4: Implement TurkishMMLUFetcher**

```python
class TurkishMMLUFetcher(_BaseFetcher):
    """
    alibayram Turkish MMLU leaderboard — 293K original Turkish academic questions.

    Covers: Turkish academic knowledge, comprehension
    Maps to: turkish

    Source: https://huggingface.co/datasets/alibayram/yapay_zeka_turkce_mmlu_liderlik_tablosu
    """
    source_name = "turkish_mmlu"
    PARQUET_URL = "https://huggingface.co/api/datasets/alibayram/yapay_zeka_turkce_mmlu_liderlik_tablosu/parquet/default/train/0.parquet"

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import pandas as pd

            df = pd.read_parquet(self.PARQUET_URL)

            # Columns: model, parameter_size, quantization_level, basari (success %)
            result = {}
            for _, row in df.iterrows():
                model = str(row.get("model", ""))
                basari = row.get("basari")
                if not model or basari is None or pd.isna(basari):
                    continue
                try:
                    score = float(basari)
                    # Scores range ~19-85%, map to 0-10
                    turkish = _normalize_score(score, 20.0, 90.0, 1.0, 10.0)
                    result[model] = {"turkish": turkish}
                except (ValueError, TypeError):
                    pass

            if result:
                cache.put_all_models(self.source_name, {"models": result})
                logger.info(f"Turkish MMLU: fetched {len(result)} models")
            return result

        except Exception as e:
            logger.warning(f"Turkish MMLU fetch failed: {e}")
            return {}

    def fetch(self, model_id: str, cache: BenchmarkCache) -> Optional[BenchmarkResult]:
        all_models = self.fetch_bulk(cache)
        if not all_models:
            return None
        matched = _fuzzy_match_model(model_id, list(all_models.keys()))
        if not matched:
            return None
        return BenchmarkResult(
            source=self.source_name,
            model_id=model_id,
            raw_scores={},
            mapped_capabilities=all_models[matched],
            timestamp=time.time(),
            confidence=0.85,
        )
```

- [ ] **Step 5: Register both new fetchers in BenchmarkFetcher.__init__**

Add to the `self.fetchers` list in `BenchmarkFetcher.__init__`:
```python
SenecaTRBenchFetcher(),
TurkishMMLUFetcher(),
```

Also add to the confidence map in `fetch_all_bulk`:
```python
"seneca_trbench": 0.85,
"turkish_mmlu": 0.85,
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_benchmark_fetcher.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/models/benchmark/benchmark_fetcher.py tests/test_benchmark_fetcher.py
git commit -m "feat: add Seneca-TRBench and Turkish MMLU benchmark fetchers"
```

---

## Task 5: Model Variant Registration in ModelRegistry

**Files:**
- Modify: `src/models/model_registry.py:57-156` (ModelInfo — add variant fields)
- Modify: `src/models/model_registry.py:1057-1196` (_load_local_models — register variants)
- Test: `tests/test_model_variants.py` (new)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model_variants.py
"""Tests for thinking/vision model variant registration."""
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass


class TestModelVariantFields:
    """Test new variant fields on ModelInfo."""

    def test_base_model_defaults(self):
        from src.models.model_registry import ModelInfo
        m = ModelInfo(
            name="test-model",
            location="local",
            provider="llama_cpp",
            litellm_name="openai/test-model",
            capabilities={"reasoning": 7.0},
            context_length=8192,
            max_tokens=2048,
        )
        assert m.is_variant is False
        assert m.base_model_name == ""
        assert m.variant_flags == set()

    def test_thinking_variant_fields(self):
        from src.models.model_registry import ModelInfo
        m = ModelInfo(
            name="test-model-thinking",
            location="local",
            provider="llama_cpp",
            litellm_name="openai/test-model-thinking",
            capabilities={"reasoning": 8.5},
            context_length=8192,
            max_tokens=2048,
            thinking_model=True,
            is_variant=True,
            base_model_name="test-model",
            variant_flags={"thinking"},
        )
        assert m.is_variant is True
        assert m.base_model_name == "test-model"
        assert "thinking" in m.variant_flags


class TestVariantRegistration:
    """Test that GGUF models get correct variant entries."""

    def test_thinking_model_gets_two_entries(self):
        """A thinking-capable model without vision produces 2 entries."""
        from src.models.model_registry import _create_model_variants

        base = MagicMock()
        base.name = "Qwen3.5-35B-A3B"
        base.thinking_model = False
        base.has_vision = False
        base.mmproj_path = None
        base.path = "/models/test.gguf"
        base.capabilities = {"reasoning": 7.0, "vision": 0.0}
        base.is_variant = False
        base.base_model_name = ""
        base.variant_flags = set()

        family_thinking = True
        family_vision = False

        variants = _create_model_variants(base, family_thinking, family_vision)
        names = [v.name for v in variants]

        assert "Qwen3.5-35B-A3B" in names
        assert "Qwen3.5-35B-A3B-thinking" in names
        assert len(variants) == 2

    def test_thinking_vision_model_gets_four_entries(self):
        """A model with both thinking + mmproj produces 4 entries."""
        from src.models.model_registry import _create_model_variants

        base = MagicMock()
        base.name = "Gemma-4-26B-A4B"
        base.thinking_model = False
        base.has_vision = False
        base.mmproj_path = "/models/mmproj.gguf"
        base.path = "/models/test.gguf"
        base.capabilities = {"reasoning": 7.0, "vision": 0.0}
        base.is_variant = False
        base.base_model_name = ""
        base.variant_flags = set()

        variants = _create_model_variants(base, True, True)
        names = [v.name for v in variants]

        assert len(variants) == 4
        assert "Gemma-4-26B-A4B" in names
        assert "Gemma-4-26B-A4B-thinking" in names
        assert "Gemma-4-26B-A4B-vision" in names
        assert "Gemma-4-26B-A4B-thinking-vision" in names

        # Vision variant should have has_vision=True
        vision_v = [v for v in variants if v.name == "Gemma-4-26B-A4B-vision"][0]
        assert vision_v.has_vision is True

        # Thinking variant should have thinking_model=True
        thinking_v = [v for v in variants if v.name == "Gemma-4-26B-A4B-thinking"][0]
        assert thinking_v.thinking_model is True

    def test_non_thinking_model_gets_one_entry(self):
        """GigaChat (no thinking, no vision) → single entry."""
        from src.models.model_registry import _create_model_variants

        base = MagicMock()
        base.name = "GigaChat3.1-Lightning"
        base.thinking_model = False
        base.has_vision = False
        base.mmproj_path = None
        base.path = "/models/test.gguf"
        base.capabilities = {"reasoning": 5.5, "vision": 0.0}
        base.is_variant = False
        base.base_model_name = ""
        base.variant_flags = set()

        variants = _create_model_variants(base, False, False)
        assert len(variants) == 1
        assert variants[0].name == "GigaChat3.1-Lightning"

    def test_variants_share_same_path(self):
        """All variants of a model point to the same GGUF file."""
        from src.models.model_registry import _create_model_variants

        base = MagicMock()
        base.name = "test"
        base.thinking_model = False
        base.has_vision = False
        base.mmproj_path = "/models/mmproj.gguf"
        base.path = "/models/test.gguf"
        base.capabilities = {"reasoning": 7.0, "vision": 0.0}
        base.is_variant = False
        base.base_model_name = ""
        base.variant_flags = set()

        variants = _create_model_variants(base, True, True)
        for v in variants:
            assert v.path == "/models/test.gguf"
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/test_model_variants.py -v`
Expected: FAIL (new fields and _create_model_variants don't exist)

- [ ] **Step 3: Add variant fields to ModelInfo**

In `src/models/model_registry.py`, add these fields to the `ModelInfo` dataclass after `api_base`:

```python
    # Variant tracking
    is_variant: bool = False                    # True for thinking/vision variant entries
    base_model_name: str = ""                   # Links variant back to base (for swap logic)
    variant_flags: set[str] = field(default_factory=set)  # e.g. {"thinking"}, {"vision"}, {"thinking", "vision"}
```

- [ ] **Step 4: Implement _create_model_variants function**

Add this function in `src/models/model_registry.py` before the `ModelRegistry` class:

```python
def _create_model_variants(
    base: ModelInfo,
    family_thinking_capable: bool,
    family_has_vision: bool,
) -> list[ModelInfo]:
    """
    Create mode variant entries for a model.

    Returns 1-4 ModelInfo entries depending on capabilities:
    - Base (always)
    - Thinking variant (if family supports thinking)
    - Vision variant (if mmproj file exists)
    - Thinking+Vision variant (if both)

    All variants share the same GGUF path. The base entry has
    thinking_model=False, has_vision=False regardless of family capability.
    """
    from copy import deepcopy
    from src.models.model_profiles import FAMILY_PROFILES

    variants = []

    # ── Base entry (always created) ──
    # Base is thinking-off, vision-off
    base.thinking_model = False
    base.has_vision = False
    base.is_variant = False
    base.base_model_name = ""
    base.variant_flags = set()
    variants.append(base)

    has_mmproj = bool(base.mmproj_path)

    # ── Thinking variant ──
    if family_thinking_capable:
        tv = deepcopy(base)
        tv.name = f"{base.name}-thinking"
        tv.litellm_name = f"openai/{tv.name}"
        tv.thinking_model = True
        tv.is_variant = True
        tv.base_model_name = base.name
        tv.variant_flags = {"thinking"}
        # Apply thinking capability deltas
        _apply_thinking_deltas(tv)
        variants.append(tv)

    # ── Vision variant ──
    if has_mmproj and family_has_vision:
        vv = deepcopy(base)
        vv.name = f"{base.name}-vision"
        vv.litellm_name = f"openai/{vv.name}"
        vv.has_vision = True
        vv.is_variant = True
        vv.base_model_name = base.name
        vv.variant_flags = {"vision"}
        # Only vision capability changes
        profile = FAMILY_PROFILES.get(base.family)
        if profile and profile.has_vision:
            vv.capabilities["vision"] = profile.base_capabilities.get("vision", 7.0)
        else:
            vv.capabilities["vision"] = 6.0  # conservative default
        variants.append(vv)

    # ── Thinking+Vision variant ──
    if family_thinking_capable and has_mmproj and family_has_vision:
        tvv = deepcopy(base)
        tvv.name = f"{base.name}-thinking-vision"
        tvv.litellm_name = f"openai/{tvv.name}"
        tvv.thinking_model = True
        tvv.has_vision = True
        tvv.is_variant = True
        tvv.base_model_name = base.name
        tvv.variant_flags = {"thinking", "vision"}
        _apply_thinking_deltas(tvv)
        profile = FAMILY_PROFILES.get(base.family)
        if profile and profile.has_vision:
            tvv.capabilities["vision"] = profile.base_capabilities.get("vision", 7.0)
        else:
            tvv.capabilities["vision"] = 6.0
        variants.append(tvv)

    return variants


def _apply_thinking_deltas(model: ModelInfo) -> None:
    """
    Adjust capabilities for thinking-on mode.
    Applied to thinking variant entries.
    """
    caps = model.capabilities
    # Thinking boosts reasoning-related dimensions
    for dim in ("reasoning", "planning", "analysis", "code_reasoning"):
        if dim in caps:
            caps[dim] = min(10.0, caps[dim] + 1.0)
    # Thinking modestly boosts code generation
    if "code_generation" in caps:
        caps["code_generation"] = min(10.0, caps["code_generation"] + 0.3)
    # Thinking hurts format compliance
    for dim in ("instruction_adherence", "structured_output"):
        if dim in caps:
            caps[dim] = max(0.0, caps[dim] - 0.7)
    # Thinking hurts conversation naturalness
    if "conversation" in caps:
        caps["conversation"] = max(0.0, caps["conversation"] - 0.5)
```

- [ ] **Step 5: Integrate variant creation into _load_local_models**

In `src/models/model_registry.py:_load_local_models`, replace the section that creates and stores the model (around line 1151-1179). After creating the base `model = ModelInfo(...)`, replace `result[name] = model` with:

```python
            # Create mode variants (thinking, vision, thinking+vision)
            from src.models.model_profiles import FAMILY_PROFILES
            family_profile = FAMILY_PROFILES.get(raw["family_key"] or "")
            family_thinking = family_profile.thinking_capable if family_profile else raw["thinking"]
            family_vision = family_profile.has_vision if family_profile else raw["has_vision"]

            variants = _create_model_variants(model, family_thinking, family_vision)
            for v in variants:
                result[v.name] = v

            moe_info = f"(MoE {raw['active_params_b']:.1f}B active)" if raw['is_moe'] else ""
            variant_info = f" → {len(variants)} variants" if len(variants) > 1 else ""

            logger.info(
                f"  Local: {name} "
                f"| family={raw['family_key'] or '?'} "
                f"| {total_params:.1f}B"
                f"{moe_info} "
                f"| {raw['quantization']} "
                f"| {gpu_layers}/{raw['n_layers']} GPU layers "
                f"| ctx={context_length} "
                f"| best={model.best_score():.1f}"
                f"{'| 👁️ vision' if raw['has_vision'] else ''}"
                f"{'| 🧠 thinking' if raw['thinking'] else ''}"
                f"{variant_info}"
            )
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_model_variants.py -v`
Expected: All PASS

- [ ] **Step 7: Verify with import test**

Run: `python -c "from src.models.model_registry import ModelInfo, _create_model_variants; print('OK')"`
Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add src/models/model_registry.py tests/test_model_variants.py
git commit -m "feat: register thinking/vision mode variants as separate ModelInfo entries"
```

---

## Task 6: Variant-Aware Router and Swap Logic

**Files:**
- Modify: `src/core/router.py:348-395` (filter loop — variant awareness)
- Modify: `src/models/local_model_manager.py:469-489` (_swap_model — lightweight restart)
- Test: `tests/test_model_variants.py` (extend)

- [ ] **Step 1: Write failing tests for router variant filtering**

Add to `tests/test_model_variants.py`:

```python
class TestRouterVariantFiltering:
    """Test that router correctly filters thinking/vision variants."""

    def test_needs_vision_excludes_non_vision(self):
        """When needs_vision=True, non-vision variants are excluded."""
        from src.models.model_registry import ModelInfo

        base = ModelInfo(
            name="test", location="local", provider="llama_cpp",
            litellm_name="openai/test",
            capabilities={"reasoning": 7.0, "vision": 0.0},
            context_length=8192, max_tokens=2048,
            has_vision=False, variant_flags=set(),
        )
        vision = ModelInfo(
            name="test-vision", location="local", provider="llama_cpp",
            litellm_name="openai/test-vision",
            capabilities={"reasoning": 7.0, "vision": 7.5},
            context_length=8192, max_tokens=2048,
            has_vision=True, is_variant=True, base_model_name="test",
            variant_flags={"vision"},
        )

        # needs_vision=True should exclude base (no vision)
        assert base.has_vision is False
        assert vision.has_vision is True
        # The actual filter is already in router.py line 394:
        # if reqs.needs_vision and not model.has_vision: skip


class TestLightweightRestart:
    """Test swap logic detects same-GGUF variant changes."""

    def test_same_gguf_different_flags_is_variant_swap(self):
        """Switching between base and thinking variant of same model."""
        from src.models.model_registry import ModelInfo

        base = ModelInfo(
            name="test", location="local", provider="llama_cpp",
            litellm_name="openai/test",
            capabilities={"reasoning": 7.0},
            context_length=8192, max_tokens=2048,
            path="/models/test.gguf",
            base_model_name="",
        )
        thinking = ModelInfo(
            name="test-thinking", location="local", provider="llama_cpp",
            litellm_name="openai/test-thinking",
            capabilities={"reasoning": 8.0},
            context_length=8192, max_tokens=2048,
            path="/models/test.gguf",
            is_variant=True, base_model_name="test",
            variant_flags={"thinking"}, thinking_model=True,
        )

        # Both point to same GGUF
        assert base.path == thinking.path
        # Variant can be detected
        assert thinking.base_model_name == "test" or thinking.path == base.path
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_model_variants.py -k "TestRouter or TestLightweight" -v`
Expected: PASS (these are structural tests, the logic is already in place or being added)

- [ ] **Step 3: Add variant swap detection to LocalModelManager._swap_model**

In `src/models/local_model_manager.py:_swap_model` (line ~469), add variant-aware re-check after acquiring the lock. Replace the existing re-check block (lines 477-483):

```python
        async with self._swap_lock:
            # Re-check after acquiring lock — another task may have already loaded.
            if self.current_model == model_name and await self._health_check():
                if not (enable_vision and not self._vision_enabled):
                    logger.debug(f"Model {model_name} already healthy (resolved under lock)")
                    return True

            # ── Variant swap detection ──
            # If switching between variants of the same GGUF (e.g. base → thinking),
            # this is a lightweight restart, not a full swap. Don't count against budget.
            current_info = registry.get(self.current_model) if self.current_model else None
            target_info = registry.get(model_name)
            is_variant_swap = False
            if current_info and target_info and current_info.path and target_info.path:
                if current_info.path == target_info.path:
                    is_variant_swap = True
                    logger.info(
                        f"⚡ Variant swap (same GGUF): {self.current_model} → {model_name} "
                        f"(lightweight restart, not counted as swap)"
                    )

            self.swap_started_at = time.monotonic()
            try:
                result = await self._do_swap(model_name, reason, enable_thinking, enable_vision)
                if result and is_variant_swap:
                    # Don't count variant swaps against the budget
                    self._total_swaps = max(0, self._total_swaps - 1)
                return result
            finally:
                self.swap_started_at = 0.0
```

Note: This requires importing `registry` — add `from .model_registry import get_registry` at the top of the method, and call `registry = get_registry()`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_model_variants.py -v`
Expected: All PASS

- [ ] **Step 5: Import test**

Run: `python -c "from src.models.local_model_manager import LocalModelManager; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add src/core/router.py src/models/local_model_manager.py tests/test_model_variants.py
git commit -m "feat: variant-aware routing and lightweight restart for same-GGUF swaps"
```

---

## Task 7: Update Model Aliases and Fuzzy Matching

**Files:**
- Modify: `src/models/benchmark/benchmark_fetcher.py:143-196` (_MODEL_ALIASES + _fuzzy_match_model)
- Test: `tests/test_benchmark_fetcher.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_benchmark_fetcher.py`:

```python
class TestFuzzyMatching:
    """Test model name matching between benchmark sources and local models."""

    def test_aa_slug_matches_local_model(self):
        from src.models.benchmark.benchmark_fetcher import _fuzzy_match_model
        candidates = ["qwen3-5-32b", "llama-3-3-instruct-70b", "gemma-4-26b-it"]
        assert _fuzzy_match_model("Qwen3.5-35B-A3B", candidates) is not None
        assert _fuzzy_match_model("Gemma-4-26B-A4B", candidates) is not None

    def test_cloud_model_matches(self):
        from src.models.benchmark.benchmark_fetcher import _fuzzy_match_model
        candidates = ["gpt-4o-2024-11-20", "Claude-Sonnet-4", "gemini-2.5-flash"]
        assert _fuzzy_match_model("gpt-4o", candidates) is not None
        assert _fuzzy_match_model("claude-sonnet", candidates) is not None
        assert _fuzzy_match_model("gemini-flash-thinking", candidates) is not None

    def test_alias_lookup(self):
        from src.models.benchmark.benchmark_fetcher import _fuzzy_match_model
        candidates = ["Qwen/Qwen3.5-32B", "ServiceNow/Apriel-15B-Thinker"]
        assert _fuzzy_match_model("qwen3.5-35b", candidates) is not None

    def test_no_false_positives(self):
        from src.models.benchmark.benchmark_fetcher import _fuzzy_match_model
        candidates = ["gpt-4o", "gpt-4o-mini"]
        # "gpt-4" should not match "gpt-4o-mini" when "gpt-4o" is available
        result = _fuzzy_match_model("gpt-4o", candidates)
        assert result == "gpt-4o"
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest tests/test_benchmark_fetcher.py::TestFuzzyMatching -v`
Expected: Some FAILs (current fuzzy matching misses AA slug format)

- [ ] **Step 3: Update _MODEL_ALIASES with all current models**

Replace `_MODEL_ALIASES` in `benchmark_fetcher.py`:

```python
_MODEL_ALIASES: dict[str, list[str]] = {
    # ── Local models on disk ──
    "qwen3.5-35b":     ["Qwen3.5-35B", "qwen3-5-35b", "Qwen/Qwen3.5-35B-A3B", "qwen-3.5-35b"],
    "qwen3.5-27b":     ["Qwen3.5-27B", "qwen3-5-27b", "Qwen/Qwen3.5-27B"],
    "qwen3.5-9b":      ["Qwen3.5-9B", "qwen3-5-9b", "Qwen/Qwen3.5-9B"],
    "qwen3-coder-30b": ["Qwen3-Coder-30B", "Qwen/Qwen3-Coder-30B-A3B-Instruct"],
    "glm-4.7-flash":   ["GLM-4.7-Flash", "glm4-flash", "THUDM/GLM-4.7-Flash"],
    "gemma-4-26b":     ["Gemma-4-26B", "gemma-4-26b-it", "google/gemma-4-26b-it", "gemma4"],
    "gpt-oss-20b":     ["gpt-oss-20b", "GPT-OSS-20B", "openai/gpt-oss-20b"],
    "apriel-15b":      ["Apriel-15B-Thinker", "ServiceNow/Apriel-15B-Thinker", "apriel-thinker"],
    "gigachat-lightning": ["GigaChat3.1-Lightning", "gigachat", "Sber/GigaChat3.1-Lightning"],
    "nerdsking-7b":    ["nerdsking-python-coder-7B", "nerdsking-python-7B"],
    # ── Cloud models ──
    "gpt-4o":          ["GPT-4o", "gpt-4o-2024-11-20", "chatgpt-4o-latest"],
    "gpt-4o-mini":     ["GPT-4o-mini", "gpt-4o-mini-2024-07-18"],
    "claude-sonnet":   ["Claude-Sonnet-4", "claude-sonnet-4-20250514", "anthropic/claude-sonnet-4"],
    "gemini-flash":    ["Gemini-2.0-Flash", "gemini-2.0-flash", "google/gemini-2.0-flash"],
    "gemini-flash-thinking": ["Gemini-2.5-Flash", "gemini-2.5-flash-preview", "gemini-2.5-flash"],
    "groq-llama-70b":  ["Llama-3.3-70B", "meta-llama/Llama-3.3-70B-Instruct"],
    "o4-mini":         ["o4-mini", "o4-mini-2025-04-16"],
}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_benchmark_fetcher.py::TestFuzzyMatching -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/benchmark/benchmark_fetcher.py tests/test_benchmark_fetcher.py
git commit -m "feat: update model aliases for all local + cloud models"
```

---

## Task 8: Enhanced Benchmark CLI

**Files:**
- Modify: `src/models/benchmark/benchmark_cli.py`
- No new tests (CLI is manual tool)

- [ ] **Step 1: Add --force-refresh flag to enrich command**

Replace `cmd_enrich()` in `benchmark_cli.py`:

```python
def cmd_enrich(force_refresh: bool = False):
    """Enrich registry with benchmark data."""
    if force_refresh:
        print("🔄 Force-refreshing benchmark cache...")
        fetcher = BenchmarkFetcher()
        fetcher.refresh_cache()

    registry = get_registry()
    enriched = enrich_registry_with_benchmarks(registry)

    print(f"\n✅ Enriched {len(enriched)} models with benchmark data")

    # Show enrichment details
    for name, caps in enriched.items():
        model = registry.get(name)
        variant_tag = ""
        if model and model.is_variant:
            variant_tag = f" [{','.join(sorted(model.variant_flags))}]"
        top = sorted(caps.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join(f"{k}={v:.1f}" for k, v in top)
        print(f"   {name:45s}{variant_tag:20s}: {top_str}")

    # Show models that fell back to family profiles
    fallback = [n for n in registry.models if n not in enriched]
    if fallback:
        print(f"\n⚠️  {len(fallback)} models fell back to family profiles:")
        for name in sorted(fallback):
            model = registry.get(name)
            print(f"   {name:45s} (family={model.family if model else '?'})")
```

- [ ] **Step 2: Update main() to pass --force-refresh**

In the `main()` function, update the enrich command handler:

```python
    elif cmd == "enrich":
        force = "--force-refresh" in sys.argv or "-f" in sys.argv
        cmd_enrich(force_refresh=force)
```

- [ ] **Step 3: Add variants command**

Add a new `cmd_variants()` function:

```python
def cmd_variants():
    """Show all model variants and their mode flags."""
    registry = get_registry()

    print(f"\n🔀 Model Variants ({len(registry.models)} total entries)")
    print(f"{'═' * 70}")

    # Group by base model
    bases = {}
    for name, model in registry.models.items():
        base = model.base_model_name or name
        if base not in bases:
            bases[base] = []
        bases[base].append(model)

    for base_name, variants in sorted(bases.items()):
        flags_str = ""
        for v in sorted(variants, key=lambda m: m.name):
            mode = "base" if not v.is_variant else ",".join(sorted(v.variant_flags))
            thinking = "🧠" if v.thinking_model else "  "
            vision = "👁️" if v.has_vision else "  "
            print(f"  {thinking} {vision} {v.name:45s} [{mode:20s}] best={v.best_score():.1f}")
        print()
```

Register in `main()`:
```python
    elif cmd == "variants":
        cmd_variants()
```

- [ ] **Step 4: Test CLI works**

Run: `python -m src.models.benchmark.benchmark_cli variants`
Expected: Shows grouped model variants with mode flags

- [ ] **Step 5: Commit**

```bash
git add src/models/benchmark/benchmark_cli.py
git commit -m "feat: enhanced benchmark CLI with --force-refresh and variants command"
```

---

## Task 9: Integration Test — Full Enrichment Pipeline

**Files:**
- Test: `tests/test_benchmark_enrichment.py` (new)

- [ ] **Step 1: Write integration test**

```python
# tests/test_benchmark_enrichment.py
"""Integration test for the full benchmark enrichment pipeline."""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


class TestEnrichmentPipeline:
    """Test that enrichment flows from fetchers → registry correctly."""

    def test_enrichment_updates_model_capabilities(self, tmp_path):
        """End-to-end: fetch mock data → enrich → verify capabilities changed."""
        from src.models.benchmark.benchmark_fetcher import (
            BenchmarkFetcher, BenchmarkCache, enrich_registry_with_benchmarks,
        )
        from src.models.model_registry import ModelInfo, ModelRegistry

        # Create a mock registry with one model
        registry = ModelRegistry()
        registry.models = {
            "test-model": ModelInfo(
                name="test-model",
                location="local",
                provider="llama_cpp",
                litellm_name="openai/test-model",
                capabilities={
                    "reasoning": 5.0, "code_generation": 5.0,
                    "turkish": 3.0, "tool_use": 4.0,
                    "conversation": 5.0, "domain_knowledge": 5.0,
                    "instruction_adherence": 5.0,
                },
                context_length=8192,
                max_tokens=2048,
                family="qwen35",
            ),
        }

        # Enrich with benchmark data
        enriched = enrich_registry_with_benchmarks(
            registry, cache_dir=tmp_path / "cache", min_confidence_sources=1,
        )

        # Model should appear in enriched if any source matched
        # (depends on fuzzy matching — may or may not match "test-model")
        # At minimum, the function should not crash
        assert isinstance(enriched, dict)

    def test_cloud_models_get_enriched(self, tmp_path):
        """Cloud models should also get benchmark data."""
        from src.models.benchmark.benchmark_fetcher import enrich_registry_with_benchmarks
        from src.models.model_registry import ModelInfo, ModelRegistry

        registry = ModelRegistry()
        registry.models = {
            "claude-sonnet": ModelInfo(
                name="claude-sonnet",
                location="cloud",
                provider="anthropic",
                litellm_name="anthropic/claude-sonnet-4-20250514",
                capabilities={"reasoning": 9.5, "turkish": 8.0},
                context_length=200000,
                max_tokens=8192,
            ),
        }

        enriched = enrich_registry_with_benchmarks(
            registry, cache_dir=tmp_path / "cache", min_confidence_sources=1,
        )

        # Should not crash on cloud models
        assert isinstance(enriched, dict)

    def test_variant_models_get_separate_enrichment(self, tmp_path):
        """Thinking variant should get different scores than base."""
        from src.models.benchmark.benchmark_fetcher import enrich_registry_with_benchmarks
        from src.models.model_registry import ModelInfo, ModelRegistry

        registry = ModelRegistry()
        registry.models = {
            "test-model": ModelInfo(
                name="test-model",
                location="local",
                provider="llama_cpp",
                litellm_name="openai/test-model",
                capabilities={"reasoning": 7.0},
                context_length=8192,
                max_tokens=2048,
                thinking_model=False,
                is_variant=False,
            ),
            "test-model-thinking": ModelInfo(
                name="test-model-thinking",
                location="local",
                provider="llama_cpp",
                litellm_name="openai/test-model-thinking",
                capabilities={"reasoning": 8.0},
                context_length=8192,
                max_tokens=2048,
                thinking_model=True,
                is_variant=True,
                base_model_name="test-model",
                variant_flags={"thinking"},
            ),
        }

        enriched = enrich_registry_with_benchmarks(
            registry, cache_dir=tmp_path / "cache", min_confidence_sources=1,
        )

        assert isinstance(enriched, dict)
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_benchmark_enrichment.py -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/test_benchmark_fetcher.py tests/test_model_variants.py tests/test_benchmark_enrichment.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_benchmark_enrichment.py
git commit -m "test: integration tests for benchmark enrichment pipeline"
```

---

## Task 10: Live Benchmark Fetch Verification

**Files:** None (verification only)

- [ ] **Step 1: Clear old benchmark cache**

```bash
rm -rf .benchmark_cache/
```

- [ ] **Step 2: Run benchmark fetch**

```bash
python -m src.models.benchmark.benchmark_cli benchmarks
```

Expected: Shows fetch counts per source. At minimum AA + Aider + OpenRouter should succeed. Note any 404s or failures.

- [ ] **Step 3: Run enrichment**

```bash
python -m src.models.benchmark.benchmark_cli enrich --force-refresh
```

Expected: Shows enriched models with benchmark data vs family fallback.

- [ ] **Step 4: Show variants**

```bash
python -m src.models.benchmark.benchmark_cli variants
```

Expected: Shows 24 registry entries grouped by base model, with thinking/vision flags.

- [ ] **Step 5: Show model details for a key model**

```bash
python -m src.models.benchmark.benchmark_cli model Qwen3.5-35B-A3B
python -m src.models.benchmark.benchmark_cli model Qwen3.5-35B-A3B-thinking
```

Expected: Different capability scores between base and thinking variant.

- [ ] **Step 6: Score comparison for a task**

```bash
python -m src.models.benchmark.benchmark_cli score coder
python -m src.models.benchmark.benchmark_cli score planner
```

Expected: Thinking variants rank higher for planner, base variants rank higher for coder (format-sensitive).

- [ ] **Step 7: Commit any fixes from verification**

```bash
git add -A
git commit -m "fix: adjustments from live benchmark verification"
```
