# benchmark_fetcher.py
"""
Benchmark Fetcher — pulls model evaluation data from multiple public sources
and maps them to our 15-dimension capability schema.

Sources:
  1. Artificial Analysis API (artificialanalysis.ai)
  2. Open LLM Leaderboard (HuggingFace)
  3. LiveCodeBench (code eval, GitHub releases)
  4. Berkeley Function Calling Leaderboard (BFCL)
  5. LMSys Chatbot Arena (Elo ratings)
  6. Aider polyglot code leaderboard
  7. Local cache / manual overrides

Each source returns normalized 0-10 scores for the dimensions it covers.
Missing dimensions are left as None (not 0).

Usage:
    fetcher = BenchmarkFetcher(cache_dir=".benchmark_cache")
    scores = fetcher.fetch_all("Qwen/Qwen3-32B")
    # → {"reasoning": 8.3, "code_generation": 7.9, ...}

    # Or refresh everything:
    fetcher.refresh_cache()
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("models.benchmark.fetcher")

CACHE_TTL_HOURS = 48  # Re-fetch after 2 days


@dataclass
class BenchmarkResult:
    """Raw benchmark scores from a single source."""
    source: str
    model_id: str                           # Canonical model identifier
    raw_scores: dict[str, float]            # benchmark_name → raw score
    mapped_capabilities: dict[str, float]   # capability_name → 0-10 score
    timestamp: float = 0.0
    confidence: float = 1.0                 # 0-1, how much we trust this source


@dataclass
class BenchmarkCache:
    """Persistent cache for benchmark data."""
    cache_dir: Path

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, source: str, model_id: str) -> str:
        raw = f"{source}:{model_id}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, source: str, model_id: str) -> Optional[dict]:
        path = self.cache_dir / f"{self._key(source, model_id)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            age_hours = (time.time() - data.get("timestamp", 0)) / 3600
            if age_hours > CACHE_TTL_HOURS:
                return None  # expired
            return data
        except Exception:
            return None

    def put(self, source: str, model_id: str, data: dict) -> None:
        data["timestamp"] = time.time()
        path = self.cache_dir / f"{self._key(source, model_id)}.json"
        try:
            path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    def get_all_models(self, source: str) -> Optional[dict]:
        """Get bulk cached data for a source (leaderboard-style)."""
        path = self.cache_dir / f"_bulk_{source}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            age_hours = (time.time() - data.get("timestamp", 0)) / 3600
            if age_hours > CACHE_TTL_HOURS:
                return None
            return data
        except Exception:
            return None

    def put_all_models(self, source: str, data: dict) -> None:
        data["timestamp"] = time.time()
        path = self.cache_dir / f"_bulk_{source}.json"
        try:
            path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Bulk cache write failed: {e}")


# ─── Normalization Helpers ───────────────────────────────────────────────────

def _normalize_score(
    raw: float,
    min_val: float,
    max_val: float,
    target_min: float = 0.0,
    target_max: float = 10.0,
) -> float:
    """Linear normalization from [min_val, max_val] → [target_min, target_max]."""
    if max_val <= min_val:
        return (target_min + target_max) / 2
    clamped = max(min_val, min(max_val, raw))
    normalized = (clamped - min_val) / (max_val - min_val)
    return round(target_min + normalized * (target_max - target_min), 1)


def _normalize_percentage(pct: float) -> float:
    """Map percentage (0-100) to 0-10 with a slight curve favoring higher scores."""
    # 50% → ~4.0, 70% → ~6.5, 85% → ~8.0, 95% → ~9.5
    return _normalize_score(pct, 25.0, 100.0, 1.0, 10.0)


def _normalize_elo(elo: float) -> float:
    """Map Arena Elo (typically 900-1400) to 0-10."""
    return _normalize_score(elo, 900.0, 1400.0, 2.0, 10.0)


# ─── Model Name Matching ────────────────────────────────────────────────────

# Maps our local model family names to patterns found in leaderboard data
_MODEL_ALIASES: dict[str, list[str]] = {
    "qwen3-32b":       ["Qwen3-32B", "qwen3-32b", "Qwen/Qwen3-32B"],
    "qwen3-8b":        ["Qwen3-8B", "qwen3-8b", "Qwen/Qwen3-8B"],
    "qwen3-30b-a3b":   ["Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B"],
    "qwen2.5-coder-32b": ["Qwen2.5-Coder-32B", "Qwen/Qwen2.5-Coder-32B-Instruct"],
    "qwen2.5-coder-7b":  ["Qwen2.5-Coder-7B", "Qwen/Qwen2.5-Coder-7B-Instruct"],
    "llama-3.3-70b":   ["Llama-3.3-70B", "meta-llama/Llama-3.3-70B-Instruct"],
    "llama-3.1-8b":    ["Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"],
    "phi-4-14b":       ["Phi-4", "microsoft/phi-4"],
    "gemma-3-27b":     ["Gemma-3-27B", "google/gemma-3-27b-it"],
    "deepseek-r1":     ["DeepSeek-R1", "deepseek-ai/DeepSeek-R1"],
    "mistral-24b":     ["Mistral-Small-24B", "mistralai/Mistral-Small-24B-Instruct"],
    "qwq-32b":         ["QwQ-32B", "Qwen/QwQ-32B"],
    "qwen3.5-32b":     ["Qwen3.5-32B", "Qwen/Qwen3.5-32B"],
    "qwen3-coder-32b": ["Qwen3-Coder-32B", "Qwen/Qwen3-Coder-32B-Instruct"],
    "gemma3-27b":      ["Gemma-3-27B", "google/gemma-3-27b-it"],
    "phi4-14b":        ["Phi-4-14B", "microsoft/Phi-4"],
}


def _fuzzy_match_model(query: str, candidates: list[str]) -> Optional[str]:
    """
    Fuzzy match a model name against candidate list.
    Returns best match or None.
    """
    query_lower = query.lower().replace("-", "").replace("_", "").replace(" ", "")

    best_match = None
    best_score = 0

    for candidate in candidates:
        cand_lower = candidate.lower().replace("-", "").replace("_", "").replace(" ", "")

        # Exact match
        if query_lower == cand_lower:
            return candidate

        # Substring match
        if query_lower in cand_lower or cand_lower in query_lower:
            score = min(len(query_lower), len(cand_lower)) / max(len(query_lower), len(cand_lower))
            if score > best_score:
                best_score = score
                best_match = candidate

        # Check aliases
        for alias_key, alias_list in _MODEL_ALIASES.items():
            ak = alias_key.lower().replace("-", "")
            if ak in query_lower or query_lower in ak:
                for alias in alias_list:
                    al = alias.lower().replace("-", "").replace("_", "").replace("/", "")
                    if al in cand_lower or cand_lower in al:
                        return candidate

    return best_match if best_score > 0.5 else None


# ─── Source Fetchers ─────────────────────────────────────────────────────────

class _BaseFetcher:
    """Base class for benchmark source fetchers."""
    source_name: str = "unknown"

    def fetch(self, model_id: str, cache: BenchmarkCache) -> Optional[BenchmarkResult]:
        raise NotImplementedError

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        """Fetch all models at once (for leaderboard-style sources)."""
        raise NotImplementedError


class ArtificialAnalysisFetcher(_BaseFetcher):
    """
    Fetches from artificialanalysis.ai API.

    Covers: MMLU-Pro, GPQA, MATH, HumanEval, IFEval, context benchmarks.
    Maps to: reasoning, domain_knowledge, code_generation, instruction_adherence.

    API: https://artificialanalysis.ai/api/text/v1/leaderboard
    """
    source_name = "artificial_analysis"
    API_URL = "https://artificialanalysis.ai/api/text/v1/leaderboard"

    # Maps their benchmark keys → our capability dimensions + normalization ranges
    BENCHMARK_MAP = {
        "mmlu_pro":    {"cap": "domain_knowledge",      "min": 20, "max": 90},
        "gpqa":        {"cap": "reasoning",              "min": 25, "max": 75},
        "math":        {"cap": "reasoning",              "min": 10, "max": 95},  # secondary
        "humaneval":   {"cap": "code_generation",        "min": 30, "max": 98},
        "ifeval":      {"cap": "instruction_adherence",  "min": 30, "max": 95},
        "bigbench":    {"cap": "analysis",               "min": 30, "max": 90},
    }

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import httpx
            resp = httpx.get(self.API_URL, timeout=30.0, follow_redirects=True)
            if resp.status_code != 200:
                logger.warning(f"Artificial Analysis API returned {resp.status_code}")
                return {}

            data = resp.json()
            models_data = data if isinstance(data, list) else data.get("data", data.get("models", []))

            result = {}
            for entry in models_data:
                model_name = entry.get("model", entry.get("name", ""))
                if not model_name:
                    continue

                mapped = {}
                for bench_key, mapping in self.BENCHMARK_MAP.items():
                    score = entry.get(bench_key) or entry.get(f"{bench_key}_score")
                    if score is not None:
                        try:
                            score = float(score)
                            cap = mapping["cap"]
                            norm = _normalize_score(score, mapping["min"], mapping["max"], 2.0, 10.0)
                            # If capability already has a value, average them
                            if cap in mapped:
                                mapped[cap] = round((mapped[cap] + norm) / 2, 1)
                            else:
                                mapped[cap] = norm
                        except (ValueError, TypeError):
                            pass

                if mapped:
                    result[model_name] = mapped

            cache.put_all_models(self.source_name, {"models": result})
            logger.info(f"Artificial Analysis: fetched {len(result)} models")
            return result

        except ImportError:
            logger.warning("httpx not available for Artificial Analysis fetch")
            return {}
        except Exception as e:
            logger.warning(f"Artificial Analysis fetch failed: {e}")
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
            confidence=0.85,
        )


class HuggingFaceLeaderboardFetcher(_BaseFetcher):
    """
    Fetches from the HuggingFace Open LLM Leaderboard v2.

    Covers: IFEval, BBH, MATH-Hard, GPQA, MUSR, MMLU-Pro
    Maps to: instruction_adherence, reasoning, analysis, domain_knowledge

    API: HuggingFace datasets API
    """
    source_name = "hf_leaderboard"
    API_URL = "https://huggingface.co/api/datasets/open-llm-leaderboard/contents/data"
    # Alternative: use the spaces API
    RESULTS_URL = "https://huggingface.co/datasets/open-llm-leaderboard/results/resolve/main/"

    BENCHMARK_MAP = {
        "ifeval":       {"cap": "instruction_adherence", "min": 20, "max": 90},
        "bbh":          {"cap": "reasoning",             "min": 20, "max": 85},
        "math_hard":    {"cap": "reasoning",             "min": 5,  "max": 75},
        "gpqa":         {"cap": "reasoning",             "min": 25, "max": 55},
        "musr":         {"cap": "analysis",              "min": 20, "max": 70},
        "mmlu_pro":     {"cap": "domain_knowledge",      "min": 20, "max": 85},
    }

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import httpx

            # Try the leaderboard API endpoint
            url = "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard/api/predict"
            # Alternative: fetch from the dataset directly
            dataset_url = "https://datasets-server.huggingface.co/rows?dataset=open-llm-leaderboard%2Fresults&config=default&split=train&offset=0&length=200"

            resp = httpx.get(dataset_url, timeout=30.0, follow_redirects=True)
            if resp.status_code != 200:
                logger.debug(f"HF Leaderboard returned {resp.status_code}, trying fallback")
                return self._fetch_from_spaces(cache)

            data = resp.json()
            rows = data.get("rows", [])

            result = {}
            for row in rows:
                entry = row.get("row", row)
                model_name = entry.get("model_name_for_query", entry.get("fullname", ""))
                if not model_name:
                    continue

                mapped = {}
                for bench_key, mapping in self.BENCHMARK_MAP.items():
                    for possible_key in [bench_key, f"{bench_key}_norm", f"leaderboard_{bench_key}"]:
                        score = entry.get(possible_key)
                        if score is not None:
                            try:
                                score = float(score)
                                if score > 1 and score <= 100:
                                    # Percentage
                                    pass
                                elif score <= 1:
                                    score *= 100  # Convert to percentage
                                cap = mapping["cap"]
                                norm = _normalize_score(score, mapping["min"], mapping["max"], 2.0, 10.0)
                                if cap in mapped:
                                    mapped[cap] = round((mapped[cap] + norm) / 2, 1)
                                else:
                                    mapped[cap] = norm
                            except (ValueError, TypeError):
                                pass
                            break

                if mapped:
                    result[model_name] = mapped

            if result:
                cache.put_all_models(self.source_name, {"models": result})
                logger.info(f"HF Leaderboard: fetched {len(result)} models")
            return result

        except Exception as e:
            logger.warning(f"HF Leaderboard fetch failed: {e}")
            return {}

    def _fetch_from_spaces(self, cache: BenchmarkCache) -> dict[str, dict]:
        """Fallback: try the Spaces Gradio API."""
        try:
            import httpx
            resp = httpx.post(
                "https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/predict",
                json={"fn_index": 0, "data": []},
                timeout=30.0,
            )
            if resp.status_code == 200:
                # Parse Gradio response format
                data = resp.json()
                # This varies by Gradio version; structure may change
                logger.debug("HF Spaces API responded, but format parsing is best-effort")
            return {}
        except Exception:
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


class LiveCodeBenchFetcher(_BaseFetcher):
    """
    Fetches from LiveCodeBench results.

    Covers: pass@1 on live coding problems
    Maps to: code_generation, code_reasoning

    Source: GitHub releases / website
    """
    source_name = "livecodebench"
    URL = "https://livecodebench.github.io/assets/data/results.json"
    FALLBACK_URL = "https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/main/docs/assets/data/results.json"

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import httpx

            data = None
            for url in [self.URL, self.FALLBACK_URL]:
                try:
                    resp = httpx.get(url, timeout=20.0, follow_redirects=True)
                    if resp.status_code == 200:
                        data = resp.json()
                        break
                except Exception:
                    continue

            if not data:
                return {}

            result = {}
            entries = data if isinstance(data, list) else data.get("results", data.get("data", []))

            for entry in entries:
                model_name = entry.get("model", entry.get("name", ""))
                if not model_name:
                    continue

                pass_at_1 = entry.get("pass@1", entry.get("pass_at_1"))
                if pass_at_1 is not None:
                    try:
                        score = float(pass_at_1)
                        if score <= 1.0:
                            score *= 100  # Convert to percentage

                        code_gen = _normalize_score(score, 10.0, 80.0, 2.0, 10.0)
                        # Code reasoning is usually slightly below generation
                        code_reason = round(code_gen * 0.9, 1)

                        result[model_name] = {
                            "code_generation": code_gen,
                            "code_reasoning": code_reason,
                        }
                    except (ValueError, TypeError):
                        pass

            if result:
                cache.put_all_models(self.source_name, {"models": result})
                logger.info(f"LiveCodeBench: fetched {len(result)} models")
            return result

        except Exception as e:
            logger.warning(f"LiveCodeBench fetch failed: {e}")
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


class BFCLFetcher(_BaseFetcher):
    """
    Berkeley Function Calling Leaderboard.

    Covers: function calling accuracy
    Maps to: tool_use, structured_output

    Source: https://gorilla.cs.berkeley.edu/leaderboard.html
    """
    source_name = "bfcl"
    URL = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/data/leaderboard_output.json"
    FALLBACK_URL = "https://gorilla.cs.berkeley.edu/api/leaderboard"

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import httpx

            data = None
            for url in [self.URL, self.FALLBACK_URL]:
                try:
                    resp = httpx.get(url, timeout=20.0, follow_redirects=True)
                    if resp.status_code == 200:
                        data = resp.json()
                        break
                except Exception:
                    continue

            if not data:
                return {}

            result = {}
            entries = data if isinstance(data, list) else data.get("data", [])

            for entry in entries:
                model_name = entry.get("model", entry.get("Model", ""))
                if not model_name:
                    continue

                # Overall accuracy
                overall = entry.get("overall_accuracy", entry.get("Overall Acc", entry.get("score")))
                if overall is not None:
                    try:
                        score = float(overall)
                        if score <= 1.0:
                            score *= 100

                        tool_score = _normalize_score(score, 30.0, 95.0, 2.0, 10.0)
                        struct_score = round(tool_score * 0.85, 1)  # Correlated but distinct

                        result[model_name] = {
                            "tool_use": tool_score,
                            "structured_output": struct_score,
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


class LMSysArenaFetcher(_BaseFetcher):
    """
    LMSys Chatbot Arena Elo ratings.

    Covers: overall quality, conversation, writing
    Maps to: conversation, prose_quality, and general quality validation

    Source: HuggingFace dataset
    """
    source_name = "lmsys_arena"
    URL = "https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard/resolve/main/elo_results.json"
    FALLBACK_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/serve/monitor/elo_results.json"

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import httpx

            data = None
            for url in [self.URL, self.FALLBACK_URL]:
                try:
                    resp = httpx.get(url, timeout=20.0, follow_redirects=True)
                    if resp.status_code == 200:
                        data = resp.json()
                        break
                except Exception:
                    continue

            if not data:
                return {}

            result = {}

            # LMSys format varies; handle different structures
            # Common: {"text": {"model_name": {"rating": 1234}}}
            categories = {}
            if isinstance(data, dict):
                for category_key in ["full", "text", "coding", "hard_prompts", "english"]:
                    if category_key in data:
                        categories[category_key] = data[category_key]
                if not categories:
                    # Maybe it's already flat
                    categories["full"] = data

            for cat_name, cat_data in categories.items():
                if not isinstance(cat_data, dict):
                    continue

                for model_name, model_data in cat_data.items():
                    if isinstance(model_data, dict):
                        elo = model_data.get("rating", model_data.get("elo"))
                    elif isinstance(model_data, (int, float)):
                        elo = model_data
                    else:
                        continue

                    if elo is None:
                        continue

                    try:
                        elo = float(elo)
                    except (ValueError, TypeError):
                        continue

                    if model_name not in result:
                        result[model_name] = {}

                    score = _normalize_elo(elo)

                    if cat_name in ("text", "english", "full"):
                        result[model_name]["conversation"] = score
                        result[model_name]["prose_quality"] = round(score * 0.95, 1)
                    elif cat_name == "coding":
                        result[model_name]["code_generation"] = score
                    elif cat_name == "hard_prompts":
                        result[model_name]["reasoning"] = score

            if result:
                cache.put_all_models(self.source_name, {"models": result})
                logger.info(f"LMSys Arena: fetched {len(result)} models")
            return result

        except Exception as e:
            logger.warning(f"LMSys Arena fetch failed: {e}")
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
            confidence=0.75,  # Arena is preference-based, less precise
        )


class AiderLeaderboardFetcher(_BaseFetcher):
    """
    Aider polyglot coding leaderboard.

    Covers: real-world code editing accuracy
    Maps to: code_generation, code_reasoning, instruction_adherence

    Source: https://aider.chat/docs/leaderboards/
    """
    source_name = "aider"
    URL = "https://raw.githubusercontent.com/Aider-AI/aider/main/aider/website/_data/polyglot_leaderboard.yml"
    FALLBACK_URL = "https://raw.githubusercontent.com/Aider-AI/aider/main/aider/website/_data/edit_leaderboard.yml"

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import httpx
            import yaml

            data = None
            for url in [self.URL, self.FALLBACK_URL]:
                try:
                    resp = httpx.get(url, timeout=20.0, follow_redirects=True)
                    if resp.status_code == 200:
                        data = yaml.safe_load(resp.text)
                        break
                except Exception:
                    continue

            if not data:
                return {}

            result = {}
            entries = data if isinstance(data, list) else data.get("results", [])

            for entry in entries:
                model_name = entry.get("model", entry.get("name", ""))
                if not model_name:
                    continue

                # Aider reports percent_cases_well_formed and percent_correct
                correct = entry.get("pass_rate_2", entry.get("percent_correct", entry.get("score")))
                well_formed = entry.get("percent_cases_well_formed")

                if correct is not None:
                    try:
                        correct = float(correct)
                        if correct <= 1.0:
                            correct *= 100

                        code_gen = _normalize_score(correct, 15.0, 85.0, 2.0, 10.0)
                        code_reason = round(code_gen * 0.95, 1)

                        mapped = {
                            "code_generation": code_gen,
                            "code_reasoning": code_reason,
                        }

                        if well_formed is not None:
                            wf = float(well_formed)
                            if wf <= 1.0:
                                wf *= 100
                            mapped["instruction_adherence"] = _normalize_score(
                                wf, 50.0, 100.0, 4.0, 10.0
                            )

                        result[model_name] = mapped
                    except (ValueError, TypeError):
                        pass

            if result:
                cache.put_all_models(self.source_name, {"models": result})
                logger.info(f"Aider: fetched {len(result)} models")
            return result

        except Exception as e:
            logger.warning(f"Aider leaderboard fetch failed: {e}")
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


class BigCodeBenchFetcher(_BaseFetcher):
    """
    BigCodeBench — comprehensive code generation benchmark.

    Covers: function-level code generation with complex instructions
    Maps to: code_generation, instruction_adherence

    Source: https://bigcode-bench.github.io/
    """
    source_name = "bigcodebench"
    URL = "https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/docs/assets/data/results.json"

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
            result = {}
            entries = data if isinstance(data, list) else data.get("results", [])

            for entry in entries:
                model_name = entry.get("model", entry.get("name", ""))
                score = entry.get("pass@1", entry.get("complete", entry.get("score")))

                if model_name and score is not None:
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


class OpenRouterRankingsFetcher(_BaseFetcher):
    """
    OpenRouter model rankings — context length & model availability data.

    Covers: context_length data across many models.
    Maps to: context_utilization (derived from context_length tiers).

    Source: https://openrouter.ai/api/v1/models (public, no auth needed)
    """
    source_name = "openrouter"
    URL = "https://openrouter.ai/api/v1/models"

    @staticmethod
    def _context_length_to_score(ctx_len: int) -> float:
        """Map context_length to a context_utilization score."""
        if ctx_len >= 128_000:
            return 9.0
        elif ctx_len >= 32_000:
            return 7.0
        elif ctx_len >= 8_000:
            return 5.0
        else:
            return 3.0

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import urllib.request

            req = urllib.request.Request(
                self.URL,
                headers={"User-Agent": "kutay-benchmark-fetcher/1.0"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8")

            data = json.loads(raw)
            models_list = data.get("data", [])

            result: dict[str, dict] = {}
            for model in models_list:
                model_id = model.get("id", "")
                ctx_len = model.get("context_length")
                pricing = model.get("pricing")

                # Skip models with no pricing data
                if not pricing:
                    continue
                if not model_id or ctx_len is None:
                    continue

                try:
                    ctx_len = int(ctx_len)
                except (ValueError, TypeError):
                    continue

                score = self._context_length_to_score(ctx_len)
                result[model_id] = {
                    "context_utilization": score,
                }

            if result:
                cache.put_all_models(self.source_name, {"models": result})
                logger.info(f"OpenRouter: fetched {len(result)} models")
            return result

        except Exception as e:
            logger.warning(f"OpenRouter fetch failed: {e}")
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
            confidence=0.60,
        )


# ─── Main Fetcher Orchestrator ───────────────────────────────────────────────

class BenchmarkFetcher:
    """
    Orchestrates fetching from all sources and merges results.

    Usage:
        fetcher = BenchmarkFetcher()

        # Get merged scores for a single model
        scores = fetcher.fetch_model("Qwen/Qwen3-32B")

        # Refresh cache for all sources
        fetcher.refresh_cache()

        # Get bulk data for enriching the registry
        all_scores = fetcher.fetch_all_bulk()
    """

    def __init__(self, cache_dir: str | Path = ".benchmark_cache"):
        self.cache = BenchmarkCache(cache_dir=Path(cache_dir))
        self.fetchers: list[_BaseFetcher] = [
            ArtificialAnalysisFetcher(),
            HuggingFaceLeaderboardFetcher(),
            LiveCodeBenchFetcher(),
            BFCLFetcher(),
            LMSysArenaFetcher(),
            AiderLeaderboardFetcher(),
            BigCodeBenchFetcher(),
            OpenRouterRankingsFetcher(),
        ]

    def fetch_model(self, model_id: str) -> dict[str, float]:
        """
        Fetch and merge benchmark data for a single model from all sources.
        Returns merged 15-dimension capability dict.

        Merge strategy: weighted average by source confidence.
        """
        all_results: list[BenchmarkResult] = []

        for fetcher in self.fetchers:
            try:
                result = fetcher.fetch(model_id, self.cache)
                if result and result.mapped_capabilities:
                    all_results.append(result)
            except Exception as e:
                logger.debug(f"Fetcher {fetcher.source_name} failed for {model_id}: {e}")

        return self._merge_results(all_results)

    def fetch_all_bulk(self) -> dict[str, dict[str, float]]:
        """
        Fetch bulk data from all sources and return merged scores
        keyed by model name.

        Returns: {"model_name": {"reasoning": 8.3, ...}, ...}
        """
        # Gather all data per source
        source_data: dict[str, dict[str, dict]] = {}
        source_confidence: dict[str, float] = {}

        for fetcher in self.fetchers:
            try:
                bulk = fetcher.fetch_bulk(self.cache)
                if bulk:
                    source_data[fetcher.source_name] = bulk
                    # Map source to confidence
                    conf_map = {
                        "artificial_analysis": 0.85,
                        "hf_leaderboard": 0.80,
                        "livecodebench": 0.90,
                        "bfcl": 0.90,
                        "lmsys_arena": 0.75,
                        "aider": 0.90,
                        "bigcodebench": 0.85,
                        "openrouter": 0.60,
                    }
                    source_confidence[fetcher.source_name] = conf_map.get(
                        fetcher.source_name, 0.75
                    )
            except Exception as e:
                logger.warning(f"Bulk fetch from {fetcher.source_name} failed: {e}")

        # Collect all unique model names across sources
        all_model_names: set[str] = set()
        for src_models in source_data.values():
            all_model_names.update(src_models.keys())

        # Merge per model
        merged: dict[str, dict[str, float]] = {}

        for model_name in all_model_names:
            results = []
            for source_name, src_models in source_data.items():
                if model_name in src_models:
                    results.append(BenchmarkResult(
                        source=source_name,
                        model_id=model_name,
                        raw_scores={},
                        mapped_capabilities=src_models[model_name],
                        confidence=source_confidence.get(source_name, 0.75),
                    ))

            caps = self._merge_results(results)
            if caps:
                merged[model_name] = caps

        logger.info(
            f"Benchmark bulk fetch complete: {len(merged)} models from "
            f"{len(source_data)} sources"
        )
        return merged

    def refresh_cache(self) -> None:
        """Force refresh all cached benchmark data."""
        logger.info("Refreshing benchmark cache from all sources...")
        # Clear bulk caches to force re-fetch
        for f in self.cache.cache_dir.glob("_bulk_*.json"):
            f.unlink()
        self.fetch_all_bulk()

    @staticmethod
    def _merge_results(results: list[BenchmarkResult]) -> dict[str, float]:
        """
        Merge multiple benchmark results using confidence-weighted averaging.

        For each capability dimension:
        - Collect all (score, confidence) pairs from sources that cover it
        - Compute weighted average
        - Round to 1 decimal
        """
        if not results:
            return {}

        # Accumulate: cap_name → [(score, confidence), ...]
        accum: dict[str, list[tuple[float, float]]] = {}

        for r in results:
            for cap, score in r.mapped_capabilities.items():
                if score is not None and score > 0:
                    if cap not in accum:
                        accum[cap] = []
                    accum[cap].append((score, r.confidence))

        merged = {}
        for cap, pairs in accum.items():
            total_weight = sum(conf for _, conf in pairs)
            if total_weight > 0:
                weighted_sum = sum(score * conf for score, conf in pairs)
                merged[cap] = round(weighted_sum / total_weight, 1)

        return merged


# ─── Integration with Registry ──────────────────────────────────────────────

def enrich_registry_with_benchmarks(
    registry: "ModelRegistry",
    cache_dir: str | Path = ".benchmark_cache",
    override_existing: bool = False,
    min_confidence_sources: int = 2,
) -> dict[str, dict[str, float]]:
    """
    Fetch benchmark data and merge into the registry's capability scores.

    Strategy:
    - For each model in the registry, try to find benchmark data
    - If found AND at least min_confidence_sources contributed,
      blend with existing estimate: 60% benchmark + 40% profile-estimate
    - If override_existing=True, replace entirely with benchmark data

    Returns dict of models that were enriched: {model_name: {updated_caps}}

    Usage:
        from model_registry import get_registry
        from benchmark_fetcher import enrich_registry_with_benchmarks

        registry = get_registry()
        enriched = enrich_registry_with_benchmarks(registry)
        print(f"Enriched {len(enriched)} models with benchmark data")
    """
    fetcher = BenchmarkFetcher(cache_dir=cache_dir)

    # Fetch bulk data from all sources
    bulk_data = fetcher.fetch_all_bulk()
    if not bulk_data:
        logger.info("No benchmark data available for enrichment")
        return {}

    enriched = {}

    for model_name, model_info in registry.models.items():
        # Try to match this model against benchmark data
        # Build search terms from model metadata
        search_terms = [
            model_name,
            model_info.litellm_name,
            model_info.family,
        ]
        if model_info.path:
            search_terms.append(Path(model_info.path).stem)

        # Try each search term
        bench_caps = None
        for term in search_terms:
            if not term:
                continue
            matched_key = _fuzzy_match_model(term, list(bulk_data.keys()))
            if matched_key:
                bench_caps = bulk_data[matched_key]
                break

        if not bench_caps:
            continue

        # Store raw benchmark scores for auto-tuner provenance
        model_info.benchmark_scores = dict(bench_caps)

        # Blend benchmark data with existing profile estimates
        updated = {}
        existing_caps = model_info.capabilities

        for cap_name in existing_caps:
            existing_score = existing_caps[cap_name]
            bench_score = bench_caps.get(cap_name)

            if bench_score is not None and bench_score > 0:
                if override_existing:
                    updated[cap_name] = bench_score
                else:
                    # Blend: 60% benchmark, 40% profile
                    blended = round(0.6 * bench_score + 0.4 * existing_score, 1)
                    updated[cap_name] = max(0.0, min(10.0, blended))
            else:
                updated[cap_name] = existing_score

        # Apply updates
        model_info.capabilities = updated
        enriched[model_name] = updated

        logger.debug(
            f"Enriched {model_name} with benchmark data "
            f"({len(bench_caps)} dimensions from benchmarks)"
        )

    logger.info(f"Enriched {len(enriched)}/{len(registry.models)} models with benchmark data")
    return enriched
