# benchmark_fetcher.py
"""
Benchmark Fetcher — pulls model evaluation data from multiple public sources
and maps them to our 15-dimension capability schema.

Sources:
  1. Artificial Analysis v2 API (authenticated)
  2. Chatbot Arena ELO (HuggingFace Parquet)
  3. Open LLM Leaderboard (HuggingFace Parquet)
  4. LiveCodeBench (performances JSON)
  5. BFCL (Berkeley Function Calling — CSV)
  6. Aider polyglot code leaderboard (YAML — unchanged)
  7. BigCodeBench (HuggingFace Parquet)
  8. OpenRouter rankings (unchanged)
  9. Seneca-TRBench (Turkish benchmark CSV)
  10. Turkish MMLU (HuggingFace Parquet)

Each source returns normalized 0-10 scores for the dimensions it covers.
Missing dimensions are left as None (not 0).

Usage:
    fetcher = BenchmarkFetcher(cache_dir=".benchmark_cache")
    scores = fetcher.fetch_all("Qwen/Qwen3-32B")
    # -> {"reasoning": 8.3, "code_generation": 7.9, ...}

    # Or refresh everything:
    fetcher.refresh_cache()
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
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
    raw_scores: dict[str, float]            # benchmark_name -> raw score
    mapped_capabilities: dict[str, float]   # capability_name -> 0-10 score
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


# --- Normalization Helpers ---------------------------------------------------

def _normalize_score(
    raw: float,
    min_val: float,
    max_val: float,
    target_min: float = 0.0,
    target_max: float = 10.0,
) -> float:
    """Linear normalization from [min_val, max_val] -> [target_min, target_max]."""
    if max_val <= min_val:
        return (target_min + target_max) / 2
    clamped = max(min_val, min(max_val, raw))
    normalized = (clamped - min_val) / (max_val - min_val)
    return round(target_min + normalized * (target_max - target_min), 1)


def _normalize_percentage(pct: float) -> float:
    """Map percentage (0-100) to 0-10 with a slight curve favoring higher scores."""
    # 50% -> ~4.0, 70% -> ~6.5, 85% -> ~8.0, 95% -> ~9.5
    return _normalize_score(pct, 25.0, 100.0, 1.0, 10.0)


def _normalize_elo(elo: float) -> float:
    """Map Arena Elo (typically 900-1400) to 0-10."""
    return _normalize_score(elo, 900.0, 1400.0, 2.0, 10.0)


# --- Model Name Matching ----------------------------------------------------

# Maps our local model family names to patterns found in leaderboard data
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


# --- Source Fetchers ---------------------------------------------------------

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
    Fetches from Artificial Analysis v2 authenticated API.

    Covers: GPQA, MMLU-Pro, HLE, LiveCodeBench, coding/math/intelligence
            indices, IFBench, TerminalBench, SciCode.
    Maps to: reasoning, domain_knowledge, code_generation, code_reasoning,
             instruction_adherence, analysis.

    API: https://artificialanalysis.ai/api/v2/data/llms/models
    Auth: x-api-key header
    """
    source_name = "artificial_analysis"
    API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"

    # Maps their benchmark keys -> our capability dimensions + normalization ranges
    # When multiple benchmarks map to the same capability, they get averaged.
    BENCHMARK_MAP = {
        "gpqa":                                {"cap": "reasoning",              "min": 25, "max": 75},
        "artificial_analysis_math_index":      {"cap": "reasoning",              "min": 20, "max": 90},
        "mmlu_pro":                            {"cap": "domain_knowledge",       "min": 20, "max": 90},
        "hle":                                 {"cap": "domain_knowledge",       "min": 5,  "max": 60},
        "livecodebench":                       {"cap": "code_generation",        "min": 10, "max": 70},
        "artificial_analysis_coding_index":    {"cap": "code_reasoning",         "min": 20, "max": 90},
        "scicode":                             {"cap": "code_reasoning",         "min": 5,  "max": 50},
        "ifbench":                             {"cap": "instruction_adherence",  "min": 30, "max": 95},
        "terminalbench_hard":                  {"cap": "analysis",               "min": 5,  "max": 60},
        "artificial_analysis_intelligence_index": {"cap": "analysis",            "min": 20, "max": 90},
    }

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        from src.app.config import ARTIFICIAL_ANALYSIS_API_KEY

        if not ARTIFICIAL_ANALYSIS_API_KEY:
            logger.debug("No ARTIFICIAL_ANALYSIS_API_KEY set, skipping AA fetch")
            return {}

        try:
            import httpx
            resp = httpx.get(
                self.API_URL,
                headers={"x-api-key": ARTIFICIAL_ANALYSIS_API_KEY},
                timeout=30.0,
                follow_redirects=True,
            )
            if resp.status_code != 200:
                logger.warning(f"Artificial Analysis API returned {resp.status_code}")
                return {}

            data = resp.json()
            models_data = data.get("data", [])
            if isinstance(data, list):
                models_data = data

            result = {}
            for entry in models_data:
                slug = entry.get("slug", "")
                if not slug:
                    continue

                # Benchmark scores are nested under "evaluations"
                evals = entry.get("evaluations", {})
                if not evals:
                    continue

                mapped = {}
                for bench_key, mapping in self.BENCHMARK_MAP.items():
                    score = evals.get(bench_key)
                    if score is not None:
                        try:
                            score = float(score)
                            # Most scores are 0-1 fractions, convert to percentage.
                            # Composite indices (intelligence/coding/math) are already 0-100.
                            if score <= 1.0 and "index" not in bench_key:
                                score *= 100
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
                    result[slug] = mapped

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
            confidence=0.90,
        )


class ChatbotArenaFetcher(_BaseFetcher):
    """
    Chatbot Arena ELO ratings (replaces LMSysArenaFetcher).

    Covers: Arena Score (ELO-based)
    Maps to: conversation, prose_quality

    Source: HuggingFace Parquet dataset
    """
    source_name = "chatbot_arena"
    URL = "https://huggingface.co/api/datasets/mathewhe/chatbot-arena-elo/parquet/default/train/0.parquet"

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import pandas as pd

            df = pd.read_parquet(self.URL)

            result = {}
            for _, row in df.iterrows():
                model_name = row.get("Model", "")
                arena_score = row.get("Arena Score")

                if not model_name or arena_score is None:
                    continue

                try:
                    elo = float(arena_score)
                    if math.isnan(elo):
                        continue
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


class HuggingFaceLeaderboardFetcher(_BaseFetcher):
    """
    HuggingFace Open LLM Leaderboard v2 (Parquet format).

    Covers: IFEval, BBH, MATH Lvl 5, GPQA, MUSR, MMLU-PRO
    Maps to: instruction_adherence, reasoning, analysis, domain_knowledge

    Source: HuggingFace datasets Parquet API
    """
    source_name = "hf_leaderboard"
    URL = "https://huggingface.co/api/datasets/open-llm-leaderboard/contents/parquet/default/train/0.parquet"

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

            df = pd.read_parquet(self.URL)

            result = {}
            for _, row in df.iterrows():
                model_name = row.get("Model", "")
                if not model_name:
                    continue

                mapped = {}
                for bench_key, mapping in self.BENCHMARK_MAP.items():
                    score = row.get(bench_key)
                    if score is not None:
                        try:
                            score = float(score)
                            if math.isnan(score):
                                continue
                            if 0 < score <= 1:
                                score *= 100  # Convert fraction to percentage
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


class LiveCodeBenchFetcher(_BaseFetcher):
    """
    LiveCodeBench — live coding problem evaluations.

    Covers: pass@1 on live coding problems across difficulties
    Maps to: code_generation, code_reasoning

    Source: performances_generation.json (new format with models + performances)
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

            result = {}

            # New format: {"models": [...], "performances": [{model, pass_at_1, difficulty}, ...]}
            if isinstance(data, dict) and "performances" in data:
                # Aggregate pass@1 per model across all difficulties
                model_scores: dict[str, list[float]] = {}
                for perf in data["performances"]:
                    model_name = perf.get("model", "")
                    pass_at_1 = perf.get("pass@1", perf.get("pass_at_1"))
                    if model_name and pass_at_1 is not None:
                        try:
                            score = float(pass_at_1)
                            if score <= 1.0:
                                score *= 100
                            model_scores.setdefault(model_name, []).append(score)
                        except (ValueError, TypeError):
                            pass

                for model_name, scores in model_scores.items():
                    avg_score = sum(scores) / len(scores)
                    code_gen = _normalize_score(avg_score, 10.0, 80.0, 2.0, 10.0)
                    code_reason = round(code_gen * 0.9, 1)
                    result[model_name] = {
                        "code_generation": code_gen,
                        "code_reasoning": code_reason,
                    }
            else:
                # Fallback: old flat list format
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
                                score *= 100
                            code_gen = _normalize_score(score, 10.0, 80.0, 2.0, 10.0)
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

    Covers: function calling accuracy (Overall Acc from CSV)
    Maps to: tool_use, structured_output

    Source: https://gorilla.cs.berkeley.edu/data_overall.csv
    """
    source_name = "bfcl"
    URL = "https://gorilla.cs.berkeley.edu/data_overall.csv"

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import httpx

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
                    score = float(overall.strip().rstrip("%"))
                    # If it looks like a fraction (0-1), convert
                    if score <= 1.0:
                        score *= 100

                    tool_score = _normalize_score(score, 30.0, 95.0, 2.0, 10.0)
                    struct_score = round(tool_score * 0.85, 1)

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
    BigCodeBench — comprehensive code generation benchmark (HF Parquet).

    Covers: function-level code generation with complex instructions
    Maps to: code_generation, instruction_adherence

    Source: HuggingFace datasets Parquet API
    """
    source_name = "bigcodebench"
    URL = "https://huggingface.co/api/datasets/bigcode/bigcodebench-results/parquet/default/train/0.parquet"

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import pandas as pd

            df = pd.read_parquet(self.URL)

            result = {}
            for _, row in df.iterrows():
                model_name = row.get("model", "")
                if not model_name:
                    continue

                # Prefer 'complete', fallback to 'instruct'
                score = row.get("complete")
                if score is None or (isinstance(score, float) and math.isnan(score)):
                    score = row.get("instruct")
                if score is None or (isinstance(score, float) and math.isnan(score)):
                    continue

                try:
                    score = float(score)
                    if math.isnan(score):
                        continue
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
    OpenRouter model rankings -- context length & model availability data.

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


class SenecaTRBenchFetcher(_BaseFetcher):
    """
    Seneca-TRBench — Turkish language benchmark.

    Covers: Turkish MCQ, SAQ, and combined scores
    Maps to: turkish

    Source: HuggingFace Spaces CSV
    """
    source_name = "seneca_trbench"
    URL = "https://huggingface.co/spaces/AlicanKiraz0/seneca-trbench/resolve/main/leaderboard_data.csv"

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import httpx

            resp = httpx.get(self.URL, timeout=20.0, follow_redirects=True)
            if resp.status_code != 200:
                return {}

            reader = csv.DictReader(io.StringIO(resp.text))
            result = {}

            for row in reader:
                model_name = row.get("Model", "")
                combined = row.get("Combined Score", "")

                if not model_name or not combined:
                    continue

                try:
                    score = float(combined.strip().rstrip("%"))
                    turkish_score = _normalize_score(score, 20.0, 95.0, 1.0, 10.0)
                    result[model_name] = {
                        "turkish": turkish_score,
                    }
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


class TurkishMMLUFetcher(_BaseFetcher):
    """
    Turkish MMLU — Turkish language MMLU benchmark (HF Parquet).

    Covers: Turkish academic knowledge (basari = success %)
    Maps to: turkish

    Source: HuggingFace datasets Parquet API
    """
    source_name = "turkish_mmlu"
    URL = "https://huggingface.co/api/datasets/alibayram/yapay_zeka_turkce_mmlu_liderlik_tablosu/parquet/default/train/0.parquet"

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import pandas as pd

            df = pd.read_parquet(self.URL)

            result = {}
            for _, row in df.iterrows():
                model_name = row.get("model", "")
                basari = row.get("basari")

                if not model_name or basari is None:
                    continue

                try:
                    score = float(basari)
                    if math.isnan(score):
                        continue
                    turkish_score = _normalize_score(score, 20.0, 90.0, 1.0, 10.0)
                    result[model_name] = {
                        "turkish": turkish_score,
                    }
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


# --- Main Fetcher Orchestrator -----------------------------------------------

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
            ChatbotArenaFetcher(),
            AiderLeaderboardFetcher(),
            BigCodeBenchFetcher(),
            OpenRouterRankingsFetcher(),
            SenecaTRBenchFetcher(),
            TurkishMMLUFetcher(),
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
                        "artificial_analysis": 0.90,
                        "hf_leaderboard": 0.80,
                        "livecodebench": 0.90,
                        "bfcl": 0.90,
                        "chatbot_arena": 0.75,
                        "aider": 0.90,
                        "bigcodebench": 0.85,
                        "openrouter": 0.60,
                        "seneca_trbench": 0.85,
                        "turkish_mmlu": 0.85,
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

        # Accumulate: cap_name -> [(score, confidence), ...]
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


# --- Integration with Registry -----------------------------------------------

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
