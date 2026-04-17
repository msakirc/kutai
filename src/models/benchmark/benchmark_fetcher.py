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
    # ── Local models on disk (verified against live AA cache 2026-04-17) ──
    # Qwen3 local trio — three distinct models, three distinct AA keys
    "qwen3-30b-a3b":       ["qwen3-30b-a3b-instruct::thinking", "qwen3-30b-a3b-instruct", "qwen3-30b-a3b-2507"],
    "qwen3-32b":           ["qwen3-32b-instruct", "qwen3-32b-instruct::thinking"],
    "qwen3-coder-30b":     ["qwen3-coder-30b-a3b-instruct"],  # 30B Coder DOES exist in AA
    "qwen3-8b":            ["qwen3-8b-instruct", "qwen3-8b-instruct::thinking"],
    "qwen3-14b":           ["qwen3-14b-instruct", "qwen3-14b-instruct::thinking"],
    "qwen3-235b":          ["qwen3-235b-a22b-instruct-2507", "qwen3-235b-a22b-instruct"],

    "gpt-oss-20b":         ["gpt-oss-20b"],
    "gpt-oss-120b":        ["gpt-oss-120b"],
    "apriel-15b":          ["apriel-v1-6-15b-thinker", "apriel-v1-5-15b-thinker"],
    "llama-3-3-70b":       ["llama-3-3-instruct-70b"],

    # ── Cloud models (live AA key order: creator-version-family, not family-version) ──
    "gpt-4o":              ["gpt-4o", "gpt-4o-2024-08-06"],
    "gpt-4o-mini":         ["gpt-4o-mini"],
    "claude-sonnet-4-5":   ["claude-4-5-sonnet", "claude-4-5-sonnet::thinking"],
    "claude-sonnet-4-6":   ["claude-sonnet-4-6", "claude-sonnet-4-6-adaptive"],
    "claude-haiku-4-5":    ["claude-4-5-haiku", "claude-4-5-haiku::thinking"],
    "gemini-flash":        ["gemini-3-flash", "gemini-3-flash::thinking"],
    "o4-mini":             ["o4-mini"],
    "deepseek-r1":         ["deepseek-r1"],
}


def _strip_gguf_suffixes(name: str) -> str:
    """Strip GGUF quantization and upscale suffixes from model names.

    Examples:
        Qwen3.5-27B.Q4_K_M          -> Qwen3.5-27B
        gemma-4-26B-A4B-it-UD-IQ4_NL -> gemma-4-26B-A4B-it
        GLM-4.7-Flash-UD-Q4_K_XL    -> GLM-4.7-Flash
        ServiceNow-AI_Apriel-1.6-15b-Thinker-Q4_K_L -> ServiceNow-AI_Apriel-1.6-15b-Thinker
    """
    import re as _re
    # Strip quantization suffix: .Q4_K_M, -Q4_K_XL, -IQ4_NL, _Q5_k_m, etc.
    name = _re.sub(r"[.\-_](?:I?Q\d+[_\-][A-Za-z0-9_]+)$", "", name)
    # Strip UD- prefix (ultra-dense) that comes before quant
    name = _re.sub(r"[\-_]UD$", "", name)
    # Strip .gguf extension
    name = _re.sub(r"\.gguf$", "", name, flags=_re.IGNORECASE)
    # Strip .i1- prefix (importance matrix indicator, e.g. .i1-Q4_K_M -> already stripped)
    name = _re.sub(r"\.i1$", "", name)
    return name


_MODE_SUFFIXES: tuple[str, ...] = ("::thinking", "::reasoning", "::nonthinking")
_FAMILY_SHIFTERS: frozenset[str] = frozenset({
    "coder", "vl", "omni", "next", "max",
    "nemotron", "thinker", "thinking", "reasoning", "vision",
})


def _extract_size_tokens(segments: list[str]) -> set[str]:
    """Extract size-tokens from a list of segments.

    A size token is a segment ending in 'b' whose body (after optional leading
    'a' for active-params marker) is purely numeric. Examples: '30b', '480b',
    '7b', 'a3b', 'a22b', 'a35b'.
    """
    out: set[str] = set()
    for s in segments:
        if s.endswith("b") and len(s) > 1:
            core = s[:-1]
            if core.startswith("a"):
                core = core[1:]
            if core.isdigit():
                out.add(s)
    return out


def _normalize_for_matching(name: str) -> str:
    """Normalize a model name for fuzzy matching.

    Preserves version dots as dashes to avoid false matches
    (e.g., 'qwen3.5' -> 'qwen3-5', not 'qwen35').
    Strips AA mode suffixes ('::thinking', '::reasoning', '::nonthinking') so
    a local GGUF can still match a thinking-flagged AA entry.
    """
    n = name.lower()
    # Strip AA mode suffix BEFORE other transforms — :: is preserved literally otherwise
    for mode_suffix in _MODE_SUFFIXES:
        if n.endswith(mode_suffix):
            n = n[: -len(mode_suffix)]
    # Remove org prefixes
    if "/" in n:
        n = n.rsplit("/", 1)[-1]
    # Remove common suffixes
    for suffix in [":latest", ":free", "-instruct", "-chat", "-it", "-hf"]:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    # Replace dots with dashes (preserve version boundaries: 3.5 -> 3-5)
    n = n.replace(".", "-")
    # Normalize separators (keep dashes as single separator)
    n = n.replace("_", "-").replace(" ", "-").replace(":", "-")
    # Collapse multiple dashes
    while "--" in n:
        n = n.replace("--", "-")
    return n.strip("-")


def _fuzzy_match_model(query: str, candidates: list[str]) -> Optional[str]:
    """
    Fuzzy match a model name against candidate list.
    Returns best match or None.

    Matching strategy:
    1. Exact match (after normalization)
    2. Alias-based match
    3. Stripped GGUF name match
    4. Substring match with length-ratio scoring
    """
    query_stripped = _strip_gguf_suffixes(query)
    query_norm = _normalize_for_matching(query_stripped)
    # Also decompose into segments for boundary-aware matching
    query_segs = query_norm.split("-")

    best_match = None
    best_score = 0.0

    for candidate in candidates:
        cand_stripped = _strip_gguf_suffixes(candidate)
        cand_norm = _normalize_for_matching(cand_stripped)

        # Exact match after normalization
        if query_norm == cand_norm:
            return candidate

        # Segment-prefix match: check if one is a prefix of the other
        # at segment boundaries (prevents qwen3 matching qwen3-5)
        cand_segs = cand_norm.split("-")
        min_len = min(len(query_segs), len(cand_segs))
        if min_len >= 2 and query_segs[:min_len] == cand_segs[:min_len]:
            score = min_len / max(len(query_segs), len(cand_segs))
            if score > best_score:
                best_score = score
                best_match = candidate

        # Full substring match (only if high overlap ratio AND size tokens agree)
        if query_norm in cand_norm or cand_norm in query_norm:
            score = min(len(query_norm), len(cand_norm)) / max(len(query_norm), len(cand_norm))
            # Reject if size-tokens differ (30b vs 480b, 8b vs 80b, etc.)
            q_sizes = _extract_size_tokens(query_segs)
            c_sizes = _extract_size_tokens(cand_norm.split("-"))
            if q_sizes and c_sizes and not (q_sizes & c_sizes):
                # Different sizes — skip substring match
                pass
            elif score > 0.7 and score > best_score:
                best_score = score
                best_match = candidate

    # Check aliases (separate pass — aliases are authoritative)
    # Family-shifter segments: tokens that change which model family this is.
    # If the query contains one of these and the alias key does NOT, the alias
    # must not match (e.g. 'qwen3-30b-a3b' alias must not match 'qwen3-coder-30b-a3b').
    for alias_key, alias_list in _MODEL_ALIASES.items():
        ak = _normalize_for_matching(alias_key)
        query_test = _normalize_for_matching(query_stripped)
        if ak == query_test or _segments_contain(query_test, ak):
            # Family-shifter guard: if query carries a shifter (e.g. 'coder', 'vl')
            # that is absent from BOTH the alias key and ALL alias targets, the
            # alias does not actually describe this family — skip it.
            ak_segs = set(ak.split("-"))
            q_segs = set(query_test.split("-"))
            target_segs: set[str] = set()
            for alias in alias_list:
                target_segs.update(_normalize_for_matching(alias).split("-"))
            extra_shifters = (q_segs - ak_segs - target_segs) & _FAMILY_SHIFTERS
            if extra_shifters:
                continue
            for alias in alias_list:
                an = _normalize_for_matching(alias)
                for candidate in candidates:
                    cn = _normalize_for_matching(_strip_gguf_suffixes(candidate))
                    if an == cn or _segments_contain(cn, an) or _segments_contain(an, cn):
                        return candidate

    return best_match if best_score > 0.5 else None


def _segments_contain(haystack: str, needle: str) -> bool:
    """Check if all segments of needle appear in haystack (order-preserving).

    Example: needle='apriel-15b' matches haystack='servicenow-ai-apriel-1-6-15b-thinker'
    because 'apriel' and '15b' appear in order.
    Requires ALL needle segments to match and needle has at least 2 segments.
    """
    h_segs = haystack.split("-")
    n_segs = needle.split("-")
    if len(n_segs) < 2:
        return False
    h_idx = 0
    matched = 0
    for n_seg in n_segs:
        while h_idx < len(h_segs):
            if h_segs[h_idx] == n_seg:
                matched += 1
                h_idx += 1
                break
            h_idx += 1
    return matched == len(n_segs)


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

            # Post-process: detect thinking/non-thinking pairs
            # AA uses several patterns:
            # 1. "model-name" (default=thinking) + "model-name-non-reasoning"
            # 2. "model-name-instruct" (default=non-thinking) + "model-name-instruct-reasoning"
            # 3. "model-name-thinking" (thinking variant, no explicit non-thinking)
            #
            # We normalize to: "clean-name" (non-thinking) and "clean-name::thinking"
            processed = {}

            for slug, caps in result.items():
                if slug.endswith("-non-reasoning"):
                    # This is explicitly non-thinking
                    clean = slug.replace("-non-reasoning", "")
                    processed[clean] = caps
                elif slug.endswith("-reasoning") and not slug.endswith("-non-reasoning"):
                    # This is explicitly thinking
                    clean = slug.replace("-reasoning", "")
                    processed[f"{clean}::thinking"] = caps
                elif slug.endswith("-thinking"):
                    # Cloud model thinking variant
                    clean = slug.replace("-thinking", "")
                    processed[f"{clean}::thinking"] = caps
                else:
                    # Check if there's a corresponding -non-reasoning entry
                    # If so, this slug IS the thinking version
                    non_reasoning_slug = f"{slug}-non-reasoning"
                    if non_reasoning_slug in result:
                        # This is the thinking version
                        processed[f"{slug}::thinking"] = caps
                    else:
                        # Regular model, no thinking pair detected
                        processed[slug] = caps

            result = processed

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


class LMArenaFetcher(_BaseFetcher):
    """
    LM Arena (lmarena-ai) official leaderboard — per-category ELO ratings.

    Covers: 27 categories including coding, math, creative_writing,
            instruction_following, multi_turn, hard_prompts, expert, etc.
    Maps to: code_generation, code_reasoning, reasoning, instruction_adherence,
             prose_quality, conversation, context_utilization, domain_knowledge,
             analysis

    Source: Official HuggingFace Parquet dataset (updated frequently)
    """
    source_name = "lm_arena"
    URL = "https://huggingface.co/api/datasets/lmarena-ai/leaderboard-dataset/parquet/text/latest/0.parquet"

    # Maps LM Arena category -> (our capability, ELO weight multiplier)
    CATEGORY_MAP = {
        "overall":                ("conversation", 1.0),
        "coding":                 ("code_generation", 1.0),
        "math":                   ("reasoning", 1.0),
        "creative_writing":       ("prose_quality", 1.0),
        "instruction_following":  ("instruction_adherence", 1.0),
        "hard_prompts":           ("reasoning", 0.8),
        "multi_turn":             ("conversation", 0.9),
        "expert":                 ("analysis", 1.0),
        "longer_query":           ("context_utilization", 1.0),
    }

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import pandas as pd

            df = pd.read_parquet(self.URL)

            # Group by model, collect per-category ELO
            model_cats: dict[str, dict[str, float]] = {}
            for _, row in df.iterrows():
                model_name = row.get("model_name", "")
                category = row.get("category", "")
                rating = row.get("rating")

                if not model_name or not category or rating is None:
                    continue
                try:
                    elo = float(rating)
                    if math.isnan(elo):
                        continue
                except (ValueError, TypeError):
                    continue

                model_cats.setdefault(model_name, {})[category] = elo

            # Map categories to capabilities
            result: dict[str, dict[str, float]] = {}
            for model_name, cats in model_cats.items():
                caps: dict[str, list[float]] = {}
                for cat, (cap, weight) in self.CATEGORY_MAP.items():
                    if cat in cats:
                        score = _normalize_elo(cats[cat]) * weight
                        caps.setdefault(cap, []).append(score)

                if caps:
                    mapped = {}
                    for cap, scores in caps.items():
                        mapped[cap] = round(sum(scores) / len(scores), 1)
                    # coding category also feeds code_reasoning
                    if "coding" in cats:
                        mapped["code_reasoning"] = round(
                            _normalize_elo(cats["coding"]) * 0.95, 1
                        )
                    result[model_name] = mapped

            if result:
                cache.put_all_models(self.source_name, {"models": result})
                logger.info(f"LM Arena: fetched {len(result)} models")
            return result

        except Exception as e:
            logger.warning(f"LM Arena fetch failed: {e}")
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


class UGILeaderboardFetcher(_BaseFetcher):
    """
    UGI (Uncensored General Intelligence) Leaderboard.

    Covers: UGI composite, Writing quality, Natural Intelligence, Textbook knowledge
    Maps to: prose_quality, domain_knowledge, analysis, conversation

    Source: HuggingFace Spaces CSV (1000+ models, updated frequently)
    Notable: Only source with Apriel-15B scores.
    """
    source_name = "ugi"
    URL = "https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard/resolve/main/ugi-leaderboard-data.csv"

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]

        try:
            import httpx

            resp = httpx.get(self.URL, timeout=30.0, follow_redirects=True)
            if resp.status_code != 200:
                return {}

            # Strip BOM if present
            text = resp.text.lstrip("\ufeff")
            reader = csv.DictReader(io.StringIO(text))
            result = {}

            for row in reader:
                # Column name may have BOM prefix or vary
                model_name = (
                    row.get("author/model_name")
                    or row.get("\ufeffauthor/model_name")
                    or row.get("Model", "")
                )
                if not model_name:
                    continue

                mapped = {}

                # Find columns by prefix (they have emoji suffixes)
                def _get_col(prefix: str) -> Optional[str]:
                    for k in row:
                        if k.startswith(prefix):
                            return row[k]
                    return None

                # UGI composite -> general quality indicator
                ugi = _get_col("UGI ")  # "UGI 🏆"
                if ugi:
                    try:
                        ugi_val = float(ugi)
                        mapped["analysis"] = _normalize_score(ugi_val, 30, 80, 2.0, 10.0)
                    except (ValueError, TypeError):
                        pass

                # Writing score -> prose_quality
                writing = _get_col("Writing")  # "Writing ✍️"
                if writing:
                    try:
                        w_val = float(writing)
                        mapped["prose_quality"] = _normalize_score(w_val, 30, 90, 2.0, 10.0)
                    except (ValueError, TypeError):
                        pass

                # NatInt (natural intelligence) -> domain_knowledge
                natint = _get_col("NatInt")  # "NatInt 💡"
                if natint:
                    try:
                        n_val = float(natint)
                        mapped["domain_knowledge"] = _normalize_score(n_val, 20, 80, 2.0, 10.0)
                    except (ValueError, TypeError):
                        pass

                if mapped:
                    result[model_name] = mapped

            if result:
                cache.put_all_models(self.source_name, {"models": result})
                logger.info(f"UGI: fetched {len(result)} models")
            return result

        except Exception as e:
            logger.warning(f"UGI leaderboard fetch failed: {e}")
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
            confidence=0.70,
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
            LMArenaFetcher(),
            BFCLFetcher(),
            OpenRouterRankingsFetcher(),
            SenecaTRBenchFetcher(),
            TurkishMMLUFetcher(),
            UGILeaderboardFetcher(),
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
                        "bfcl": 0.90,
                        "chatbot_arena": 0.75,
                        "aider": 0.90,
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

def _search_source_for_model(
    source_models: dict[str, dict],
    search_terms: list[str],
    is_thinking: bool,
) -> Optional[dict[str, float]]:
    """Search a single source's model dict for the best match.

    For thinking variants, tries ::thinking keys first.
    Returns matched capabilities dict or None.
    """
    all_keys = list(source_models.keys())

    if is_thinking:
        # Try thinking-specific keys first
        thinking_keys = [k for k in all_keys if k.endswith("::thinking")]
        if thinking_keys:
            for term in search_terms:
                if not term:
                    continue
                match = _fuzzy_match_model(term, thinking_keys)
                if match:
                    return source_models[match]

    # Search non-thinking keys (or all keys for non-thinking models)
    non_thinking_keys = [k for k in all_keys if not k.endswith("::thinking")]
    for term in search_terms:
        if not term:
            continue
        match = _fuzzy_match_model(term, non_thinking_keys)
        if match:
            return source_models[match]

    return None


def enrich_registry_with_benchmarks(
    registry: "ModelRegistry",
    cache_dir: str | Path = ".benchmark_cache",
    override_existing: bool = False,
    min_confidence_sources: int = 1,
) -> dict[str, dict[str, float]]:
    """
    Fetch benchmark data and merge into the registry's capability scores.

    Strategy:
    - For each model, search EACH source independently (different sources
      use different naming conventions, so per-source matching is essential)
    - Merge results across sources using confidence-weighted averaging
    - Blend with existing profile estimates: 60% benchmark + 40% profile

    Returns dict of models that were enriched: {model_name: {updated_caps}}
    """
    fetcher = BenchmarkFetcher(cache_dir=cache_dir)

    # Fetch per-source data (each source has its own naming convention)
    source_confidence = {
        "artificial_analysis": 0.90,
        "lm_arena": 0.85,
        "bfcl": 0.90,
        "openrouter": 0.60,
        "seneca_trbench": 0.85,
        "turkish_mmlu": 0.85,
        "ugi": 0.70,
    }

    per_source_data: dict[str, dict[str, dict]] = {}
    for f_obj in fetcher.fetchers:
        try:
            bulk = f_obj.fetch_bulk(fetcher.cache)
            if bulk:
                per_source_data[f_obj.source_name] = bulk
        except Exception as e:
            logger.warning(f"Bulk fetch from {f_obj.source_name} failed: {e}")

    if not per_source_data:
        logger.info("No benchmark data available for enrichment")
        return {}

    enriched = {}

    for model_name, model_info in registry.models.items():
        # Build search terms from model metadata
        search_terms = [model_name]
        base_name = getattr(model_info, 'base_model_name', '')
        if base_name and base_name != model_name:
            search_terms.append(base_name)
        if model_info.litellm_name:
            search_terms.append(model_info.litellm_name)
        if model_info.path:
            search_terms.append(Path(model_info.path).stem)
        # Also add the stripped GGUF name
        for term in list(search_terms):
            stripped = _strip_gguf_suffixes(term)
            if stripped != term and stripped not in search_terms:
                search_terms.append(stripped)

        is_thinking = getattr(model_info, 'is_variant', False) and \
                      'thinking' in getattr(model_info, 'variant_flags', set())

        # Search each source independently and collect results
        results: list[BenchmarkResult] = []
        for source_name, source_models in per_source_data.items():
            matched_caps = _search_source_for_model(
                source_models, search_terms, is_thinking
            )
            if matched_caps:
                results.append(BenchmarkResult(
                    source=source_name,
                    model_id=model_name,
                    raw_scores={},
                    mapped_capabilities=matched_caps,
                    confidence=source_confidence.get(source_name, 0.75),
                ))

        if not results:
            continue

        # Merge across sources using confidence-weighted averaging
        bench_caps = BenchmarkFetcher._merge_results(results)
        if not bench_caps:
            continue

        # Store raw benchmark scores for auto-tuner provenance
        model_info.benchmark_scores = dict(bench_caps)

        # Blend benchmark data with existing profile estimates
        updated = {}
        existing_caps = model_info.capabilities
        bench_dims = 0

        for cap_name in existing_caps:
            existing_score = existing_caps[cap_name]
            bench_score = bench_caps.get(cap_name)

            if bench_score is not None and bench_score > 0:
                bench_dims += 1
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

        sources_hit = [r.source for r in results]
        logger.debug(
            f"Enriched {model_name} with benchmark data "
            f"({bench_dims} dimensions from {len(results)} sources: "
            f"{', '.join(sources_hit)})"
        )

    logger.info(f"Enriched {len(enriched)}/{len(registry.models)} models with benchmark data")
    return enriched
