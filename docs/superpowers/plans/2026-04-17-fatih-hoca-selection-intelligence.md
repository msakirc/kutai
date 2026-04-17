# Fatih Hoca Selection Intelligence — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Fatih Hoca's model selection actually use the benchmark signal it already caches, fix three scoring-hygiene bugs, and add pick telemetry so future tuning is evidence-based instead of guessed.

**Architecture:** Benchmark data exists on disk (`.benchmark_cache/_bulk_*.json`) but never reaches ranking. Phase 1 wires it end-to-end: extend alias table with real AA keys for locals actually on disk, call `enrich_registry_with_benchmarks()` in `fatih_hoca.init()`, invoke `auto_tuner.blend_capability_scores()` so `ModelInfo.capabilities` is benchmark-informed by the time `rank_candidates()` reads it. Then remove the `×10`-then-clamp in capability scoring, replace the hardcoded `perf_score = 50` with measured tps from Nerd Herd, narrow the provider-wide failure penalty to model+error-class scope, and persist every pick's candidate breakdown to a new `model_pick_log` table for offline analysis. No ranking algorithm rewrite — only signal quality + plumbing.

**Tech Stack:** Python 3.10, aiosqlite, existing fatih_hoca/benchmark/auto_tuner/nerd_herd packages. No new dependencies.

**Related prior work:** `docs/superpowers/plans/2026-04-06-benchmark-driven-capabilities.md` built the fetch + blend infrastructure. This plan completes the integration that plan stopped short of.

**Out of scope for Phase 1:**
- Learned weights / weight auto-calibration (needs telemetry data first — collect for 2+ weeks before tuning)
- Background benchmark refresh scheduler (manual CLI still works; revisit when pick telemetry shows stale-data harm)
- Pareto / multi-objective optimization
- Task-profile expansion beyond the 16 existing entries

---

## File Structure

**New files:**
- `tests/fatih_hoca/test_alias_disambiguation.py` — triple-test for Qwen3 variants and other multi-variant families
- `tests/fatih_hoca/test_init_enrichment.py` — verifies `init()` wires benchmarks + auto-tuner
- `tests/fatih_hoca/test_scoring_hygiene.py` — ×10-clamp removal, perf_score from tps, narrowed failure penalty
- `tests/fatih_hoca/test_pick_telemetry.py` — candidate breakdown persisted to DB
- `tests/fatih_hoca/conftest.py` — shared fixtures (canned AA cache, fake Nerd Herd, in-memory ModelRegistry)
- `src/infra/db_migrations/006_model_pick_log.sql` — new table for pick telemetry

**Modified files:**
- `src/models/benchmark/benchmark_fetcher.py` — extend `_MODEL_ALIASES`, add stale-cache purge in `BenchmarkCache.load()`
- `packages/fatih_hoca/src/fatih_hoca/__init__.py:22–81` — `init()` calls enrich + blend, emits coverage log
- `packages/fatih_hoca/src/fatih_hoca/ranking.py:218` — remove `×10`-then-clamp
- `packages/fatih_hoca/src/fatih_hoca/ranking.py:293–295` — replace hardcoded `perf_score = 50` with measured-tps-derived score
- `packages/fatih_hoca/src/fatih_hoca/ranking.py:52–107` — narrow `_failure_penalty` scope for `rate_limit`
- `packages/fatih_hoca/src/fatih_hoca/selector.py:176–180` — emit full candidate breakdown to logger + persist top-5 to DB
- `src/infra/db.py` — run new migration

---

## Task 1: Shared test fixtures

**Files:**
- Create: `tests/fatih_hoca/__init__.py` (empty)
- Create: `tests/fatih_hoca/conftest.py`

- [ ] **Step 1: Create empty package marker**

```bash
touch tests/fatih_hoca/__init__.py
```

- [ ] **Step 2: Write conftest with three reusable fixtures**

```python
# tests/fatih_hoca/conftest.py
"""Shared fixtures for fatih_hoca tests: canned AA cache, fake Nerd Herd, registry."""
from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import pytest

from fatih_hoca.registry import ModelInfo, ModelRegistry
from nerd_herd.types import SystemSnapshot, LocalModelState


@pytest.fixture
def canned_aa_cache(tmp_path: Path) -> Path:
    """Write a realistic AA bulk cache with the Qwen3 trio + a coder fine-tune + a dense small model."""
    cache_dir = tmp_path / ".benchmark_cache"
    cache_dir.mkdir()
    payload = {
        "timestamp": time.time(),  # fresh
        "models": {
            # Base Qwen3 30B A3B MoE (thinking variant) — strong reasoning, weak coding
            "qwen3-30b-a3b-instruct::thinking": {
                "reasoning": 7.5, "analysis": 7.0, "domain_knowledge": 6.5,
                "code_generation": 5.0, "code_reasoning": 5.5,
                "instruction_adherence": 7.0,
            },
            # Dense 32B Qwen3 — balanced, mid coder
            "qwen3-32b-instruct": {
                "reasoning": 7.0, "analysis": 6.8, "domain_knowledge": 6.5,
                "code_generation": 6.5, "code_reasoning": 6.5,
                "instruction_adherence": 7.0,
            },
            # Qwen3 Coder 480B — strong coder (this is the ONLY AA coder variant; no 30B-coder exists)
            "qwen3-coder-480b-a35b-instruct": {
                "reasoning": 7.5, "analysis": 7.2, "domain_knowledge": 7.0,
                "code_generation": 9.0, "code_reasoning": 8.8,
                "instruction_adherence": 7.5,
            },
            # Llama 3.3 70B cloud
            "llama-3-3-70b-instruct": {
                "reasoning": 7.0, "code_generation": 6.0, "domain_knowledge": 7.5,
                "instruction_adherence": 7.2,
            },
        },
    }
    cache_path = cache_dir / "_bulk_artificial_analysis.json"
    cache_path.write_text(json.dumps(payload))
    return cache_dir


@dataclass
class FakeNerdHerd:
    """Minimal Nerd Herd stub that returns a controllable SystemSnapshot."""
    loaded_model: str | None = None
    measured_tps: float = 0.0
    vram_available_mb: int = 24000

    def snapshot(self) -> SystemSnapshot:
        snap = SystemSnapshot(vram_available_mb=self.vram_available_mb)
        snap.local = LocalModelState(
            model_name=self.loaded_model,
            measured_tps=self.measured_tps,
        )
        return snap


@pytest.fixture
def fake_nerd_herd() -> FakeNerdHerd:
    return FakeNerdHerd()


@pytest.fixture
def registry_with_qwen_trio() -> ModelRegistry:
    """Registry seeded with three Qwen3 local models covering the disambiguation trap."""
    reg = ModelRegistry()
    reg._models["qwen3-30b-a3b"] = ModelInfo(
        name="qwen3-30b-a3b",
        location="local",
        path="/fake/Qwen3-30B-A3B-Instruct-Q4_K_M.gguf",
        total_params_b=30.0,
        active_params_b=3.0,
        family="qwen3",
        capabilities={"reasoning": 6.0, "code_generation": 5.0},
        thinking_model=True,
    )
    reg._models["qwen3-coder-30b"] = ModelInfo(
        name="qwen3-coder-30b",
        location="local",
        path="/fake/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        total_params_b=30.0,
        active_params_b=3.0,
        family="qwen3-coder",
        capabilities={"reasoning": 6.0, "code_generation": 7.5},
        specialty="coding",
    )
    reg._models["qwen3-32b"] = ModelInfo(
        name="qwen3-32b",
        location="local",
        path="/fake/Qwen3-32B-Instruct-Q4_K_M.gguf",
        total_params_b=32.0,
        active_params_b=32.0,
        family="qwen3",
        capabilities={"reasoning": 6.5, "code_generation": 6.0},
    )
    return reg
```

- [ ] **Step 3: Run fixtures sanity check**

```bash
timeout 30 pytest tests/fatih_hoca/conftest.py --collect-only -q
```

Expected: collected 0 items (conftest has no tests, just fixtures — just want to confirm imports succeed).

- [ ] **Step 4: Commit**

```bash
git add tests/fatih_hoca/__init__.py tests/fatih_hoca/conftest.py
git commit -m "test(fatih_hoca): shared fixtures for Qwen trio + canned AA cache"
```

---

## Task 2: Alias table audit — replace fabricated entries with real AA keys

**Context:** Current `_MODEL_ALIASES` (benchmark_fetcher.py:149) contains entries like `"qwen3-coder-30b"` that point to names AA has no record of (AA's coder line is 480B only). We need (a) entries that match what's actually on disk and (b) alias targets that correspond to real AA cache keys.

Real AA Qwen keys confirmed in `.benchmark_cache/_bulk_artificial_analysis.json`:
- `qwen3-30b-a3b-instruct::thinking` — base 30B MoE, thinking mode
- `qwen3-32b-instruct` — dense 32B
- `qwen3-coder-480b-a35b-instruct` — coder is only 480B in AA
- `qwen3-8b-instruct`, `qwen3-14b-instruct`, `qwen3-235b-a22b-instruct-2507`, etc.

**Files:**
- Modify: `src/models/benchmark/benchmark_fetcher.py:149–169` (`_MODEL_ALIASES`)
- Modify: `src/models/benchmark/benchmark_fetcher.py:217–273` (`_fuzzy_match_model` — strip `::thinking` suffix when comparing)
- Test: `tests/fatih_hoca/test_alias_disambiguation.py` (new)

- [ ] **Step 1: Write the failing disambiguation test**

```python
# tests/fatih_hoca/test_alias_disambiguation.py
"""Qwen3 trio must never cross-match. Each local GGUF → its own AA key."""
from src.models.benchmark.benchmark_fetcher import _fuzzy_match_model


# Realistic AA key list (subset of _bulk_artificial_analysis.json as of 2026-04-13)
AA_KEYS = [
    "qwen3-30b-a3b-instruct::thinking",
    "qwen3-32b-instruct",
    "qwen3-coder-480b-a35b-instruct",
    "qwen3-8b-instruct",
    "qwen3-14b-instruct",
    "qwen3-235b-a22b-instruct-2507",
    "llama-3-3-70b-instruct",
    "apriel-1-6-15b-thinker",
]


class TestQwenTrioDisambiguation:
    def test_base_a3b_matches_a3b(self):
        """Qwen3-30B-A3B-Instruct GGUF → qwen3-30b-a3b-instruct::thinking, NOT qwen3-32b."""
        m = _fuzzy_match_model("Qwen3-30B-A3B-Instruct-Q4_K_M", AA_KEYS)
        assert m == "qwen3-30b-a3b-instruct::thinking"

    def test_base_a3b_does_not_match_coder(self):
        """Base A3B must NOT match the coder variant (wrong training)."""
        m = _fuzzy_match_model("Qwen3-30B-A3B-Instruct-Q4_K_M", AA_KEYS)
        assert m != "qwen3-coder-480b-a35b-instruct"

    def test_base_a3b_does_not_match_dense_32b(self):
        """Different architecture (MoE vs dense) — must not match."""
        m = _fuzzy_match_model("Qwen3-30B-A3B-Instruct-Q4_K_M", AA_KEYS)
        assert m != "qwen3-32b-instruct"

    def test_coder_30b_gguf_returns_none(self):
        """No AA entry exists for 30B Coder. Must return None, not misroute to 480B coder or 32B dense."""
        m = _fuzzy_match_model("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M", AA_KEYS)
        assert m is None, f"Expected None (no 30B coder in AA), got {m!r}"

    def test_dense_32b_matches_itself(self):
        m = _fuzzy_match_model("Qwen3-32B-Instruct-Q4_K_M", AA_KEYS)
        assert m == "qwen3-32b-instruct"

    def test_dense_32b_does_not_match_a3b(self):
        m = _fuzzy_match_model("Qwen3-32B-Instruct-Q4_K_M", AA_KEYS)
        assert m != "qwen3-30b-a3b-instruct::thinking"


class TestOtherLocals:
    def test_apriel_matches(self):
        m = _fuzzy_match_model("ServiceNow-AI_Apriel-1.6-15b-Thinker-Q4_K_L", AA_KEYS)
        assert m == "apriel-1-6-15b-thinker"

    def test_llama_3_3_matches(self):
        m = _fuzzy_match_model("Llama-3.3-70B-Instruct-Q4_K_M", AA_KEYS)
        assert m == "llama-3-3-70b-instruct"
```

- [ ] **Step 2: Run the test — expect failures**

```bash
timeout 30 pytest tests/fatih_hoca/test_alias_disambiguation.py -v
```

Expected failures: `test_coder_30b_gguf_returns_none` (current fuzzy substring match may route to coder-480b or 32b), possibly `test_base_a3b_matches_a3b` (the `::thinking` suffix breaks exact-normalize match).

- [ ] **Step 3: Fix `_fuzzy_match_model` to strip `::thinking` and `::reasoning` suffixes before comparison**

In `src/models/benchmark/benchmark_fetcher.py`, update `_normalize_for_matching` (line 193) to strip mode suffixes from candidates:

```python
def _normalize_for_matching(name: str) -> str:
    """Normalize a model name for fuzzy matching.

    Preserves version dots as dashes to avoid false matches
    (e.g., 'qwen3.5' -> 'qwen3-5', not 'qwen35').
    Strips AA mode suffixes ('::thinking', '::reasoning') so a local
    GGUF can still match a thinking-flagged AA entry.
    """
    n = name.lower()
    # Strip AA mode suffix BEFORE other transforms — :: is preserved literally otherwise
    for mode_suffix in ("::thinking", "::reasoning", "::nonthinking"):
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
```

- [ ] **Step 4: Tighten substring match to prevent 30B → 480B bleed**

At `benchmark_fetcher.py:254–259`, substring match accepts overlap ratio > 0.7. `qwen3-coder-30b-a3b` (13 chars non-sep) inside `qwen3-coder-480b-a35b` (15 chars) has ratio ~0.77 — too permissive. Require that size-token segments match exactly:

```python
        # Full substring match (only if high overlap ratio AND size tokens agree)
        if query_norm in cand_norm or cand_norm in query_norm:
            score = min(len(query_norm), len(cand_norm)) / max(len(query_norm), len(cand_norm))
            # Reject if size-tokens differ (30b vs 480b, 8b vs 80b, etc.)
            q_sizes = {s for s in query_segs if s.endswith("b") and s[:-1].replace("a","").isdigit()}
            c_sizes = {s for s in cand_norm.split("-") if s.endswith("b") and s[:-1].replace("a","").isdigit()}
            if q_sizes and c_sizes and not (q_sizes & c_sizes):
                # Different sizes — skip substring match
                pass
            elif score > 0.7 and score > best_score:
                best_score = score
                best_match = candidate
```

- [ ] **Step 5: Replace fabricated `_MODEL_ALIASES` with real AA keys**

At `benchmark_fetcher.py:149`, replace the dict (the `qwen3.5-*` entries are speculative; keep only aliases whose targets exist in AA):

```python
_MODEL_ALIASES: dict[str, list[str]] = {
    # ── Local models on disk (verified against AA bulk cache 2026-04-17) ──
    # Qwen3 trio — three distinct models, three distinct AA keys
    "qwen3-30b-a3b":   ["qwen3-30b-a3b-instruct::thinking", "qwen3-30b-a3b-instruct"],
    "qwen3-32b":       ["qwen3-32b-instruct", "qwen3-32b-instruct::thinking"],
    # Note: NO entry for qwen3-coder-30b — AA has no 30B coder.
    # The base A3B alias covers Thinker/Instruct variants.
    "qwen3-8b":        ["qwen3-8b-instruct"],
    "qwen3-14b":       ["qwen3-14b-instruct"],
    "qwen3-235b":      ["qwen3-235b-a22b-instruct-2507", "qwen3-235b-a22b-instruct"],

    "gpt-oss-20b":     ["gpt-oss-20b"],
    "gpt-oss-120b":    ["gpt-oss-120b"],
    "apriel-15b":      ["apriel-1-6-15b-thinker"],
    "llama-3-3-70b":   ["llama-3-3-70b-instruct"],

    # ── Cloud models ──
    "gpt-4o":          ["gpt-4o", "gpt-4o-2024-11-20"],
    "gpt-4o-mini":     ["gpt-4o-mini", "gpt-4o-mini-2024-07-18"],
    "claude-sonnet-4": ["claude-sonnet-4-5", "claude-sonnet-4-20250514"],
    "gemini-flash":    ["gemini-2-5-flash", "gemini-2-0-flash"],
    "o4-mini":         ["o4-mini", "o4-mini-2025-04-16"],
    "deepseek-r1":     ["deepseek-r1", "deepseek-r1-0528"],
}
```

**Note for executor:** re-verify each alias target against the current live `.benchmark_cache/_bulk_artificial_analysis.json` keyset before committing. If an entry doesn't exist there, drop it rather than guessing — silent non-matches are better than wrong-matches.

- [ ] **Step 6: Re-run the test — expect all pass**

```bash
timeout 30 pytest tests/fatih_hoca/test_alias_disambiguation.py -v
```

Expected: 8 passed.

- [ ] **Step 7: Run existing benchmark fetcher tests to catch regressions**

```bash
timeout 60 pytest tests/test_benchmark_fetcher.py -v
```

Expected: all existing tests still pass (the normalization change preserves old behavior for names without `::` suffix).

- [ ] **Step 8: Commit**

```bash
git add src/models/benchmark/benchmark_fetcher.py tests/fatih_hoca/test_alias_disambiguation.py
git commit -m "fix(benchmark): Qwen3 trio disambiguation, real AA alias keys, ::thinking suffix handling"
```

---

## Task 3: Stale-cache purge — enforce 48h TTL

**Context:** `BenchmarkCache` currently serves stale data (AA cache observed 4 days old, TTL is 48h). The TTL only gates "should I refetch," not "should I still use what's on disk." Fix: purge entries older than TTL at load time so stale data can't silently flow into scoring.

**Files:**
- Modify: `src/models/benchmark/benchmark_fetcher.py` — `BenchmarkCache.load()` (find via grep)
- Test: `tests/test_benchmark_fetcher.py` (add new test class)

- [ ] **Step 1: Locate `BenchmarkCache.load()`**

```bash
grep -n "def load" src/models/benchmark/benchmark_fetcher.py
```

Note the line number of `BenchmarkCache.load()` for the next step.

- [ ] **Step 2: Write failing test for TTL purge**

Append to `tests/test_benchmark_fetcher.py`:

```python
class TestBenchmarkCacheStaleness:
    """Stale cache entries (age > TTL) must be purged, not served."""

    def test_fresh_cache_loads_normally(self, tmp_path):
        from src.models.benchmark.benchmark_fetcher import BenchmarkCache
        import json, time

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
        import json, time, logging

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
```

- [ ] **Step 3: Run the failing test**

```bash
timeout 30 pytest tests/test_benchmark_fetcher.py::TestBenchmarkCacheStaleness -v
```

Expected: `test_stale_cache_returns_none_with_warning` fails (current code serves stale data).

- [ ] **Step 4: Add TTL enforcement to `BenchmarkCache.load()`**

In `src/models/benchmark/benchmark_fetcher.py`, modify `BenchmarkCache.load()` to check timestamp:

```python
    def load(self, source: str) -> dict | None:
        """Load cached data for a source. Returns None if missing or stale."""
        import time
        import logging
        logger = logging.getLogger(__name__)

        path = self.cache_dir / f"_bulk_{source}.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("benchmark cache unreadable: source=%s err=%s", source, e)
            return None

        ts = data.get("timestamp", 0)
        age_hours = (time.time() - ts) / 3600
        if age_hours > CACHE_TTL_HOURS:
            logger.warning(
                "benchmark cache stale: source=%s age=%.1fh ttl=%dh — returning None",
                source, age_hours, CACHE_TTL_HOURS,
            )
            return None
        return data
```

- [ ] **Step 5: Run all cache tests**

```bash
timeout 30 pytest tests/test_benchmark_fetcher.py -v -k "Cache or Staleness"
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/models/benchmark/benchmark_fetcher.py tests/test_benchmark_fetcher.py
git commit -m "fix(benchmark): purge stale cache entries (>TTL) instead of serving them silently"
```

---

## Task 4: Wire enrich + blend into `fatih_hoca.init()`

**Context:** `enrich_registry_with_benchmarks()` and `auto_tuner.blend_capability_scores()` both exist but nothing calls them at startup. The registry goes live with hand-authored profile capabilities only.

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/__init__.py:22–81` (`init()`)
- Test: `tests/fatih_hoca/test_init_enrichment.py` (new)

- [ ] **Step 1: Write failing test for enrichment-at-init**

```python
# tests/fatih_hoca/test_init_enrichment.py
"""init() must populate benchmark_scores on registered models from cached AA data."""
from __future__ import annotations

import logging
import pytest

import fatih_hoca
from fatih_hoca.registry import ModelInfo


def _seed_registry_with_aa_matching_models(monkeypatch, canned_aa_cache):
    """Monkeypatch ModelRegistry.load_gguf_dir to return three Qwen locals whose names match the canned AA cache."""
    from fatih_hoca import registry as reg_mod

    def fake_load_gguf_dir(self, models_dir):
        self._models["qwen3-30b-a3b"] = ModelInfo(
            name="qwen3-30b-a3b", location="local",
            path="/fake/Qwen3-30B-A3B-Instruct-Q4_K_M.gguf",
            total_params_b=30.0, active_params_b=3.0,
            family="qwen3",
            capabilities={"reasoning": 5.0, "code_generation": 5.0},
        )
        self._models["qwen3-32b"] = ModelInfo(
            name="qwen3-32b", location="local",
            path="/fake/Qwen3-32B-Instruct-Q4_K_M.gguf",
            total_params_b=32.0, active_params_b=32.0,
            family="qwen3",
            capabilities={"reasoning": 5.0, "code_generation": 5.0},
        )
        return list(self._models.values())

    monkeypatch.setattr(reg_mod.ModelRegistry, "load_gguf_dir", fake_load_gguf_dir)


def test_init_populates_benchmark_scores(monkeypatch, canned_aa_cache, tmp_path):
    _seed_registry_with_aa_matching_models(monkeypatch, canned_aa_cache)

    # Point enrichment at our canned cache dir
    monkeypatch.chdir(canned_aa_cache.parent)

    fatih_hoca.init(models_dir=str(tmp_path / "fake_gguf_dir"))

    models = {m.name: m for m in fatih_hoca.all_models()}
    assert "qwen3-30b-a3b" in models
    assert "qwen3-32b" in models

    # Benchmark scores should be populated from AA cache
    a3b = models["qwen3-30b-a3b"]
    d32 = models["qwen3-32b"]
    assert a3b.benchmark_scores, "qwen3-30b-a3b should have benchmark_scores from AA"
    assert d32.benchmark_scores, "qwen3-32b should have benchmark_scores from AA"

    # After blending, capabilities should reflect AA signal (AA has d32 code_gen=6.5 vs a3b=5.0)
    assert d32.capabilities["code_generation"] > a3b.capabilities["code_generation"], \
        "post-blend capabilities must reflect AA's stronger coder signal for dense 32B"


def test_init_logs_coverage(monkeypatch, canned_aa_cache, tmp_path, caplog):
    _seed_registry_with_aa_matching_models(monkeypatch, canned_aa_cache)
    monkeypatch.chdir(canned_aa_cache.parent)

    with caplog.at_level(logging.INFO, logger="fatih_hoca"):
        fatih_hoca.init(models_dir=str(tmp_path / "fake_gguf_dir"))

    coverage_logs = [r for r in caplog.records if "benchmark coverage" in r.message.lower()]
    assert coverage_logs, "init() must emit a 'benchmark coverage: N/M matched' log line"
    msg = coverage_logs[0].message
    assert "2/2" in msg or "matched=2" in msg, f"expected 2/2 coverage, got: {msg}"


def test_init_warns_on_unmatched_models(monkeypatch, canned_aa_cache, tmp_path, caplog):
    """If a model doesn't match any AA entry, init must log the unmatched name at WARNING."""
    from fatih_hoca import registry as reg_mod

    def fake_load_gguf_dir(self, models_dir):
        self._models["some-obscure-local"] = ModelInfo(
            name="some-obscure-local", location="local",
            path="/fake/some-obscure-local.gguf",
            total_params_b=7.0, family="unknown",
            capabilities={"reasoning": 5.0},
        )
        return list(self._models.values())

    monkeypatch.setattr(reg_mod.ModelRegistry, "load_gguf_dir", fake_load_gguf_dir)
    monkeypatch.chdir(canned_aa_cache.parent)

    with caplog.at_level(logging.WARNING, logger="fatih_hoca"):
        fatih_hoca.init(models_dir=str(tmp_path / "fake_gguf_dir"))

    unmatched_logs = [r for r in caplog.records if "unmatched" in r.message.lower()]
    assert unmatched_logs, "init() must warn on models without benchmark coverage"
    assert any("some-obscure-local" in r.message for r in unmatched_logs)
```

- [ ] **Step 2: Run the failing tests**

```bash
timeout 30 pytest tests/fatih_hoca/test_init_enrichment.py -v
```

Expected: all three fail (`init()` does not call enrich, does not log coverage, does not warn).

- [ ] **Step 3: Modify `init()` to call enrich + blend + log coverage**

Replace `packages/fatih_hoca/src/fatih_hoca/__init__.py` body of `init()` (lines 50–81):

```python
    global _selector, _registry

    if nerd_herd is None:
        from nerd_herd.types import SystemSnapshot

        class _NoopNerdHerd:
            def snapshot(self) -> SystemSnapshot:
                return SystemSnapshot()

        nerd_herd = _NoopNerdHerd()

    _registry = ModelRegistry()
    model_names: list[str] = []

    if catalog_path:
        models = _registry.load_yaml(catalog_path)
        model_names.extend(m.name for m in models)

    if models_dir:
        models = _registry.load_gguf_dir(models_dir)
        model_names.extend(m.name for m in models)

    _registry._load_speed_cache()

    # ── Benchmark enrichment: populate ModelInfo.benchmark_scores from cached AA data ──
    import logging
    logger = logging.getLogger(__name__)
    try:
        from src.models.benchmark.benchmark_fetcher import enrich_registry_with_benchmarks

        enriched = enrich_registry_with_benchmarks(_registry)
        all_models = _registry.all_models()
        matched = sum(1 for m in all_models if m.benchmark_scores)
        total = len(all_models)
        unmatched = [m.name for m in all_models if not m.benchmark_scores]

        logger.info(
            "benchmark coverage: %d/%d matched (unmatched=%d)",
            matched, total, len(unmatched),
        )
        if unmatched:
            logger.warning(
                "benchmark coverage: %d unmatched models — %s",
                len(unmatched), ", ".join(unmatched[:10]),
            )
    except Exception as e:
        logger.warning("benchmark enrichment failed at init: %s", e)

    # ── Blend profile + benchmark into final capabilities vector ──
    try:
        from src.models.auto_tuner import blend_capability_scores

        for m in _registry.all_models():
            if not m.benchmark_scores:
                continue
            blended = blend_capability_scores(
                profile_scores=dict(m.capabilities),
                benchmark_scores=dict(m.benchmark_scores),
                grading_scores={},           # no runtime data at startup
                grading_call_count=0,
            )
            m.capabilities = blended
    except Exception as e:
        logger.warning("capability blending failed at init: %s", e)

    _selector = Selector(
        registry=_registry,
        nerd_herd=nerd_herd,
        available_providers=available_providers,
    )
    return model_names
```

- [ ] **Step 4: Run the tests again — expect pass**

```bash
timeout 30 pytest tests/fatih_hoca/test_init_enrichment.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run all fatih_hoca + init tests to catch regressions**

```bash
timeout 60 pytest packages/fatih_hoca/tests/ tests/fatih_hoca/ -v
```

Expected: all green. If existing `test_init.py` breaks because it didn't expect enrich to run, the fix is to skip enrichment when `_registry` is empty (already handled by the try/except + empty-models iteration).

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/__init__.py tests/fatih_hoca/test_init_enrichment.py
git commit -m "feat(fatih_hoca): wire benchmark enrichment + capability blending into init()"
```

---

## Task 5: Remove `×10`-then-clamp in capability composite

**Context:** `ranking.py:218` does `cap_score = min(cap_score_raw * 10, 100)`. If `score_model_for_task` returns raw 0–10, `×10` maps to 0–100, but any value ≥ 10 flattens to 100 — losing signal above the clamp. Change to a clean 0–100 scale with no clamp ceiling.

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/ranking.py:218`
- Test: `tests/fatih_hoca/test_scoring_hygiene.py` (new)

- [ ] **Step 1: Verify `score_model_for_task` output range**

```bash
grep -n "def score_model_for_task\|return " packages/fatih_hoca/src/fatih_hoca/capabilities.py | head -20
```

Note the return type and range (expected: weighted sum of 0–10 per-dim scores × 0–1 weights → raw result in 0–10 when normalized).

- [ ] **Step 2: Write failing test**

```python
# tests/fatih_hoca/test_scoring_hygiene.py
"""Scoring hygiene: no clamp-flattening, real perf_score, narrowed failure penalty."""
from __future__ import annotations

import pytest

from fatih_hoca.ranking import rank_candidates
from fatih_hoca.registry import ModelInfo
from fatih_hoca.requirements import ModelRequirements
from fatih_hoca.types import Failure
from nerd_herd.types import SystemSnapshot


def _make_model(name: str, caps: dict, **kw) -> ModelInfo:
    return ModelInfo(
        name=name, location=kw.get("location", "local"),
        provider=kw.get("provider", "local"),
        litellm_name=kw.get("litellm_name", name),
        capabilities=caps,
        total_params_b=kw.get("total_params_b", 8.0),
        active_params_b=kw.get("active_params_b", 8.0),
        tokens_per_second=kw.get("tps", 20.0),
        **{k: v for k, v in kw.items() if k not in {"location", "provider", "litellm_name", "total_params_b", "active_params_b", "tps"}},
    )


class TestCapabilityClampRemoval:
    def test_very_strong_model_separates_from_strong_model(self):
        """A 10-across model should rank ABOVE a 9-across model. Under the ×10-clamp, both cap at 100."""
        ten = _make_model("ten", {c: 10.0 for c in [
            "reasoning","planning","analysis","code_generation","code_reasoning",
            "system_design","prose_quality","instruction_adherence","domain_knowledge",
            "context_utilization","structured_output","tool_use","conversation","turkish","vision",
        ]})
        nine = _make_model("nine", {c: 9.0 for c in [
            "reasoning","planning","analysis","code_generation","code_reasoning",
            "system_design","prose_quality","instruction_adherence","domain_knowledge",
            "context_utilization","structured_output","tool_use","conversation","turkish","vision",
        ]})
        reqs = ModelRequirements(
            primary_capability="reasoning",
            difficulty=7,
            estimated_input_tokens=500,
            estimated_output_tokens=500,
        )
        snap = SystemSnapshot()
        ranked = rank_candidates([ten, nine], reqs, snap, failures=[])
        assert ranked[0].model.name == "ten", \
            "after clamp removal, the stronger model must beat the weaker one"
        assert ranked[0].composite > ranked[1].composite + 0.5, \
            "separation must be meaningful, not tie-broken"
```

- [ ] **Step 3: Run the failing test**

```bash
timeout 30 pytest tests/fatih_hoca/test_scoring_hygiene.py::TestCapabilityClampRemoval -v
```

Expected: may pass OR fail depending on whether composite includes enough non-cap signal to break the tie. If it passes already (cost/speed differ), still proceed — the fix is about signal preservation, and the stronger assertion (`> ... + 0.5`) should fail.

- [ ] **Step 4: Apply the fix at `ranking.py:218`**

Replace:

```python
        cap_score = min(cap_score_raw * 10, 100)
        reasons.append(f"cap={cap_score_raw:.1f}")
```

with:

```python
        # cap_score_raw is 0–10 (weighted mean of per-dim 0–10 scores).
        # Scale to 0–100 cleanly; trust the upstream clamp in score_model_for_task.
        cap_score = cap_score_raw * 10.0
        reasons.append(f"cap={cap_score_raw:.2f}")
```

- [ ] **Step 5: Re-run test and the full ranking test module**

```bash
timeout 60 pytest tests/fatih_hoca/test_scoring_hygiene.py packages/fatih_hoca/tests/test_ranking.py -v
```

Expected: scoring_hygiene tests pass, existing test_ranking.py still green (the change is monotonic — no existing test should depend on `cap_score ≤ 100`).

If a ranking test breaks because it asserted `cap_score <= 100`, update it to assert `cap_score <= 100 + epsilon` where the actual upper bound is 100 (raw ≤ 10 by `score_model_for_task` design).

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/ranking.py tests/fatih_hoca/test_scoring_hygiene.py
git commit -m "fix(fatih_hoca): drop ×10-then-clamp that flattened signal above cap_score_raw=10"
```

---

## Task 6: Replace hardcoded `perf_score = 50` with measured-tps-derived score

**Context:** `ranking.py:294` has `# TODO: wire up performance cache` and `perf_score = 50` — one of five composite dimensions is pure noise. Phase 1 fix: derive `perf_score` from the loaded model's measured tps via Nerd Herd, falling back to a conservative 50 when no measurement exists. This is a stop-gap until a proper per-model quality history (grading scores) lands in a later phase.

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/ranking.py:293–295`
- Test: `tests/fatih_hoca/test_scoring_hygiene.py` (add test class)

- [ ] **Step 1: Write failing test**

Append to `tests/fatih_hoca/test_scoring_hygiene.py`:

```python
class TestPerfScoreFromTps:
    """perf_score must reflect measured tps, not a hardcoded 50."""

    def test_loaded_model_with_high_tps_scores_above_50(self):
        m = _make_model("fast", {"reasoning": 7.0}, tps=40.0)
        m.is_loaded = True
        reqs = ModelRequirements(primary_capability="reasoning", difficulty=5)

        from nerd_herd.types import LocalModelState
        snap = SystemSnapshot()
        snap.local = LocalModelState(model_name="fast", measured_tps=40.0)

        ranked = rank_candidates([m], reqs, snap, failures=[])
        # Decode perf contribution from reasons list
        reasons = ranked[0].reasons
        perf_reason = next((r for r in reasons if r.startswith("perf=")), None)
        assert perf_reason is not None, "ranking must expose perf= in reasons"
        perf_val = float(perf_reason.split("=")[1])
        assert perf_val > 50, f"40 tps should beat baseline 50, got perf={perf_val}"

    def test_unmeasured_cloud_model_falls_back_to_50(self):
        m = _make_model("cloud", {"reasoning": 7.0}, location="cloud",
                        provider="anthropic", litellm_name="claude/sonnet")
        reqs = ModelRequirements(primary_capability="reasoning", difficulty=5)
        snap = SystemSnapshot()

        ranked = rank_candidates([m], reqs, snap, failures=[])
        reasons = ranked[0].reasons
        perf_reason = next((r for r in reasons if r.startswith("perf=")), None)
        assert perf_reason is not None
        perf_val = float(perf_reason.split("=")[1])
        assert perf_val == 50.0, f"cloud model with no history should fall back to 50, got {perf_val}"
```

- [ ] **Step 2: Run the failing test**

```bash
timeout 30 pytest tests/fatih_hoca/test_scoring_hygiene.py::TestPerfScoreFromTps -v
```

Expected: both fail (perf is always 50, no `perf=` reason emitted).

- [ ] **Step 3: Apply the fix at `ranking.py:293–295`**

Replace:

```python
        # ── 4. Performance History (0–100) ──
        # TODO: wire up performance cache (refresh from DB stats)
        perf_score = 50
```

with:

```python
        # ── 4. Performance History (0–100) ──
        # Derive from measured tps when this is the loaded local model.
        # TODO(phase-2): replace with grading-based quality score from model_stats.
        if model.is_local and model.is_loaded and \
           local_state.model_name == model.name and local_state.measured_tps > 0:
            tps = local_state.measured_tps
            # 10 tps → 50, 20 tps → 65, 40 tps → 80, 80+ tps → 95
            perf_score = min(95.0, 50.0 + (tps - 10) * 1.5) if tps >= 10 else max(20.0, 20.0 + tps * 3.0)
        elif model.is_local and model.tokens_per_second > 0:
            tps = model.tokens_per_second
            perf_score = min(90.0, 45.0 + (tps - 10) * 1.2) if tps >= 10 else max(15.0, 15.0 + tps * 3.0)
        else:
            perf_score = 50.0
        reasons.append(f"perf={perf_score:.0f}")
```

- [ ] **Step 4: Re-run tests**

```bash
timeout 60 pytest tests/fatih_hoca/test_scoring_hygiene.py packages/fatih_hoca/tests/test_ranking.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/ranking.py tests/fatih_hoca/test_scoring_hygiene.py
git commit -m "fix(fatih_hoca): replace hardcoded perf_score=50 with measured-tps-derived score"
```

---

## Task 7: Narrow failure penalty scope — don't blacklist provider siblings on single-model 429

**Context:** `ranking.py:_failure_penalty` (lines 52–107) applies a 0.3× multiplier to every model from a provider when any model from that provider hits a rate limit. A single `groq/llama-70b` 429 shouldn't tank `groq/mixtral-8x7b`. Narrow the provider-wide penalty to require either (a) the same `litellm_name`, or (b) Nerd Herd's `consecutive_failures >= 3` for the provider (indicating real provider-level trouble, not single-model quota).

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/ranking.py:52–107` (`_failure_penalty`)
- Modify: `packages/fatih_hoca/src/fatih_hoca/ranking.py` — pass snapshot into failure penalty or inspect at call site
- Test: `tests/fatih_hoca/test_scoring_hygiene.py` (add test class)

- [ ] **Step 1: Read current `_failure_penalty` to understand shape**

```bash
sed -n '52,110p' packages/fatih_hoca/src/fatih_hoca/ranking.py
```

Note how failures are keyed (by `litellm_name`? `provider`?) and how `rate_limit` is distinguished from per-model failures.

- [ ] **Step 2: Write failing test**

Append to `tests/fatih_hoca/test_scoring_hygiene.py`:

```python
class TestFailurePenaltyScope:
    """A single model's 429 must not poison its provider siblings."""

    def test_single_model_rate_limit_does_not_penalize_siblings(self):
        # Two Groq models — one had a 429, the other is healthy
        rate_limited = _make_model("groq-llama-70b",
            {"reasoning": 7.0}, location="cloud",
            provider="groq", litellm_name="groq/llama-3.3-70b")
        sibling = _make_model("groq-mixtral",
            {"reasoning": 7.0}, location="cloud",
            provider="groq", litellm_name="groq/mixtral-8x7b")

        reqs = ModelRequirements(primary_capability="reasoning", difficulty=5)
        snap = SystemSnapshot()
        # Groq provider has 0 consecutive failures — only one model is rate-limited
        from nerd_herd.types import CloudProviderState
        snap.cloud["groq"] = CloudProviderState(provider="groq", consecutive_failures=0)

        failures = [Failure(
            model="groq/llama-3.3-70b",
            error_type="rate_limit",
            message="429 Too Many Requests",
        )]

        ranked = rank_candidates([rate_limited, sibling], reqs, snap, failures=failures)
        by_name = {r.model.name: r for r in ranked}

        # Sibling's reasons must NOT contain provider-wide rate-limit penalty
        sibling_reasons = by_name["groq-mixtral"].reasons
        assert not any("rate_limit" in r.lower() for r in sibling_reasons), \
            f"sibling should not be penalized for another model's 429, got reasons={sibling_reasons}"
        # Rate-limited model SHOULD have penalty
        rl_reasons = by_name["groq-llama-70b"].reasons
        assert any("rate_limit" in r.lower() for r in rl_reasons), \
            "rate-limited model must show rate_limit in reasons"

    def test_provider_wide_penalty_applies_when_circuit_breaker_trips(self):
        """When Nerd Herd reports consecutive_failures >= 3, treat as provider outage."""
        m1 = _make_model("groq-a", {"reasoning": 7.0}, location="cloud",
                         provider="groq", litellm_name="groq/a")
        m2 = _make_model("groq-b", {"reasoning": 7.0}, location="cloud",
                         provider="groq", litellm_name="groq/b")

        reqs = ModelRequirements(primary_capability="reasoning", difficulty=5)
        snap = SystemSnapshot()
        from nerd_herd.types import CloudProviderState
        snap.cloud["groq"] = CloudProviderState(provider="groq", consecutive_failures=3)

        failures = [Failure(model="groq/a", error_type="rate_limit", message="429")]

        ranked = rank_candidates([m1, m2], reqs, snap, failures=failures)
        by_name = {r.model.name: r for r in ranked}
        # Both models should be penalized (provider-wide when circuit tripped)
        assert any("provider_rate_limit" in r.lower() or "provider" in r.lower()
                   for r in by_name["groq-b"].reasons), \
            "sibling should be penalized when circuit breaker shows provider outage"
```

- [ ] **Step 3: Run the failing test**

```bash
timeout 30 pytest tests/fatih_hoca/test_scoring_hygiene.py::TestFailurePenaltyScope -v
```

Expected: both fail.

- [ ] **Step 4: Modify `_failure_penalty` to require snapshot context for provider-wide scope**

Signature change: `_failure_penalty(model, failure_idx, failures, snapshot)`. Narrow the `rate_limit` branch so it only applies provider-wide when `snapshot.cloud[provider].consecutive_failures >= 3`:

```python
def _failure_penalty(
    model: ModelInfo,
    failure_idx: dict,
    failures: list["Failure"],
    snapshot: "SystemSnapshot",
) -> tuple[float, bool, list[str]]:
    """Return (multiplier, exclude, reasons) for this model given past failures.

    Rate-limit policy (narrowed 2026-04-17):
    - Per-model 429 → penalize ONLY that litellm_name (0.3×).
    - Provider-wide penalty (0.3× on siblings) applies ONLY when
      snapshot reports consecutive_failures >= 3 for that provider.
      This prevents single-model quota from poisoning healthy siblings.
    """
    mult = 1.0
    exclude = False
    reasons: list[str] = []

    # Direct failures for this model's litellm_name
    direct = [f for f in failures if f.model == model.litellm_name]
    for f in direct:
        if f.error_type == "loading":
            return (0.0, True, [f"excluded_loading:{f.message[:40]}"])
        if f.error_type == "timeout":
            mult = min(mult, 0.2); reasons.append("timeout")
        elif f.error_type == "quality_failure":
            mult = min(mult, 0.5); reasons.append("quality_failure")
        elif f.error_type == "server_error":
            mult = min(mult, 0.3); reasons.append("server_error")
        elif f.error_type == "rate_limit":
            mult = min(mult, 0.3); reasons.append("rate_limit")

    # Provider-wide 429 — ONLY when circuit breaker shows real provider trouble
    if model.location == "cloud" and model.provider:
        prov_state = snapshot.cloud.get(model.provider)
        consec = prov_state.consecutive_failures if prov_state else 0
        if consec >= 3:
            prov_rate_limits = [
                f for f in failures
                if f.error_type == "rate_limit"
                and f.model != model.litellm_name  # direct already handled above
                and f.model.startswith(f"{model.provider}/")
            ]
            if prov_rate_limits:
                mult = min(mult, 0.3)
                reasons.append(f"provider_rate_limit(consec={consec})")

    return (mult, exclude, reasons)
```

- [ ] **Step 5: Update the caller at `ranking.py:155` to pass `snapshot`**

```python
        fail_mult, fail_exclude, fail_reasons = _failure_penalty(
            model, failure_idx, failures, snapshot
        )
```

- [ ] **Step 6: Re-run tests**

```bash
timeout 60 pytest tests/fatih_hoca/test_scoring_hygiene.py packages/fatih_hoca/tests/test_ranking.py -v
```

Expected: all pass. If existing `test_ranking.py` had a test that asserted provider-wide penalty on a single 429, update it to set `snapshot.cloud[provider].consecutive_failures = 3`.

- [ ] **Step 7: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/ranking.py tests/fatih_hoca/test_scoring_hygiene.py
git commit -m "fix(fatih_hoca): narrow rate-limit failure penalty to prevent single-model provider poisoning"
```

---

## Task 8: Pick telemetry schema + DB migration

**Context:** We need per-pick evidence: every candidate's composite + component scores + reasons. Without this, weight tuning is blind. Schema designed for ~1 year of data at ~1000 picks/day = 365k rows; stays small (single-digit MB).

**Files:**
- Create: `src/infra/db_migrations/006_model_pick_log.sql`
- Modify: `src/infra/db.py` — register migration

- [ ] **Step 1: Check existing migration registration pattern**

```bash
grep -n "db_migrations\|006_\|005_\|MIGRATIONS" src/infra/db.py | head -20
```

Note the pattern (inline CREATE TABLE IF NOT EXISTS, or migration-file runner — likely the former per earlier grep showing CREATE TABLE inline in db.py).

- [ ] **Step 2: Add table definition inline in `db.py`**

Find the end of the existing `CREATE TABLE IF NOT EXISTS` block in `src/infra/db.py` (after `file_locks` per earlier grep; use `grep -n "file_locks" src/infra/db.py` to locate). Add immediately after it:

```python
    # Model pick telemetry (Phase 1 selection-intelligence plan)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS model_pick_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            task_name TEXT NOT NULL,
            agent_type TEXT,
            difficulty INTEGER,
            call_category TEXT,
            picked_model TEXT NOT NULL,
            picked_score REAL NOT NULL,
            picked_reasons TEXT,
            candidates_json TEXT NOT NULL,
            failures_json TEXT,
            snapshot_summary TEXT
        )
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_pick_log_task
          ON model_pick_log(task_name, timestamp DESC)
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_pick_log_model
          ON model_pick_log(picked_model, timestamp DESC)
    """)
```

- [ ] **Step 3: Write a smoke test that the table exists after init**

Append to `tests/fatih_hoca/test_pick_telemetry.py`:

```python
# tests/fatih_hoca/test_pick_telemetry.py
"""Pick telemetry: every selection is logged to model_pick_log with full candidate breakdown."""
from __future__ import annotations

import json
import pytest
import aiosqlite


@pytest.mark.asyncio
async def test_model_pick_log_table_exists(tmp_path, monkeypatch):
    """After init_db(), model_pick_log must exist with expected columns."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))

    from src.infra.db import init_db
    await init_db()

    async with aiosqlite.connect(tmp_path / "test.db") as db:
        cur = await db.execute("PRAGMA table_info(model_pick_log)")
        cols = {row[1] for row in await cur.fetchall()}

    expected = {
        "id", "timestamp", "task_name", "agent_type", "difficulty",
        "call_category", "picked_model", "picked_score", "picked_reasons",
        "candidates_json", "failures_json", "snapshot_summary",
    }
    assert expected.issubset(cols), f"missing columns: {expected - cols}"
```

- [ ] **Step 4: Run the smoke test**

```bash
timeout 30 pytest tests/fatih_hoca/test_pick_telemetry.py::test_model_pick_log_table_exists -v
```

Expected: pass after db.py edit applied.

- [ ] **Step 5: Commit**

```bash
git add src/infra/db.py tests/fatih_hoca/test_pick_telemetry.py
git commit -m "feat(db): add model_pick_log table for selection telemetry"
```

---

## Task 9: Selector emits telemetry + full reasons to logger

**Context:** After a pick, `selector.py:176–180` logs a terse line. The `reasons` list per candidate is computed in `rank_candidates` but discarded. Change: (a) log the top-3 candidates with their scores and reasons, (b) async-persist the full breakdown to `model_pick_log`. Persistence must be fire-and-forget to avoid slowing the selection path.

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/selector.py:176–180` (pick log line)
- Modify: `packages/fatih_hoca/src/fatih_hoca/selector.py` — new `_persist_pick_telemetry()` helper
- Test: `tests/fatih_hoca/test_pick_telemetry.py` (extend)

- [ ] **Step 1: Write failing test for log + persistence**

Append to `tests/fatih_hoca/test_pick_telemetry.py`:

```python
@pytest.mark.asyncio
async def test_select_persists_pick_to_db(tmp_path, monkeypatch, fake_nerd_herd, caplog):
    """A successful select() must write one row to model_pick_log with top candidates."""
    import logging
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))

    from src.infra.db import init_db
    await init_db()

    import fatih_hoca
    from fatih_hoca.registry import ModelInfo

    # Minimal registry with two candidates
    fatih_hoca._registry = None
    fatih_hoca._selector = None

    from fatih_hoca.registry import ModelRegistry
    from fatih_hoca.selector import Selector

    reg = ModelRegistry()
    reg._models["a"] = ModelInfo(
        name="a", location="local", path="/fake/a.gguf",
        total_params_b=8.0, active_params_b=8.0,
        capabilities={c: 7.0 for c in [
            "reasoning","code_generation","analysis","instruction_adherence"
        ]},
    )
    reg._models["b"] = ModelInfo(
        name="b", location="local", path="/fake/b.gguf",
        total_params_b=8.0, active_params_b=8.0,
        capabilities={c: 5.0 for c in [
            "reasoning","code_generation","analysis","instruction_adherence"
        ]},
    )
    fatih_hoca._registry = reg
    fatih_hoca._selector = Selector(registry=reg, nerd_herd=fake_nerd_herd)

    with caplog.at_level(logging.INFO, logger="fatih_hoca.selector"):
        pick = fatih_hoca.select(
            task="coder", agent_type="coder", difficulty=5,
            estimated_input_tokens=500, estimated_output_tokens=500,
            call_category="main_work",
        )

    # Logger: top-3 line emitted with 'picked='
    top3_logs = [r for r in caplog.records if "picked=" in r.message and "candidates=" in r.message]
    assert top3_logs, "selector must emit a top-candidates summary log line"

    # DB: one row in model_pick_log
    async with aiosqlite.connect(tmp_path / "test.db") as db:
        cur = await db.execute(
            "SELECT picked_model, picked_score, candidates_json FROM model_pick_log"
        )
        rows = await cur.fetchall()
    assert len(rows) == 1, f"expected 1 pick row, got {len(rows)}"
    picked_model, picked_score, cand_json = rows[0]
    assert picked_model == pick.model.name
    cands = json.loads(cand_json)
    assert len(cands) >= 2, "candidates_json must include all ranked candidates"
    # Each candidate entry has name, composite, and reasons
    assert all("name" in c and "composite" in c and "reasons" in c for c in cands)
```

- [ ] **Step 2: Run the failing test**

```bash
timeout 30 pytest tests/fatih_hoca/test_pick_telemetry.py -v
```

Expected: fails (no persistence, no top-3 log).

- [ ] **Step 3: Read current `selector.select()` pick-log section**

```bash
sed -n '140,200p' packages/fatih_hoca/src/fatih_hoca/selector.py
```

Note where the final `Pick` is constructed and logged.

- [ ] **Step 4: Add persistence + rich log line**

At the end of `Selector.select()`, just before `return Pick(...)`, add:

```python
        # Pick telemetry — async fire-and-forget DB write + structured log
        try:
            top_n = min(len(ranked), 5)
            top_summary = ", ".join(
                f"{r.model.name}={r.composite:.1f}"
                for r in ranked[:top_n]
            )
            logger.info(
                "picked=%s score=%.1f task=%s diff=%d category=%s candidates=[%s]",
                ranked[0].model.name, ranked[0].composite,
                effective_task, reqs.difficulty, call_category,
                top_summary,
            )
            self._persist_pick_telemetry(ranked, reqs, effective_task, call_category, failures, snapshot)
        except Exception as e:
            logger.debug("pick telemetry log/persist failed: %s", e)
```

And add the helper method on `Selector`:

```python
    def _persist_pick_telemetry(
        self,
        ranked: list,
        reqs,
        task_name: str,
        call_category: str,
        failures: list,
        snapshot,
    ) -> None:
        """Fire-and-forget write to model_pick_log. Never raises; never blocks selection."""
        import json, asyncio

        candidates_payload = [
            {
                "name": r.model.name,
                "composite": round(r.composite, 2),
                "reasons": r.reasons,
            }
            for r in ranked[:10]  # top 10 only — rest is noise
        ]
        failures_payload = [
            {"model": f.model, "type": f.error_type, "msg": f.message[:100]}
            for f in (failures or [])
        ]
        snapshot_summary = {
            "vram_free_mb": snapshot.vram_available_mb,
            "loaded": snapshot.local.model_name,
        }

        async def _write():
            try:
                from src.infra.db import get_db
                async with get_db() as db:
                    await db.execute(
                        """
                        INSERT INTO model_pick_log
                          (task_name, agent_type, difficulty, call_category,
                           picked_model, picked_score, picked_reasons,
                           candidates_json, failures_json, snapshot_summary)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            task_name or "unknown",
                            getattr(reqs, "agent_type", None),
                            reqs.difficulty,
                            call_category,
                            ranked[0].model.name,
                            round(ranked[0].composite, 2),
                            json.dumps(ranked[0].reasons),
                            json.dumps(candidates_payload),
                            json.dumps(failures_payload),
                            json.dumps(snapshot_summary),
                        ),
                    )
                    await db.commit()
            except Exception:
                pass  # telemetry must never break selection

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_write())
            else:
                loop.run_until_complete(_write())
        except RuntimeError:
            # No event loop (sync context) — skip telemetry this call
            pass
```

**Note for executor:** verify `src.infra.db.get_db()` exists as an async context manager. If the helper is named differently (e.g., `db_connection`), update the import accordingly.

- [ ] **Step 5: Re-run telemetry tests**

```bash
timeout 30 pytest tests/fatih_hoca/test_pick_telemetry.py -v
```

Expected: all pass.

- [ ] **Step 6: Run the full selector test suite**

```bash
timeout 60 pytest packages/fatih_hoca/tests/test_selector.py -v
```

Expected: green. If any test breaks because it didn't mock `get_db`, the fix is to catch the import failure silently in the telemetry helper (already in the `try/except Exception: pass`).

- [ ] **Step 7: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/selector.py tests/fatih_hoca/test_pick_telemetry.py
git commit -m "feat(fatih_hoca): pick telemetry — full candidate breakdown to log + model_pick_log"
```

---

## Task 10: End-to-end integration test

**Context:** Prove the whole chain works on realistic data. Seed a registry with the Qwen trio, point enrichment at the canned AA cache, verify that after `init()` the dense 32B (stronger coder per AA) ranks above the base A3B (weaker coder per AA) for a `coder` task.

**Files:**
- Create: `tests/fatih_hoca/test_e2e_benchmark_driven_ranking.py`

- [ ] **Step 1: Write the E2E test**

```python
# tests/fatih_hoca/test_e2e_benchmark_driven_ranking.py
"""End-to-end: AA benchmark signal flows through init → blend → ranking → correct winner."""
from __future__ import annotations

import pytest

import fatih_hoca
from fatih_hoca.registry import ModelInfo, ModelRegistry


def _seed(monkeypatch):
    from fatih_hoca import registry as reg_mod
    seeded = {
        "qwen3-30b-a3b": ModelInfo(
            name="qwen3-30b-a3b", location="local",
            path="/fake/Qwen3-30B-A3B-Instruct-Q4_K_M.gguf",
            total_params_b=30.0, active_params_b=3.0, family="qwen3",
            tokens_per_second=25.0,
            # Profile guess: weaker coder, stronger generalist
            capabilities={
                "reasoning": 6.0, "code_generation": 5.5, "code_reasoning": 5.5,
                "analysis": 6.0, "instruction_adherence": 6.5,
                "tool_use": 5.5, "domain_knowledge": 6.0,
            },
        ),
        "qwen3-32b": ModelInfo(
            name="qwen3-32b", location="local",
            path="/fake/Qwen3-32B-Instruct-Q4_K_M.gguf",
            total_params_b=32.0, active_params_b=32.0, family="qwen3",
            tokens_per_second=10.0,
            # Profile guess: balanced — AA will push coder up
            capabilities={
                "reasoning": 6.5, "code_generation": 5.5, "code_reasoning": 5.5,
                "analysis": 6.0, "instruction_adherence": 6.5,
                "tool_use": 5.5, "domain_knowledge": 6.0,
            },
        ),
    }

    def fake_load_gguf_dir(self, models_dir):
        self._models.update(seeded)
        return list(seeded.values())

    monkeypatch.setattr(reg_mod.ModelRegistry, "load_gguf_dir", fake_load_gguf_dir)


def test_aa_signal_promotes_dense_32b_for_coder_over_base_a3b(
    monkeypatch, canned_aa_cache, tmp_path
):
    _seed(monkeypatch)
    monkeypatch.chdir(canned_aa_cache.parent)

    # Reset module state
    fatih_hoca._registry = None
    fatih_hoca._selector = None

    fatih_hoca.init(models_dir=str(tmp_path / "fake"))

    # After blend: qwen3-32b code_gen should move toward AA's 6.5 (up from profile 5.5),
    # qwen3-30b-a3b code_gen should move toward AA's 5.0 (down from profile 5.5).
    models = {m.name: m for m in fatih_hoca.all_models()}
    a3b_code = models["qwen3-30b-a3b"].capabilities["code_generation"]
    d32_code = models["qwen3-32b"].capabilities["code_generation"]
    assert d32_code > a3b_code, (
        f"AA signal must promote dense 32B coder above base A3B: "
        f"32b={d32_code:.2f} a3b={a3b_code:.2f}"
    )

    # Rank for a coder task — dense 32B must win
    pick = fatih_hoca.select(
        task="coder", agent_type="coder", difficulty=5,
        estimated_input_tokens=500, estimated_output_tokens=1000,
        call_category="main_work",
    )
    assert pick is not None
    assert pick.model.name == "qwen3-32b", (
        f"expected qwen3-32b (AA-stronger coder), got {pick.model.name}"
    )
```

- [ ] **Step 2: Run the E2E test**

```bash
timeout 60 pytest tests/fatih_hoca/test_e2e_benchmark_driven_ranking.py -v
```

Expected: pass. If it fails with a different winner, investigate:
1. Confirm `enrich_registry_with_benchmarks` actually matched the fixtures (check `model.benchmark_scores` populated).
2. Confirm `blend_capability_scores` produced the expected delta.
3. If blending works but ranking still picks A3B, check whether swap-budget stickiness or performance_score (a3b tps=25 vs 32b tps=10) is dominating — in that case, the test may need to pass `prefer_quality=True` to down-weight speed, OR the test is catching a real issue where speed dominates benchmark signal for medium-difficulty coder tasks (worth flagging to the user, not silencing).

- [ ] **Step 3: Run the entire fatih_hoca test suite as a regression guard**

```bash
timeout 120 pytest packages/fatih_hoca/tests/ tests/fatih_hoca/ -v
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/fatih_hoca/test_e2e_benchmark_driven_ranking.py
git commit -m "test(fatih_hoca): E2E — AA signal promotes dense 32B over base A3B for coder task"
```

---

## Task 11: Verification sweep + documentation touch-up

**Files:**
- Modify: `CLAUDE.md` (update the "Fatih Hoca owns all model knowledge" paragraph + remove the outdated shopping_advisor-flat-profile note if verified wrong)
- Modify: `docs/architecture-modularization.md` (if benchmark wiring diagram exists, update)

- [ ] **Step 1: Run full test suite with timeout**

```bash
timeout 300 pytest packages/fatih_hoca/tests/ tests/fatih_hoca/ tests/test_benchmark_fetcher.py -v
```

Expected: all pass.

- [ ] **Step 2: Smoke-test the real thing**

```bash
# From repo root
python -c "
import fatih_hoca
names = fatih_hoca.init(models_dir='models/', catalog_path='src/models/models.yaml')
print(f'registered {len(names)} models')
for m in fatih_hoca.all_models()[:5]:
    print(f'  {m.name}: benchmark={bool(m.benchmark_scores)} caps_code_gen={m.capabilities.get(\"code_generation\", \"?\"):.2f}')
"
```

Expected: log shows `benchmark coverage: X/Y matched (unmatched=Z)` with X > 0. Any "unmatched" warnings identify real alias gaps — add entries or document in a follow-up.

- [ ] **Step 3: Update CLAUDE.md**

In the "LLM Dispatch & Model Routing" section of `CLAUDE.md`, update the Fatih Hoca description to mention benchmark wiring:

```
- **Fatih Hoca** owns all model knowledge: catalog (YAML+GGUF), benchmark enrichment (AA + 9 sources, cached in `.benchmark_cache/`, wired into `init()`), 15-dimension scoring, task profiles, swap budget (max 3/5min), failure adaptation, quota planning. Queries Nerd Herd for system state via `snapshot()`. Every pick is persisted to `model_pick_log` for offline weight tuning.
```

And in "Common Pitfalls":

```
- **Benchmark cache staleness**: `.benchmark_cache/_bulk_*.json` TTL is 48h. When stale, `BenchmarkCache.load()` returns None (no silent fallback). Refresh with `python -m src.models.benchmark.benchmark_cli benchmarks`. Check `model_pick_log` snapshot_summary if selection quality drops.
```

Remove the outdated note if you confirmed `shopping_advisor` IS in `TASK_PROFILES`:

```
~~- **`shopping_advisor` task profile** must exist in `TASK_PROFILES`...~~
```

(Verify first: `grep -n shopping_advisor packages/fatih_hoca/src/fatih_hoca/capabilities.py`. If present, strike the pitfall.)

- [ ] **Step 4: Final commit**

```bash
git add CLAUDE.md
git commit -m "docs: reflect benchmark wiring + pick telemetry in CLAUDE.md"
```

- [ ] **Step 5: Ship-readiness check**

```bash
git log --oneline | head -12
```

Expected: 10 commits from this plan, in order.

---

## Post-Phase-1 handoff

Once this plan is complete:

1. **Collect telemetry for 2 weeks.** Let `model_pick_log` fill with real picks across varied tasks. Query: `SELECT task_name, picked_model, AVG(picked_score), COUNT(*) FROM model_pick_log GROUP BY task_name, picked_model ORDER BY task_name, 3 DESC`.

2. **Phase 2 candidates** (separate plans):
   - Background benchmark refresh scheduler (cron or orchestrator task every 24h, triggered when cache age > 24h).
   - Weight auto-calibration using `model_pick_log` outcomes as training signal.
   - Replace `perf_score` tps-derived score with grading-score-derived score from `model_stats`.
   - Audit specialty-bonus double-counting (Task 2 from analysis, deferred).
   - Extend `_MODEL_ALIASES` whenever `init()` logs unmatched locals.

3. **Do not** tune scoring weights by hand in response to complaints until you have 2+ weeks of `model_pick_log` data. The whole point of this plan was to replace guessing with evidence — skipping the data-collection phase defeats it.
