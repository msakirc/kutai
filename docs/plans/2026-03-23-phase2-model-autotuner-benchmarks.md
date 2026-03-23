# Phase 2 Completion: Hybrid Auto-Tuner + Benchmark Refresh + Prometheus Metrics

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the two remaining Phase 2 gaps: (1) a hybrid auto-tuner that blends internal grading history with benchmark-derived capability scores and exposes results via Prometheus, and (2) refresh the benchmark fetcher with more up-to-date sources.

**Architecture:** The auto-tuner lives in `src/models/auto_tuner.py`. It reads `model_stats` (grading history) from the DB and blends those empirical scores with the benchmark-derived + profile-derived capability scores already in the registry. The blended scores feed back into `ModelInfo.capabilities` so the router's `select_model()` automatically benefits. New Prometheus metrics expose per-model per-capability scores, grading history, and tuning events at the existing `/metrics` endpoint.

**Tech Stack:** Python 3.12, aiosqlite, existing `model_stats` table, existing `/metrics` Prometheus endpoint in `api.py`, existing `BenchmarkFetcher` in `benchmark_fetcher.py`

---

## Task 1: Auto-Tuner Core — Blend Internal Grades with Precoded Scores

**Files:**
- Create: `src/models/auto_tuner.py`
- Modify: `src/models/model_registry.py` (import + call auto-tuner on refresh)
- Test: `tests/test_auto_tuner.py`

**Context:**
- `model_stats` table has per-model per-agent_type rows with `avg_grade` (1-5 scale), `success_rate`, `total_calls`
- `ModelInfo.capabilities` dict has 14-dimension scores (0-10)
- `TASK_PROFILES` maps agent_type → weighted capability vector (tells us which capabilities matter for each agent)
- `update_quality_from_grading()` already does single-sample EMA updates — the auto-tuner does batch recalculation
- `enrich_registry_with_benchmarks()` already blends 60% benchmark + 40% profile — we add a third signal

**Step 1: Write the failing test**

```python
# tests/test_auto_tuner.py
import unittest
from unittest.mock import AsyncMock, patch, MagicMock


class TestAutoTuner(unittest.TestCase):
    """Test hybrid capability score blending."""

    def test_blend_scores_all_sources(self):
        """When we have profile, benchmark, AND grading data, blend all three."""
        from src.models.auto_tuner import blend_capability_scores

        profile_scores = {"code_generation": 7.0, "reasoning": 6.0}
        benchmark_scores = {"code_generation": 8.0, "reasoning": 7.5}
        grading_scores = {"code_generation": 9.0}  # only some caps have grades

        result = blend_capability_scores(
            profile_scores, benchmark_scores, grading_scores,
            grading_call_count=25,
        )
        # With 25 calls, grading weight should be meaningful
        self.assertGreater(result["code_generation"], 7.5)
        self.assertLess(result["code_generation"], 9.0)
        # reasoning has no grading data — falls back to benchmark+profile blend
        self.assertAlmostEqual(result["reasoning"], 7.5 * 0.6 + 6.0 * 0.4, places=1)

    def test_blend_scores_no_grading(self):
        """Without grading history, use benchmark+profile only (existing behavior)."""
        from src.models.auto_tuner import blend_capability_scores

        profile = {"code_generation": 7.0}
        benchmark = {"code_generation": 8.0}

        result = blend_capability_scores(profile, benchmark, {}, grading_call_count=0)
        self.assertAlmostEqual(result["code_generation"], 8.0 * 0.6 + 7.0 * 0.4, places=1)

    def test_blend_scores_few_calls_low_grading_weight(self):
        """With < 5 calls, grading signal gets very low weight."""
        from src.models.auto_tuner import blend_capability_scores

        profile = {"code_generation": 7.0}
        benchmark = {"code_generation": 8.0}
        grading = {"code_generation": 2.0}  # bad grade but few samples

        result = blend_capability_scores(profile, benchmark, grading, grading_call_count=2)
        # Should still be close to benchmark+profile, grading barely matters
        no_grading = 8.0 * 0.6 + 7.0 * 0.4
        self.assertGreater(result["code_generation"], no_grading - 0.5)

    def test_grading_scores_from_model_stats(self):
        """Convert model_stats rows into per-capability grading scores."""
        from src.models.auto_tuner import compute_grading_scores

        # Simulate model_stats rows for one model
        stats_rows = [
            {"model": "qwen3-32b", "agent_type": "coder", "avg_grade": 4.5, "success_rate": 0.95, "total_calls": 30},
            {"model": "qwen3-32b", "agent_type": "reviewer", "avg_grade": 3.5, "success_rate": 0.80, "total_calls": 15},
        ]

        scores, total_calls = compute_grading_scores("qwen3-32b", stats_rows)

        # coder maps primarily to code_generation → should have a score
        self.assertIn("code_generation", scores)
        self.assertGreater(scores["code_generation"], 0)
        self.assertEqual(total_calls, 45)


class TestAutoTunerAsync(unittest.IsolatedAsyncioTestCase):

    async def test_run_tuning_cycle(self):
        """Full tuning cycle reads DB, computes blended scores, updates registry."""
        from src.models.auto_tuner import run_tuning_cycle

        mock_stats = [
            {"model": "test-model", "agent_type": "coder",
             "avg_grade": 4.0, "success_rate": 0.90, "total_calls": 20,
             "avg_cost": 0.001, "avg_latency": 500},
        ]

        mock_model = MagicMock()
        mock_model.capabilities = {"code_generation": 7.0, "reasoning": 6.0}
        mock_model.benchmark_scores = {"code_generation": 8.0}
        mock_model.profile_scores = {"code_generation": 7.0, "reasoning": 6.0}

        mock_registry = MagicMock()
        mock_registry.models = {"test-model": mock_model}

        with patch("src.models.auto_tuner.get_model_stats", new_callable=AsyncMock, return_value=mock_stats), \
             patch("src.models.auto_tuner.get_registry", return_value=mock_registry):
            report = await run_tuning_cycle()

        self.assertIn("test-model", report["tuned_models"])


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_auto_tuner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.auto_tuner'`

**Step 3: Implement auto_tuner.py**

```python
# src/models/auto_tuner.py
"""
Hybrid Auto-Tuner — blends three signal sources for model capability scores:

1. **Profile scores** — hardcoded family-based estimates from model_profiles.py
2. **Benchmark scores** — fetched from public leaderboards (Aider, LiveCodeBench, etc.)
3. **Grading scores** — empirical quality grades from actual task execution (model_stats DB)

Blend weights adapt based on how much grading data we have:
  - 0-4 calls:  0% grading, 60% benchmark, 40% profile  (existing behavior)
  - 5-19 calls: 20% grading, 48% benchmark, 32% profile
  - 20-49 calls: 35% grading, 39% benchmark, 26% profile
  - 50+ calls:  50% grading, 30% benchmark, 20% profile

The tuning cycle runs periodically (every 6 hours or on-demand) and updates
ModelInfo.capabilities in-place so the router's select_model() automatically
benefits on its next scoring pass.
"""

from __future__ import annotations

import time
from typing import Any

from src.infra.logging_config import get_logger
from src.models.capabilities import TASK_PROFILES, Cap

logger = get_logger("models.auto_tuner")

# ── Weight Schedule ─────────────────────────────────────────────────────────

def _grading_weight(call_count: int) -> float:
    """Grading signal weight increases with sample count."""
    if call_count < 5:
        return 0.0
    if call_count < 20:
        return 0.20
    if call_count < 50:
        return 0.35
    return 0.50


# ── Core Blending ───────────────────────────────────────────────────────────

def blend_capability_scores(
    profile_scores: dict[str, float],
    benchmark_scores: dict[str, float],
    grading_scores: dict[str, float],
    grading_call_count: int = 0,
) -> dict[str, float]:
    """
    Blend three signal sources into final capability scores.

    For each capability dimension:
    - If grading data exists for that dim AND enough calls: use 3-way blend
    - Otherwise: use existing 2-way blend (60% benchmark, 40% profile)

    Returns dict of capability_name → blended score (0-10).
    """
    all_caps = set(profile_scores) | set(benchmark_scores) | set(grading_scores)
    result: dict[str, float] = {}

    gw = _grading_weight(grading_call_count)

    for cap in all_caps:
        profile_val = profile_scores.get(cap)
        bench_val = benchmark_scores.get(cap)
        grade_val = grading_scores.get(cap)

        if grade_val is not None and gw > 0:
            # 3-way blend: grading + benchmark + profile
            remaining = 1.0 - gw
            bw = remaining * 0.6  # benchmark share of remainder
            pw = remaining * 0.4  # profile share of remainder

            # Use whatever sources are available
            total_weight = 0.0
            weighted_sum = 0.0

            if grade_val is not None:
                weighted_sum += gw * grade_val
                total_weight += gw
            if bench_val is not None:
                weighted_sum += bw * bench_val
                total_weight += bw
            if profile_val is not None:
                weighted_sum += pw * profile_val
                total_weight += pw

            result[cap] = round(weighted_sum / total_weight, 1) if total_weight > 0 else 5.0
        else:
            # 2-way blend (existing behavior): 60% benchmark, 40% profile
            if bench_val is not None and profile_val is not None:
                result[cap] = round(bench_val * 0.6 + profile_val * 0.4, 1)
            elif bench_val is not None:
                result[cap] = round(bench_val, 1)
            elif profile_val is not None:
                result[cap] = round(profile_val, 1)
            # else: skip — no data at all

        # Clamp
        if cap in result:
            result[cap] = max(0.0, min(10.0, result[cap]))

    return result


# ── Convert model_stats → per-capability grading scores ─────────────────────

def compute_grading_scores(
    model_name: str,
    stats_rows: list[dict],
) -> tuple[dict[str, float], int]:
    """
    Convert model_stats rows into per-capability quality scores.

    Each stats row has (model, agent_type, avg_grade, success_rate, total_calls).
    We map agent_type → TASK_PROFILES → dominant capabilities, then distribute
    the grade (scaled 1-5 → 2-10) across those capabilities weighted by
    the task profile weights.

    Returns (capability_scores, total_call_count).
    """
    # Accumulate: cap → [(score, weight), ...]
    cap_accum: dict[str, list[tuple[float, float]]] = {}
    total_calls = 0

    for row in stats_rows:
        if row["model"] != model_name:
            continue

        agent_type = row["agent_type"]
        avg_grade = row.get("avg_grade", 0.0) or 0.0
        success_rate = row.get("success_rate", 0.0) or 0.0
        calls = row.get("total_calls", 0) or 0
        total_calls += calls

        if calls < 3 or avg_grade <= 0:
            continue

        # Map grade (1-5) to capability scale (2-10)
        quality_score = avg_grade * 2.0

        # Factor in success rate: penalize unreliable models
        quality_score *= (0.5 + 0.5 * success_rate)

        # Find the task profile for this agent_type
        profile = TASK_PROFILES.get(agent_type)
        if not profile:
            continue

        # Distribute quality across capabilities weighted by profile
        for cap_name, weight in profile.items():
            cap_key = cap_name.value if hasattr(cap_name, "value") else cap_name
            if weight < 0.3:
                continue  # skip low-relevance capabilities

            if cap_key not in cap_accum:
                cap_accum[cap_key] = []
            # Weight by both task profile weight and call count (more data = more influence)
            effective_weight = weight * min(calls, 50)  # cap influence at 50 calls
            cap_accum[cap_key].append((quality_score, effective_weight))

    # Merge into single score per capability
    scores: dict[str, float] = {}
    for cap, pairs in cap_accum.items():
        total_w = sum(w for _, w in pairs)
        if total_w > 0:
            scores[cap] = round(sum(s * w for s, w in pairs) / total_w, 1)
            scores[cap] = max(0.0, min(10.0, scores[cap]))

    return scores, total_calls


# ── Tuning Cycle ────────────────────────────────────────────────────────────

_last_tuning: float = 0.0
TUNING_INTERVAL_SECS = 6 * 3600  # 6 hours


async def run_tuning_cycle(force: bool = False) -> dict[str, Any]:
    """
    Run a full tuning cycle: read model_stats, blend with benchmarks + profiles,
    update registry capabilities in-place.

    Returns a report dict with what was tuned and by how much.
    """
    global _last_tuning
    from src.infra.db import get_model_stats
    from src.models.model_registry import get_registry

    registry = get_registry()
    stats = await get_model_stats()

    report: dict[str, Any] = {
        "tuned_models": {},
        "skipped": [],
        "timestamp": time.time(),
    }

    for model_name, model_info in registry.models.items():
        # Get the three signal sources
        profile_scores = getattr(model_info, "profile_scores", None) or {}
        benchmark_scores = getattr(model_info, "benchmark_scores", None) or {}

        # Compute grading-derived scores from model_stats
        grading_scores, total_calls = compute_grading_scores(model_name, stats)

        # Also try matching by litellm_name (cloud models use that in stats)
        if total_calls == 0 and model_info.litellm_name != model_name:
            grading_scores, total_calls = compute_grading_scores(
                model_info.litellm_name, stats,
            )

        if not profile_scores and not benchmark_scores and total_calls == 0:
            report["skipped"].append(model_name)
            continue

        # If no profile/benchmark stored separately, use current caps as profile fallback
        if not profile_scores:
            profile_scores = dict(model_info.capabilities)
        if not benchmark_scores:
            benchmark_scores = {}

        blended = blend_capability_scores(
            profile_scores, benchmark_scores, grading_scores,
            grading_call_count=total_calls,
        )

        if not blended:
            continue

        # Track changes
        changes: dict[str, tuple[float, float]] = {}
        for cap, new_val in blended.items():
            old_val = model_info.capabilities.get(cap, 0.0)
            if abs(new_val - old_val) > 0.2:
                changes[cap] = (old_val, new_val)

        if changes:
            # Apply blended scores
            for cap, new_val in blended.items():
                model_info.capabilities[cap] = new_val
            report["tuned_models"][model_name] = {
                "changes": {k: {"old": v[0], "new": v[1]} for k, v in changes.items()},
                "grading_calls": total_calls,
                "grading_weight": _grading_weight(total_calls),
            }
            logger.info(
                f"Auto-tuned {model_name}: {len(changes)} caps adjusted "
                f"(grading_calls={total_calls}, gw={_grading_weight(total_calls):.0%})"
            )

    _last_tuning = time.time()
    logger.info(
        f"Tuning cycle complete: {len(report['tuned_models'])} models tuned, "
        f"{len(report['skipped'])} skipped"
    )
    return report


async def maybe_run_tuning() -> dict[str, Any] | None:
    """Run tuning if enough time has passed since last run."""
    if time.time() - _last_tuning < TUNING_INTERVAL_SECS:
        return None
    return await run_tuning_cycle()


# ── Prometheus Metrics ──────────────────────────────────────────────────────

def get_prometheus_lines() -> list[str]:
    """
    Generate Prometheus-format metrics for model quality and tuning state.
    Called by the /metrics endpoint in api.py.
    """
    from src.models.model_registry import get_registry

    lines: list[str] = []
    registry = get_registry()

    # Per-model per-capability scores
    lines.append("# HELP kutay_model_capability Model capability score by dimension")
    lines.append("# TYPE kutay_model_capability gauge")
    for name, model in registry.models.items():
        safe_name = name.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        for cap, score in model.capabilities.items():
            safe_cap = cap.replace('\\', '\\\\').replace('"', '\\"')
            lines.append(
                f'kutay_model_capability{{model="{safe_name}",capability="{safe_cap}"}} {score:.1f}'
            )

    # Per-model composite quality (avg of all capability scores)
    lines.append("# HELP kutay_model_quality_avg Average capability score across all dimensions")
    lines.append("# TYPE kutay_model_quality_avg gauge")
    for name, model in registry.models.items():
        safe_name = name.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        caps = model.capabilities
        if caps:
            avg = sum(caps.values()) / len(caps)
            lines.append(f'kutay_model_quality_avg{{model="{safe_name}"}} {avg:.2f}')

    # Tuning metadata
    lines.append("# HELP kutay_autotuner_last_run_timestamp Unix timestamp of last tuning cycle")
    lines.append("# TYPE kutay_autotuner_last_run_timestamp gauge")
    lines.append(f"kutay_autotuner_last_run_timestamp {_last_tuning:.0f}")

    # Grading weight schedule info (for each weight tier, how many models qualify)
    lines.append("# HELP kutay_autotuner_interval_seconds Tuning cycle interval in seconds")
    lines.append("# TYPE kutay_autotuner_interval_seconds gauge")
    lines.append(f"kutay_autotuner_interval_seconds {TUNING_INTERVAL_SECS}")

    return lines
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_auto_tuner.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/models/auto_tuner.py tests/test_auto_tuner.py
git commit -m "feat(phase2): add hybrid auto-tuner with 3-way capability blending"
```

---

## Task 2: Store Profile + Benchmark Scores Separately on ModelInfo

**Files:**
- Modify: `src/models/model_registry.py` (add fields to ModelInfo, store during enrichment)
- Modify: `src/models/benchmark/benchmark_fetcher.py` (store benchmark_scores on model)

**Context:**
The auto-tuner needs access to the original profile scores and benchmark scores separately (before they were blended into `capabilities`). Currently `enrich_registry_with_benchmarks()` blends them in-place and the originals are lost.

**Step 1: Add fields to ModelInfo**

In `src/models/model_registry.py`, add two fields to the `ModelInfo` dataclass:

```python
    # Score provenance (for auto-tuner blending)
    profile_scores: dict[str, float] = field(default_factory=dict)
    benchmark_scores: dict[str, float] = field(default_factory=dict)
```

**Step 2: Store profile scores during model registration**

In `model_registry.py`, after `_compute_capabilities()` returns, store a copy:

```python
model_info.profile_scores = dict(model_info.capabilities)
```

This goes in the `_register_local_model()` and `_register_cloud_model()` methods, right after capabilities are set.

**Step 3: Store benchmark scores during enrichment**

In `benchmark_fetcher.py` `enrich_registry_with_benchmarks()`, before the blend, store:

```python
model_info.benchmark_scores = dict(benchmark_caps)
```

**Step 4: Run existing tests**

Run: `python -m pytest tests/ -v -x --timeout=30`
Expected: No regressions

**Step 5: Commit**

```bash
git add src/models/model_registry.py src/models/benchmark/benchmark_fetcher.py
git commit -m "feat(phase2): store profile + benchmark scores separately for auto-tuner"
```

---

## Task 3: Wire Auto-Tuner into Orchestrator Loop + Prometheus Endpoint

**Files:**
- Modify: `src/core/orchestrator.py` (call `maybe_run_tuning()` periodically)
- Modify: `src/app/api.py` (add auto-tuner metrics to `/metrics` endpoint)

**Step 1: Add tuning call to orchestrator main loop**

In `src/core/orchestrator.py`, in the main processing loop (where `maybe_persist()` and other periodic tasks are called), add:

```python
from src.models.auto_tuner import maybe_run_tuning

# Inside the main loop, alongside other periodic checks:
try:
    await maybe_run_tuning()
except Exception as e:
    logger.debug("auto-tuning cycle failed", error=str(e))
```

**Step 2: Add Prometheus metrics to /metrics endpoint**

In `src/app/api.py`, in the `prometheus_metrics()` function, before the final `return`, add:

```python
        # ── Auto-tuner quality metrics ──
        try:
            from src.models.auto_tuner import get_prometheus_lines
            lines.extend(get_prometheus_lines())
        except Exception as e:
            logger.debug(f"Auto-tuner metrics unavailable: {e}")
```

**Step 3: Run tests**

Run: `python -m pytest tests/test_auto_tuner.py tests/ -v -x --timeout=30`
Expected: All pass

**Step 4: Commit**

```bash
git add src/core/orchestrator.py src/app/api.py
git commit -m "feat(phase2): wire auto-tuner into orchestrator loop + Prometheus metrics"
```

---

## Task 4: Add /tune Telegram Command

**Files:**
- Modify: `src/app/telegram_bot.py` (add /tune command)

**Step 1: Add command handler**

In `telegram_bot.py`, add a handler in `_setup_handlers()`:

```python
self.app.add_handler(CommandHandler("tune", self.cmd_tune))
```

**Step 2: Implement the command**

```python
async def cmd_tune(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force a tuning cycle and report results."""
    if not self._is_admin(update):
        return
    await update.message.reply_text("Running tuning cycle...")
    try:
        from src.models.auto_tuner import run_tuning_cycle
        report = await run_tuning_cycle(force=True)

        tuned = report.get("tuned_models", {})
        if not tuned:
            await update.message.reply_text("No models needed tuning adjustment.")
            return

        lines = ["🎛 *Auto-Tuning Report*\n"]
        for model, info in tuned.items():
            changes = info["changes"]
            gw = info["grading_weight"]
            lines.append(f"*{model}* (grading weight: {gw:.0%})")
            for cap, vals in changes.items():
                arrow = "↑" if vals["new"] > vals["old"] else "↓"
                lines.append(f"  {cap}: {vals['old']:.1f} {arrow} {vals['new']:.1f}")
        lines.append(f"\n_{len(report.get('skipped', []))} models skipped (no data)_")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Tuning failed: {e}")
```

**Step 3: Commit**

```bash
git add src/app/telegram_bot.py
git commit -m "feat(phase2): add /tune Telegram command for on-demand auto-tuning"
```

---

## Task 5: Refresh Benchmark Sources

**Files:**
- Modify: `src/models/benchmark/benchmark_fetcher.py` (update/add sources)

**Context:**
Current 7 sources: Artificial Analysis, HuggingFace Leaderboard, LiveCodeBench, BFCL, LMSys Arena, Aider, BigCodeBench. We want to:
1. Update the HuggingFace source to use the Open LLM Leaderboard v2 API (new schema)
2. Add OpenRouter rankings as a new source (covers many models with pricing + quality data)
3. Update model alias mappings with newer models (Qwen3.5, Llama 4, etc.)
4. Reduce cache TTL from 72h to 48h for fresher data

**Step 1: Update cache TTL**

Change `CACHE_TTL_HOURS = 72` to `CACHE_TTL_HOURS = 48` at the top of the file.

**Step 2: Add OpenRouter rankings source**

Add a new fetcher class after the existing ones:

```python
class OpenRouterRankingsFetcher(_BaseFetcher):
    """
    Fetch model quality data from OpenRouter's /api/v1/models endpoint.
    Provides pricing + context length + moderation data across many providers.
    Maps top_provider pricing tiers to rough quality estimates.
    """
    source_name = "openrouter"

    def _fetch_from_api(self) -> dict[str, dict[str, float]]:
        import urllib.request
        url = "https://openrouter.ai/api/v1/models"
        req = urllib.request.Request(url, headers={"User-Agent": "kutay-benchmark/1.0"})

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        models = data.get("data", [])
        results: dict[str, dict[str, float]] = {}

        for m in models:
            model_id = m.get("id", "")
            name = model_id.split("/")[-1] if "/" in model_id else model_id
            ctx = m.get("context_length", 0)

            # Use pricing as a proxy signal — better models cost more
            pricing = m.get("pricing", {})
            prompt_cost = float(pricing.get("prompt", "0") or "0")
            completion_cost = float(pricing.get("completion", "0") or "0")

            # Skip models with no pricing data (usually broken/stale)
            if prompt_cost == 0 and completion_cost == 0:
                continue

            # Context utilization score based on context length
            if ctx >= 128000:
                ctx_score = 9.0
            elif ctx >= 32000:
                ctx_score = 7.0
            elif ctx >= 8000:
                ctx_score = 5.0
            else:
                ctx_score = 3.0

            caps: dict[str, float] = {
                "context_utilization": ctx_score,
            }
            results[name] = caps

        return results

    def fetch(self, model_id: str, cache: BenchmarkCache) -> Optional[BenchmarkResult]:
        cached = cache.get(self.source_name, model_id)
        if cached:
            return BenchmarkResult(
                source=self.source_name,
                model_id=model_id,
                raw_scores={},
                mapped_capabilities=cached.get("capabilities", {}),
                timestamp=cached.get("timestamp", 0),
                confidence=0.60,
            )
        # OpenRouter is bulk-only; individual model lookups use the bulk cache
        bulk = self.fetch_bulk(cache)
        matched = _fuzzy_match_model(model_id, list(bulk.keys())) if bulk else None
        if matched and matched in bulk:
            return BenchmarkResult(
                source=self.source_name,
                model_id=model_id,
                raw_scores={},
                mapped_capabilities=bulk[matched],
                confidence=0.60,
            )
        return None

    def fetch_bulk(self, cache: BenchmarkCache) -> dict[str, dict[str, float]]:
        cached = cache.get_all_models(self.source_name)
        if cached and "models" in cached:
            return cached["models"]
        try:
            models = self._fetch_from_api()
            cache.put_all_models(self.source_name, {"models": models})
            logger.info(f"OpenRouter: fetched {len(models)} models")
            return models
        except Exception as e:
            logger.warning(f"OpenRouter fetch failed: {e}")
            return {}
```

**Step 3: Register the new fetcher**

In `BenchmarkFetcher.__init__()`, add to the fetchers list:

```python
OpenRouterRankingsFetcher(),
```

And add to the confidence map in `fetch_all_bulk()`:

```python
"openrouter": 0.60,
```

**Step 4: Update model aliases**

Add newer models to `_MODEL_ALIASES`:

```python
    "qwen3.5-32b":     ["Qwen3.5-32B", "Qwen/Qwen3.5-32B"],
    "qwen3-coder-32b": ["Qwen3-Coder-32B", "Qwen/Qwen3-Coder-32B-Instruct"],
    "gemma3-27b":      ["Gemma-3-27B", "google/gemma-3-27b-it"],
    "phi4-14b":        ["Phi-4-14B", "microsoft/Phi-4"],
```

**Step 5: Commit**

```bash
git add src/models/benchmark/benchmark_fetcher.py
git commit -m "feat(phase2): add OpenRouter source, update aliases, reduce cache TTL to 48h"
```

---

## Task 6: Rolling Model Health Metrics (Phase 2.6 Gap)

**Files:**
- Modify: `src/infra/metrics.py` (add model health counters)
- Modify: `src/models/auto_tuner.py` (add health assessment to Prometheus output)
- Modify: `src/app/api.py` (add model_stats to Prometheus)

**Context:**
The router already has performance history (section 4 of scoring, lines 466-483) and demotes models with <50% success rate. But there are no Prometheus metrics exposing per-model success_rate, avg_grade, or health status for Grafana dashboards.

**Step 1: Add model_stats Prometheus metrics**

Add to `auto_tuner.py`'s `get_prometheus_lines()`:

```python
async def get_prometheus_lines_async() -> list[str]:
    """Extended Prometheus lines that include DB-sourced model_stats."""
    lines = get_prometheus_lines()  # sync capability scores

    try:
        from src.infra.db import get_model_stats
        stats = await get_model_stats()

        lines.append("# HELP kutay_model_success_rate Model success rate by agent type")
        lines.append("# TYPE kutay_model_success_rate gauge")
        lines.append("# HELP kutay_model_avg_grade Model average grade by agent type")
        lines.append("# TYPE kutay_model_avg_grade gauge")
        lines.append("# HELP kutay_model_total_calls Model total calls by agent type")
        lines.append("# TYPE kutay_model_total_calls counter")
        lines.append("# HELP kutay_model_avg_latency_ms Model average latency by agent type")
        lines.append("# TYPE kutay_model_avg_latency_ms gauge")
        lines.append("# HELP kutay_model_avg_cost Model average cost by agent type")
        lines.append("# TYPE kutay_model_avg_cost gauge")

        for row in stats:
            model = row["model"].replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            agent = row["agent_type"].replace('\\', '\\\\').replace('"', '\\"')
            sr = row.get("success_rate", 0) or 0
            grade = row.get("avg_grade", 0) or 0
            calls = row.get("total_calls", 0) or 0
            latency = row.get("avg_latency", 0) or 0
            cost = row.get("avg_cost", 0) or 0

            labels = f'model="{model}",agent_type="{agent}"'
            lines.append(f'kutay_model_success_rate{{{labels}}} {sr:.3f}')
            lines.append(f'kutay_model_avg_grade{{{labels}}} {grade:.2f}')
            lines.append(f'kutay_model_total_calls{{{labels}}} {calls}')
            lines.append(f'kutay_model_avg_latency_ms{{{labels}}} {latency:.1f}')
            lines.append(f'kutay_model_avg_cost{{{labels}}} {cost:.6f}')
    except Exception as e:
        logger.debug(f"model_stats prometheus export failed: {e}")

    return lines
```

**Step 2: Update api.py to use the async version**

In the `/metrics` endpoint, change the auto-tuner import to use the async version:

```python
        try:
            from src.models.auto_tuner import get_prometheus_lines_async
            lines.extend(await get_prometheus_lines_async())
        except Exception as e:
            logger.debug(f"Auto-tuner metrics unavailable: {e}")
```

**Step 3: Commit**

```bash
git add src/models/auto_tuner.py src/app/api.py
git commit -m "feat(phase2): expose model_stats as Prometheus metrics (success_rate, grade, latency, cost)"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Auto-tuner core: 3-way blending + grading→capability conversion | `auto_tuner.py`, `test_auto_tuner.py` |
| 2 | Store profile/benchmark scores separately on ModelInfo | `model_registry.py`, `benchmark_fetcher.py` |
| 3 | Wire into orchestrator loop + Prometheus | `orchestrator.py`, `api.py` |
| 4 | /tune Telegram command | `telegram_bot.py` |
| 5 | Benchmark source refresh: OpenRouter, updated aliases, 48h TTL | `benchmark_fetcher.py` |
| 6 | Rolling model health as Prometheus metrics | `auto_tuner.py`, `api.py` |
