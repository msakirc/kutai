# tests/test_auto_tuner.py
"""
Tests for the auto-tuner: blend logic, grading computation, tuning cycle,
and Prometheus metric generation.
"""

import asyncio
import os
import sys
import time
import unittest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.auto_tuner import (
    MIN_CHANGE_THRESHOLD,
    TUNING_INTERVAL_SECONDS,
    _get_blend_weights,
    blend_capability_scores,
    compute_grading_scores,
    get_prometheus_lines,
    maybe_run_tuning,
    run_tuning_cycle,
)
from src.models.capabilities import Cap


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _make_caps(value: float = 5.0) -> dict[str, float]:
    """Return a full capability dict with every Cap set to `value`."""
    return {c.value: value for c in Cap}


@dataclass
class FakeModelInfo:
    """Minimal stand-in for ModelInfo used in tests."""
    name: str = "test-model"
    location: str = "cloud"
    provider: str = "openai"
    litellm_name: str = "openai/gpt-4o"
    capabilities: dict = field(default_factory=lambda: _make_caps(5.0))
    context_length: int = 128000
    max_tokens: int = 4096
    supports_function_calling: bool = True
    supports_json_mode: bool = True
    thinking_model: bool = False
    has_vision: bool = False
    # profile_scores / benchmark_scores may be absent (Task 2 adds them)


class FakeRegistry:
    """Minimal stand-in for ModelRegistry."""

    def __init__(self, models: dict):
        self.models = models


# ═════════════════════════════════════════════════════════════════════════════
# 1 — Blend Weight Schedule
# ═════════════════════════════════════════════════════════════════════════════

class TestBlendWeightSchedule(unittest.TestCase):

    def test_zero_calls(self):
        gw, bw, pw = _get_blend_weights(0)
        self.assertAlmostEqual(gw, 0.0)
        self.assertAlmostEqual(bw, 0.60)
        self.assertAlmostEqual(pw, 0.40)

    def test_4_calls(self):
        gw, bw, pw = _get_blend_weights(4)
        self.assertAlmostEqual(gw, 0.0)
        self.assertAlmostEqual(bw, 0.60)
        self.assertAlmostEqual(pw, 0.40)

    def test_5_calls(self):
        gw, bw, pw = _get_blend_weights(5)
        self.assertAlmostEqual(gw, 0.20)
        self.assertAlmostEqual(bw, 0.48)
        self.assertAlmostEqual(pw, 0.32)

    def test_19_calls(self):
        gw, bw, pw = _get_blend_weights(19)
        self.assertAlmostEqual(gw, 0.20)
        self.assertAlmostEqual(bw, 0.48)
        self.assertAlmostEqual(pw, 0.32)

    def test_20_calls(self):
        gw, bw, pw = _get_blend_weights(20)
        self.assertAlmostEqual(gw, 0.35)
        self.assertAlmostEqual(bw, 0.39)
        self.assertAlmostEqual(pw, 0.26)

    def test_50_calls(self):
        gw, bw, pw = _get_blend_weights(50)
        self.assertAlmostEqual(gw, 0.50)
        self.assertAlmostEqual(bw, 0.30)
        self.assertAlmostEqual(pw, 0.20)

    def test_1000_calls(self):
        gw, bw, pw = _get_blend_weights(1000)
        self.assertAlmostEqual(gw, 0.50)
        self.assertAlmostEqual(bw, 0.30)
        self.assertAlmostEqual(pw, 0.20)

    def test_weights_sum_to_one(self):
        for n in [0, 4, 5, 19, 20, 49, 50, 100]:
            gw, bw, pw = _get_blend_weights(n)
            self.assertAlmostEqual(gw + bw + pw, 1.0, places=5,
                                   msg=f"weights don't sum to 1 at n={n}")


# ═════════════════════════════════════════════════════════════════════════════
# 2 — blend_capability_scores
# ═════════════════════════════════════════════════════════════════════════════

class TestBlendCapabilityScores(unittest.TestCase):

    def test_three_way_blend_50_calls(self):
        """Full 3-way blend at 50+ calls: 50% grading, 30% bench, 20% profile."""
        profile = {"reasoning": 6.0, "code_generation": 7.0}
        bench = {"reasoning": 8.0, "code_generation": 9.0}
        grading = {"reasoning": 7.0, "code_generation": 5.0}

        result = blend_capability_scores(profile, bench, grading, 50)

        # reasoning: 0.50*7 + 0.30*8 + 0.20*6 = 3.5 + 2.4 + 1.2 = 7.1
        self.assertAlmostEqual(result["reasoning"], 7.1, places=2)
        # code_gen: 0.50*5 + 0.30*9 + 0.20*7 = 2.5 + 2.7 + 1.4 = 6.6
        self.assertAlmostEqual(result["code_generation"], 6.6, places=2)

    def test_no_grading_falls_back_to_two_way(self):
        """When no grading data exists, use 60% bench + 40% profile."""
        profile = {"reasoning": 6.0}
        bench = {"reasoning": 8.0}
        grading: dict[str, float] = {}

        result = blend_capability_scores(profile, bench, grading, 0)

        # reasoning: 0.60*8 + 0.40*6 = 4.8 + 2.4 = 7.2
        self.assertAlmostEqual(result["reasoning"], 7.2, places=2)

    def test_few_calls_low_grading_weight(self):
        """With 5 calls, grading gets 20% weight."""
        profile = {"reasoning": 6.0}
        bench = {"reasoning": 8.0}
        grading = {"reasoning": 10.0}

        result = blend_capability_scores(profile, bench, grading, 5)

        # reasoning: 0.20*10 + 0.48*8 + 0.32*6 = 2.0 + 3.84 + 1.92 = 7.76
        self.assertAlmostEqual(result["reasoning"], 7.76, places=2)

    def test_grading_present_but_under_5_calls_uses_two_way(self):
        """Grading data exists but < 5 calls: falls back to 2-way."""
        profile = {"reasoning": 6.0}
        bench = {"reasoning": 8.0}
        grading = {"reasoning": 10.0}

        result = blend_capability_scores(profile, bench, grading, 3)

        # fallback: 0.60*8 + 0.40*6 = 7.2
        self.assertAlmostEqual(result["reasoning"], 7.2, places=2)

    def test_clamped_to_0_10(self):
        """Scores are clamped to [0, 10]."""
        profile = {"reasoning": 10.0}
        bench = {"reasoning": 10.0}
        grading = {"reasoning": 12.0}  # over 10

        result = blend_capability_scores(profile, bench, grading, 100)
        self.assertLessEqual(result["reasoning"], 10.0)
        self.assertGreaterEqual(result["reasoning"], 0.0)

    def test_union_of_capabilities(self):
        """All capabilities from all sources appear in the output."""
        profile = {"reasoning": 5.0}
        bench = {"code_generation": 7.0}
        grading = {"vision": 3.0}

        result = blend_capability_scores(profile, bench, grading, 50)
        self.assertIn("reasoning", result)
        self.assertIn("code_generation", result)
        self.assertIn("vision", result)


# ═════════════════════════════════════════════════════════════════════════════
# 3 — compute_grading_scores
# ═════════════════════════════════════════════════════════════════════════════

class TestComputeGradingScores(unittest.TestCase):

    def test_basic_conversion(self):
        """Single agent_type row converts correctly."""
        rows = [{
            "model": "gpt-4o",
            "agent_type": "coder",
            "avg_grade": 4.0,    # 1-5 scale
            "success_rate": 1.0,  # perfect
            "total_calls": 25,
        }]
        scores, total = compute_grading_scores("gpt-4o", rows)

        self.assertEqual(total, 25)
        # quality = 4.0 * 2.0 * (0.5 + 0.5*1.0) = 8.0
        # "coder" profile has code_generation=1.0, instruction_adherence=0.9,
        # domain_knowledge=0.7, reasoning=0.5, code_reasoning=0.5,
        # context_utilization=0.5, etc.
        # Only caps with weight >= 0.3 are included
        self.assertIn("code_generation", scores)
        self.assertIn("instruction_adherence", scores)
        self.assertIn("reasoning", scores)
        # code_generation: quality * 1.0 = 8.0
        self.assertAlmostEqual(scores["code_generation"], 8.0, places=1)
        # conversation has weight 0.0 in coder → should not appear
        self.assertNotIn("conversation", scores)

    def test_partial_success_rate(self):
        """success_rate < 1 reduces quality."""
        rows = [{
            "model": "m1",
            "agent_type": "coder",
            "avg_grade": 4.0,
            "success_rate": 0.5,
            "total_calls": 10,
        }]
        scores, total = compute_grading_scores("m1", rows)

        # quality = 4.0 * 2.0 * (0.5 + 0.5*0.5) = 8.0 * 0.75 = 6.0
        # code_generation weight=1.0 → 6.0
        self.assertAlmostEqual(scores["code_generation"], 6.0, places=1)

    def test_unknown_agent_type_skipped(self):
        """Rows with unknown agent_type are skipped."""
        rows = [{
            "model": "m1",
            "agent_type": "nonexistent_type",
            "avg_grade": 5.0,
            "success_rate": 1.0,
            "total_calls": 100,
        }]
        scores, total = compute_grading_scores("m1", rows)
        self.assertEqual(scores, {})
        self.assertEqual(total, 0)

    def test_multiple_agent_types_call_weighted(self):
        """Scores from multiple agent_types use call-count-weighted averaging."""
        rows = [
            {
                "model": "m1",
                "agent_type": "coder",
                "avg_grade": 4.0,
                "success_rate": 1.0,
                "total_calls": 20,
            },
            {
                "model": "m1",
                "agent_type": "fixer",
                "avg_grade": 3.0,
                "success_rate": 1.0,
                "total_calls": 10,
            },
        ]
        scores, total = compute_grading_scores("m1", rows)
        self.assertEqual(total, 30)

        # Both coder and fixer contribute to "reasoning".
        # coder: quality=8.0, profile_weight=0.5, eff_calls=min(20,50)=20
        #   w = 0.5*20 = 10,  numerator += 8.0*10 = 80
        # fixer: quality=6.0, profile_weight=0.8, eff_calls=min(10,50)=10
        #   w = 0.8*10 = 8,   numerator += 6.0*8 = 48
        # weighted_avg = (80+48) / (10+8) = 128/18 ≈ 7.11
        self.assertIn("reasoning", scores)
        self.assertAlmostEqual(scores["reasoning"], 128.0 / 18.0, places=1)

    def test_scores_clamped(self):
        """Output scores are clamped to 0-10."""
        rows = [{
            "model": "m1",
            "agent_type": "coder",
            "avg_grade": 5.0,
            "success_rate": 1.0,
            "total_calls": 10,
        }]
        scores, _ = compute_grading_scores("m1", rows)
        for val in scores.values():
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 10.0)


# ═════════════════════════════════════════════════════════════════════════════
# 4 — run_tuning_cycle (async)
# ═════════════════════════════════════════════════════════════════════════════

class TestRunTuningCycle(unittest.TestCase):

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_tuning_cycle_with_grading_data(self):
        """Tuning cycle updates model capabilities from DB stats."""
        model = FakeModelInfo(name="test-model", capabilities=_make_caps(5.0))
        fake_registry = FakeRegistry({"test-model": model})

        stats = [
            {
                "model": "test-model",
                "agent_type": "coder",
                "avg_grade": 5.0,
                "success_rate": 1.0,
                "total_calls": 60,
            },
        ]

        mock_get_stats = AsyncMock(return_value=stats)

        with patch("src.infra.db.get_model_stats", mock_get_stats), \
             patch("src.models.model_registry.get_registry", return_value=fake_registry):
            import src.models.auto_tuner as at_mod
            orig_last = at_mod._last_run_ts
            try:
                report = self._run(run_tuning_cycle())
            finally:
                at_mod._last_run_ts = orig_last

        self.assertIn("tuned_models", report)
        self.assertIn("skipped", report)
        self.assertIn("timestamp", report)
        self.assertIsInstance(report["timestamp"], float)

    def test_tuning_cycle_no_stats_skips(self):
        """Models with no grading data are still processed (2-way blend)."""
        model = FakeModelInfo(name="m1", capabilities=_make_caps(5.0))
        fake_registry = FakeRegistry({"m1": model})

        mock_get_stats = AsyncMock(return_value=[])

        with patch("src.infra.db.get_model_stats", mock_get_stats), \
             patch("src.models.model_registry.get_registry", return_value=fake_registry):
            import src.models.auto_tuner as at_mod
            orig_last = at_mod._last_run_ts
            try:
                report = self._run(run_tuning_cycle())
            finally:
                at_mod._last_run_ts = orig_last

        # With no stats and profile==benchmark==capabilities (both fallback
        # to current caps), the 2-way blend yields exactly the same values →
        # change = 0 → model is skipped.
        self.assertIn("m1", report["skipped"])

    def test_tuning_cycle_applies_threshold(self):
        """Changes below MIN_CHANGE_THRESHOLD are not applied."""
        caps = _make_caps(5.0)
        model = FakeModelInfo(name="m1", capabilities=caps.copy())
        fake_registry = FakeRegistry({"m1": model})

        # Stats that produce a tiny change (grade ~ 2.5 on 1-5 → quality=5.0)
        # quality*weight for caps with weight>=0.3 ≈ 5.0*w, very close to 5.0
        stats = [{
            "model": "m1",
            "agent_type": "coder",
            "avg_grade": 2.5,
            "success_rate": 1.0,
            "total_calls": 50,
        }]

        mock_get_stats = AsyncMock(return_value=stats)

        with patch("src.infra.db.get_model_stats", mock_get_stats), \
             patch("src.models.model_registry.get_registry", return_value=fake_registry):
            import src.models.auto_tuner as at_mod
            orig = at_mod._last_run_ts
            try:
                report = self._run(run_tuning_cycle())
            finally:
                at_mod._last_run_ts = orig

        # Whether tuned or skipped, the report should be well-formed
        self.assertIn("tuned_models", report)
        self.assertIn("skipped", report)


# ═════════════════════════════════════════════════════════════════════════════
# 5 — maybe_run_tuning
# ═════════════════════════════════════════════════════════════════════════════

class TestMaybeRunTuning(unittest.TestCase):

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_skips_when_recent(self):
        """Does not run if last run was recent."""
        import src.models.auto_tuner as at_mod
        orig = at_mod._last_run_ts
        try:
            at_mod._last_run_ts = time.time()  # just ran
            result = self._run(maybe_run_tuning())
            self.assertIsNone(result)
        finally:
            at_mod._last_run_ts = orig

    def test_runs_when_stale(self):
        """Runs if 6+ hours since last run."""
        import src.models.auto_tuner as at_mod
        model = FakeModelInfo(name="m1", capabilities=_make_caps(5.0))
        fake_registry = FakeRegistry({"m1": model})

        mock_get_stats = AsyncMock(return_value=[])

        orig = at_mod._last_run_ts
        try:
            at_mod._last_run_ts = 0.0  # ancient
            with patch("src.infra.db.get_model_stats", mock_get_stats), \
                 patch("src.models.model_registry.get_registry", return_value=fake_registry):
                result = self._run(maybe_run_tuning())
            self.assertIsNotNone(result)
            self.assertIn("tuned_models", result)
        finally:
            at_mod._last_run_ts = orig


# ═════════════════════════════════════════════════════════════════════════════
# 6 — Prometheus metrics
# ═════════════════════════════════════════════════════════════════════════════

class TestPrometheusLines(unittest.TestCase):

    def test_valid_format(self):
        """get_prometheus_lines returns valid Prometheus gauge lines."""
        caps = {"reasoning": 7.5, "code_generation": 8.0}
        model = FakeModelInfo(name="test-model", capabilities=caps)
        fake_registry = FakeRegistry({"test-model": model})

        with patch("src.models.model_registry.get_registry", return_value=fake_registry):
            lines = get_prometheus_lines()

        text = "\n".join(lines)

        # Check HELP and TYPE headers for each metric family
        self.assertIn("# HELP kutay_model_capability", text)
        self.assertIn("# TYPE kutay_model_capability gauge", text)
        self.assertIn("# HELP kutay_model_quality_avg", text)
        self.assertIn("# TYPE kutay_model_quality_avg gauge", text)
        self.assertIn("# HELP kutay_autotuner_last_run_timestamp", text)
        self.assertIn("# TYPE kutay_autotuner_last_run_timestamp gauge", text)
        self.assertIn("# HELP kutay_autotuner_interval_seconds", text)
        self.assertIn("# TYPE kutay_autotuner_interval_seconds gauge", text)

        # Check capability data lines (exclude comments)
        cap_lines = [l for l in lines
                     if l.startswith("kutay_model_capability{")]
        self.assertEqual(len(cap_lines), 2)  # two capabilities

        for line in cap_lines:
            self.assertIn('model="test-model"', line)
            self.assertIn("capability=", line)
            # Should end with a float
            parts = line.rsplit(" ", 1)
            float(parts[1])  # should not raise

        # Check avg quality line
        avg_lines = [l for l in lines
                     if l.startswith("kutay_model_quality_avg{")]
        self.assertEqual(len(avg_lines), 1)
        avg_val = float(avg_lines[0].rsplit(" ", 1)[1])
        self.assertAlmostEqual(avg_val, 7.75, places=2)

        # Check autotuner metadata
        ts_lines = [l for l in lines
                    if l.startswith("kutay_autotuner_last_run_timestamp ")]
        self.assertEqual(len(ts_lines), 1)

        interval_lines = [l for l in lines
                          if l.startswith("kutay_autotuner_interval_seconds ")]
        self.assertEqual(len(interval_lines), 1)
        interval_val = float(interval_lines[0].rsplit(" ", 1)[1])
        self.assertEqual(interval_val, TUNING_INTERVAL_SECONDS)

    def test_no_models(self):
        """Works with an empty registry — still emits headers + metadata."""
        fake_registry = FakeRegistry({})

        with patch("src.models.model_registry.get_registry", return_value=fake_registry):
            lines = get_prometheus_lines()

        # 4 metric families x 2 header lines = 8, plus 2 data lines for
        # last_run_timestamp and interval_seconds = 10 total
        self.assertEqual(len(lines), 10)


# ═════════════════════════════════════════════════════════════════════════════
# 7 — litellm_name fallback
# ═════════════════════════════════════════════════════════════════════════════

class TestLitellmNameFallback(unittest.TestCase):

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_stats_found_via_litellm_name(self):
        """When model_stats are keyed by litellm_name, tuning still finds them."""
        model = FakeModelInfo(
            name="gemini-flash",
            litellm_name="gemini/gemini-2.0-flash",
            capabilities=_make_caps(5.0),
        )
        fake_registry = FakeRegistry({"gemini-flash": model})

        # Stats recorded under the litellm_name, not the registry name
        stats = [{
            "model": "gemini/gemini-2.0-flash",
            "agent_type": "coder",
            "avg_grade": 5.0,
            "success_rate": 1.0,
            "total_calls": 60,
        }]

        mock_get_stats = AsyncMock(return_value=stats)

        with patch("src.infra.db.get_model_stats", mock_get_stats), \
             patch("src.models.model_registry.get_registry", return_value=fake_registry):
            import src.models.auto_tuner as at_mod
            orig = at_mod._last_run_ts
            try:
                report = self._run(run_tuning_cycle())
            finally:
                at_mod._last_run_ts = orig

        # The model should have been tuned (not skipped), because the
        # litellm_name fallback found the stats.
        self.assertIn("gemini-flash", report["tuned_models"])
        self.assertEqual(
            report["tuned_models"]["gemini-flash"]["grading_calls"], 60
        )

    def test_no_fallback_when_registry_name_has_stats(self):
        """When stats exist under registry name, litellm_name is not used."""
        model = FakeModelInfo(
            name="my-model",
            litellm_name="openai/gpt-4o",
            capabilities=_make_caps(5.0),
        )
        fake_registry = FakeRegistry({"my-model": model})

        stats = [
            {
                "model": "my-model",
                "agent_type": "coder",
                "avg_grade": 5.0,
                "success_rate": 1.0,
                "total_calls": 60,
            },
            {
                "model": "openai/gpt-4o",
                "agent_type": "coder",
                "avg_grade": 1.0,
                "success_rate": 0.1,
                "total_calls": 100,
            },
        ]

        mock_get_stats = AsyncMock(return_value=stats)

        with patch("src.infra.db.get_model_stats", mock_get_stats), \
             patch("src.models.model_registry.get_registry", return_value=fake_registry):
            import src.models.auto_tuner as at_mod
            orig = at_mod._last_run_ts
            try:
                report = self._run(run_tuning_cycle())
            finally:
                at_mod._last_run_ts = orig

        # Should use the registry-name stats (60 calls), not litellm (100)
        self.assertIn("my-model", report["tuned_models"])
        self.assertEqual(
            report["tuned_models"]["my-model"]["grading_calls"], 60
        )


# ═════════════════════════════════════════════════════════════════════════════
# 8 — _last_run_ts update & force bypass
# ═════════════════════════════════════════════════════════════════════════════

class TestLastRunTimestamp(unittest.TestCase):

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_run_tuning_cycle_updates_last_run_ts(self):
        """run_tuning_cycle sets _last_run_ts to a recent timestamp."""
        model = FakeModelInfo(name="m1", capabilities=_make_caps(5.0))
        fake_registry = FakeRegistry({"m1": model})
        mock_get_stats = AsyncMock(return_value=[])

        import src.models.auto_tuner as at_mod
        orig = at_mod._last_run_ts
        try:
            at_mod._last_run_ts = 0.0
            before = time.time()
            with patch("src.infra.db.get_model_stats", mock_get_stats), \
                 patch("src.models.model_registry.get_registry", return_value=fake_registry):
                report = self._run(run_tuning_cycle())
            after = time.time()

            self.assertGreaterEqual(at_mod._last_run_ts, before)
            self.assertLessEqual(at_mod._last_run_ts, after)
            self.assertEqual(report["timestamp"], at_mod._last_run_ts)
        finally:
            at_mod._last_run_ts = orig

    def test_maybe_run_tuning_force_bypasses_interval(self):
        """force=True on maybe_run_tuning runs even if interval hasn't elapsed."""
        model = FakeModelInfo(name="m1", capabilities=_make_caps(5.0))
        fake_registry = FakeRegistry({"m1": model})
        mock_get_stats = AsyncMock(return_value=[])

        import src.models.auto_tuner as at_mod
        orig = at_mod._last_run_ts
        try:
            at_mod._last_run_ts = time.time()  # just ran
            with patch("src.infra.db.get_model_stats", mock_get_stats), \
                 patch("src.models.model_registry.get_registry", return_value=fake_registry):
                # Without force, should return None
                result_no_force = self._run(maybe_run_tuning(force=False))
                self.assertIsNone(result_no_force)

                # With force, should run
                result_forced = self._run(maybe_run_tuning(force=True))
                self.assertIsNotNone(result_forced)
                self.assertIn("tuned_models", result_forced)
        finally:
            at_mod._last_run_ts = orig


# ═════════════════════════════════════════════════════════════════════════════
# 9 — compute_grading_scores model_name filtering
# ═════════════════════════════════════════════════════════════════════════════

class TestGradingScoresModelFiltering(unittest.TestCase):

    def test_filters_by_model_name(self):
        """Only rows matching the given model_name are used."""
        rows = [
            {
                "model": "gpt-4o",
                "agent_type": "coder",
                "avg_grade": 5.0,
                "success_rate": 1.0,
                "total_calls": 30,
            },
            {
                "model": "claude-sonnet",
                "agent_type": "coder",
                "avg_grade": 2.0,
                "success_rate": 0.5,
                "total_calls": 20,
            },
        ]
        scores, total = compute_grading_scores("gpt-4o", rows)

        # Only gpt-4o's 30 calls should count
        self.assertEqual(total, 30)
        # quality = 5.0 * 2.0 * (0.5 + 0.5*1.0) = 10.0
        # code_generation weight=1.0 → 10.0
        self.assertAlmostEqual(scores["code_generation"], 10.0, places=1)

    def test_empty_when_no_matching_model(self):
        """Returns empty scores when no rows match model_name."""
        rows = [{
            "model": "other-model",
            "agent_type": "coder",
            "avg_grade": 5.0,
            "success_rate": 1.0,
            "total_calls": 50,
        }]
        scores, total = compute_grading_scores("my-model", rows)
        self.assertEqual(scores, {})
        self.assertEqual(total, 0)

    def test_call_count_capped_at_50(self):
        """effective_calls is capped at 50 even with more actual calls."""
        # Two agent types with different call counts: 100 and 10
        rows = [
            {
                "model": "m1",
                "agent_type": "coder",
                "avg_grade": 4.0,
                "success_rate": 1.0,
                "total_calls": 100,  # capped to 50
            },
            {
                "model": "m1",
                "agent_type": "fixer",
                "avg_grade": 2.0,
                "success_rate": 1.0,
                "total_calls": 10,
            },
        ]
        scores, total = compute_grading_scores("m1", rows)
        self.assertEqual(total, 110)

        # For code_generation: only coder contributes (fixer weight=0.6)
        # coder: quality=8.0, weight=1.0, eff_calls=50 → w=50, num=8.0*50=400
        # fixer: quality=4.0, weight=0.6, eff_calls=10 → w=6, num=4.0*6=24
        # result = (400+24)/(50+6) = 424/56 ≈ 7.57
        self.assertIn("code_generation", scores)
        self.assertAlmostEqual(scores["code_generation"], 424.0 / 56.0, places=1)


# ═════════════════════════════════════════════════════════════════════════════
# 10 — Prometheus label escaping
# ═════════════════════════════════════════════════════════════════════════════

class TestPrometheusLabelEscaping(unittest.TestCase):

    def test_backslash_and_quote_escaped(self):
        """Model names with backslash and double-quote are escaped."""
        from src.models.auto_tuner import _prom_escape_label
        caps = {"reasoning": 5.0}
        model = FakeModelInfo(
            name='model\\"weird',
            capabilities=caps,
        )
        fake_registry = FakeRegistry({'model\\"weird': model})

        with patch("src.models.model_registry.get_registry", return_value=fake_registry):
            lines = get_prometheus_lines()

        cap_lines = [l for l in lines if l.startswith("kutay_model_capability{")]
        self.assertEqual(len(cap_lines), 1)
        # The backslash should become \\\\ and " should become \\"
        self.assertIn('model="model\\\\\\"weird"', cap_lines[0])

    def test_newline_escaped(self):
        """Newlines in label values are escaped to \\n."""
        from src.models.auto_tuner import _prom_escape_label
        self.assertEqual(_prom_escape_label("line1\nline2"), "line1\\nline2")
        self.assertEqual(_prom_escape_label('a"b'), 'a\\"b')
        self.assertEqual(_prom_escape_label("a\\b"), "a\\\\b")

    def test_normal_names_unchanged(self):
        """Normal model names pass through unmodified."""
        from src.models.auto_tuner import _prom_escape_label
        self.assertEqual(_prom_escape_label("gpt-4o"), "gpt-4o")
        self.assertEqual(_prom_escape_label("gemini/gemini-2.0-flash"),
                         "gemini/gemini-2.0-flash")


if __name__ == "__main__":
    unittest.main()
