"""
Tests for S6: Runtime State Tracking.

Covers:
  - ModelRuntimeState dataclass: construction, field defaults
  - get_runtime_state() module helper

NOTE: the router.select_model() runtime-aware scoring tests that used to
live here were removed 2026-06-07 with the dead select_model() scorer.
Live runtime-aware ranking (runtime ctx filter, thinking-mismatch
stickiness, measured_tps) is covered by
packages/fatih_hoca/tests/test_ranking_s6_rollup.py and
packages/fatih_hoca/tests/test_swap_policy.py.
"""
from __future__ import annotations

import time
import sys
import os
import unittest
from dataclasses import fields
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── ModelRuntimeState unit tests ─────────────────────────────────────────────

class TestModelRuntimeState(unittest.TestCase):

    def test_fields_present(self):
        from src.models.local_model_manager import ModelRuntimeState
        field_names = {f.name for f in fields(ModelRuntimeState)}
        for expected in ("model_name", "thinking_enabled", "context_length",
                         "gpu_layers", "measured_tps", "loaded_at"):
            self.assertIn(expected, field_names)

    def test_defaults(self):
        from src.models.local_model_manager import ModelRuntimeState
        rs = ModelRuntimeState(
            model_name="llama3",
            thinking_enabled=False,
            context_length=8192,
            gpu_layers=33,
        )
        self.assertEqual(rs.measured_tps, 0.0)
        self.assertAlmostEqual(rs.loaded_at, time.time(), delta=2.0)

    def test_thinking_enabled_flag(self):
        from src.models.local_model_manager import ModelRuntimeState
        rs = ModelRuntimeState(
            model_name="qwq",
            thinking_enabled=True,
            context_length=32768,
            gpu_layers=33,
        )
        self.assertTrue(rs.thinking_enabled)

    def test_measured_tps_mutable(self):
        from src.models.local_model_manager import ModelRuntimeState
        rs = ModelRuntimeState("m", False, 4096, 20)
        rs.measured_tps = 42.5
        self.assertEqual(rs.measured_tps, 42.5)


# ── get_runtime_state helper ──────────────────────────────────────────────────

class TestGetRuntimeState(unittest.TestCase):

    def test_returns_none_when_no_manager(self):
        """get_runtime_state() returns None if the singleton has never been init'd."""
        import src.models.local_model_manager as lmm_mod
        original = lmm_mod._manager
        try:
            lmm_mod._manager = None
            from src.models.local_model_manager import get_runtime_state
            self.assertIsNone(get_runtime_state())
        finally:
            lmm_mod._manager = original

    def test_returns_runtime_state_from_manager(self):
        from src.models.local_model_manager import ModelRuntimeState, get_runtime_state
        import src.models.local_model_manager as lmm_mod

        fake_rs = ModelRuntimeState("llama3", False, 8192, 33)
        fake_mgr = MagicMock()
        fake_mgr.runtime_state = fake_rs

        original = lmm_mod._manager
        try:
            lmm_mod._manager = fake_mgr
            result = get_runtime_state()
            self.assertIs(result, fake_rs)
        finally:
            lmm_mod._manager = original


# LocalModelManager.get_metrics() now delegates to dallama._metrics.fetch;
# coverage moved to packages/dallama/tests/test_metrics.py.


if __name__ == "__main__":
    unittest.main()
