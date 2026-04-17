"""
Integration tests for orchestrator routing helpers and LLMDispatcher.

Covers:
  - _parse_task_difficulty: extraction from classification context, defaults, clamping
  - _reorder_by_model_affinity: empty list, no loaded model, boost matching tasks,
    never override 2-priority gap, single task
"""
from __future__ import annotations

import asyncio
import json
import sys
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_task(priority=5, agent_type="executor", difficulty=None,
               created_at="2026-01-01"):
    """Create a minimal task dict for testing."""
    ctx: dict = {}
    if difficulty is not None:
        ctx["classification"] = {"difficulty": difficulty, "agent_type": agent_type}
    return {
        "id": 1,
        "priority": priority,
        "agent_type": agent_type,
        "context": ctx,
        "created_at": created_at,
    }


def _make_task_json_ctx(priority=5, agent_type="executor", difficulty=5):
    """Create a task whose context is a JSON string (as stored in DB)."""
    return {
        "id": 2,
        "priority": priority,
        "agent_type": agent_type,
        "context": json.dumps({"classification": {"difficulty": difficulty}}),
        "created_at": "2026-01-01",
    }


def _make_model_info(name="test-model", is_local=True, demoted=False,
                     capabilities=None, location="local"):
    """Return a MagicMock that looks like ModelInfo."""
    info = MagicMock()
    info.name = name
    info.is_local = is_local
    info.demoted = demoted
    info.location = location
    info.provider = "llama_cpp"
    info.litellm_name = f"openai/{name}"
    info.capabilities = capabilities if capabilities is not None else {}
    info.context_length = 8192
    info.max_tokens = 4096
    info.supports_function_calling = True
    info.specialty = None
    info.has_vision = False
    info.total_params_b = 7.0
    info.active_params_b = 7.0
    info.tokens_per_second = 30.0
    info.tier = "free"
    info.rate_limit_rpm = 30
    info.model_type = "dense"
    info.operational_dict.return_value = {
        "location": location,
        "provider": "llama_cpp",
        "context_length": 8192,
        "max_tokens": 4096,
        "supports_function_calling": True,
        "supports_json_mode": False,
        "thinking_model": False,
        "has_vision": False,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "tokens_per_second": 30.0,
        "tier": "free",
        "rate_limit_rpm": 30,
        "model_type": "dense",
        "total_params_b": 7.0,
        "active_params_b": 7.0,
    }
    return info


ALL_CAPS = [
    "reasoning", "planning", "analysis",
    "code_generation", "code_reasoning",
    "system_design", "prose_quality",
    "instruction_adherence",
    "domain_knowledge", "context_utilization",
    "structured_output", "tool_use",
    "vision", "conversation",
]


# ═══════════════════════════════════════════════════════════════════════════════
# _parse_task_difficulty Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseTaskDifficulty(unittest.TestCase):

    def _fn(self, task):
        from src.core.orchestrator import _parse_task_difficulty
        return _parse_task_difficulty(task)

    # 1. difficulty extracted from classification context dict
    def test_parse_difficulty_from_classification(self):
        task = _make_task(difficulty=7)
        self.assertEqual(self._fn(task), 7)

    # 2. no classification context → defaults to 5
    def test_parse_difficulty_default(self):
        task = {"id": 1, "priority": 5, "agent_type": "executor", "context": {}}
        self.assertEqual(self._fn(task), 5)

    # 3. context stored as JSON string (as it comes from DB)
    def test_parse_difficulty_string_context(self):
        task = _make_task_json_ctx(difficulty=8)
        self.assertEqual(self._fn(task), 8)

    # 4. values outside 1-10 are clamped
    def test_parse_difficulty_clamped(self):
        task_low = {"id": 1, "priority": 5, "agent_type": "x",
                    "context": {"classification": {"difficulty": -5}}}
        task_high = {"id": 2, "priority": 5, "agent_type": "x",
                     "context": {"classification": {"difficulty": 99}}}
        self.assertEqual(self._fn(task_low), 1)
        self.assertEqual(self._fn(task_high), 10)


# ═══════════════════════════════════════════════════════════════════════════════
# _reorder_by_model_affinity Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestReorderByModelAffinity(unittest.TestCase):

    def _fn(self, tasks):
        from src.core.orchestrator import _reorder_by_model_affinity
        return _reorder_by_model_affinity(tasks)

    # 5. empty list returns empty
    def test_reorder_no_tasks(self):
        result = self._fn([])
        self.assertEqual(result, [])

    # 6. no model loaded → returns tasks unchanged (same order)
    def test_reorder_no_loaded_model(self):
        tasks = [_make_task(priority=5), _make_task(priority=3)]
        mock_manager = MagicMock()
        mock_manager.current_model = None

        with patch("src.models.local_model_manager.get_local_manager",
                   return_value=mock_manager):
            result = self._fn(tasks)

        # With no loaded model, function returns early without sorting
        self.assertEqual(result, tasks)

    # 7. tasks matching loaded model sort higher within same base priority
    def test_reorder_boosts_matching_tasks(self):
        from src.models.capabilities import TASK_PROFILES
        # Pick a real task_key known to exist in TASK_PROFILES
        known_task = list(TASK_PROFILES.keys())[0]  # e.g. "planner"

        # Two tasks at same priority; one has agent_type matching a known task profile
        task_matching = _make_task(priority=5, agent_type=known_task,
                                   difficulty=3, created_at="2026-01-01 00:00:01")
        task_other = _make_task(priority=5, agent_type="unknown_agent_xyz",
                                difficulty=3, created_at="2026-01-01 00:00:02")

        model_info = _make_model_info(
            capabilities={k: 8.0 for k in ALL_CAPS}
        )

        mock_manager = MagicMock()
        mock_manager.current_model = "test-model"
        mock_registry = MagicMock()
        mock_registry.get.return_value = model_info

        # score_model_for_task returns 8.0 for the known_task, 0 for others
        def score_side_effect(caps, ops, reqs):
            if reqs.task_name == known_task:
                return 8.0
            return 0.0

        with patch("src.models.local_model_manager.get_local_manager",
                   return_value=mock_manager), \
             patch("src.models.model_registry.get_registry",
                   return_value=mock_registry), \
             patch("src.models.capabilities.score_model_for_task",
                   side_effect=score_side_effect):
            result = self._fn([task_other, task_matching])

        # The matching task should sort first (higher effective priority)
        self.assertEqual(result[0]["agent_type"], known_task)

    # 8. 2-priority gap is NEVER overridden (max boost < 1.0)
    def test_reorder_never_overrides_2_priority_gap(self):
        # High priority task (8) vs low priority task (5) with perfect model fit
        task_high = _make_task(priority=8, agent_type="planner",
                               difficulty=1, created_at="2026-01-01")
        task_low = _make_task(priority=5, agent_type="coder",
                              difficulty=1, created_at="2026-01-01")

        model_info = _make_model_info(capabilities={k: 10.0 for k in ALL_CAPS})

        mock_manager = MagicMock()
        mock_manager.current_model = "test-model"
        mock_registry = MagicMock()
        mock_registry.get.return_value = model_info

        # task_low (coder) gets perfect score; task_high (planner) gets zero
        def score_side_effect(caps, ops, reqs):
            if reqs.task_name == "coder":
                return 10.0
            return 0.0

        with patch("src.models.local_model_manager.get_local_manager",
                   return_value=mock_manager), \
             patch("src.models.model_registry.get_registry",
                   return_value=mock_registry), \
             patch("src.models.capabilities.score_model_for_task",
                   side_effect=score_side_effect):
            result = self._fn([task_low, task_high])

        # The high priority task (8) must still be first despite zero fit
        self.assertEqual(result[0]["priority"], 8)

    # 9. single task returns unchanged
    def test_reorder_single_task(self):
        task = _make_task(priority=5)
        result = self._fn([task])
        self.assertEqual(result, [task])


# ═══════════════════════════════════════════════════════════════════════════════
# LLMDispatcher model selection Tests
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# _handle_exhausted / _handle_failed method-existence regression tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHandleExhaustedExists(unittest.TestCase):
    """Plan A Task 1: regression — the exhausted handler must exist as a method."""

    def test_handle_exhausted_is_method(self):
        from src.core.orchestrator import Orchestrator
        self.assertTrue(hasattr(Orchestrator, "_handle_exhausted"),
                        "Orchestrator._handle_exhausted should be defined as a method")
        self.assertTrue(asyncio.iscoroutinefunction(Orchestrator._handle_exhausted))


class TestHandleFailedExists(unittest.TestCase):
    """Plan A Task 2: regression — the failed handler must exist as a method."""

    def test_handle_failed_is_method(self):
        from src.core.orchestrator import Orchestrator
        self.assertTrue(hasattr(Orchestrator, "_handle_failed"),
                        "Orchestrator._handle_failed should be defined as a method")
        self.assertTrue(asyncio.iscoroutinefunction(Orchestrator._handle_failed))


if __name__ == "__main__":
    unittest.main()
