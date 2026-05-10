"""Z10 T4B — calibration note injection into the system prompt.

Verifies the prompt-builder feedback loop: well-calibrated models earn a
'trusted' note, poorly-calibrated ones earn a 'be conservative' note,
mid-range ones get nothing.
"""
from __future__ import annotations

import pytest

from coulson.context import (
    build_system_prompt, seed_calibration_cache, reset_calibration_cache,
)


class _FakeProfile:
    """Minimal stand-in for an agent profile in build_system_prompt."""
    agent_type = "coder"
    max_iterations = 1
    tools: dict = {}
    allowed_tools: list = []  # explicit empty → no tools section
    _suppress_clarification = True
    _prompt_version_override = None
    _active_model_id = "gpt-oss-20b"

    def get_system_prompt(self, task):
        return "You are a Coder. Follow the rules."


def _seed(rows):
    reset_calibration_cache()
    seed_calibration_cache(rows)


def _task() -> dict:
    return {"id": 1, "title": "x", "description": "y", "agent_type": "coder"}


def test_low_reliability_emits_conservative_note(monkeypatch):
    monkeypatch.setenv("CALIBRATION_MIN_SAMPLE_N", "30")
    _seed([
        {"model_id": "gpt-oss-20b", "task_kind": "coder",
         "confidence_bucket": "high", "sample_n": 89, "correct_n": 50,
         "reliability": 0.56, "updated_at": "2026-05-11"},
    ])
    prompt = build_system_prompt(_FakeProfile(), _task())
    assert "[CALIBRATION NOTE]" in prompt
    assert "be more conservative" in prompt
    assert "0.56" in prompt


def test_high_reliability_emits_trusted_note(monkeypatch):
    monkeypatch.setenv("CALIBRATION_MIN_SAMPLE_N", "30")
    _seed([
        {"model_id": "gpt-oss-20b", "task_kind": "coder",
         "confidence_bucket": "high", "sample_n": 200, "correct_n": 180,
         "reliability": 0.90, "updated_at": "2026-05-11"},
    ])
    prompt = build_system_prompt(_FakeProfile(), _task())
    assert "[CALIBRATION NOTE]" in prompt
    assert "trusted" in prompt
    assert "0.90" in prompt


def test_mid_range_emits_no_note(monkeypatch):
    monkeypatch.setenv("CALIBRATION_MIN_SAMPLE_N", "30")
    _seed([
        {"model_id": "gpt-oss-20b", "task_kind": "coder",
         "confidence_bucket": "high", "sample_n": 100, "correct_n": 75,
         "reliability": 0.75, "updated_at": "2026-05-11"},
    ])
    prompt = build_system_prompt(_FakeProfile(), _task())
    assert "[CALIBRATION NOTE]" not in prompt


def test_low_samples_emit_no_note_even_when_poor(monkeypatch):
    monkeypatch.setenv("CALIBRATION_MIN_SAMPLE_N", "30")
    _seed([
        {"model_id": "gpt-oss-20b", "task_kind": "coder",
         "confidence_bucket": "high", "sample_n": 12, "correct_n": 4,
         "reliability": 0.33, "updated_at": "2026-05-11"},
    ])
    prompt = build_system_prompt(_FakeProfile(), _task())
    assert "[CALIBRATION NOTE]" not in prompt


def test_no_cache_no_note():
    """Empty cache → no injection, no crash."""
    reset_calibration_cache()
    prompt = build_system_prompt(_FakeProfile(), _task())
    assert "[CALIBRATION NOTE]" not in prompt
