"""LocalModelState.idle_seconds populates from inference timestamps.

InferenceCollector tracks when requests_processing last dropped to 0 and
exposes idle_seconds via collect(). NerdHerd.snapshot() merges this into
the returned LocalModelState.
"""
from __future__ import annotations
import time

import pytest

from nerd_herd.inference import InferenceCollector
from nerd_herd.types import LocalModelState


# ---------------------------------------------------------------------------
# Helpers: minimal Prometheus text snippets
# ---------------------------------------------------------------------------

def _metrics(requests_processing: int = 0, tokens_total: int = 1000) -> str:
    return (
        f"llamacpp:tokens_predicted_total {tokens_total}\n"
        f"llamacpp:requests_processing {requests_processing}\n"
        f"llamacpp:requests_pending 0\n"
        f"llamacpp:kv_cache_usage_ratio 0.1\n"
    )


# ---------------------------------------------------------------------------
# idle_seconds behaviour tests
# ---------------------------------------------------------------------------

def test_idle_seconds_zero_when_no_inference_yet():
    """Fresh collector with no parse calls → idle_seconds is 0.0."""
    collector = InferenceCollector()
    assert collector.collect()["idle_seconds"] == 0.0


def test_idle_seconds_zero_on_fresh_completion(monkeypatch):
    """Right after inference completes (requests_processing → 0), idle should be ~0."""
    collector = InferenceCollector()
    t0 = time.time()
    monkeypatch.setattr(time, "time", lambda: t0)
    collector._parse_and_record(_metrics(requests_processing=0), ts=t0)
    result = collector.collect()
    assert result["idle_seconds"] < 0.1


def test_idle_seconds_grows_with_time(monkeypatch):
    """idle_seconds grows as real time passes after requests_processing hits 0."""
    collector = InferenceCollector()
    t0 = time.time()
    # Record metrics showing inference just finished
    monkeypatch.setattr(time, "time", lambda: t0)
    collector._parse_and_record(_metrics(requests_processing=0), ts=t0)

    # Advance clock by 42.5s
    monkeypatch.setattr(time, "time", lambda: t0 + 42.5)
    result = collector.collect()
    assert 42.0 <= result["idle_seconds"] <= 43.0


def test_idle_seconds_zero_while_in_flight(monkeypatch):
    """While requests_processing > 0, idle_seconds should be 0."""
    collector = InferenceCollector()
    t0 = time.time()
    monkeypatch.setattr(time, "time", lambda: t0)
    # First, let it go idle
    collector._parse_and_record(_metrics(requests_processing=0), ts=t0)
    # Now advance time and simulate an in-flight call
    monkeypatch.setattr(time, "time", lambda: t0 + 30.0)
    collector._parse_and_record(_metrics(requests_processing=1), ts=t0 + 30.0)
    result = collector.collect()
    assert result["idle_seconds"] == 0.0


def test_idle_seconds_resets_after_new_inference(monkeypatch):
    """After a new inference completes, idle_seconds resets to ~0."""
    collector = InferenceCollector()
    t0 = time.time()
    monkeypatch.setattr(time, "time", lambda: t0)
    # First idle period
    collector._parse_and_record(_metrics(requests_processing=0), ts=t0)

    # Active inference 30s later
    monkeypatch.setattr(time, "time", lambda: t0 + 30.0)
    collector._parse_and_record(_metrics(requests_processing=1), ts=t0 + 30.0)

    # Inference completes at t0 + 50
    monkeypatch.setattr(time, "time", lambda: t0 + 50.0)
    collector._parse_and_record(_metrics(requests_processing=0), ts=t0 + 50.0)

    # Right after completion — idle should be ~0
    result = collector.collect()
    assert result["idle_seconds"] < 0.1


# ---------------------------------------------------------------------------
# LocalModelState field existence
# ---------------------------------------------------------------------------

def test_local_model_state_has_idle_seconds_field():
    """LocalModelState.idle_seconds exists and defaults to 0.0."""
    state = LocalModelState()
    assert hasattr(state, "idle_seconds")
    assert state.idle_seconds == 0.0


def test_local_model_state_idle_seconds_assignable():
    """idle_seconds can be set on construction."""
    state = LocalModelState(idle_seconds=123.4)
    assert state.idle_seconds == 123.4
