"""Tests for stateful simulator state (Phase 2d)."""
from sim.state import (  # type: ignore
    SimState, SimPoolCounter, SimLocalModel,
)


def test_simstate_init_defaults():
    s = SimState()
    assert s.virtual_clock == 0.0
    assert s.time_bucketed == {}
    assert s.per_call == {}
    assert s.locals == {}


def test_advance_clock():
    s = SimState()
    s.advance_clock(30.5)
    assert s.virtual_clock == 30.5
    s.advance_clock(10.0)
    assert s.virtual_clock == 40.5


def test_time_bucketed_decrement_and_reset():
    s = SimState()
    s.time_bucketed["groq"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=3600.0)
    s.time_bucketed["groq"].remaining -= 1
    assert s.time_bucketed["groq"].remaining == 999

    # Reset fires once clock crosses reset_at
    s.virtual_clock = 3601.0
    s.maybe_reset_buckets()
    assert s.time_bucketed["groq"].remaining == 1000
    # reset_at rolls forward by 86400 (daily)
    assert s.time_bucketed["groq"].reset_at == 3600.0 + 86400.0


def test_per_call_spend_accumulates():
    s = SimState()
    s.per_call["anthropic"] = SimPoolCounter(remaining=30, limit=30, reset_at=86400.0)
    s.per_call["anthropic"].remaining -= 1
    assert s.per_call["anthropic"].remaining == 29


def test_local_idle_increments_when_unused():
    s = SimState()
    s.locals["llama-3"] = SimLocalModel(is_loaded=True, idle_seconds=0.0)
    s.tick_locals(delta_seconds=30.0, used_local_name=None)
    assert s.locals["llama-3"].idle_seconds == 30.0


def test_local_idle_resets_when_used():
    s = SimState()
    s.locals["llama-3"] = SimLocalModel(is_loaded=True, idle_seconds=120.0)
    s.tick_locals(delta_seconds=5.0, used_local_name="llama-3")
    assert s.locals["llama-3"].idle_seconds == 0.0
