"""Tests for general_beckman.lookahead."""
import pytest
from types import SimpleNamespace


def _snap_with_cloud(remaining: int | None):
    """Build a minimal SystemSnapshot-like object with one cloud provider."""
    rpm = SimpleNamespace(remaining=remaining, limit=None, reset_at=None)
    limits = SimpleNamespace(rpm=rpm)
    provider = SimpleNamespace(provider="anthropic", limits=limits)
    return SimpleNamespace(cloud={"anthropic": provider}, local=None, vram_available_mb=0)


def test_hold_back_when_cloud_queue_exceeds_headroom():
    from general_beckman.lookahead import should_hold_back
    snap = _snap_with_cloud(remaining=2)
    # 5 cloud tasks queued * 1.5 factor = 8 required > 2 remaining
    assert should_hold_back({"agent_type": "researcher"}, snap, cloud_queue_depth=5) is True


def test_dont_hold_back_when_quota_ample():
    from general_beckman.lookahead import should_hold_back
    snap = _snap_with_cloud(remaining=1000)
    assert should_hold_back({"agent_type": "researcher"}, snap, cloud_queue_depth=5) is False


def test_mechanical_never_held_back():
    from general_beckman.lookahead import should_hold_back
    snap = _snap_with_cloud(remaining=0)
    assert should_hold_back({"agent_type": "mechanical"}, snap, cloud_queue_depth=100) is False


def test_local_agent_never_held_back():
    from general_beckman.lookahead import should_hold_back
    snap = _snap_with_cloud(remaining=0)
    assert should_hold_back({"agent_type": "coder"}, snap, cloud_queue_depth=100) is False


def test_no_cloud_signal_no_gate():
    from general_beckman.lookahead import should_hold_back
    snap = SimpleNamespace(cloud={}, local=None, vram_available_mb=0)
    assert should_hold_back({"agent_type": "researcher"}, snap, cloud_queue_depth=100) is False


def test_none_snapshot_no_gate():
    from general_beckman.lookahead import should_hold_back
    assert should_hold_back({"agent_type": "researcher"}, None, cloud_queue_depth=100) is False


def test_missing_remaining_treated_as_no_signal():
    from general_beckman.lookahead import should_hold_back
    snap = _snap_with_cloud(remaining=None)
    # Treated as no signal → no gating
    assert should_hold_back({"agent_type": "researcher"}, snap, cloud_queue_depth=10) is False
