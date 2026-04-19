"""Pool classification + urgency formulas (pure functions, no I/O)."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from fatih_hoca.pools import (
    Pool,
    classify_pool,
    compute_urgency,
    URGENCY_MAX_BONUS,
    LOCAL_IDLE_SATURATION_SECS,
    RESET_HORIZON_SECS,
)


@dataclass
class _FakeModel:
    name: str
    is_local: bool = False
    is_free: bool = False
    provider: str = ""


@dataclass
class _FakeRateLimit:
    remaining: int | None = None
    limit: int | None = None
    reset_at: int | None = None


@dataclass
class _FakeRateLimits:
    rpd: _FakeRateLimit = field(default_factory=_FakeRateLimit)


@dataclass
class _FakeModelState:
    limits: _FakeRateLimits = field(default_factory=_FakeRateLimits)


@dataclass
class _FakeProviderState:
    models: dict[str, _FakeModelState] = field(default_factory=dict)
    limits: _FakeRateLimits = field(default_factory=_FakeRateLimits)


@dataclass
class _FakeLocal:
    idle_seconds: float = 0.0


@dataclass
class _FakeSnapshot:
    local: _FakeLocal = field(default_factory=_FakeLocal)
    cloud: dict[str, _FakeProviderState] = field(default_factory=dict)


# ── Classification ──

def test_local_model_classifies_as_local():
    m = _FakeModel(name="qwen", is_local=True)
    assert classify_pool(m) is Pool.LOCAL


def test_free_cloud_classifies_as_time_bucketed():
    m = _FakeModel(name="groq-llama", is_local=False, is_free=True, provider="groq")
    assert classify_pool(m) is Pool.TIME_BUCKETED


def test_paid_cloud_classifies_as_per_call():
    m = _FakeModel(name="claude-sonnet", is_local=False, is_free=False, provider="anthropic")
    assert classify_pool(m) is Pool.PER_CALL


# ── Urgency: local ──

def test_local_urgency_zero_when_active():
    m = _FakeModel(name="qwen", is_local=True)
    snap = _FakeSnapshot(local=_FakeLocal(idle_seconds=0.0))
    assert compute_urgency(m, snap) == 0.0


def test_local_urgency_scales_linearly_to_saturation():
    m = _FakeModel(name="qwen", is_local=True)
    snap = _FakeSnapshot(local=_FakeLocal(idle_seconds=LOCAL_IDLE_SATURATION_SECS / 2))
    u = compute_urgency(m, snap)
    assert 0.45 <= u <= 0.55


def test_local_urgency_saturates_at_one():
    m = _FakeModel(name="qwen", is_local=True)
    snap = _FakeSnapshot(local=_FakeLocal(idle_seconds=LOCAL_IDLE_SATURATION_SECS * 5))
    assert compute_urgency(m, snap) == 1.0


# ── Urgency: time-bucketed cloud ──

def _make_bucketed_snapshot(remaining, limit, reset_in_seconds, provider="groq", model_id="llama-70b"):
    now = time.time()
    rl = _FakeRateLimit(remaining=remaining, limit=limit, reset_at=int(now + reset_in_seconds))
    mstate = _FakeModelState(limits=_FakeRateLimits(rpd=rl))
    prov = _FakeProviderState(models={model_id: mstate}, limits=_FakeRateLimits(rpd=rl))
    return _FakeSnapshot(cloud={provider: prov})


def test_bucketed_urgency_high_when_unused_and_close_to_reset():
    m = _FakeModel(name="llama-70b", is_free=True, provider="groq")
    snap = _make_bucketed_snapshot(remaining=900, limit=1000, reset_in_seconds=300)
    u = compute_urgency(m, snap)
    # remaining_frac = 0.9, reset_proximity = 1 - 300/3600 ≈ 0.917 → urgency ≈ 0.825
    assert 0.78 <= u <= 0.88


def test_bucketed_urgency_low_when_reset_far_away():
    m = _FakeModel(name="llama-70b", is_free=True, provider="groq")
    snap = _make_bucketed_snapshot(remaining=900, limit=1000, reset_in_seconds=86400)
    u = compute_urgency(m, snap)
    # reset_proximity clamped to 0
    assert u == 0.0


def test_bucketed_urgency_zero_when_quota_exhausted():
    m = _FakeModel(name="llama-70b", is_free=True, provider="groq")
    snap = _make_bucketed_snapshot(remaining=0, limit=1000, reset_in_seconds=300)
    assert compute_urgency(m, snap) == 0.0


def test_bucketed_urgency_midnight_utc_fallback_when_reset_missing():
    """If reset_at is None, fall back to midnight UTC assumption."""
    m = _FakeModel(name="llama-70b", is_free=True, provider="groq")
    rl = _FakeRateLimit(remaining=900, limit=1000, reset_at=None)
    mstate = _FakeModelState(limits=_FakeRateLimits(rpd=rl))
    prov = _FakeProviderState(models={"llama-70b": mstate}, limits=_FakeRateLimits(rpd=rl))
    snap = _FakeSnapshot(cloud={"groq": prov})
    u = compute_urgency(m, snap)
    assert 0.0 <= u <= 1.0  # fallback yields a defined number


# ── Urgency: per-call ──

def test_per_call_urgency_always_zero():
    m = _FakeModel(name="claude-sonnet", is_free=False, provider="anthropic")
    snap = _FakeSnapshot()
    assert compute_urgency(m, snap) == 0.0


# ── Missing telemetry ──

def test_missing_local_snapshot_returns_zero():
    m = _FakeModel(name="qwen", is_local=True)
    snap = _FakeSnapshot()  # idle_seconds=0.0 default
    assert compute_urgency(m, snap) == 0.0


def test_unknown_provider_returns_zero():
    m = _FakeModel(name="mystery", is_free=True, provider="unknown-provider")
    snap = _FakeSnapshot()
    assert compute_urgency(m, snap) == 0.0
