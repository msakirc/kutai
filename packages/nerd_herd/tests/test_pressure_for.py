import time
from unittest.mock import MagicMock
from nerd_herd.types import (
    SystemSnapshot, CloudProviderState, CloudModelState, RateLimit, RateLimitMatrix,
    LocalModelState, QueueProfile,
)


def _snap_with_cloud(provider, model_name, remaining, limit, reset_at, in_flight=0):
    m = CloudModelState(model_id=model_name)
    m.limits.rpd = RateLimit(limit=limit, remaining=remaining, reset_at=reset_at, in_flight=in_flight)
    prov = CloudProviderState(provider=provider, models={model_name: m})
    return SystemSnapshot(cloud={provider: prov})


def _kwargs(**overrides):
    """Default new-signature kwargs with optional overrides."""
    base = dict(
        task_difficulty=5,
        est_per_call_tokens=0,
        est_per_task_tokens=0,
        est_iterations=1,
        est_call_cost=0.0,
        cap_needed=5.0,
        consecutive_failures=0,
    )
    base.update(overrides)
    return base


def test_pressure_for_cloud_model_depletion_negative():
    snap = _snap_with_cloud("anthropic", "claude-sonnet-4-6",
                            remaining=5, limit=100, reset_at=int(time.time()) + 3600)
    # is_free=False → per_call profile (depletion_max=-1.0).
    fake = MagicMock(is_local=False, is_free=False, provider="anthropic", cap_score=5.0)
    fake.name = "claude-sonnet-4-6"
    result = snap.pressure_for(fake, **_kwargs())
    assert result.scalar < -0.5


def test_pressure_for_missing_model_returns_zero():
    snap = SystemSnapshot()
    fake = MagicMock(is_local=False, provider="unknown", is_free=False, cap_score=5.0)
    fake.name = "x"
    result = snap.pressure_for(fake, **_kwargs())
    assert -1.0 <= result.scalar <= 1.0


def test_pressure_for_local_busy_negative_or_zero():
    snap = SystemSnapshot(local=LocalModelState(model_name="qwen3-8b"))
    fake = MagicMock(is_local=True, cap_score=5.0)
    fake.name = "qwen3-8b"
    val = snap.pressure_for(fake, **_kwargs())
    assert -1.0 <= val.scalar <= 1.0


# ── New full-path smoke test ────────────────────────────────────────────────

class FakeModel:
    def __init__(self, name, provider, *, is_local=False, is_free=False, size_mb=0,
                 cap_score=5.0, capabilities=None):
        self.name = name
        self.provider = provider
        self.is_local = is_local
        self.is_free = is_free
        self.is_loaded = False
        self.size_mb = size_mb
        self.cap_score = cap_score
        self.capabilities = capabilities or set()


def test_pressure_for_full_path_returns_breakdown():
    """End-to-end smoke test: pressure_for with all 10 signals."""
    snap = SystemSnapshot(
        vram_available_mb=8000,
        local=LocalModelState(),
        cloud={
            "groq": CloudProviderState(
                provider="groq",
                limits=RateLimitMatrix(rpd=RateLimit(limit=14_400, remaining=14_000)),
                models={
                    "groq/llama": CloudModelState(
                        model_id="groq/llama",
                        limits=RateLimitMatrix(
                            rpm=RateLimit(limit=30, remaining=29),
                            tpm=RateLimit(limit=6000, remaining=5800),
                            rpd=RateLimit(limit=14_400, remaining=14_000),
                        ),
                    ),
                },
            ),
        },
        queue_profile=QueueProfile(
            total_ready_count=5, hard_tasks_count=1,
            by_difficulty={3: 4, 7: 1},
            by_capability={"function_calling": 5},
            projected_tokens=20_000, projected_calls=15,
        ),
    )
    model = FakeModel("groq/llama", "groq", is_free=True, cap_score=5.0)
    breakdown = snap.pressure_for(
        model,
        task_difficulty=5,
        est_per_call_tokens=2000,
        est_per_task_tokens=20_000,
        est_iterations=10,
        est_call_cost=0.0,
        cap_needed=5.0,
        consecutive_failures=0,
    )
    # Returns a PressureBreakdown; .scalar is in [-1, +1]
    assert -1.0 <= breakdown.scalar <= 1.0
    # All 10 signal keys populated
    assert all(k in breakdown.signals for k in
               ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9", "S10", "S11"))


# ── Admission reservations subtract from effective remaining ───────────────


def test_pressure_for_subtracts_in_flight_est_tokens_from_remaining():
    """Beckman-admitted-but-not-yet-called tasks carry est_tokens in
    InFlightCall. Pool pressure must subtract them from tpm/cpm
    remaining so a second admission tick on the same model doesn't
    see fresh budget. Production 2026-05-02: 5 parallel admissions
    on gemini all saw fresh tpm_remaining and overshot the quota."""
    from nerd_herd.types import InFlightCall

    # Provider with 12K TPM remaining.
    m = CloudModelState(model_id="gemini/flash")
    m.limits.tpm = RateLimit(limit=12000, remaining=12000)
    m.limits.rpm = RateLimit(limit=10, remaining=10)
    prov = CloudProviderState(provider="gemini", models={"gemini/flash": m})

    # Two tasks already admitted, each projecting 4500 tokens.
    in_flight = [
        InFlightCall(
            call_id=f"task-{i}", task_id=i, category="main_work",
            model="gemini/flash", provider="gemini", is_local=False,
            started_at=time.time(), est_tokens=4500,
        )
        for i in (1, 2)
    ]
    snap = SystemSnapshot(cloud={"gemini": prov}, in_flight_calls=in_flight)

    fake = MagicMock(is_local=False, is_free=False, provider="gemini", cap_score=5.0)
    fake.name = "gemini/flash"

    # Without subtraction, the matrix shows 12K remaining and a 4K-call
    # has plenty of headroom (S2 ≈ 0). With subtraction, effective
    # tpm_remaining = 12K - 9K = 3K, and a 4K-token call now exceeds
    # capacity → S2 fires negative.
    breakdown = snap.pressure_for(fake, **_kwargs(est_per_call_tokens=4000))
    assert breakdown.signals["S2"] < 0, (
        f"S2 should fire negative when in-flight reservations leave "
        f"insufficient headroom: signals={breakdown.signals}"
    )
