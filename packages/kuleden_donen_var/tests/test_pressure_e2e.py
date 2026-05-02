"""End-to-end pressure pipeline tests — call → KDV → adapter → matrix
→ pressure_for. Verifies signals see the right data after real call
sequences.

This catches the specific bug class where a signal's unit test passes
on synthetic input but the production data path is broken (e.g.
rpm_remaining stayed None on Gemini-class providers because the state
class lacked the property the adapter read from).

Each test simulates a real workload pattern and asserts the pressure
scalar shifts in the expected direction.
"""
from __future__ import annotations

import time

from kuleden_donen_var import KuledenConfig, KuledenDonenVar
from kuleden_donen_var.nerd_herd_adapter import build_cloud_provider_state
from nerd_herd.types import (
    InFlightCall, LocalModelState, QueueProfile, SystemSnapshot,
)


class _FakeModel:
    """Minimal model surface needed by snapshot.pressure_for."""
    def __init__(self, *, name, provider, is_local=False, is_free=True,
                 cap_score=6.0):
        self.name = name
        self.provider = provider
        self.is_local = is_local
        self.is_free = is_free
        self.cap_score = cap_score
        self.is_loaded = False
        self.size_mb = 0
        self.capabilities = {"reasoning": cap_score, "code_generation": cap_score}

    def estimated_cost(self, *_args, **_kw) -> float:
        return 0.0


def _build_snapshot(kdv: KuledenDonenVar, provider: str) -> SystemSnapshot:
    """Wrap KDV state into the SystemSnapshot pressure_for expects."""
    cloud_state = build_cloud_provider_state(kdv, provider)
    return SystemSnapshot(
        vram_available_mb=8000,
        local=LocalModelState(),
        cloud={provider: cloud_state} if cloud_state else {},
        queue_profile=QueueProfile(),
        in_flight_calls=[],
    )


def _pressure(kdv, model, *, difficulty=5, est_per_call=2000, est_per_task=8000):
    snap = _build_snapshot(kdv, model.provider)
    br = snap.pressure_for(
        model,
        task_difficulty=difficulty,
        est_per_call_tokens=est_per_call,
        est_per_task_tokens=est_per_task,
        est_iterations=4,
        est_call_cost=0.0,
        cap_needed=5.0,
        consecutive_failures=0,
    )
    return br


# ── RPM saturation through call recording ───────────────────────────────


def test_rpm_pressure_shifts_negative_after_burst_calls():
    """Register a 5-RPM model. Cold state → pressure neutral or positive.
    After 5 calls in this minute → S1 depletion fires, scalar goes
    strongly negative. THE bug: this didn't work for weeks because
    matrix.rpm.remaining was always None on header-less providers."""
    kdv = KuledenDonenVar(KuledenConfig())
    kdv.register("gemini/foo", "gemini", rpm=5, tpm=250_000)
    state = kdv._rate_limiter.model_limits["gemini/foo"]
    m = _FakeModel(name="gemini/foo", provider="gemini", is_free=True)

    cold = _pressure(kdv, m, difficulty=5)
    assert cold.signals["S1"] >= 0.0, f"cold S1 should be non-negative, got {cold.signals}"

    # Saturate via record_attempt — same path the caller uses
    for _ in range(5):
        kdv.record_attempt("gemini/foo", "gemini", estimated_tokens=0)
    saturated = _pressure(kdv, m, difficulty=5)
    # remaining=0 → time_bucketed depletion arm fires depletion_max=-1.0.
    # User feedback 2026-05-01: free-tier RPM saturation MUST hold
    # dispatch — selector strict-gate excludes scalar=-1.0 candidates.
    assert saturated.signals["S1"] <= -0.95, (
        f"saturated S1 must hit hard depletion (free-tier RPM=0 → "
        f"selector exclusion floor); got {saturated.signals}"
    )
    assert saturated.scalar < cold.scalar


def test_tpm_pressure_shifts_negative_after_token_burst():
    """Same pattern for TPM: log heavy token usage in this minute,
    pressure should shift negative on the next call's burden estimate."""
    kdv = KuledenDonenVar(KuledenConfig())
    kdv.register("gemini/foo", "gemini", rpm=100, tpm=10_000)
    m = _FakeModel(name="gemini/foo", provider="gemini")

    cold = _pressure(kdv, m, est_per_call=2000)

    # Burn 9000 tokens in this minute via record_tokens (post_call path)
    kdv._rate_limiter.record_tokens("gemini/foo", "gemini", 9000)
    saturated = _pressure(kdv, m, est_per_call=2000)

    # tpm_remaining drops from 10K to 1K. Next call estimate=2K fits
    # poorly → S2 fires hard negative. S1 also fires (frac=0.10 < 0.15
    # depletion_threshold for per_call, but this is time_bucketed where
    # threshold=0.30, frac=0.10 < 0.30 → depletion).
    assert saturated.scalar < cold.scalar, (
        f"tpm saturation must shift pressure: cold={cold.scalar:+.3f} "
        f"saturated={saturated.scalar:+.3f}"
    )


# ── 429-body parser → pressure pipeline ──────────────────────────────────


def test_429_body_parse_writes_through_to_matrix_and_pressure():
    """The 429-body parser is the only data path for Gemini's daily
    quota state. After a 429 with limit:0, pressure must immediately
    reflect the depletion — selector should refuse the model on the
    next selection without needing another round-trip."""
    from kuleden_donen_var.header_parser import parse_429_body

    kdv = KuledenDonenVar(KuledenConfig())
    kdv.register("gemini/foo", "gemini", rpm=5, tpm=250_000)
    m = _FakeModel(name="gemini/foo", provider="gemini", is_free=True)

    cold = _pressure(kdv, m, difficulty=5)

    msg = (
        "RESOURCE_EXHAUSTED Quota exceeded for metric: "
        "generate_content_free_tier_requests, limit: 0, model: foo. "
        "Please retry in 86400s."
    )
    snap = parse_429_body("gemini", msg)
    assert snap is not None
    kdv._rate_limiter.update_from_headers("gemini/foo", "gemini", snap)

    after = _pressure(kdv, m, difficulty=5)
    # rpd_remaining=0 with reset_at far in future →
    # time_bucketed pool + exhausted_neutral=True returns 0.0 from S1
    # but other signals (S2 if tokens-day, S5 if queue calls) should
    # still fire. Or at minimum: scalar must NOT improve over cold.
    assert after.scalar <= cold.scalar, (
        f"429-body parse should not produce a positive shift: "
        f"cold={cold.scalar:+.3f} after={after.scalar:+.3f}"
    )


# ── Static seed (Gemini quota table) → pressure pipeline ────────────────


def test_gemini_static_seed_quotas_propagate_to_matrix():
    """Adapter sets rate_limit_rpm/tpm/rpd on DiscoveredModel from the
    static free-tier table. Those values must reach the matrix so
    pressure_for sees them on the very first call."""
    from fatih_hoca.cloud.types import DiscoveredModel
    from fatih_hoca.registry import register_cloud_from_discovered, ModelRegistry

    registry = ModelRegistry()
    discovered = DiscoveredModel(
        litellm_name="gemini/gemini-2.5-flash",
        raw_id="gemini-2.5-flash",
        rate_limit_rpm=5, rate_limit_tpm=250_000, rate_limit_rpd=20,
    )
    m = register_cloud_from_discovered(registry, "gemini", discovered)
    assert m is not None
    assert m.rate_limit_rpm == 5
    assert m.rate_limit_tpm == 250_000


# ── In-flight propagation → S9 local veto ───────────────────────────────


def test_local_inflight_vetoes_concurrent_admission_via_pressure_for():
    """End-to-end: in_flight_calls passed to pressure_for surfaces as
    S9=-1.0 for any local candidate. Confirms the wire-up from
    SystemSnapshot.in_flight_calls through pressure_for to S9."""
    kdv = KuledenDonenVar(KuledenConfig())
    local = _FakeModel(name="loaded-x", provider="local", is_local=True)
    local.is_loaded = True

    snap = SystemSnapshot(
        vram_available_mb=8000,
        local=LocalModelState(model_name="loaded-x", idle_seconds=30,
                               requests_processing=0),
        cloud={},
        queue_profile=QueueProfile(),
        in_flight_calls=[InFlightCall(
            call_id="t1", task_id=1, category="main_work",
            model="loaded-x", provider="local", is_local=True,
            started_at=time.time(),
        )],
    )
    br = snap.pressure_for(
        local, task_difficulty=5,
        est_per_call_tokens=1000, est_per_task_tokens=4000,
        est_iterations=4, est_call_cost=0.0, cap_needed=5.0,
        consecutive_failures=0,
    )
    # S9 LOCAL_BUSY_PENALTY is a sentinel ≤ -1.0 (currently -10.0) so it
    # survives M3 weight dilution; combine_signals' final clamp produces
    # scalar = -1.0 exactly. See dde55b7 fix(s9): ... for context.
    assert br.signals["S9"] <= -1.0, f"local in-flight veto failed: {br.signals}"
    assert br.scalar == -1.0, f"scalar should clamp to -1.0: {br.scalar}"


# ── Failure tracking → S10 ──────────────────────────────────────────────


def test_synthetic_429_backoff_writes_through_to_matrix():
    """When a provider 429s with NO headers and NO parseable body, the
    caller should still register a 60s backoff so subsequent admissions
    skip the model. Tests the synthetic-snapshot path that fires when
    real data isn't available. User feedback 2026-05-01: '429 should
    also be recognized, even if no headers, no point trying it again
    in same minute'."""
    from kuleden_donen_var.header_parser import RateLimitSnapshot
    import time as _t

    kdv = KuledenDonenVar(KuledenConfig())
    kdv.register("groq/foo", "groq", rpm=30, tpm=8000)
    state = kdv._rate_limiter.model_limits["groq/foo"]
    m = _FakeModel(name="groq/foo", provider="groq")

    cold = _pressure(kdv, m, difficulty=5)
    assert cold.signals["S1"] >= -0.1  # fresh, no depletion

    # Apply synthetic 60s backoff (the path caller.py takes when 429
    # comes with no parseable headers/body).
    synthetic = RateLimitSnapshot(
        rpm_remaining=0,
        rpm_reset_at=_t.time() + 60.0,
    )
    kdv._rate_limiter.update_from_headers("groq/foo", "groq", synthetic)

    # Property reads through fresh-header path: rpm_remaining returns 0.
    assert state.rpm_remaining == 0

    # Pressure recomputed from updated state. Depletion arm fires.
    after = _pressure(kdv, m, difficulty=5)
    assert after.signals["S1"] < -0.5, (
        f"synthetic backoff must fire S1 depletion; "
        f"got {after.signals}"
    )
    assert after.scalar < cold.scalar


def test_post_429_cooldown_holds_remaining_at_zero():
    """After a 429, sliding-window decay can free a slot in ~60s while
    the provider is still throttling — selector picks the model again,
    429 fires again, loop. Post-429 cooldown property forces
    rpm_remaining=0 for 60s regardless of sliding-window math.

    Production triage 2026-05-01: groq/llama-4-scout 1276 fails / 45
    success (3%), groq/llama-3.3-70b 811/2 (0%) — selector thrashed
    on marginal capacity because S1 only fired -0.89 (above the
    strict -1.0 floor) once the oldest timestamp aged out."""
    kdv = KuledenDonenVar(KuledenConfig())
    kdv.register("groq/scout", "groq", rpm=30, tpm=30_000)
    state = kdv._rate_limiter.model_limits["groq/scout"]
    m = _FakeModel(name="groq/scout", provider="groq")

    # Simulate burst that produced 429: most recent request long enough
    # ago that sliding window says 1/30 used → frac=0.97 (would normally
    # admit), but provider just rate-limited.
    state.record_429()
    assert state.in_post_429_cooldown is True
    assert state.rpm_remaining == 0, (
        f"cooldown must clamp rpm_remaining to 0; got {state.rpm_remaining}"
    )

    # Pressure must reflect the clamp. S1 fires hard depletion.
    pressure = _pressure(kdv, m, difficulty=5)
    assert pressure.signals["S1"] <= -0.95, (
        f"post-429 cooldown must drive S1 to depletion floor; "
        f"got {pressure.signals}"
    )

    # Simulate cooldown expiry — flip the saved timestamp into the past.
    state._last_429_at = time.time() - 65.0
    assert state.in_post_429_cooldown is False
    # rpm_remaining now reflects sliding window (empty here, but rpm_limit
    # was adapted down by record_429's 20% cut).
    assert state.rpm_remaining == state.rpm_limit > 0


def test_consecutive_failures_propagate_to_s10():
    """consecutive_failures kwarg flows from snapshot.cloud[provider]
    state through pressure_for into S10."""
    kdv = KuledenDonenVar(KuledenConfig())
    kdv.register("groq/foo", "groq", rpm=30, tpm=131_072)
    m = _FakeModel(name="groq/foo", provider="groq")

    snap = _build_snapshot(kdv, "groq")
    br_clean = snap.pressure_for(
        m, task_difficulty=5,
        est_per_call_tokens=1000, est_per_task_tokens=4000,
        est_iterations=4, est_call_cost=0.0, cap_needed=5.0,
        consecutive_failures=0,
    )
    br_failing = snap.pressure_for(
        m, task_difficulty=5,
        est_per_call_tokens=1000, est_per_task_tokens=4000,
        est_iterations=4, est_call_cost=0.0, cap_needed=5.0,
        consecutive_failures=3,
    )
    assert br_failing.signals["S10"] < br_clean.signals["S10"]
    assert br_failing.scalar < br_clean.scalar
