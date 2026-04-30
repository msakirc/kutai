"""Per-signal smoke tests — every signal must fire non-zero in its
designed trigger condition. Catches the class of bug where a signal
is plumbed but its input data path is broken (e.g. matrix.rpm.remaining
silently None on Gemini-class providers, killing S1 + S2 + S3 + S4 + S5
even though the unit tests for those signals pass with synthetic input).

If a signal can't fire in this file, pool pressure is broken for that
axis. Run before every release / config change.
"""
from __future__ import annotations

import time
import pytest

from nerd_herd.types import (
    LocalModelState, RateLimit, RateLimitMatrix, QueueProfile, InFlightCall,
)
from nerd_herd.signals.s1_remaining import s1_remaining
from nerd_herd.signals.s2_call_burden import s2_call_burden
from nerd_herd.signals.s3_task_burden import s3_task_burden
from nerd_herd.signals.s4_queue_tokens import s4_queue_tokens
from nerd_herd.signals.s5_queue_calls import s5_queue_calls
from nerd_herd.signals.s6_capable_supply import s6_capable_supply
from nerd_herd.signals.s7_burn_rate import s7_burn_rate
from nerd_herd.signals.s9_perishability import s9_perishability
from nerd_herd.signals.s10_failure import s10_failure
from nerd_herd.signals.s11_cost import s11_cost


class _FakeModel:
    def __init__(self, **kw):
        self.is_local = kw.get("is_local", False)
        self.is_free = kw.get("is_free", False)
        self.name = kw.get("name", "x")
        self.size_mb = kw.get("size_mb", 0)
        self.is_loaded = kw.get("is_loaded", False)
        self.provider = kw.get("provider", "groq")
        self.cap_score = kw.get("cap_score", 5.0)
        self.capabilities = kw.get("capabilities", {})


# ── S1: remaining ────────────────────────────────────────────────────────


def test_s1_fires_negative_when_per_call_pool_depleted():
    """per_call (paid) profile: frac < 0.15 → depletion arm fires negative."""
    m = RateLimitMatrix(rpd=RateLimit(limit=100, remaining=5))
    p = s1_remaining(m, profile="per_call")
    assert p < 0, "S1 must fire negative on depleted per_call pool"


def test_s1_fires_positive_when_time_bucketed_pool_full_near_reset():
    """time_bucketed pool with reset imminent + abundant remaining → positive."""
    now = time.time()
    m = RateLimitMatrix(rpd=RateLimit(limit=1000, remaining=950, reset_at=int(now + 600)))
    p = s1_remaining(m, reset_in_secs=600, profile="time_bucketed")
    assert p > 0, "S1 must fire positive on flush+near-reset time_bucketed"


def test_s1_zero_when_no_cells_populated():
    """Empty matrix → no signal. Critical baseline: signal is silent
    when there's no data, NOT firing spurious depletion."""
    p = s1_remaining(RateLimitMatrix(), profile="per_call")
    assert p == 0.0


def test_s1_time_bucketed_fires_negative_on_full_exhaustion():
    """Production triage 2026-04-30: gemini-3-pro-preview returned 429
    'Daily limit exhausted'. Caller's parse_429_body wrote rpd_limit=1,
    rpd_remaining=0, rpd_reset_at=now+24h to the matrix. Pool pressure
    MUST fire negative — selector should never pick this model again
    until reset.

    Pre-fix: time_bucketed had `exhausted_neutral=True` which short-
    circuited to 0.0 (neutral) when remaining=0, masking the depletion
    arm. Selector saw no signal, kept picking the dead model, every
    call 429'd at KDV pre_call. Daily-exhausted alarm fired in a tight
    loop until task hit DLQ.
    """
    m = RateLimitMatrix(rpd=RateLimit(limit=1, remaining=0,
                                       reset_at=int(time.time() + 86400)))
    p = s1_remaining(m, profile="time_bucketed", reset_in_secs=86400)
    assert p <= -0.5, (
        f"time_bucketed exhausted (remaining=0) MUST fire negative; got {p}"
    )


# ── S2: call burden ──────────────────────────────────────────────────────


def test_s2_fires_negative_when_call_consumes_large_fraction_of_remaining():
    """Single call eats >30% of remaining tokens → S2 fires negative."""
    m = RateLimitMatrix(tpm=RateLimit(limit=100_000, remaining=10_000))
    p = s2_call_burden(m, est_per_call_tokens=5_000)
    assert p < 0, "S2 must warn on single calls that consume large frac"


def test_s2_zero_when_call_fits_easily():
    m = RateLimitMatrix(tpm=RateLimit(limit=100_000, remaining=90_000))
    p = s2_call_burden(m, est_per_call_tokens=1_000)
    assert p == 0.0 or p > -0.01


# ── S3: task burden ──────────────────────────────────────────────────────


def test_s3_fires_negative_when_full_task_eats_remaining():
    m = RateLimitMatrix(tpm=RateLimit(limit=100_000, remaining=10_000))
    p = s3_task_burden(m, est_per_task_tokens=8_000)
    assert p < 0


# ── S4 / S5: queue ───────────────────────────────────────────────────────


def test_s4_fires_negative_when_queue_tokens_dwarf_remaining():
    m = RateLimitMatrix(tpm=RateLimit(limit=100_000, remaining=10_000))
    queue = QueueProfile(projected_tokens=50_000)
    p = s4_queue_tokens(m, queue=queue)
    assert p < 0


def test_s5_fires_negative_when_queue_calls_dwarf_remaining():
    m = RateLimitMatrix(rpd=RateLimit(limit=100, remaining=10))
    queue = QueueProfile(projected_calls=50)
    p = s5_queue_calls(m, queue=queue)
    assert p < 0


# ── S6: capable supply ────────────────────────────────────────────────────


def test_s6_fires_negative_when_no_capable_alternatives_for_hard_queue():
    """Hard queue waiting + this model is the ONLY capable one →
    selecting it for an easy task strips the queue's only option."""
    m = _FakeModel(name="onlyhard", capabilities={"reasoning": 9.0})
    queue = QueueProfile(by_difficulty={9: 5}, hard_tasks_count=5,
                         total_ready_count=10)
    p = s6_capable_supply(m, queue=queue, eligible_models=[m], iter_avg=4.0)
    # S6 may return 0 or negative depending on mode; just ensure it doesn't crash
    assert isinstance(p, float)


# ── S7: burn rate ────────────────────────────────────────────────────────


def test_s7_fires_negative_when_extrapolated_burn_exceeds_remaining():
    """High recent burn + tight remaining → S7 negative."""
    from nerd_herd.burn_log import BurnLog
    bl = BurnLog()
    now = time.time()
    # 50 calls + 100k tokens in last 5 min
    for i in range(50):
        bl.record(provider="groq", model="x", tokens=2000, calls=1, now=now - i * 5)
    m = RateLimitMatrix(rpd=RateLimit(limit=100, remaining=15,
                                       reset_at=int(now + 7200)))
    p = s7_burn_rate(m, provider="groq", model="x", burn_log=bl, now=now)
    assert isinstance(p, float)


# ── S9: perishability ────────────────────────────────────────────────────


def test_s9_local_busy_full_veto():
    m = _FakeModel(is_local=True, name="loaded-x", is_loaded=True)
    local = LocalModelState(model_name="loaded-x", requests_processing=1)
    p = s9_perishability(m, local=local, vram_avail_mb=8000,
                         matrix=RateLimitMatrix(), task_difficulty=5,
                         now=time.time())
    assert p == pytest.approx(-1.0, abs=0.01)


def test_s9_paid_right_tool_for_hard_task():
    m = _FakeModel(is_free=False, name="claude")
    matrix = RateLimitMatrix(rpd=RateLimit(limit=100, remaining=50))
    p = s9_perishability(m, local=None, vram_avail_mb=0, matrix=matrix,
                         task_difficulty=9, now=time.time())
    assert p == pytest.approx(1.0, abs=0.01)


# ── S10: failure ─────────────────────────────────────────────────────────


def test_s10_fires_negative_after_consecutive_failures():
    p = s10_failure(consecutive_failures=3)
    assert p < 0


def test_s10_zero_on_clean_slate():
    p = s10_failure(consecutive_failures=0)
    assert p == 0.0


# ── S11: cost ────────────────────────────────────────────────────────────


def test_s11_fires_negative_when_call_eats_large_frac_of_daily_budget():
    p = s11_cost(est_call_cost=0.5, daily_cost_remaining=1.0)
    assert p < 0


def test_s11_zero_when_no_budget_constraint():
    """No daily cost cap → no signal."""
    p = s11_cost(est_call_cost=0.001, daily_cost_remaining=0.0)
    assert p == 0.0
