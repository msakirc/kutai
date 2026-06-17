import pytest
from nerd_herd.types import QueueProfile, RateLimit, RateLimitMatrix
from nerd_herd.signals.s4_queue_tokens import s4_queue_tokens


def _matrix(**cells):
    m = RateLimitMatrix()
    for k, v in cells.items():
        setattr(m, k, v)
    return m


def test_s4_zero_when_no_queue():
    m = _matrix(tpd=RateLimit(limit=1_000_000, remaining=500_000))
    qp = QueueProfile(projected_tokens=0)
    assert s4_queue_tokens(m, queue=qp) == 0.0


def test_s4_zero_below_70pct_demand_ratio():
    m = _matrix(tpd=RateLimit(limit=1_000_000, remaining=500_000))
    qp = QueueProfile(projected_tokens=300_000)  # 60% of 500k
    assert s4_queue_tokens(m, queue=qp) == 0.0


def test_s4_negative_at_95pct_demand():
    m = _matrix(tpd=RateLimit(limit=1_000_000, remaining=500_000))
    qp = QueueProfile(projected_tokens=475_000)  # 95% of remaining
    p = s4_queue_tokens(m, queue=qp)
    assert -0.6 < p < -0.4


def test_s4_clipped_at_oversubscription():
    m = _matrix(tpd=RateLimit(limit=1_000_000, remaining=500_000))
    qp = QueueProfile(projected_tokens=750_000)  # 150% — over budget
    p = s4_queue_tokens(m, queue=qp)
    assert p == pytest.approx(-1.0, abs=0.05)


# ── Per-minute axes are PACING, not conservation (2026-06-18) ─────────────────
# A per-minute window refills every ~60s — a deep queue drains over many
# minutes and never "exhausts" it, so queue-CONSERVATION (S4) must not fire on
# it. Per-task fit (does THIS call fit a minute's tokens) stays with S2/S3;
# per-minute pacing stays with lane caps + in-flight reservation.

def test_s4_ignores_per_minute_window():
    # tpm dwarfed 10x by the whole-queue projection — but tpm is per-minute,
    # so S4 must stay 0 (no conservation pressure from a refilling window).
    m = _matrix(tpm=RateLimit(limit=30_000, remaining=30_000))
    qp = QueueProfile(projected_tokens=300_000)
    assert s4_queue_tokens(m, queue=qp) == 0.0


def test_s4_reads_daily_ignores_minute_when_both_present():
    # Healthy daily budget + tight per-minute window: S4 reads the daily
    # (cycle) axis only -> 30% -> 0. The per-minute tightness is S2/S3's job,
    # not a reason to conserve/erase this model.
    m = _matrix(
        tpm=RateLimit(limit=30_000, remaining=30_000),
        tpd=RateLimit(limit=1_000_000, remaining=1_000_000),
    )
    qp = QueueProfile(projected_tokens=300_000)
    assert s4_queue_tokens(m, queue=qp) == 0.0
