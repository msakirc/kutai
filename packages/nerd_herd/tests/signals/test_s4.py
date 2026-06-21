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


# ── Fleet-capacity denominator (2026-06-21) ───────────────────────────────────

def test_s4_fleet_denominator_dilutes_small_window():
    # This model's own tpd window is tiny (20k), but the FLEET has 1.02M on the
    # tpd axis. A 400k queue projection is 5% of the fleet -> no conservation,
    # even though it is 20x this model's own window.
    m = _matrix(tpd=RateLimit(limit=20_000, remaining=20_000))
    qp = QueueProfile(projected_tokens=400_000)
    assert s4_queue_tokens(m, queue=qp, fleet_remaining={"tpd": 1_020_000}) == 0.0


def test_s4_per_model_fallback_when_no_fleet():
    # No fleet_remaining -> falls back to this model's own remaining (old behavior).
    m = _matrix(tpd=RateLimit(limit=20_000, remaining=20_000))
    qp = QueueProfile(projected_tokens=400_000)  # 20x this model's window
    assert s4_queue_tokens(m, queue=qp) == pytest.approx(-1.0, abs=0.05)


def test_s4_fleet_of_one_equals_per_model():
    # Fleet sum of a fleet-of-one == this model's remaining -> still conserves
    # (the genuine "no escape hatch" case, e.g. pp11).
    m = _matrix(tpd=RateLimit(limit=20_000, remaining=20_000))
    qp = QueueProfile(projected_tokens=40_000)  # 2x the only window
    assert s4_queue_tokens(m, queue=qp, fleet_remaining={"tpd": 20_000}) == pytest.approx(-1.0, abs=0.05)
