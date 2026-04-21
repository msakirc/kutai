"""compute_urgency + threshold — Task 18."""
import time

from general_beckman.admission import compute_urgency, threshold


def _task(priority=5, difficulty=3, age_s=0, unblocks=0):
    return {
        "id": 1,
        "priority": priority,
        "difficulty": difficulty,
        "created_at": time.time() - age_s,
        "downstream_unblocks_count": unblocks,
    }


def test_priority_5_baseline():
    u = compute_urgency(_task(priority=5))
    assert abs(u - 0.5) < 0.01


def test_age_scales_over_24h():
    u0 = compute_urgency(_task(priority=5, age_s=0))
    u1 = compute_urgency(_task(priority=5, age_s=86400))
    assert u1 > u0
    assert u1 - u0 <= 0.05 + 1e-6


def test_blocker_bump_capped():
    u1 = compute_urgency(_task(priority=5, unblocks=5))
    u2 = compute_urgency(_task(priority=5, unblocks=50))
    assert u2 == u1


def test_urgency_clamped_0_1():
    u = compute_urgency(_task(priority=10, age_s=10**9, unblocks=10**6))
    assert u <= 1.0


def test_priority_0_does_not_drag_below_zero():
    u = compute_urgency(_task(priority=0))
    assert u >= 0.0


def test_threshold_linear():
    assert abs(threshold(0.0) - 0.5) < 1e-6
    assert abs(threshold(0.5) - 0.0) < 1e-6
    assert abs(threshold(1.0) - (-0.5)) < 1e-6
