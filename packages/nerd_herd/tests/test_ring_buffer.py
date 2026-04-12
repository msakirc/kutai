import time
from nerd_herd.ring_buffer import RingBuffer


def test_empty_rate():
    rb = RingBuffer(capacity=60)
    assert rb.rate(60) == 0.0


def test_single_sample_rate():
    rb = RingBuffer(capacity=60)
    rb.append(1000.0, 100)
    assert rb.rate(60) == 0.0  # need at least 2 samples


def test_two_samples_rate():
    rb = RingBuffer(capacity=60)
    rb.append(1000.0, 0)
    rb.append(1010.0, 50)
    # rate = (50 - 0) / (1010 - 1000) = 5.0
    assert rb.rate(60) == 5.0


def test_rate_window_filtering():
    rb = RingBuffer(capacity=60)
    # Old sample outside window
    rb.append(900.0, 0)
    # Samples inside 60s window (relative to newest = 1060)
    rb.append(1010.0, 100)
    rb.append(1060.0, 200)
    # rate over 60s window: oldest in window is t=1010, newest t=1060
    # (200 - 100) / (1060 - 1010) = 2.0
    assert rb.rate(60, now=1060.0) == 2.0


def test_eviction():
    rb = RingBuffer(capacity=3)
    rb.append(1.0, 10)
    rb.append(2.0, 20)
    rb.append(3.0, 30)
    rb.append(4.0, 40)  # evicts (1.0, 10)
    assert len(rb) == 3
    # rate: (40 - 20) / (4 - 2) = 10.0
    assert rb.rate(60) == 10.0


def test_latest():
    rb = RingBuffer(capacity=10)
    assert rb.latest() is None
    rb.append(1.0, 42)
    assert rb.latest() == 42


def test_negative_rate_returns_zero():
    """Counter resets (e.g. server restart) should return 0, not negative."""
    rb = RingBuffer(capacity=60)
    rb.append(1000.0, 100)
    rb.append(1010.0, 20)  # counter reset
    assert rb.rate(60) == 0.0
