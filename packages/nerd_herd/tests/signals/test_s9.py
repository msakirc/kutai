"""Tests for S9 universal perishability signal."""
import time
import pytest

from nerd_herd.types import (
    LocalModelState, RateLimit, RateLimitMatrix,
)
from nerd_herd.signals.s9_perishability import s9_perishability


class FakeModel:
    def __init__(self, *, is_local=False, is_free=False, name="x", size_mb=0):
        self.is_local = is_local
        self.is_free = is_free
        self.name = name
        self.size_mb = size_mb
        self.is_loaded = False


# Loaded local + idle
def test_s9_loaded_local_idle_positive():
    m = FakeModel(is_local=True, name="loaded-x")
    m.is_loaded = True
    local = LocalModelState(model_name="loaded-x", idle_seconds=30, requests_processing=0)
    p = s9_perishability(m, local=local, vram_avail_mb=8000, matrix=RateLimitMatrix(),
                         task_difficulty=5, now=time.time())
    assert p == pytest.approx(0.25, abs=0.05)


# Loaded local + processing → hard busy veto. -1.0 not -0.10: llama-server
# is serial (--parallel 1) so a second admission must NEVER squeeze through
# even at max urgency.
def test_s9_loaded_local_busy_negative():
    m = FakeModel(is_local=True, name="loaded-x")
    m.is_loaded = True
    local = LocalModelState(model_name="loaded-x", idle_seconds=0, requests_processing=1)
    p = s9_perishability(m, local=local, vram_avail_mb=8000, matrix=RateLimitMatrix(),
                         task_difficulty=5, now=time.time())
    assert p == pytest.approx(-1.0, abs=0.01)


# In-flight registry signals busy even before requests_processing catches up.
# Production triage 2026-04-30: tasks #4464 + #4457 admitted 15s apart on
# the same local model because requests_processing was still 0 during the
# admission→prompt-processing gap. The in_flight_calls list has the
# admitted-not-yet-running entry (reserve_task fires synchronously at
# admission); S9 must veto on it.
def test_s9_local_in_flight_blocks_concurrent_admission():
    from nerd_herd.types import InFlightCall
    m = FakeModel(is_local=True, name="loaded-x")
    m.is_loaded = True
    local = LocalModelState(
        model_name="loaded-x", idle_seconds=30, requests_processing=0,
    )
    in_flight = [InFlightCall(
        call_id="task-4464", task_id=4464, category="main_work",
        model="loaded-x", provider="local", is_local=True,
        started_at=time.time(),
    )]
    # Even when requests_processing reads 0 (admission window before
    # llama-server picks up the request), in_flight presence vetoes a
    # second local admission.
    p = s9_perishability(
        m, local=local, vram_avail_mb=8000, matrix=RateLimitMatrix(),
        task_difficulty=5, now=time.time(), in_flight_calls=in_flight,
    )
    assert p == pytest.approx(-1.0, abs=0.01)


# Cold local during another local's in-flight: also vetoed (GPU is shared).
def test_s9_cold_local_in_flight_blocks_swap():
    from nerd_herd.types import InFlightCall
    m = FakeModel(is_local=True, name="cold-x", size_mb=4000)
    m.is_loaded = False
    local = LocalModelState(model_name="loaded-y")
    in_flight = [InFlightCall(
        call_id="task-1", task_id=1, category="main_work",
        model="loaded-y", provider="local", is_local=True,
        started_at=time.time(),
    )]
    p = s9_perishability(
        m, local=local, vram_avail_mb=8000, matrix=RateLimitMatrix(),
        task_difficulty=5, now=time.time(), in_flight_calls=in_flight,
    )
    assert p == pytest.approx(-1.0, abs=0.01)


# Cold local + VRAM available
def test_s9_cold_local_vram_available_positive():
    m = FakeModel(is_local=True, name="cold-x", size_mb=4000)
    m.is_loaded = False
    local = LocalModelState(model_name="other-loaded")
    p = s9_perishability(m, local=local, vram_avail_mb=8000, matrix=RateLimitMatrix(),
                         task_difficulty=5, now=time.time())
    assert p == pytest.approx(0.4, abs=0.05)


# Cold local + VRAM unavailable
def test_s9_cold_local_no_vram_negative():
    m = FakeModel(is_local=True, name="cold-x", size_mb=8000)
    m.is_loaded = False
    local = LocalModelState(model_name="other-loaded")
    p = s9_perishability(m, local=local, vram_avail_mb=4000, matrix=RateLimitMatrix(),
                         task_difficulty=5, now=time.time())
    assert p == pytest.approx(-0.5, abs=0.05)


# Free cloud + reset imminent + flush
def test_s9_free_cloud_reset_imminent_positive():
    m = FakeModel(is_free=True, name="groq-x")
    now = time.time()
    matrix = RateLimitMatrix(
        rpd=RateLimit(limit=1000, remaining=950, reset_at=int(now + 600)),  # 10min
    )
    p = s9_perishability(m, local=None, vram_avail_mb=0, matrix=matrix,
                         task_difficulty=5, now=now)
    assert p > 0.7  # strong abundance


# Paid cloud + budget flush + hard task
def test_s9_paid_cloud_flush_hard_task_positive():
    m = FakeModel(is_free=False, name="claude-opus")
    matrix = RateLimitMatrix(
        rpd=RateLimit(limit=200, remaining=190),
    )
    p = s9_perishability(m, local=None, vram_avail_mb=0, matrix=matrix,
                         task_difficulty=9, now=time.time())
    assert p == pytest.approx(1.0, abs=0.05)


# Paid cloud + budget flush + easy task → 0 (no perishability fires)
def test_s9_paid_cloud_flush_easy_task_zero():
    m = FakeModel(is_free=False, name="claude-opus")
    matrix = RateLimitMatrix(
        rpd=RateLimit(limit=200, remaining=190),
    )
    p = s9_perishability(m, local=None, vram_avail_mb=0, matrix=matrix,
                         task_difficulty=3, now=time.time())
    assert p == 0.0
