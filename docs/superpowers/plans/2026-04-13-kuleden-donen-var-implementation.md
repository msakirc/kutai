# Kuleden Dönen Var Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract cloud provider health management (rate limits, quotas, circuit breakers, header parsing) into a standalone package at `packages/kuleden_donen_var/`.

**Architecture:** KDV is a dumb pipe — it tracks cloud provider capacity and reports changes via an `on_capacity_change` callback. Host pushes model config via `register()`. Router calls `pre_call()` / `post_call()` / `record_failure()`. KDV never discovers models, picks models, or makes LLM calls.

**Tech Stack:** Python 3.10+, stdlib only (no external deps), pytest for tests.

**Spec:** `docs/superpowers/specs/2026-04-13-cloud-operator-design.md`

---

## File Structure

### New files (package)

| File | Responsibility |
|------|---------------|
| `packages/kuleden_donen_var/pyproject.toml` | Package metadata |
| `packages/kuleden_donen_var/src/kuleden_donen_var/__init__.py` | Public API re-exports |
| `packages/kuleden_donen_var/src/kuleden_donen_var/config.py` | Config, dataclasses, types (`KuledenConfig`, `ProviderStatus`, `ModelStatus`, `CapacityEvent`, `PreCallResult`) |
| `packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py` | Main class composing modules (~100 lines) |
| `packages/kuleden_donen_var/src/kuleden_donen_var/header_parser.py` | Provider-specific header normalization (moved from `src/models/header_parser.py`) |
| `packages/kuleden_donen_var/src/kuleden_donen_var/rate_limiter.py` | Two-tier rate limiting (moved from `src/models/rate_limiter.py`, stripped of `src/` imports) |
| `packages/kuleden_donen_var/src/kuleden_donen_var/circuit_breaker.py` | Per-provider failure tracking + cooldown (moved from `src/core/router.py`) |
| `packages/kuleden_donen_var/tests/test_config.py` | Type/config tests |
| `packages/kuleden_donen_var/tests/test_header_parser.py` | Header parsing tests |
| `packages/kuleden_donen_var/tests/test_rate_limiter.py` | Rate limiting tests |
| `packages/kuleden_donen_var/tests/test_circuit_breaker.py` | Circuit breaker tests |
| `packages/kuleden_donen_var/tests/test_kdv.py` | Integration tests for main class |

### Modified files (KutAI shims + integration)

| File | Change |
|------|--------|
| `src/models/header_parser.py` | Becomes thin re-export shim |
| `src/models/rate_limiter.py` | Becomes thin shim delegating to KDV |
| `src/models/quota_planner.py` | Keeps threshold logic, delegates utilization tracking to KDV |
| `src/core/router.py` | Delete `CircuitBreaker` class (~50 lines), replace rate_limiter/header_parser/circuit_breaker calls with KDV calls |
| `requirements.txt` | Add `-e ./packages/kuleden_donen_var` |

---

### Task 1: Package scaffold + config types

**Files:**
- Create: `packages/kuleden_donen_var/pyproject.toml`
- Create: `packages/kuleden_donen_var/src/kuleden_donen_var/__init__.py`
- Create: `packages/kuleden_donen_var/src/kuleden_donen_var/config.py`
- Create: `packages/kuleden_donen_var/tests/test_config.py`

- [ ] **Step 1: Write failing test for config types**

```python
# packages/kuleden_donen_var/tests/test_config.py
"""Tests for config dataclasses."""
from kuleden_donen_var.config import (
    KuledenConfig,
    ProviderStatus,
    ModelStatus,
    CapacityEvent,
    PreCallResult,
)


def test_kuleden_config_defaults():
    cfg = KuledenConfig()
    assert cfg.circuit_breaker_threshold == 3
    assert cfg.circuit_breaker_cooldown_seconds == 600.0
    assert cfg.circuit_breaker_window_seconds == 300.0
    assert cfg.on_capacity_change is None


def test_provider_status_defaults():
    ps = ProviderStatus(provider="groq")
    assert ps.circuit_breaker_open is False
    assert ps.utilization_pct == 0.0
    assert ps.rpm_remaining is None
    assert ps.tpm_remaining is None
    assert ps.rpd_remaining is None
    assert ps.reset_in_seconds is None
    assert ps.models == {}


def test_model_status_defaults():
    ms = ModelStatus(model_id="groq/llama-8b")
    assert ms.utilization_pct == 0.0
    assert ms.has_capacity is True
    assert ms.daily_exhausted is False


def test_capacity_event():
    ps = ProviderStatus(provider="groq")
    evt = CapacityEvent(
        provider="groq",
        model_id="groq/llama-8b",
        event_type="capacity_restored",
        snapshot=ps,
    )
    assert evt.event_type == "capacity_restored"
    assert evt.snapshot.provider == "groq"


def test_pre_call_result_allowed():
    r = PreCallResult(allowed=True, wait_seconds=0.0, daily_exhausted=False)
    assert r.allowed is True
    assert r.wait_seconds == 0.0


def test_pre_call_result_denied():
    r = PreCallResult(allowed=False, wait_seconds=12.5, daily_exhausted=False)
    assert r.allowed is False
    assert r.wait_seconds == 12.5


def test_pre_call_result_daily_exhausted():
    r = PreCallResult(allowed=False, wait_seconds=0.0, daily_exhausted=True)
    assert r.daily_exhausted is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest packages/kuleden_donen_var/tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kuleden_donen_var'`

- [ ] **Step 3: Create pyproject.toml**

```toml
# packages/kuleden_donen_var/pyproject.toml
[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"

[project]
name = "kuleden-donen-var"
version = "0.1.0"
description = "Cloud LLM provider capacity tracker"
requires-python = ">=3.10"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 4: Create config.py with all types**

```python
# packages/kuleden_donen_var/src/kuleden_donen_var/config.py
"""Configuration dataclasses for Kuleden Dönen Var."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class KuledenConfig:
    """Engine settings — configured once at startup."""
    circuit_breaker_threshold: int = 3
    circuit_breaker_window_seconds: float = 300.0
    circuit_breaker_cooldown_seconds: float = 600.0
    on_capacity_change: Callable[[CapacityEvent], None] | None = None


@dataclass
class ModelStatus:
    """Per-model capacity state."""
    model_id: str
    utilization_pct: float = 0.0
    has_capacity: bool = True
    daily_exhausted: bool = False
    rpm_remaining: int | None = None
    tpm_remaining: int | None = None
    rpd_remaining: int | None = None


@dataclass
class ProviderStatus:
    """Per-provider capacity state."""
    provider: str
    circuit_breaker_open: bool = False
    utilization_pct: float = 0.0
    rpm_remaining: int | None = None
    tpm_remaining: int | None = None
    rpd_remaining: int | None = None
    reset_in_seconds: float | None = None
    models: dict[str, ModelStatus] = field(default_factory=dict)


@dataclass
class CapacityEvent:
    """Fired on meaningful capacity state changes."""
    provider: str
    model_id: str | None
    event_type: str  # capacity_restored, limit_hit, circuit_breaker_tripped, circuit_breaker_reset, daily_exhausted
    snapshot: ProviderStatus


@dataclass
class PreCallResult:
    """Result of pre_call check."""
    allowed: bool
    wait_seconds: float
    daily_exhausted: bool
```

- [ ] **Step 5: Create __init__.py**

```python
# packages/kuleden_donen_var/src/kuleden_donen_var/__init__.py
"""Kuleden Dönen Var — cloud LLM provider capacity tracker."""
from .config import (
    KuledenConfig,
    CapacityEvent,
    ModelStatus,
    PreCallResult,
    ProviderStatus,
)

__all__ = [
    "KuledenConfig",
    "CapacityEvent",
    "ModelStatus",
    "PreCallResult",
    "ProviderStatus",
]
```

- [ ] **Step 6: Install package and run tests**

Run:
```bash
pip install -e packages/kuleden_donen_var
pytest packages/kuleden_donen_var/tests/test_config.py -v
```
Expected: All 7 tests PASS

- [ ] **Step 7: Commit**

```bash
git add packages/kuleden_donen_var/pyproject.toml \
  packages/kuleden_donen_var/src/kuleden_donen_var/__init__.py \
  packages/kuleden_donen_var/src/kuleden_donen_var/config.py \
  packages/kuleden_donen_var/tests/test_config.py
git commit -m "feat(kuleden_donen_var): package scaffold + config types"
```

---

### Task 2: Header parser

Move `src/models/header_parser.py` into the package. The only dependency to cut is `src.infra.logging_config.get_logger` → stdlib `logging.getLogger`.

**Files:**
- Create: `packages/kuleden_donen_var/src/kuleden_donen_var/header_parser.py`
- Create: `packages/kuleden_donen_var/tests/test_header_parser.py`

- [ ] **Step 1: Write failing tests for header parsing**

```python
# packages/kuleden_donen_var/tests/test_header_parser.py
"""Tests for provider-specific rate limit header parsing."""
import time
from kuleden_donen_var.header_parser import parse_rate_limit_headers, RateLimitSnapshot


def test_snapshot_has_any_data_empty():
    snap = RateLimitSnapshot()
    assert snap.has_any_data() is False


def test_snapshot_has_any_data_with_rpm():
    snap = RateLimitSnapshot(rpm_limit=30)
    assert snap.has_any_data() is True


def test_openai_style_headers():
    headers = {
        "x-ratelimit-limit-requests": "30",
        "x-ratelimit-remaining-requests": "25",
        "x-ratelimit-reset-requests": "6s",
        "x-ratelimit-limit-tokens": "131072",
        "x-ratelimit-remaining-tokens": "100000",
        "x-ratelimit-reset-tokens": "12ms",
    }
    snap = parse_rate_limit_headers("groq", headers)
    assert snap is not None
    assert snap.rpm_limit == 30
    assert snap.rpm_remaining == 25
    assert snap.tpm_limit == 131072
    assert snap.tpm_remaining == 100000
    assert snap.rpm_reset_at is not None
    assert snap.rpm_reset_at > time.time()


def test_anthropic_headers():
    headers = {
        "anthropic-ratelimit-requests-limit": "50",
        "anthropic-ratelimit-requests-remaining": "48",
        "anthropic-ratelimit-requests-reset": "2026-01-27T12:00:30Z",
        "anthropic-ratelimit-tokens-limit": "80000",
        "anthropic-ratelimit-tokens-remaining": "75000",
        "anthropic-ratelimit-tokens-reset": "2026-01-27T12:00:30Z",
    }
    snap = parse_rate_limit_headers("anthropic", headers)
    assert snap is not None
    assert snap.rpm_limit == 50
    assert snap.rpm_remaining == 48
    assert snap.tpm_limit == 80000


def test_cerebras_daily_limits():
    headers = {
        "x-ratelimit-limit-tokens-minute": "131072",
        "x-ratelimit-remaining-tokens-minute": "100000",
        "x-ratelimit-reset-tokens-minute": "30.0",
        "x-ratelimit-limit-requests-day": "1000",
        "x-ratelimit-remaining-requests-day": "950",
        "x-ratelimit-reset-requests-day": "33011.382867",
    }
    snap = parse_rate_limit_headers("cerebras", headers)
    assert snap is not None
    assert snap.tpm_limit == 131072
    assert snap.rpd_limit == 1000
    assert snap.rpd_remaining == 950


def test_llm_provider_prefix_stripped():
    headers = {
        "llm_provider-x-ratelimit-limit-requests": "15",
        "llm_provider-x-ratelimit-remaining-requests": "10",
    }
    snap = parse_rate_limit_headers("gemini", headers)
    assert snap is not None
    assert snap.rpm_limit == 15
    assert snap.rpm_remaining == 10


def test_empty_headers_returns_none():
    assert parse_rate_limit_headers("openai", {}) is None
    assert parse_rate_limit_headers("openai", None) is None


def test_unknown_provider_uses_openai_style():
    headers = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "99",
    }
    snap = parse_rate_limit_headers("some_new_provider", headers)
    assert snap is not None
    assert snap.rpm_limit == 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest packages/kuleden_donen_var/tests/test_header_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kuleden_donen_var.header_parser'`

- [ ] **Step 3: Copy header_parser.py into package, replace logger**

Copy `src/models/header_parser.py` to `packages/kuleden_donen_var/src/kuleden_donen_var/header_parser.py` with these changes:
- Replace `from src.infra.logging_config import get_logger` → `import logging`
- Replace `logger = get_logger("models.header_parser")` → `logger = logging.getLogger(__name__)`
- Everything else stays identical

- [ ] **Step 4: Export from __init__.py**

Add to `packages/kuleden_donen_var/src/kuleden_donen_var/__init__.py`:
```python
from .header_parser import RateLimitSnapshot, parse_rate_limit_headers
```

And add `"RateLimitSnapshot", "parse_rate_limit_headers"` to `__all__`.

- [ ] **Step 5: Run tests**

Run: `pytest packages/kuleden_donen_var/tests/test_header_parser.py -v`
Expected: All 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add packages/kuleden_donen_var/src/kuleden_donen_var/header_parser.py \
  packages/kuleden_donen_var/src/kuleden_donen_var/__init__.py \
  packages/kuleden_donen_var/tests/test_header_parser.py
git commit -m "feat(kuleden_donen_var): header parser module"
```

---

### Task 3: Circuit breaker

Move the `CircuitBreaker` class from `src/core/router.py` (lines 283-326) into the package.

**Files:**
- Create: `packages/kuleden_donen_var/src/kuleden_donen_var/circuit_breaker.py`
- Create: `packages/kuleden_donen_var/tests/test_circuit_breaker.py`

- [ ] **Step 1: Write failing tests**

```python
# packages/kuleden_donen_var/tests/test_circuit_breaker.py
"""Tests for per-provider circuit breaker."""
import time
from kuleden_donen_var.circuit_breaker import CircuitBreaker


def test_initially_not_degraded():
    cb = CircuitBreaker(failure_threshold=3, window_seconds=300, cooldown_seconds=600)
    assert cb.is_degraded is False


def test_single_failure_not_degraded():
    cb = CircuitBreaker(failure_threshold=3, window_seconds=300, cooldown_seconds=600)
    cb.record_failure()
    assert cb.is_degraded is False


def test_threshold_failures_trips_breaker():
    cb = CircuitBreaker(failure_threshold=3, window_seconds=300, cooldown_seconds=600)
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.is_degraded is True


def test_success_resets_failures():
    cb = CircuitBreaker(failure_threshold=3, window_seconds=300, cooldown_seconds=600)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    assert cb.is_degraded is False  # only 1 failure since reset


def test_cooldown_expires():
    cb = CircuitBreaker(failure_threshold=2, window_seconds=300, cooldown_seconds=0.1)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_degraded is True
    time.sleep(0.15)
    assert cb.is_degraded is False


def test_old_failures_outside_window_ignored():
    cb = CircuitBreaker(failure_threshold=3, window_seconds=0.1, cooldown_seconds=600)
    cb.record_failure()
    cb.record_failure()
    time.sleep(0.15)
    cb.record_failure()
    assert cb.is_degraded is False  # old 2 expired, only 1 in window
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest packages/kuleden_donen_var/tests/test_circuit_breaker.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create circuit_breaker.py**

```python
# packages/kuleden_donen_var/src/kuleden_donen_var/circuit_breaker.py
"""Per-provider circuit breaker — track failures, temporarily disable."""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Track failures per provider and temporarily disable."""

    def __init__(
        self,
        failure_threshold: int = 3,
        window_seconds: float = 300,
        cooldown_seconds: float = 600,
    ):
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.failures: list[float] = []
        self.degraded_until: float = 0.0

    def record_failure(self) -> None:
        now = time.time()
        self.failures.append(now)
        self.failures = [t for t in self.failures if now - t < self.window_seconds]
        if len(self.failures) >= self.failure_threshold:
            self.degraded_until = now + self.cooldown_seconds
            logger.warning(
                "circuit breaker tripped, cooldown_seconds=%s",
                self.cooldown_seconds,
            )

    def record_success(self) -> None:
        self.failures.clear()
        self.degraded_until = 0.0

    @property
    def is_degraded(self) -> bool:
        if time.time() >= self.degraded_until:
            if self.degraded_until > 0:
                self.degraded_until = 0.0
                self.failures.clear()
            return False
        return True
```

- [ ] **Step 4: Run tests**

Run: `pytest packages/kuleden_donen_var/tests/test_circuit_breaker.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/kuleden_donen_var/src/kuleden_donen_var/circuit_breaker.py \
  packages/kuleden_donen_var/tests/test_circuit_breaker.py
git commit -m "feat(kuleden_donen_var): circuit breaker module"
```

---

### Task 4: Rate limiter

Move the rate limiting logic from `src/models/rate_limiter.py`. This is the largest module. Key changes:
- Remove `src.infra.logging_config` → stdlib `logging`
- Remove `src.infra.db.accelerate_retries` → replaced by `on_capacity_change` callback (wired through KDV main class)
- Remove `src.models.model_registry` → `_init_from_registry()` deleted, host calls `register()` instead
- Remove `src.models.header_parser` import → use package-internal `header_parser`

**Files:**
- Create: `packages/kuleden_donen_var/src/kuleden_donen_var/rate_limiter.py`
- Create: `packages/kuleden_donen_var/tests/test_rate_limiter.py`

- [ ] **Step 1: Write failing tests**

```python
# packages/kuleden_donen_var/tests/test_rate_limiter.py
"""Tests for two-tier rate limiting."""
import asyncio
import time
import pytest
from kuleden_donen_var.rate_limiter import RateLimitState, RateLimitManager
from kuleden_donen_var.header_parser import RateLimitSnapshot


# -- RateLimitState --

def test_state_initially_has_capacity():
    state = RateLimitState(rpm_limit=30, tpm_limit=100000)
    assert state.has_capacity() is True
    assert state.current_rpm == 0
    assert state.current_tpm == 0


def test_state_headroom():
    state = RateLimitState(rpm_limit=30, tpm_limit=100000)
    assert state.rpm_headroom == 30
    assert state.tpm_headroom == 100000


def test_state_utilization_initially_zero():
    state = RateLimitState(rpm_limit=30, tpm_limit=100000)
    assert state.utilization_pct() == 0.0


def test_state_record_tokens():
    state = RateLimitState(rpm_limit=30, tpm_limit=100000)
    state.record_tokens(50000)
    assert state.current_tpm == 50000
    assert state.utilization_pct() == 50.0


def test_state_429_reduces_limits():
    state = RateLimitState(rpm_limit=30, tpm_limit=100000)
    state.record_429()
    assert state.rpm_limit < 30
    assert state.tpm_limit < 100000


def test_state_429_floors_at_half_original():
    state = RateLimitState(rpm_limit=10, tpm_limit=10000)
    for _ in range(20):
        state.record_429()
    assert state.rpm_limit >= 5  # 50% of original
    assert state.tpm_limit >= 5000


def test_state_update_from_snapshot():
    state = RateLimitState(rpm_limit=30, tpm_limit=100000)
    snap = RateLimitSnapshot(rpm_limit=60, rpm_remaining=55)
    state.update_from_snapshot(snap)
    assert state.rpm_limit == 60
    assert state._header_rpm_remaining == 55


def test_state_daily_limit_exhaustion():
    state = RateLimitState(rpm_limit=30, tpm_limit=100000)
    state.rpd_remaining = 0
    state.rpd_reset_at = time.time() + 3600
    assert state.has_capacity() is False


# -- RateLimitManager --

def test_manager_register_and_has_capacity():
    mgr = RateLimitManager()
    mgr.register_model("groq/llama-8b", "groq", rpm=30, tpm=131072)
    assert mgr.has_capacity("groq/llama-8b", "groq") is True


def test_manager_record_tokens():
    mgr = RateLimitManager()
    mgr.register_model("groq/llama-8b", "groq", rpm=30, tpm=131072)
    mgr.record_tokens("groq/llama-8b", "groq", 50000)
    util = mgr.get_utilization("groq/llama-8b")
    assert util > 0


def test_manager_record_429():
    mgr = RateLimitManager()
    mgr.register_model("groq/llama-8b", "groq", rpm=30, tpm=131072)
    mgr.record_429("groq/llama-8b", "groq")
    # Limits should be reduced
    state = mgr.model_limits["groq/llama-8b"]
    assert state.rpm_limit < 30


def test_manager_update_from_headers():
    mgr = RateLimitManager()
    mgr.register_model("groq/llama-8b", "groq", rpm=30, tpm=131072)
    snap = RateLimitSnapshot(rpm_limit=60, rpm_remaining=55)
    mgr.update_from_headers("groq/llama-8b", "groq", snap)
    state = mgr.model_limits["groq/llama-8b"]
    assert state.rpm_limit == 60


def test_manager_daily_exhausted():
    mgr = RateLimitManager()
    mgr.register_model("cerebras/llama-8b", "cerebras", rpm=30, tpm=131072)
    state = mgr.model_limits["cerebras/llama-8b"]
    state.rpd_remaining = 0
    state.rpd_reset_at = time.time() + 3600
    assert mgr.is_daily_exhausted("cerebras/llama-8b") is True


def test_manager_provider_aggregate():
    mgr = RateLimitManager()
    mgr.register_model("groq/llama-8b", "groq", rpm=30, tpm=131072,
                        provider_aggregate_rpm=100, provider_aggregate_tpm=500000)
    mgr.register_model("groq/mixtral", "groq", rpm=30, tpm=131072)
    # Provider limits should exist and be shared
    assert "groq" in mgr._provider_limits
    prov_util = mgr.get_provider_utilization("groq")
    assert prov_util == 0.0


def test_manager_restore_limits():
    mgr = RateLimitManager()
    mgr.register_model("groq/llama-8b", "groq", rpm=30, tpm=131072)
    mgr.record_429("groq/llama-8b", "groq")
    state = mgr.model_limits["groq/llama-8b"]
    reduced_rpm = state.rpm_limit
    state._last_429_at = time.time() - 700  # 11+ minutes ago
    mgr.restore_limits()
    assert state.rpm_limit > reduced_rpm


def test_manager_unregistered_model_has_capacity():
    mgr = RateLimitManager()
    assert mgr.has_capacity("unknown/model", "unknown") is True


@pytest.mark.asyncio
async def test_manager_wait_and_acquire_no_wait():
    mgr = RateLimitManager()
    mgr.register_model("groq/llama-8b", "groq", rpm=30, tpm=131072)
    waited = await mgr.wait_and_acquire("groq/llama-8b", "groq")
    assert waited == 0.0


def test_manager_get_status():
    mgr = RateLimitManager()
    mgr.register_model("groq/llama-8b", "groq", rpm=30, tpm=131072)
    status = mgr.get_status()
    assert "groq/llama-8b" in status["models"]
    assert "groq" in status["providers"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest packages/kuleden_donen_var/tests/test_rate_limiter.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create rate_limiter.py in package**

Copy `src/models/rate_limiter.py` to `packages/kuleden_donen_var/src/kuleden_donen_var/rate_limiter.py` with these changes:

1. Replace imports:
   - `from src.infra.logging_config import get_logger` → `import logging`
   - `logger = get_logger("models.rate_limiter")` → `logger = logging.getLogger(__name__)`
   - `from .header_parser import RateLimitSnapshot` (already correct for package-internal)

2. Remove the sleeping queue wake logic (lines 27-52 of original) — replace `_schedule_rate_limit_wake` with a callback:
   - Add `on_reset: Callable[[float], None] | None = None` parameter to `RateLimitState`
   - In `update_from_snapshot`, where it would call `_schedule_rate_limit_wake(delay)`, call `self.on_reset(delay)` if set

3. Remove `_init_from_registry()` function entirely (lines 567-591 of original)

4. Remove `get_rate_limit_manager()` singleton — KDV main class will own the instance

5. Remove `_INITIAL_PROVIDER_LIMITS` dict and `PROVIDER_AGGREGATE_LIMITS` alias — host passes these via `register_model()`

6. Remove the `from .model_registry import get_registry` import

Keep everything else identical: `RateLimitState`, `RateLimitManager`, all methods.

- [ ] **Step 4: Run tests**

Run: `pytest packages/kuleden_donen_var/tests/test_rate_limiter.py -v`
Expected: All 16 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/kuleden_donen_var/src/kuleden_donen_var/rate_limiter.py \
  packages/kuleden_donen_var/tests/test_rate_limiter.py
git commit -m "feat(kuleden_donen_var): rate limiter module"
```

---

### Task 5: KDV main class

Compose all modules into the main `KuledenDonenVar` class. This mirrors DaLLaMa's pattern: a thin orchestrator that wires internal modules together and exposes a simple API.

**Files:**
- Create: `packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py`
- Create: `packages/kuleden_donen_var/tests/test_kdv.py`
- Modify: `packages/kuleden_donen_var/src/kuleden_donen_var/__init__.py`

- [ ] **Step 1: Write failing tests**

```python
# packages/kuleden_donen_var/tests/test_kdv.py
"""Tests for KuledenDonenVar main class."""
import time
import pytest
from kuleden_donen_var import KuledenDonenVar, KuledenConfig
from kuleden_donen_var.config import CapacityEvent, PreCallResult


@pytest.fixture
def events():
    return []


@pytest.fixture
def kdv(events):
    cfg = KuledenConfig(
        circuit_breaker_threshold=3,
        circuit_breaker_cooldown_seconds=0.5,
        on_capacity_change=lambda evt: events.append(evt),
    )
    return KuledenDonenVar(cfg)


@pytest.fixture
def kdv_with_model(kdv):
    kdv.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
    return kdv


# -- register --

def test_register_model(kdv):
    kdv.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
    status = kdv.status
    assert "groq" in status
    assert "groq/llama-8b" in status["groq"].models


def test_register_multiple_models_same_provider(kdv):
    kdv.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
    kdv.register("groq/mixtral", "groq", rpm=30, tpm=131072)
    assert len(kdv.status["groq"].models) == 2


def test_register_with_provider_aggregate(kdv):
    kdv.register("groq/llama-8b", "groq", rpm=30, tpm=131072,
                 provider_aggregate_rpm=100, provider_aggregate_tpm=500000)
    assert "groq" in kdv.status


# -- pre_call --

def test_pre_call_allowed(kdv_with_model):
    result = kdv_with_model.pre_call("groq/llama-8b", "groq")
    assert result.allowed is True
    assert result.wait_seconds == 0.0
    assert result.daily_exhausted is False


def test_pre_call_circuit_breaker_blocks(kdv_with_model):
    for _ in range(3):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "server_error")
    result = kdv_with_model.pre_call("groq/llama-8b", "groq")
    assert result.allowed is False


def test_pre_call_daily_exhausted(kdv_with_model):
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    state.rpd_remaining = 0
    state.rpd_reset_at = time.time() + 3600
    result = kdv_with_model.pre_call("groq/llama-8b", "groq")
    assert result.allowed is False
    assert result.daily_exhausted is True


# -- post_call --

def test_post_call_records_tokens(kdv_with_model):
    kdv_with_model.post_call("groq/llama-8b", "groq", headers={}, token_count=5000)
    util = kdv_with_model._rate_limiter.get_utilization("groq/llama-8b")
    assert util > 0


def test_post_call_parses_headers(kdv_with_model):
    headers = {
        "x-ratelimit-limit-requests": "60",
        "x-ratelimit-remaining-requests": "55",
    }
    kdv_with_model.post_call("groq/llama-8b", "groq", headers=headers, token_count=1000)
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    assert state.rpm_limit == 60


def test_post_call_records_circuit_breaker_success(kdv_with_model):
    # Trip the breaker first
    for _ in range(3):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "server_error")
    assert kdv_with_model.pre_call("groq/llama-8b", "groq").allowed is False
    # Successful call resets it
    kdv_with_model.post_call("groq/llama-8b", "groq", headers={}, token_count=100)
    assert kdv_with_model.pre_call("groq/llama-8b", "groq").allowed is True


# -- record_failure --

def test_record_failure_rate_limit(kdv_with_model, events):
    kdv_with_model.record_failure("groq/llama-8b", "groq", "rate_limit")
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    assert state.rpm_limit < 30  # adaptively reduced
    assert any(e.event_type == "limit_hit" for e in events)


def test_record_failure_server_error_trips_breaker(kdv_with_model, events):
    for _ in range(3):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "server_error")
    assert any(e.event_type == "circuit_breaker_tripped" for e in events)


def test_record_failure_timeout_trips_breaker(kdv_with_model):
    for _ in range(3):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "timeout")
    result = kdv_with_model.pre_call("groq/llama-8b", "groq")
    assert result.allowed is False


def test_record_failure_auth_ignored(kdv_with_model):
    kdv_with_model.record_failure("groq/llama-8b", "groq", "auth")
    # Auth errors don't trip circuit breaker
    result = kdv_with_model.pre_call("groq/llama-8b", "groq")
    assert result.allowed is True


# -- status --

def test_status_empty(kdv):
    assert kdv.status == {}


def test_status_reflects_state(kdv_with_model):
    kdv_with_model.post_call("groq/llama-8b", "groq", headers={}, token_count=5000)
    status = kdv_with_model.status
    assert status["groq"].provider == "groq"
    assert status["groq"].circuit_breaker_open is False
    model_status = status["groq"].models["groq/llama-8b"]
    assert model_status.model_id == "groq/llama-8b"


def test_status_circuit_breaker_reflected(kdv_with_model):
    for _ in range(3):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "server_error")
    assert kdv_with_model.status["groq"].circuit_breaker_open is True


# -- on_capacity_change --

def test_capacity_change_on_limit_hit(kdv_with_model, events):
    kdv_with_model.record_failure("groq/llama-8b", "groq", "rate_limit")
    assert len(events) >= 1
    assert events[-1].event_type == "limit_hit"
    assert events[-1].provider == "groq"
    assert events[-1].snapshot.provider == "groq"


def test_capacity_change_on_circuit_breaker_trip(kdv_with_model, events):
    for _ in range(3):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "server_error")
    cb_events = [e for e in events if e.event_type == "circuit_breaker_tripped"]
    assert len(cb_events) == 1


def test_capacity_change_on_circuit_breaker_reset(kdv_with_model, events):
    for _ in range(3):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "server_error")
    events.clear()
    kdv_with_model.post_call("groq/llama-8b", "groq", headers={}, token_count=100)
    cb_events = [e for e in events if e.event_type == "circuit_breaker_reset"]
    assert len(cb_events) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest packages/kuleden_donen_var/tests/test_kdv.py -v`
Expected: FAIL — `ImportError: cannot import name 'KuledenDonenVar'`

- [ ] **Step 3: Create kdv.py**

```python
# packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py
"""KuledenDonenVar — main class composing all modules."""
from __future__ import annotations

import logging
from typing import Any

from .circuit_breaker import CircuitBreaker
from .config import (
    CapacityEvent,
    KuledenConfig,
    ModelStatus,
    PreCallResult,
    ProviderStatus,
)
from .header_parser import parse_rate_limit_headers
from .rate_limiter import RateLimitManager

logger = logging.getLogger(__name__)


class KuledenDonenVar:
    def __init__(self, config: KuledenConfig):
        self._config = config
        self._rate_limiter = RateLimitManager()
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._providers: dict[str, set[str]] = {}  # provider → {model_ids}

    def _get_cb(self, provider: str) -> CircuitBreaker:
        if provider not in self._circuit_breakers:
            self._circuit_breakers[provider] = CircuitBreaker(
                failure_threshold=self._config.circuit_breaker_threshold,
                window_seconds=self._config.circuit_breaker_window_seconds,
                cooldown_seconds=self._config.circuit_breaker_cooldown_seconds,
            )
        return self._circuit_breakers[provider]

    def _fire(self, provider: str, model_id: str | None, event_type: str) -> None:
        if self._config.on_capacity_change is None:
            return
        snapshot = self._build_provider_status(provider)
        evt = CapacityEvent(
            provider=provider,
            model_id=model_id,
            event_type=event_type,
            snapshot=snapshot,
        )
        try:
            self._config.on_capacity_change(evt)
        except Exception:
            logger.exception("on_capacity_change callback failed")

    def register(
        self,
        model_id: str,
        provider: str,
        rpm: int,
        tpm: int,
        provider_aggregate_rpm: int | None = None,
        provider_aggregate_tpm: int | None = None,
    ) -> None:
        self._rate_limiter.register_model(
            litellm_name=model_id,
            provider=provider,
            rpm=rpm,
            tpm=tpm,
            provider_aggregate_rpm=provider_aggregate_rpm,
            provider_aggregate_tpm=provider_aggregate_tpm,
        )
        self._providers.setdefault(provider, set()).add(model_id)

    def pre_call(
        self,
        model_id: str,
        provider: str,
        estimated_tokens: int = 0,
    ) -> PreCallResult:
        # Circuit breaker check
        cb = self._get_cb(provider)
        if cb.is_degraded:
            return PreCallResult(allowed=False, wait_seconds=0.0, daily_exhausted=False)

        # Daily limit check
        if self._rate_limiter.is_daily_exhausted(model_id):
            return PreCallResult(allowed=False, wait_seconds=0.0, daily_exhausted=True)

        # Rate limit capacity check
        if not self._rate_limiter.has_capacity(model_id, provider, estimated_tokens):
            # Estimate wait time from state
            state = self._rate_limiter.model_limits.get(model_id)
            wait = 0.0
            if state and state._request_timestamps:
                import time
                oldest = state._request_timestamps[0]
                wait = max(0, 60 - (time.time() - oldest) + 0.5)
            return PreCallResult(allowed=False, wait_seconds=wait, daily_exhausted=False)

        return PreCallResult(allowed=True, wait_seconds=0.0, daily_exhausted=False)

    def post_call(
        self,
        model_id: str,
        provider: str,
        headers: dict[str, Any] | None,
        token_count: int,
    ) -> None:
        # Record tokens
        self._rate_limiter.record_tokens(model_id, provider, token_count)

        # Parse and apply response headers
        if headers:
            snapshot = parse_rate_limit_headers(provider, headers)
            if snapshot is not None:
                prev_state = self._rate_limiter.model_limits.get(model_id)
                prev_rpm_remaining = prev_state._header_rpm_remaining if prev_state else None

                self._rate_limiter.update_from_headers(model_id, provider, snapshot)

                # Fire capacity_restored if significant improvement
                if (prev_rpm_remaining is not None
                        and snapshot.rpm_remaining is not None
                        and prev_rpm_remaining <= 1
                        and snapshot.rpm_remaining > 5):
                    self._fire(provider, model_id, "capacity_restored")

        # Circuit breaker success
        cb = self._get_cb(provider)
        was_degraded = cb.is_degraded
        cb.record_success()
        if was_degraded:
            self._fire(provider, model_id, "circuit_breaker_reset")

    def record_failure(
        self,
        model_id: str,
        provider: str,
        error_type: str,
    ) -> None:
        if error_type == "rate_limit":
            self._rate_limiter.record_429(model_id, provider)
            self._fire(provider, model_id, "limit_hit")
        elif error_type in ("server_error", "timeout"):
            cb = self._get_cb(provider)
            was_degraded = cb.is_degraded
            cb.record_failure()
            if cb.is_degraded and not was_degraded:
                self._fire(provider, model_id, "circuit_breaker_tripped")
        # auth errors: not tracked (permanent, not transient)

    def _build_provider_status(self, provider: str) -> ProviderStatus:
        import time
        cb = self._get_cb(provider)
        model_ids = self._providers.get(provider, set())

        models: dict[str, ModelStatus] = {}
        worst_util = 0.0
        earliest_reset: float | None = None

        for mid in model_ids:
            state = self._rate_limiter.model_limits.get(mid)
            if state is None:
                models[mid] = ModelStatus(model_id=mid)
                continue

            util = state.utilization_pct()
            worst_util = max(worst_util, util)

            daily_exhausted = (
                state.rpd_remaining is not None
                and state.rpd_remaining <= 0
                and state.rpd_reset_at is not None
                and time.time() < state.rpd_reset_at
            )

            # Track earliest reset
            for reset_at in (state._header_rpm_reset_at, state._header_tpm_reset_at, state.rpd_reset_at):
                if reset_at is not None:
                    remaining = reset_at - time.time()
                    if remaining > 0:
                        if earliest_reset is None or remaining < earliest_reset:
                            earliest_reset = remaining

            models[mid] = ModelStatus(
                model_id=mid,
                utilization_pct=util,
                has_capacity=state.has_capacity(),
                daily_exhausted=daily_exhausted,
                rpm_remaining=state._header_rpm_remaining,
                tpm_remaining=state._header_tpm_remaining,
                rpd_remaining=state.rpd_remaining,
            )

        # Provider-level utilization
        prov_state = self._rate_limiter._provider_limits.get(provider)
        prov_util = prov_state.utilization_pct() if prov_state else worst_util

        return ProviderStatus(
            provider=provider,
            circuit_breaker_open=cb.is_degraded,
            utilization_pct=max(worst_util, prov_util),
            rpm_remaining=prov_state._header_rpm_remaining if prov_state else None,
            tpm_remaining=prov_state._header_tpm_remaining if prov_state else None,
            rpd_remaining=None,
            reset_in_seconds=earliest_reset,
            models=models,
        )

    @property
    def status(self) -> dict[str, ProviderStatus]:
        return {
            provider: self._build_provider_status(provider)
            for provider in self._providers
        }
```

- [ ] **Step 4: Update __init__.py to export KuledenDonenVar**

```python
# packages/kuleden_donen_var/src/kuleden_donen_var/__init__.py
"""Kuleden Dönen Var — cloud LLM provider capacity tracker."""
from .config import (
    CapacityEvent,
    KuledenConfig,
    ModelStatus,
    PreCallResult,
    ProviderStatus,
)
from .header_parser import RateLimitSnapshot, parse_rate_limit_headers
from .kdv import KuledenDonenVar

__all__ = [
    "KuledenDonenVar",
    "KuledenConfig",
    "CapacityEvent",
    "ModelStatus",
    "PreCallResult",
    "ProviderStatus",
    "RateLimitSnapshot",
    "parse_rate_limit_headers",
]
```

- [ ] **Step 5: Run all package tests**

Run: `pytest packages/kuleden_donen_var/tests/ -v`
Expected: All tests PASS (config: 7, header_parser: 7, circuit_breaker: 6, rate_limiter: 16, kdv: 20)

- [ ] **Step 6: Commit**

```bash
git add packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py \
  packages/kuleden_donen_var/src/kuleden_donen_var/__init__.py \
  packages/kuleden_donen_var/tests/test_kdv.py
git commit -m "feat(kuleden_donen_var): main class composing all modules"
```

---

### Task 6: Install package + create shims in KutAI

Add the package to `requirements.txt` and create thin shims so existing import paths (`from src.models.rate_limiter import ...`) keep working during migration.

**Files:**
- Modify: `requirements.txt`
- Modify: `src/models/header_parser.py`
- Modify: `src/models/rate_limiter.py`

- [ ] **Step 1: Add to requirements.txt**

Add after the existing `-e ./packages/nerd_herd` line:
```
-e ./packages/kuleden_donen_var
```

- [ ] **Step 2: Replace src/models/header_parser.py with shim**

```python
# src/models/header_parser.py
"""Shim — delegates to kuleden_donen_var package.

All real logic lives in packages/kuleden_donen_var/.
This file preserves import paths during migration.
"""
from kuleden_donen_var.header_parser import (  # noqa: F401
    RateLimitSnapshot,
    parse_rate_limit_headers,
)
```

- [ ] **Step 3: Replace src/models/rate_limiter.py with shim**

The shim must preserve the `get_rate_limit_manager()` singleton and `_INITIAL_PROVIDER_LIMITS` / `PROVIDER_AGGREGATE_LIMITS` that existing code references. The singleton now wraps KDV's rate limiter.

```python
# src/models/rate_limiter.py
"""Shim — delegates to kuleden_donen_var package.

All real logic lives in packages/kuleden_donen_var/.
This file preserves import paths during migration.
"""
from kuleden_donen_var.rate_limiter import RateLimitState, RateLimitManager  # noqa: F401
from kuleden_donen_var.header_parser import RateLimitSnapshot  # noqa: F401

# ─── Initial Provider Limits (fallback before header discovery) ──────────────
_INITIAL_PROVIDER_LIMITS: dict[str, dict[str, int]] = {
    "groq": {"rpm": 30, "tpm": 131072},
    "gemini": {"rpm": 15, "tpm": 1000000},
    "cerebras": {"rpm": 30, "tpm": 131072},
    "sambanova": {"rpm": 20, "tpm": 100000},
    "openai": {"rpm": 500, "tpm": 2000000},
    "anthropic": {"rpm": 50, "tpm": 80000},
}

PROVIDER_AGGREGATE_LIMITS = _INITIAL_PROVIDER_LIMITS

# ─── Singleton ───────────────────────────────────────────────
_manager: RateLimitManager | None = None


def get_rate_limit_manager() -> RateLimitManager:
    global _manager
    if _manager is None:
        _manager = RateLimitManager()
        _init_from_registry()
    return _manager


def _init_from_registry() -> None:
    """Auto-register all cloud models from the model registry."""
    try:
        from src.models.model_registry import get_registry
        registry = get_registry()
        manager = get_rate_limit_manager()

        for model in registry.cloud_models():
            agg = _INITIAL_PROVIDER_LIMITS.get(model.provider, {})
            manager.register_model(
                litellm_name=model.litellm_name,
                provider=model.provider,
                rpm=model.rate_limit_rpm,
                tpm=model.rate_limit_tpm,
                provider_aggregate_rpm=agg.get("rpm"),
                provider_aggregate_tpm=agg.get("tpm"),
            )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Rate limit init failed: {e}")
```

- [ ] **Step 4: Install package and run existing KutAI tests**

Run:
```bash
pip install -e packages/kuleden_donen_var
python -c "from src.models.rate_limiter import get_rate_limit_manager; print('shim ok')"
python -c "from src.models.header_parser import parse_rate_limit_headers; print('shim ok')"
pytest tests/ -x -q --timeout=30
```
Expected: Import shims work, existing tests pass

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/models/header_parser.py src/models/rate_limiter.py
git commit -m "refactor: shim rate_limiter + header_parser to kuleden_donen_var"
```

---

### Task 7: Wire router.py to KDV

Replace the scattered rate_limiter/circuit_breaker/header_parser calls in `src/core/router.py` with KDV calls. This is the main integration task.

**Files:**
- Modify: `src/core/router.py`
- Modify: `src/models/quota_planner.py` (minor — keep threshold logic, KDV feeds utilization)

- [ ] **Step 1: Add KDV singleton initialization**

At module level in `src/core/router.py`, add a lazy KDV getter. The KDV instance is created once and shared. The `on_capacity_change` callback wires to the dispatcher (import lazily to avoid circular deps).

After the existing imports (around line 98), replace:
```python
from src.models.rate_limiter import get_rate_limit_manager
from src.models.header_parser import parse_rate_limit_headers
from src.models.quota_planner import get_quota_planner
```

With:
```python
from src.models.quota_planner import get_quota_planner
from kuleden_donen_var import KuledenDonenVar, KuledenConfig, CapacityEvent
```

Add a lazy singleton:
```python
_kdv: KuledenDonenVar | None = None


def get_kdv() -> KuledenDonenVar:
    global _kdv
    if _kdv is None:
        from src.models.rate_limiter import _INITIAL_PROVIDER_LIMITS
        from src.models.model_registry import get_registry

        def _on_capacity_change(evt: CapacityEvent) -> None:
            # Feed utilization to quota planner
            planner = get_quota_planner()
            snap = evt.snapshot
            if snap.utilization_pct > 0:
                reset_in = snap.reset_in_seconds or 3600
                planner.update_paid_utilization(evt.provider, snap.utilization_pct, reset_in)

            # Wake sleeping tasks on capacity restoration
            if evt.event_type in ("capacity_restored", "circuit_breaker_reset"):
                try:
                    import asyncio
                    from src.infra.db import accelerate_retries
                    asyncio.ensure_future(accelerate_retries("capacity_restored"))
                except Exception:
                    pass

        cfg = KuledenConfig(on_capacity_change=_on_capacity_change)
        _kdv = KuledenDonenVar(cfg)

        # Register all cloud models
        try:
            registry = get_registry()
            for model in registry.cloud_models():
                agg = _INITIAL_PROVIDER_LIMITS.get(model.provider, {})
                _kdv.register(
                    model_id=model.litellm_name,
                    provider=model.provider,
                    rpm=model.rate_limit_rpm,
                    tpm=model.rate_limit_tpm,
                    provider_aggregate_rpm=agg.get("rpm"),
                    provider_aggregate_tpm=agg.get("tpm"),
                )
        except Exception:
            pass
    return _kdv
```

- [ ] **Step 2: Delete CircuitBreaker class and module-level dict**

Delete lines 283-326 from `router.py` (the `CircuitBreaker` class, `_circuit_breakers` dict, and `_get_circuit_breaker` function).

- [ ] **Step 3: Replace circuit breaker check in model scoring**

At line ~474 (in the model scoring loop), replace:
```python
        if not model.is_local:
            cb = _get_circuit_breaker(model.provider)
            if cb.is_degraded:
                _skip(f"circuit_breaker({model.provider})"); continue
```

With:
```python
        if not model.is_local:
            kdv = get_kdv()
            prov_status = kdv.status.get(model.provider)
            if prov_status and prov_status.circuit_breaker_open:
                _skip(f"circuit_breaker({model.provider})"); continue
```

- [ ] **Step 4: Replace rate limit capacity check in model scoring**

In the availability scoring section (~lines 604-624), replace direct `get_rate_limit_manager()` calls with KDV status reads:

Replace:
```python
            rl_manager = get_rate_limit_manager()
            total_tokens = (
                reqs.estimated_input_tokens + reqs.estimated_output_tokens
            )
            model_util = rl_manager.get_utilization(model.litellm_name)
            provider_util = rl_manager.get_provider_utilization(model.provider)
            _daily_exhausted = rl_manager.is_daily_exhausted(model.litellm_name)
```

With:
```python
            kdv = get_kdv()
            prov_status = kdv.status.get(model.provider)
            model_status = prov_status.models.get(model.litellm_name) if prov_status else None
            model_util = model_status.utilization_pct if model_status else 0.0
            provider_util = prov_status.utilization_pct if prov_status else 0.0
            _daily_exhausted = model_status.daily_exhausted if model_status else False
```

- [ ] **Step 5: Replace S7 sibling rebalancing rate limiter reads**

At ~line 865, replace `_rl_mgr_s7 = get_rate_limit_manager()` and the subsequent `_rl_mgr_s7.get_utilization()` / `_rl_mgr_s7.get_provider_utilization()` calls with KDV status reads, using the same pattern as Step 4.

- [ ] **Step 6: Replace pre-call rate limiting**

At ~lines 1212-1229, replace the `wait_and_acquire` flow. Since KDV's `pre_call` is non-blocking, the caller must decide what to do:

Replace:
```python
        if not model.is_local:
            rl_manager = get_rate_limit_manager()
            estimated_tokens = (
                reqs.estimated_input_tokens + reqs.estimated_output_tokens
            )
            wait_time = await rl_manager.wait_and_acquire(
                litellm_name=model.litellm_name,
                provider=model.provider,
                estimated_tokens=estimated_tokens,
            )
            if wait_time < 0:
                ...
                continue
            if wait_time > 0:
                ...
```

With:
```python
        if not model.is_local:
            kdv = get_kdv()
            estimated_tokens = (
                reqs.estimated_input_tokens + reqs.estimated_output_tokens
            )
            pre = kdv.pre_call(model.litellm_name, model.provider, estimated_tokens)
            if pre.daily_exhausted:
                logger.warning("daily limit exhausted", model_name=model.name)
                last_error = f"Daily limit exhausted for {model.name}"
                continue
            if not pre.allowed:
                if pre.wait_seconds > 0:
                    logger.info(
                        "rate limiter waiting",
                        model_name=model.name,
                        wait_time_seconds=pre.wait_seconds,
                    )
                    await asyncio.sleep(pre.wait_seconds)
                else:
                    last_error = f"Rate limited for {model.name}"
                    continue
```

- [ ] **Step 7: Replace post-call token recording and header parsing**

At ~lines 1375-1384, replace:
```python
                        rl_manager.record_tokens(
                            model.litellm_name,
                            model.provider,
                            total_tokens,
                        )

                    # Update rate limits from response headers
                    if not model.is_local:
                        _update_limits_from_response(response, model, rl_manager)
```

With:
```python
                    if not model.is_local:
                        hidden = getattr(response, "_hidden_params", None)
                        headers = {}
                        if hidden:
                            headers = dict(hidden.get("additional_headers") or hidden.get("headers") or {})
                        get_kdv().post_call(
                            model.litellm_name,
                            model.provider,
                            headers=headers,
                            token_count=total_tokens,
                        )
```

- [ ] **Step 8: Replace error handling (429 + circuit breaker)**

At ~lines 1528-1571, replace the circuit breaker and rate limiter error recording:

Replace `_get_circuit_breaker(model.provider).record_failure()` with:
```python
get_kdv().record_failure(model.litellm_name, model.provider, "server_error")
```

Replace `_get_circuit_breaker(model.provider).record_success()` (at ~line 1466) with — already handled by `post_call()`, so delete this line.

Replace the rate limit recording block:
```python
                    if is_rate_limit:
                        if not model.is_local:
                            rl_manager.record_429(
                                model.litellm_name,
                                model.provider,
                            )
                            if not model.is_free:
                                get_quota_planner().record_429(model.provider)
```

With:
```python
                    if is_rate_limit:
                        if not model.is_local:
                            get_kdv().record_failure(model.litellm_name, model.provider, "rate_limit")
                            if not model.is_free:
                                get_quota_planner().record_429(model.provider)
```

- [ ] **Step 9: Delete _update_limits_from_response function**

Delete the `_update_limits_from_response` function (~lines 955-989) — its logic is now inside KDV's `post_call`.

- [ ] **Step 10: Clean up unused imports**

Remove these imports from router.py (if no longer referenced):
```python
from src.models.rate_limiter import get_rate_limit_manager
from src.models.header_parser import parse_rate_limit_headers
```

Keep `from src.models.quota_planner import get_quota_planner` — the threshold logic stays.

- [ ] **Step 11: Run tests**

Run:
```bash
python -c "from src.core.router import select_model; print('router imports ok')"
pytest tests/ -x -q --timeout=30
pytest packages/kuleden_donen_var/tests/ -v
```
Expected: All pass

- [ ] **Step 12: Commit**

```bash
git add src/core/router.py src/models/quota_planner.py
git commit -m "refactor(router): replace rate_limiter/circuit_breaker with kuleden_donen_var"
```

---

### Task 8: Update architecture doc

**Files:**
- Modify: `docs/architecture-modularization.md`

- [ ] **Step 1: Add kuleden_donen_var to the Extracted Packages table**

Add row:
```
| **kuleden_donen_var** | Cloud provider capacity tracker: rate limits, quotas, circuit breakers | `packages/kuleden_donen_var/` | Stable v0.1.0 | None |
```

- [ ] **Step 2: Update "Extract Next" section**

Move the Cloud Operator mention to "done" status or remove it, since it's now extracted.

- [ ] **Step 3: Update the "What Agents Need to Know" section**

Add:
```markdown
**If you're fixing a cloud rate limit/quota/circuit breaker error:**
The real logic is in `packages/kuleden_donen_var/src/kuleden_donen_var/`. The shims in `src/models/rate_limiter.py` and `src/models/header_parser.py` just delegate. Don't edit the shims for cloud capacity bugs — fix Kuleden Dönen Var.
```

- [ ] **Step 4: Update the Files table**

Add `packages/kuleden_donen_var/` and update the shim descriptions for `src/models/rate_limiter.py` and `src/models/header_parser.py`.

- [ ] **Step 5: Commit**

```bash
git add docs/architecture-modularization.md
git commit -m "docs: add kuleden dönen var to architecture doc"
```
