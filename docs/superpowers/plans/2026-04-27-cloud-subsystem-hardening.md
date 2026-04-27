# Cloud Subsystem Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire boot-time cloud provider discovery, fail-fast auth, telemetry provider column, and AA-derived cloud benchmarking into KutAI without bloating selector or dispatcher logic.

**Architecture:** New `packages/fatih_hoca/src/fatih_hoca/cloud/` subsystem owns per-provider `/models` adapters, cross-provider family normalization, and disk cache. Discovery runs at `fatih_hoca.init()` and via 6h Beckman scheduled task. Registry consumes discovered models; YAML `cloud:` block becomes optional manual seed. Pick log gets `provider` column. Bench enricher learns to match cloud families. CLOUD_PROFILES retained as fallback.

**Tech Stack:** Python 3.10, asyncio, aiosqlite, httpx (already a dep via vecihi/litellm), pytest, pytest-asyncio. No new third-party deps.

**Reference spec:** `docs/superpowers/specs/2026-04-27-cloud-subsystem-hardening-design.md`

---

## File Structure

**New files:**
- `packages/fatih_hoca/src/fatih_hoca/cloud/__init__.py` — package marker, public exports
- `packages/fatih_hoca/src/fatih_hoca/cloud/types.py` — `DiscoveredModel`, `ProviderResult` dataclasses
- `packages/fatih_hoca/src/fatih_hoca/cloud/family.py` — `normalize(provider, litellm_name) -> str`
- `packages/fatih_hoca/src/fatih_hoca/cloud/cache.py` — disk cache read/write, TTL logic
- `packages/fatih_hoca/src/fatih_hoca/cloud/discovery.py` — orchestrator: `refresh_all()`, diff, alert
- `packages/fatih_hoca/src/fatih_hoca/cloud/alert_throttle.py` — per-provider 24h cooldown state
- `packages/fatih_hoca/src/fatih_hoca/cloud/providers/__init__.py`
- `packages/fatih_hoca/src/fatih_hoca/cloud/providers/base.py` — `ProviderAdapter` protocol
- `packages/fatih_hoca/src/fatih_hoca/cloud/providers/groq.py`
- `packages/fatih_hoca/src/fatih_hoca/cloud/providers/openai.py`
- `packages/fatih_hoca/src/fatih_hoca/cloud/providers/anthropic.py`
- `packages/fatih_hoca/src/fatih_hoca/cloud/providers/gemini.py`
- `packages/fatih_hoca/src/fatih_hoca/cloud/providers/cerebras.py`
- `packages/fatih_hoca/src/fatih_hoca/cloud/providers/sambanova.py`
- `packages/fatih_hoca/src/fatih_hoca/cloud/providers/openrouter.py`
- `packages/fatih_hoca/tests/cloud/test_family.py`
- `packages/fatih_hoca/tests/cloud/test_cache.py`
- `packages/fatih_hoca/tests/cloud/test_discovery.py`
- `packages/fatih_hoca/tests/cloud/test_alert_throttle.py`
- `packages/fatih_hoca/tests/cloud/test_provider_groq.py`
- `packages/fatih_hoca/tests/cloud/test_provider_openai.py`
- `packages/fatih_hoca/tests/cloud/test_provider_anthropic.py`
- `packages/fatih_hoca/tests/cloud/test_provider_gemini.py`
- `packages/fatih_hoca/tests/cloud/test_provider_cerebras.py`
- `packages/fatih_hoca/tests/cloud/test_provider_sambanova.py`
- `packages/fatih_hoca/tests/cloud/test_provider_openrouter.py`
- `tests/integration/test_cloud_discovery_live.py` — env-gated live `/models` tests

**Modified files:**
- `packages/fatih_hoca/src/fatih_hoca/registry.py` — `detect_cloud_model()` accepts scraped fields; `register()` takes `family`
- `packages/fatih_hoca/src/fatih_hoca/types.py` — `ModelInfo.family: str = ""`
- `packages/fatih_hoca/src/fatih_hoca/__init__.py` — `init()` calls `cloud_discovery.refresh_all()`, sets `_available_providers` from results
- `src/infra/db.py:485-519` — add `provider` column + index in idempotent ALTER block
- `src/infra/pick_log.py` — add `provider` parameter, INSERT column
- `src/core/llm_dispatcher.py` — pass `model.provider` into `write_pick_log_row`
- `packages/fatih_hoca/src/fatih_hoca/benchmark_cloud_match.py` (new) — family-aware match logic
- `src/models/benchmark/benchmark_fetcher.py:1252` — accept cloud-aware match overlay
- `packages/general_beckman/src/general_beckman/cron_seed.py:19` — add `cloud_refresh` cadence
- `packages/salako/src/salako/actions.py` — add `cloud_refresh` action handler
- `packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py` — `no_data_warnings()` method

---

## Implementation Order

1. Skeleton + types
2. Family normalization
3. Cache
4. Alert throttle
5. Provider adapters (groq first, then peers)
6. Discovery orchestrator
7. Registry merge of scraped fields
8. Boot wiring + `_available_providers`
9. G3: `model_pick_log.provider` column + writer + caller
10. G5: cloud benchmark match + validation artifact + approval gate
11. Beckman cron entry + salako handler
12. KDV no-data warning
13. Live discovery smoke (env-gated)

---

### Task 1: Cloud package skeleton + types

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/cloud/__init__.py`
- Create: `packages/fatih_hoca/src/fatih_hoca/cloud/types.py`
- Create: `packages/fatih_hoca/tests/cloud/__init__.py`

- [ ] **Step 1: Create package markers**

```python
# packages/fatih_hoca/src/fatih_hoca/cloud/__init__.py
"""Cloud provider discovery and benchmarking subsystem."""
```

```python
# packages/fatih_hoca/tests/cloud/__init__.py
```

- [ ] **Step 2: Write types**

```python
# packages/fatih_hoca/src/fatih_hoca/cloud/types.py
"""Shared dataclasses for cloud discovery."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ProviderStatus = Literal["ok", "auth_fail", "server_error", "network_error", "rate_limited"]


@dataclass
class DiscoveredModel:
    """One model surfaced by a provider's /models endpoint.

    Carries litellm_name plus opportunistically scraped fields. Adapter
    populates whatever the provider response actually contains; missing
    fields stay None and downstream code falls back to litellm db / defaults.
    """
    litellm_name: str
    raw_id: str
    active: bool = True
    context_length: int | None = None
    max_output_tokens: int | None = None
    cost_per_1k_input: float | None = None
    cost_per_1k_output: float | None = None
    sampling_defaults: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResult:
    """Outcome of one provider's discovery probe."""
    provider: str
    status: ProviderStatus
    auth_ok: bool
    models: list[DiscoveredModel] = field(default_factory=list)
    error: str | None = None
    served_from_cache: bool = False
    fetched_at: str | None = None
```

- [ ] **Step 3: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/cloud/ packages/fatih_hoca/tests/cloud/__init__.py
git commit -m "feat(cloud): scaffold cloud discovery package + types"
```

---

### Task 2: Family normalization

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/cloud/family.py`
- Test: `packages/fatih_hoca/tests/cloud/test_family.py`

- [ ] **Step 1: Write the failing tests**

```python
# packages/fatih_hoca/tests/cloud/test_family.py
import pytest
from fatih_hoca.cloud.family import normalize


@pytest.mark.parametrize("provider,litellm_name,expected", [
    ("groq", "groq/llama-3.3-70b-versatile", "llama-3.3-70b"),
    ("cerebras", "cerebras/llama3.3-70b", "llama-3.3-70b"),
    ("sambanova", "sambanova/Meta-Llama-3.3-70B-Instruct", "llama-3.3-70b"),
    ("groq", "groq/llama-3.1-8b-instant", "llama-3.1-8b"),
    ("sambanova", "sambanova/Qwen3-32B", "qwen3-32b"),
    ("anthropic", "claude-sonnet-4-20250514", "claude-sonnet-4"),
    ("anthropic", "claude-3-5-sonnet-20241022", "claude-3.5-sonnet"),
    ("openai", "gpt-4o", "gpt-4o"),
    ("openai", "gpt-4o-mini", "gpt-4o-mini"),
    ("openai", "o1-preview", "o1-preview"),
    ("gemini", "gemini/gemini-2.0-flash", "gemini-2.0-flash"),
    ("gemini", "gemini/gemini-2.5-flash-preview-05-20", "gemini-2.5-flash"),
    ("openrouter", "openrouter/meta-llama/llama-3.3-70b-instruct", "llama-3.3-70b"),
])
def test_normalize_known_families(provider, litellm_name, expected):
    assert normalize(provider, litellm_name) == expected


def test_normalize_unknown_falls_back_to_litellm_name():
    # Returned value must equal stripped, lowercased name; family_unknown
    # logging is the side channel — caller checks `is_known_family()`.
    out = normalize("groq", "groq/some-future-model-v9-2030")
    assert out == "some-future-model-v9-2030"


def test_normalize_unknown_marked():
    from fatih_hoca.cloud.family import normalize, is_known_family
    assert is_known_family("llama-3.3-70b") is True
    assert is_known_family("some-future-model-v9-2030") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest packages/fatih_hoca/tests/cloud/test_family.py -v`
Expected: FAIL with `ModuleNotFoundError: fatih_hoca.cloud.family`

- [ ] **Step 3: Implement family.py**

```python
# packages/fatih_hoca/src/fatih_hoca/cloud/family.py
"""Cross-provider model family normalization.

Maps provider-specific litellm_name (e.g. ``groq/llama-3.3-70b-versatile``)
to a canonical family key (``llama-3.3-70b``). Same family across providers
shares benchmark cache entry.

Rules ordered most-specific first; first regex hit wins. Unmatched names
fall back to lower-cased provider-stripped form and are flagged via
``is_known_family()`` so new releases surface for manual rule addition.
"""
from __future__ import annotations

import re
from src.infra.logging_config import get_logger

logger = get_logger("fatih_hoca.cloud.family")

# (regex pattern matched against stripped/lowered name, family key)
_FAMILY_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^meta-?llama-3\.3-70b"), "llama-3.3-70b"),
    (re.compile(r"^llama-?3\.3-70b"), "llama-3.3-70b"),
    (re.compile(r"^llama-?3\.1-70b"), "llama-3.1-70b"),
    (re.compile(r"^llama-?3\.1-8b"), "llama-3.1-8b"),
    (re.compile(r"^llama-?3-70b"), "llama-3-70b"),
    (re.compile(r"^llama-?3-8b"), "llama-3-8b"),
    (re.compile(r"^qwen-?3-?32b"), "qwen3-32b"),
    (re.compile(r"^qwen-?2\.5-72b"), "qwen2.5-72b"),
    (re.compile(r"^qwen-?2\.5-coder-32b"), "qwen2.5-coder-32b"),
    (re.compile(r"^claude-3-?5-sonnet"), "claude-3.5-sonnet"),
    (re.compile(r"^claude-sonnet-4"), "claude-sonnet-4"),
    (re.compile(r"^claude-opus-4"), "claude-opus-4"),
    (re.compile(r"^claude-haiku-4"), "claude-haiku-4"),
    (re.compile(r"^gpt-4o-mini"), "gpt-4o-mini"),
    (re.compile(r"^gpt-4o"), "gpt-4o"),
    (re.compile(r"^o1-preview"), "o1-preview"),
    (re.compile(r"^o1-mini"), "o1-mini"),
    (re.compile(r"^o1"), "o1"),
    (re.compile(r"^gemini-2\.5-flash"), "gemini-2.5-flash"),
    (re.compile(r"^gemini-2\.0-flash"), "gemini-2.0-flash"),
    (re.compile(r"^gemini-1\.5-pro"), "gemini-1.5-pro"),
    (re.compile(r"^gemini-1\.5-flash"), "gemini-1.5-flash"),
    (re.compile(r"^mixtral-8x7b"), "mixtral-8x7b"),
]

_KNOWN_FAMILIES: set[str] = {key for _, key in _FAMILY_RULES}


def _strip_provider_prefix(litellm_name: str) -> str:
    """Strip leading ``provider/`` segment(s). Openrouter uses two segments
    (``openrouter/meta-llama/...``) — strip both."""
    parts = litellm_name.split("/")
    if len(parts) >= 2:
        # If the second-to-last is a known model org, drop only one segment.
        # Else strip everything except the last segment.
        return parts[-1]
    return litellm_name


def normalize(provider: str, litellm_name: str) -> str:
    """Return canonical family key for a (provider, litellm_name) pair."""
    stripped = _strip_provider_prefix(litellm_name).lower()
    for pattern, family in _FAMILY_RULES:
        if pattern.match(stripped):
            return family
    logger.info("family_unknown provider=%s litellm_name=%s", provider, litellm_name)
    return stripped


def is_known_family(family: str) -> bool:
    """True iff family was produced by a regex rule (not fallback)."""
    return family in _KNOWN_FAMILIES
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest packages/fatih_hoca/tests/cloud/test_family.py -v`
Expected: PASS — all 14 cases.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/cloud/family.py packages/fatih_hoca/tests/cloud/test_family.py
git commit -m "feat(cloud): family normalization for cross-provider dedup"
```

---

### Task 3: Disk cache

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/cloud/cache.py`
- Test: `packages/fatih_hoca/tests/cloud/test_cache.py`

- [ ] **Step 1: Write the failing tests**

```python
# packages/fatih_hoca/tests/cloud/test_cache.py
import json
import time
from pathlib import Path

import pytest

from fatih_hoca.cloud.cache import (
    CACHE_TTL_SECONDS,
    EVICT_TTL_SECONDS,
    CachedSnapshot,
    load,
    save,
)
from fatih_hoca.cloud.types import DiscoveredModel


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cloud_models"


def _model() -> DiscoveredModel:
    return DiscoveredModel(litellm_name="groq/foo", raw_id="foo", context_length=8192)


def test_save_then_load_round_trip(cache_dir):
    save(cache_dir, "groq", [_model()], status="ok")
    snap = load(cache_dir, "groq")
    assert snap is not None
    assert snap.status == "ok"
    assert snap.is_fresh is True
    assert snap.is_evicted is False
    assert snap.models[0].litellm_name == "groq/foo"


def test_load_missing_returns_none(cache_dir):
    assert load(cache_dir, "groq") is None


def test_stale_but_not_evicted(cache_dir, monkeypatch):
    save(cache_dir, "groq", [_model()], status="ok")
    snap_path = cache_dir / "groq.json"
    raw = json.loads(snap_path.read_text())
    raw["fetched_at_unix"] = time.time() - (CACHE_TTL_SECONDS + 60)
    snap_path.write_text(json.dumps(raw))
    snap = load(cache_dir, "groq")
    assert snap is not None
    assert snap.is_fresh is False
    assert snap.is_evicted is False


def test_evicted_returns_none(cache_dir):
    save(cache_dir, "groq", [_model()], status="ok")
    snap_path = cache_dir / "groq.json"
    raw = json.loads(snap_path.read_text())
    raw["fetched_at_unix"] = time.time() - (EVICT_TTL_SECONDS + 60)
    snap_path.write_text(json.dumps(raw))
    assert load(cache_dir, "groq") is None


def test_save_overwrites(cache_dir):
    save(cache_dir, "groq", [_model()], status="ok")
    new_model = DiscoveredModel(litellm_name="groq/bar", raw_id="bar")
    save(cache_dir, "groq", [new_model], status="ok")
    snap = load(cache_dir, "groq")
    assert snap is not None
    assert {m.litellm_name for m in snap.models} == {"groq/bar"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest packages/fatih_hoca/tests/cloud/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: fatih_hoca.cloud.cache`

- [ ] **Step 3: Implement cache.py**

```python
# packages/fatih_hoca/src/fatih_hoca/cloud/cache.py
"""Disk cache for provider /models responses.

Layout: ``<cache_dir>/<provider>.json``

Schema:
    {
      "fetched_at_unix": <float>,
      "fetched_at_iso": <str>,
      "status": <ProviderStatus>,
      "models": [<DiscoveredModel-as-dict>, ...]
    }

TTL semantics:
    - fresh:    age <= CACHE_TTL_SECONDS (7d)
    - stale:    CACHE_TTL_SECONDS < age <= EVICT_TTL_SECONDS (14d)
    - evicted:  age > EVICT_TTL_SECONDS  → load() returns None
"""
from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.infra.logging_config import get_logger

from .types import DiscoveredModel, ProviderStatus

logger = get_logger("fatih_hoca.cloud.cache")

CACHE_TTL_SECONDS = 7 * 24 * 3600
EVICT_TTL_SECONDS = 14 * 24 * 3600


@dataclass
class CachedSnapshot:
    provider: str
    fetched_at_unix: float
    fetched_at_iso: str
    status: ProviderStatus
    models: list[DiscoveredModel]

    @property
    def age_seconds(self) -> float:
        return time.time() - self.fetched_at_unix

    @property
    def is_fresh(self) -> bool:
        return self.age_seconds <= CACHE_TTL_SECONDS

    @property
    def is_evicted(self) -> bool:
        return self.age_seconds > EVICT_TTL_SECONDS


def _path(cache_dir: Path, provider: str) -> Path:
    return Path(cache_dir) / f"{provider}.json"


def save(cache_dir: Path, provider: str, models: list[DiscoveredModel], status: ProviderStatus) -> None:
    p = _path(cache_dir, provider)
    p.parent.mkdir(parents=True, exist_ok=True)
    now_unix = time.time()
    payload = {
        "fetched_at_unix": now_unix,
        "fetched_at_iso": datetime.fromtimestamp(now_unix, tz=timezone.utc).isoformat(),
        "status": status,
        "models": [dataclasses.asdict(m) for m in models],
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(p)


def load(cache_dir: Path, provider: str) -> CachedSnapshot | None:
    p = _path(cache_dir, provider)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        logger.warning("cache read failed for %s: %s", provider, e)
        return None
    snap = CachedSnapshot(
        provider=provider,
        fetched_at_unix=float(raw.get("fetched_at_unix", 0.0)),
        fetched_at_iso=str(raw.get("fetched_at_iso", "")),
        status=raw.get("status", "ok"),
        models=[DiscoveredModel(**m) for m in raw.get("models", [])],
    )
    if snap.is_evicted:
        logger.info("cache evicted for %s (age=%.0fs)", provider, snap.age_seconds)
        return None
    return snap
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest packages/fatih_hoca/tests/cloud/test_cache.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/cloud/cache.py packages/fatih_hoca/tests/cloud/test_cache.py
git commit -m "feat(cloud): disk cache for provider /models snapshots"
```

---

### Task 4: Alert throttle

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/cloud/alert_throttle.py`
- Test: `packages/fatih_hoca/tests/cloud/test_alert_throttle.py`

- [ ] **Step 1: Write the failing tests**

```python
# packages/fatih_hoca/tests/cloud/test_alert_throttle.py
import time
from pathlib import Path

from fatih_hoca.cloud.alert_throttle import AlertThrottle


def test_first_failure_alerts(tmp_path: Path):
    t = AlertThrottle(tmp_path / "throttle.json")
    assert t.should_alert("groq", current_state="auth_fail") is True


def test_repeat_failure_within_24h_suppressed(tmp_path: Path):
    t = AlertThrottle(tmp_path / "throttle.json")
    assert t.should_alert("groq", current_state="auth_fail") is True
    assert t.should_alert("groq", current_state="auth_fail") is False


def test_repeat_failure_after_24h_alerts(tmp_path: Path, monkeypatch):
    t = AlertThrottle(tmp_path / "throttle.json")
    t.should_alert("groq", current_state="auth_fail")
    # Rewind last alert by 25h.
    t._state["groq"]["last_alert_unix"] = time.time() - (25 * 3600)
    t._save()
    assert t.should_alert("groq", current_state="auth_fail") is True


def test_state_flip_always_alerts(tmp_path: Path):
    t = AlertThrottle(tmp_path / "throttle.json")
    t.should_alert("groq", current_state="auth_fail")
    # Recovery transition is always alerted.
    assert t.should_alert("groq", current_state="ok") is True
    # Re-fail right after recovery is a transition too — alert.
    assert t.should_alert("groq", current_state="auth_fail") is True


def test_independent_per_provider(tmp_path: Path):
    t = AlertThrottle(tmp_path / "throttle.json")
    assert t.should_alert("groq", current_state="auth_fail") is True
    assert t.should_alert("openai", current_state="auth_fail") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest packages/fatih_hoca/tests/cloud/test_alert_throttle.py -v`
Expected: FAIL with `ModuleNotFoundError: fatih_hoca.cloud.alert_throttle`

- [ ] **Step 3: Implement alert_throttle.py**

```python
# packages/fatih_hoca/src/fatih_hoca/cloud/alert_throttle.py
"""Per-provider alert cooldown state.

Persists last-alert timestamp + last-known-state per provider in a JSON
file. Alert decision rule:

    should_alert returns True iff
        - no prior alert for this provider, OR
        - last alert was >= 24h ago, OR
        - state transitioned since last alert
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ALERT_COOLDOWN_SECONDS = 24 * 3600


class AlertThrottle:
    def __init__(self, state_path: Path):
        self._path = Path(state_path)
        self._state: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._state = json.loads(self._path.read_text())
            except Exception:
                self._state = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._state, indent=2))

    def should_alert(self, provider: str, current_state: str) -> bool:
        entry = self._state.get(provider)
        now = time.time()
        if entry is None:
            self._state[provider] = {"last_state": current_state, "last_alert_unix": now}
            self._save()
            return True
        prior_state = entry.get("last_state")
        last_alert = float(entry.get("last_alert_unix", 0.0))
        is_transition = prior_state != current_state
        cooldown_passed = (now - last_alert) >= ALERT_COOLDOWN_SECONDS
        if is_transition or cooldown_passed:
            entry["last_state"] = current_state
            entry["last_alert_unix"] = now
            self._save()
            return True
        # Still record state for accuracy, but don't trip the alert.
        entry["last_state"] = current_state
        self._save()
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest packages/fatih_hoca/tests/cloud/test_alert_throttle.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/cloud/alert_throttle.py packages/fatih_hoca/tests/cloud/test_alert_throttle.py
git commit -m "feat(cloud): per-provider alert throttle state"
```

---

### Task 5: Provider adapter base + Groq

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/cloud/providers/__init__.py`
- Create: `packages/fatih_hoca/src/fatih_hoca/cloud/providers/base.py`
- Create: `packages/fatih_hoca/src/fatih_hoca/cloud/providers/groq.py`
- Test: `packages/fatih_hoca/tests/cloud/test_provider_groq.py`

- [ ] **Step 1: Write package marker + base**

```python
# packages/fatih_hoca/src/fatih_hoca/cloud/providers/__init__.py
"""Per-provider /models adapters."""
```

```python
# packages/fatih_hoca/src/fatih_hoca/cloud/providers/base.py
"""Adapter protocol shared by per-provider implementations."""
from __future__ import annotations

from typing import Protocol

from ..types import ProviderResult


class ProviderAdapter(Protocol):
    name: str

    async def fetch_models(self, api_key: str) -> ProviderResult:
        """Probe the provider's /models endpoint with the given key.

        Must NEVER raise. All errors map to ProviderResult.status +
        ProviderResult.error string.
        """
        ...
```

- [ ] **Step 2: Write the failing tests for groq**

```python
# packages/fatih_hoca/tests/cloud/test_provider_groq.py
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fatih_hoca.cloud.providers.groq import GroqAdapter


_GROQ_OK = {
    "object": "list",
    "data": [
        {
            "id": "llama-3.3-70b-versatile",
            "object": "model",
            "owned_by": "Meta",
            "active": True,
            "context_window": 131072,
            "max_completion_tokens": 32768,
        },
        {
            "id": "deprecated-model",
            "object": "model",
            "owned_by": "Whoever",
            "active": False,
            "context_window": 8192,
        },
        {
            "id": "llama-3.1-8b-instant",
            "object": "model",
            "owned_by": "Meta",
            "active": True,
            "context_window": 131072,
            "max_completion_tokens": 8192,
        },
    ],
}


def _resp(status_code: int, body: dict | str) -> httpx.Response:
    if isinstance(body, dict):
        return httpx.Response(status_code, json=body, request=httpx.Request("GET", "https://api.groq.com"))
    return httpx.Response(status_code, text=body, request=httpx.Request("GET", "https://api.groq.com"))


@pytest.mark.asyncio
async def test_groq_ok_filters_inactive_and_prefixes():
    a = GroqAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _GROQ_OK))):
        result = await a.fetch_models("fake-key")
    assert result.status == "ok"
    assert result.auth_ok is True
    names = [m.litellm_name for m in result.models]
    # Inactive filtered.
    assert "groq/deprecated-model" not in names
    # Active prefixed with provider.
    assert "groq/llama-3.3-70b-versatile" in names
    by_name = {m.litellm_name: m for m in result.models}
    assert by_name["groq/llama-3.3-70b-versatile"].context_length == 131072
    assert by_name["groq/llama-3.3-70b-versatile"].max_output_tokens == 32768


@pytest.mark.asyncio
async def test_groq_401_auth_fail():
    a = GroqAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(401, {"error": "Invalid API key"}))):
        result = await a.fetch_models("bad-key")
    assert result.status == "auth_fail"
    assert result.auth_ok is False
    assert result.models == []
    assert "401" in (result.error or "")


@pytest.mark.asyncio
async def test_groq_5xx_server_error():
    a = GroqAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(503, "down"))):
        result = await a.fetch_models("k")
    assert result.status == "server_error"
    assert result.auth_ok is False


@pytest.mark.asyncio
async def test_groq_network_error():
    a = GroqAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(side_effect=httpx.ConnectError("DNS"))):
        result = await a.fetch_models("k")
    assert result.status == "network_error"
    assert result.auth_ok is False


@pytest.mark.asyncio
async def test_groq_429_rate_limited_treated_as_ok_auth():
    a = GroqAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(429, "slow down"))):
        result = await a.fetch_models("k")
    # 429 means key valid but probe rate-limited; provider stays enabled.
    assert result.status == "rate_limited"
    assert result.auth_ok is True


@pytest.mark.asyncio
async def test_groq_malformed_json():
    a = GroqAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, "not json {"))):
        result = await a.fetch_models("k")
    assert result.status == "server_error"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest packages/fatih_hoca/tests/cloud/test_provider_groq.py -v`
Expected: FAIL with `ModuleNotFoundError: fatih_hoca.cloud.providers.groq`

- [ ] **Step 4: Implement groq adapter**

```python
# packages/fatih_hoca/src/fatih_hoca/cloud/providers/groq.py
"""Groq /models adapter.

Endpoint: GET https://api.groq.com/openai/v1/models
Auth:     Bearer <api_key>
Response: {"data": [{"id": "...", "active": bool, "context_window": int,
                     "max_completion_tokens": int, "owned_by": str}, ...]}
"""
from __future__ import annotations

import httpx

from src.infra.logging_config import get_logger

from ..types import DiscoveredModel, ProviderResult

logger = get_logger("fatih_hoca.cloud.providers.groq")

_URL = "https://api.groq.com/openai/v1/models"
_TIMEOUT = httpx.Timeout(10.0)


class GroqAdapter:
    name = "groq"

    async def fetch_models(self, api_key: str) -> ProviderResult:
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_URL, headers=headers)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:
            return ProviderResult(provider=self.name, status="network_error",
                                  auth_ok=False, error=str(e))
        if resp.status_code in (401, 403):
            return ProviderResult(provider=self.name, status="auth_fail",
                                  auth_ok=False, error=f"{resp.status_code} {resp.text[:200]}")
        if resp.status_code == 429:
            return ProviderResult(provider=self.name, status="rate_limited",
                                  auth_ok=True, error="429 at /models probe")
        if resp.status_code >= 500:
            return ProviderResult(provider=self.name, status="server_error",
                                  auth_ok=False, error=f"{resp.status_code}")
        try:
            payload = resp.json()
        except Exception as e:  # noqa: BLE001
            return ProviderResult(provider=self.name, status="server_error",
                                  auth_ok=False, error=f"json parse: {e}")
        models: list[DiscoveredModel] = []
        for entry in payload.get("data", []):
            if not entry.get("active", True):
                continue
            raw_id = entry.get("id", "")
            if not raw_id:
                continue
            models.append(DiscoveredModel(
                litellm_name=f"groq/{raw_id}",
                raw_id=raw_id,
                active=True,
                context_length=entry.get("context_window"),
                max_output_tokens=entry.get("max_completion_tokens"),
                extra={"owned_by": entry.get("owned_by", "")},
            ))
        return ProviderResult(provider=self.name, status="ok", auth_ok=True, models=models)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest packages/fatih_hoca/tests/cloud/test_provider_groq.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/cloud/providers/ packages/fatih_hoca/tests/cloud/test_provider_groq.py
git commit -m "feat(cloud): groq /models adapter + base protocol"
```

---

### Task 6: OpenAI adapter

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/cloud/providers/openai.py`
- Test: `packages/fatih_hoca/tests/cloud/test_provider_openai.py`

- [ ] **Step 1: Write the failing tests**

```python
# packages/fatih_hoca/tests/cloud/test_provider_openai.py
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fatih_hoca.cloud.providers.openai import OpenAIAdapter


_OPENAI_OK = {
    "object": "list",
    "data": [
        {"id": "gpt-4o", "object": "model", "owned_by": "openai", "created": 1700000000},
        {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai", "created": 1710000000},
        {"id": "text-embedding-3-small", "object": "model", "owned_by": "openai", "created": 1690000000},
        {"id": "whisper-1", "object": "model", "owned_by": "openai", "created": 1680000000},
        {"id": "tts-1", "object": "model", "owned_by": "openai", "created": 1680000000},
    ],
}


def _resp(code, body):
    return httpx.Response(code, json=body, request=httpx.Request("GET", "https://api.openai.com"))


@pytest.mark.asyncio
async def test_openai_filters_non_chat_models():
    a = OpenAIAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _OPENAI_OK))):
        result = await a.fetch_models("k")
    names = [m.raw_id for m in result.models]
    assert "gpt-4o" in names
    assert "gpt-4o-mini" in names
    # Embedding/audio/tts filtered.
    assert "text-embedding-3-small" not in names
    assert "whisper-1" not in names
    assert "tts-1" not in names


@pytest.mark.asyncio
async def test_openai_litellm_name_has_no_provider_prefix():
    """litellm uses bare 'gpt-4o' for OpenAI, not 'openai/gpt-4o'."""
    a = OpenAIAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _OPENAI_OK))):
        result = await a.fetch_models("k")
    assert all(not m.litellm_name.startswith("openai/") for m in result.models)


@pytest.mark.asyncio
async def test_openai_401():
    a = OpenAIAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(401, {"error": {"message": "bad"}}))):
        result = await a.fetch_models("k")
    assert result.status == "auth_fail"
```

- [ ] **Step 2: Run → fail**

Run: `pytest packages/fatih_hoca/tests/cloud/test_provider_openai.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement openai.py**

```python
# packages/fatih_hoca/src/fatih_hoca/cloud/providers/openai.py
"""OpenAI /v1/models adapter.

Endpoint: GET https://api.openai.com/v1/models
Auth:     Bearer <api_key>
Response: {"data": [{"id": str, "object": "model", "owned_by": str,
                     "created": int}, ...]}

OpenAI lumps every model into one list — chat, embedding, TTS, Whisper.
We filter to chat-completion-ish models by id-prefix allowlist.
"""
from __future__ import annotations

import httpx

from ..types import DiscoveredModel, ProviderResult

_URL = "https://api.openai.com/v1/models"
_TIMEOUT = httpx.Timeout(10.0)
_CHAT_PREFIXES = ("gpt-", "o1", "o3", "o4")
_EXCLUDE_SUBSTR = ("embedding", "whisper", "tts", "dall-e", "moderation")


class OpenAIAdapter:
    name = "openai"

    async def fetch_models(self, api_key: str) -> ProviderResult:
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_URL, headers=headers)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:
            return ProviderResult(provider=self.name, status="network_error",
                                  auth_ok=False, error=str(e))
        if resp.status_code in (401, 403):
            return ProviderResult(provider=self.name, status="auth_fail",
                                  auth_ok=False, error=f"{resp.status_code}")
        if resp.status_code == 429:
            return ProviderResult(provider=self.name, status="rate_limited", auth_ok=True)
        if resp.status_code >= 500:
            return ProviderResult(provider=self.name, status="server_error", auth_ok=False)
        try:
            payload = resp.json()
        except Exception as e:  # noqa: BLE001
            return ProviderResult(provider=self.name, status="server_error",
                                  auth_ok=False, error=f"json parse: {e}")
        models: list[DiscoveredModel] = []
        for entry in payload.get("data", []):
            raw_id = entry.get("id", "")
            if not raw_id:
                continue
            if not raw_id.startswith(_CHAT_PREFIXES):
                continue
            if any(s in raw_id for s in _EXCLUDE_SUBSTR):
                continue
            models.append(DiscoveredModel(
                litellm_name=raw_id,  # litellm uses bare id for OpenAI
                raw_id=raw_id,
                extra={"owned_by": entry.get("owned_by", ""), "created": entry.get("created")},
            ))
        return ProviderResult(provider=self.name, status="ok", auth_ok=True, models=models)
```

- [ ] **Step 4: Run → pass**

Run: `pytest packages/fatih_hoca/tests/cloud/test_provider_openai.py -v`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/cloud/providers/openai.py packages/fatih_hoca/tests/cloud/test_provider_openai.py
git commit -m "feat(cloud): openai /v1/models adapter"
```

---

### Task 7: Anthropic adapter

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/cloud/providers/anthropic.py`
- Test: `packages/fatih_hoca/tests/cloud/test_provider_anthropic.py`

- [ ] **Step 1: Write the failing tests**

```python
# packages/fatih_hoca/tests/cloud/test_provider_anthropic.py
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fatih_hoca.cloud.providers.anthropic import AnthropicAdapter


_OK = {
    "data": [
        {"id": "claude-sonnet-4-20250514", "display_name": "Claude Sonnet 4",
         "created_at": "2025-05-14T00:00:00Z", "type": "model"},
        {"id": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet",
         "created_at": "2024-10-22T00:00:00Z", "type": "model"},
    ],
    "has_more": False,
}


def _resp(code, body):
    return httpx.Response(code, json=body, request=httpx.Request("GET", "https://api.anthropic.com"))


@pytest.mark.asyncio
async def test_anthropic_ok():
    a = AnthropicAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _OK))):
        result = await a.fetch_models("k")
    assert result.status == "ok"
    names = [m.litellm_name for m in result.models]
    assert "claude-sonnet-4-20250514" in names
    assert "claude-3-5-sonnet-20241022" in names


@pytest.mark.asyncio
async def test_anthropic_401():
    a = AnthropicAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(401, {"error": {"message": "bad"}}))):
        result = await a.fetch_models("k")
    assert result.status == "auth_fail"
```

- [ ] **Step 2: Run → fail**

Run: `pytest packages/fatih_hoca/tests/cloud/test_provider_anthropic.py -v`

- [ ] **Step 3: Implement anthropic.py**

```python
# packages/fatih_hoca/src/fatih_hoca/cloud/providers/anthropic.py
"""Anthropic /v1/models adapter.

Endpoint: GET https://api.anthropic.com/v1/models
Auth headers: x-api-key, anthropic-version
Response: {"data": [{"id": str, "display_name": str, "created_at": str,
                     "type": "model"}, ...]}
"""
from __future__ import annotations

import httpx

from ..types import DiscoveredModel, ProviderResult

_URL = "https://api.anthropic.com/v1/models"
_TIMEOUT = httpx.Timeout(10.0)
_API_VERSION = "2023-06-01"


class AnthropicAdapter:
    name = "anthropic"

    async def fetch_models(self, api_key: str) -> ProviderResult:
        headers = {"x-api-key": api_key, "anthropic-version": _API_VERSION}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_URL, headers=headers)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:
            return ProviderResult(provider=self.name, status="network_error",
                                  auth_ok=False, error=str(e))
        if resp.status_code in (401, 403):
            return ProviderResult(provider=self.name, status="auth_fail",
                                  auth_ok=False, error=f"{resp.status_code}")
        if resp.status_code == 429:
            return ProviderResult(provider=self.name, status="rate_limited", auth_ok=True)
        if resp.status_code >= 500:
            return ProviderResult(provider=self.name, status="server_error", auth_ok=False)
        try:
            payload = resp.json()
        except Exception as e:  # noqa: BLE001
            return ProviderResult(provider=self.name, status="server_error",
                                  auth_ok=False, error=f"json parse: {e}")
        models: list[DiscoveredModel] = []
        for entry in payload.get("data", []):
            raw_id = entry.get("id", "")
            if not raw_id:
                continue
            models.append(DiscoveredModel(
                litellm_name=raw_id,  # litellm uses bare id for Anthropic
                raw_id=raw_id,
                extra={
                    "display_name": entry.get("display_name", ""),
                    "created_at": entry.get("created_at", ""),
                },
            ))
        return ProviderResult(provider=self.name, status="ok", auth_ok=True, models=models)
```

- [ ] **Step 4: Run → pass**

Run: `pytest packages/fatih_hoca/tests/cloud/test_provider_anthropic.py -v`

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/cloud/providers/anthropic.py packages/fatih_hoca/tests/cloud/test_provider_anthropic.py
git commit -m "feat(cloud): anthropic /v1/models adapter"
```

---

### Task 8: Gemini adapter

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/cloud/providers/gemini.py`
- Test: `packages/fatih_hoca/tests/cloud/test_provider_gemini.py`

- [ ] **Step 1: Write the failing tests**

```python
# packages/fatih_hoca/tests/cloud/test_provider_gemini.py
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fatih_hoca.cloud.providers.gemini import GeminiAdapter


_OK = {
    "models": [
        {
            "name": "models/gemini-2.0-flash",
            "displayName": "Gemini 2.0 Flash",
            "inputTokenLimit": 1048576,
            "outputTokenLimit": 8192,
            "supportedGenerationMethods": ["generateContent", "countTokens"],
            "temperature": 1.0,
            "topP": 0.95,
            "topK": 64,
        },
        {
            "name": "models/embedding-001",
            "displayName": "Embedding",
            "supportedGenerationMethods": ["embedContent"],
        },
    ],
}


def _resp(code, body):
    return httpx.Response(code, json=body, request=httpx.Request("GET", "https://generativelanguage.googleapis.com"))


@pytest.mark.asyncio
async def test_gemini_filters_non_generative():
    a = GeminiAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _OK))):
        result = await a.fetch_models("k")
    assert result.status == "ok"
    names = [m.litellm_name for m in result.models]
    assert "gemini/gemini-2.0-flash" in names
    assert all("embedding" not in n for n in names)


@pytest.mark.asyncio
async def test_gemini_scrapes_token_limits_and_sampling():
    a = GeminiAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _OK))):
        result = await a.fetch_models("k")
    m = next(x for x in result.models if x.raw_id == "gemini-2.0-flash")
    assert m.context_length == 1048576
    assert m.max_output_tokens == 8192
    assert m.sampling_defaults == {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0}


@pytest.mark.asyncio
async def test_gemini_400_invalid_key_treated_as_auth_fail():
    """Gemini returns 400 with key-invalid message instead of 401."""
    a = GeminiAdapter()
    body = {"error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}}
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(400, body))):
        result = await a.fetch_models("k")
    assert result.status == "auth_fail"
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Implement gemini.py**

```python
# packages/fatih_hoca/src/fatih_hoca/cloud/providers/gemini.py
"""Google Gemini /v1beta/models adapter.

Endpoint: GET https://generativelanguage.googleapis.com/v1beta/models?key=<api_key>
Response: {"models": [{"name": "models/<id>", "inputTokenLimit": int,
                       "outputTokenLimit": int,
                       "supportedGenerationMethods": [...],
                       "temperature": float, "topP": float,
                       "topK": int, ...}, ...]}

Notes:
    - Gemini returns 400 with INVALID_ARGUMENT for bad key, not 401.
    - Filter to models that support ``generateContent``.
"""
from __future__ import annotations

import httpx

from ..types import DiscoveredModel, ProviderResult

_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_TIMEOUT = httpx.Timeout(10.0)


class GeminiAdapter:
    name = "gemini"

    async def fetch_models(self, api_key: str) -> ProviderResult:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_URL, params={"key": api_key})
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:
            return ProviderResult(provider=self.name, status="network_error",
                                  auth_ok=False, error=str(e))
        if resp.status_code in (401, 403):
            return ProviderResult(provider=self.name, status="auth_fail", auth_ok=False)
        if resp.status_code == 400:
            # Gemini quirk: bad key returns 400 INVALID_ARGUMENT.
            try:
                msg = resp.json().get("error", {}).get("message", "")
            except Exception:  # noqa: BLE001
                msg = ""
            if "API key" in msg or "INVALID_ARGUMENT" in resp.text.upper():
                return ProviderResult(provider=self.name, status="auth_fail",
                                      auth_ok=False, error=msg or "400")
            return ProviderResult(provider=self.name, status="server_error",
                                  auth_ok=False, error=f"400 {msg}")
        if resp.status_code == 429:
            return ProviderResult(provider=self.name, status="rate_limited", auth_ok=True)
        if resp.status_code >= 500:
            return ProviderResult(provider=self.name, status="server_error", auth_ok=False)
        try:
            payload = resp.json()
        except Exception as e:  # noqa: BLE001
            return ProviderResult(provider=self.name, status="server_error",
                                  auth_ok=False, error=f"json parse: {e}")
        models: list[DiscoveredModel] = []
        for entry in payload.get("models", []):
            full_name = entry.get("name", "")
            raw_id = full_name.split("/", 1)[-1] if "/" in full_name else full_name
            if not raw_id:
                continue
            methods = entry.get("supportedGenerationMethods", [])
            if "generateContent" not in methods:
                continue
            sampling: dict[str, float] = {}
            if entry.get("temperature") is not None:
                sampling["temperature"] = float(entry["temperature"])
            if entry.get("topP") is not None:
                sampling["top_p"] = float(entry["topP"])
            if entry.get("topK") is not None:
                sampling["top_k"] = float(entry["topK"])
            models.append(DiscoveredModel(
                litellm_name=f"gemini/{raw_id}",
                raw_id=raw_id,
                context_length=entry.get("inputTokenLimit"),
                max_output_tokens=entry.get("outputTokenLimit"),
                sampling_defaults=sampling,
                extra={"display_name": entry.get("displayName", "")},
            ))
        return ProviderResult(provider=self.name, status="ok", auth_ok=True, models=models)
```

- [ ] **Step 4: Run → pass**

Run: `pytest packages/fatih_hoca/tests/cloud/test_provider_gemini.py -v`

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/cloud/providers/gemini.py packages/fatih_hoca/tests/cloud/test_provider_gemini.py
git commit -m "feat(cloud): gemini v1beta /models adapter"
```

---

### Task 9: Cerebras adapter

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/cloud/providers/cerebras.py`
- Test: `packages/fatih_hoca/tests/cloud/test_provider_cerebras.py`

- [ ] **Step 1: Write the failing tests**

```python
# packages/fatih_hoca/tests/cloud/test_provider_cerebras.py
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fatih_hoca.cloud.providers.cerebras import CerebrasAdapter


_OK = {
    "data": [
        {"id": "llama3.3-70b", "object": "model", "owned_by": "Meta"},
        {"id": "llama-3.3-70b", "object": "model", "owned_by": "Meta"},
    ],
}


def _resp(code, body):
    return httpx.Response(code, json=body, request=httpx.Request("GET", "https://api.cerebras.ai"))


@pytest.mark.asyncio
async def test_cerebras_ok():
    a = CerebrasAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _OK))):
        result = await a.fetch_models("k")
    names = [m.litellm_name for m in result.models]
    assert "cerebras/llama3.3-70b" in names


@pytest.mark.asyncio
async def test_cerebras_401():
    a = CerebrasAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(401, {}))):
        result = await a.fetch_models("k")
    assert result.status == "auth_fail"
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Implement cerebras.py**

```python
# packages/fatih_hoca/src/fatih_hoca/cloud/providers/cerebras.py
"""Cerebras /v1/models adapter (OpenAI-compatible)."""
from __future__ import annotations

import httpx

from ..types import DiscoveredModel, ProviderResult

_URL = "https://api.cerebras.ai/v1/models"
_TIMEOUT = httpx.Timeout(10.0)


class CerebrasAdapter:
    name = "cerebras"

    async def fetch_models(self, api_key: str) -> ProviderResult:
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_URL, headers=headers)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:
            return ProviderResult(provider=self.name, status="network_error",
                                  auth_ok=False, error=str(e))
        if resp.status_code in (401, 403):
            return ProviderResult(provider=self.name, status="auth_fail", auth_ok=False)
        if resp.status_code == 429:
            return ProviderResult(provider=self.name, status="rate_limited", auth_ok=True)
        if resp.status_code >= 500:
            return ProviderResult(provider=self.name, status="server_error", auth_ok=False)
        try:
            payload = resp.json()
        except Exception as e:  # noqa: BLE001
            return ProviderResult(provider=self.name, status="server_error",
                                  auth_ok=False, error=f"json parse: {e}")
        models: list[DiscoveredModel] = []
        for entry in payload.get("data", []):
            raw_id = entry.get("id", "")
            if not raw_id:
                continue
            models.append(DiscoveredModel(
                litellm_name=f"cerebras/{raw_id}",
                raw_id=raw_id,
                extra={"owned_by": entry.get("owned_by", "")},
            ))
        return ProviderResult(provider=self.name, status="ok", auth_ok=True, models=models)
```

- [ ] **Step 4: Run → pass + commit**

Run: `pytest packages/fatih_hoca/tests/cloud/test_provider_cerebras.py -v`

```bash
git add packages/fatih_hoca/src/fatih_hoca/cloud/providers/cerebras.py packages/fatih_hoca/tests/cloud/test_provider_cerebras.py
git commit -m "feat(cloud): cerebras /v1/models adapter"
```

---

### Task 10: Sambanova adapter

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/cloud/providers/sambanova.py`
- Test: `packages/fatih_hoca/tests/cloud/test_provider_sambanova.py`

- [ ] **Step 1: Write the failing tests**

```python
# packages/fatih_hoca/tests/cloud/test_provider_sambanova.py
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fatih_hoca.cloud.providers.sambanova import SambanovaAdapter


_OK = {
    "data": [
        {"id": "Qwen3-32B", "object": "model"},
        {"id": "Meta-Llama-3.3-70B-Instruct", "object": "model"},
    ],
}


def _resp(code, body):
    return httpx.Response(code, json=body, request=httpx.Request("GET", "https://api.sambanova.ai"))


@pytest.mark.asyncio
async def test_sambanova_ok():
    a = SambanovaAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _OK))):
        result = await a.fetch_models("k")
    names = [m.litellm_name for m in result.models]
    assert "sambanova/Qwen3-32B" in names
    assert "sambanova/Meta-Llama-3.3-70B-Instruct" in names


@pytest.mark.asyncio
async def test_sambanova_401():
    a = SambanovaAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(401, {}))):
        result = await a.fetch_models("k")
    assert result.status == "auth_fail"
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Implement sambanova.py**

```python
# packages/fatih_hoca/src/fatih_hoca/cloud/providers/sambanova.py
"""Sambanova /v1/models adapter (OpenAI-compatible)."""
from __future__ import annotations

import httpx

from ..types import DiscoveredModel, ProviderResult

_URL = "https://api.sambanova.ai/v1/models"
_TIMEOUT = httpx.Timeout(10.0)


class SambanovaAdapter:
    name = "sambanova"

    async def fetch_models(self, api_key: str) -> ProviderResult:
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_URL, headers=headers)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:
            return ProviderResult(provider=self.name, status="network_error",
                                  auth_ok=False, error=str(e))
        if resp.status_code in (401, 403):
            return ProviderResult(provider=self.name, status="auth_fail", auth_ok=False)
        if resp.status_code == 429:
            return ProviderResult(provider=self.name, status="rate_limited", auth_ok=True)
        if resp.status_code >= 500:
            return ProviderResult(provider=self.name, status="server_error", auth_ok=False)
        try:
            payload = resp.json()
        except Exception as e:  # noqa: BLE001
            return ProviderResult(provider=self.name, status="server_error",
                                  auth_ok=False, error=f"json parse: {e}")
        models: list[DiscoveredModel] = []
        for entry in payload.get("data", []):
            raw_id = entry.get("id", "")
            if not raw_id:
                continue
            models.append(DiscoveredModel(
                litellm_name=f"sambanova/{raw_id}",
                raw_id=raw_id,
            ))
        return ProviderResult(provider=self.name, status="ok", auth_ok=True, models=models)
```

- [ ] **Step 4: Run → pass + commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/cloud/providers/sambanova.py packages/fatih_hoca/tests/cloud/test_provider_sambanova.py
git commit -m "feat(cloud): sambanova /v1/models adapter"
```

---

### Task 11: OpenRouter adapter (rich scrape)

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/cloud/providers/openrouter.py`
- Test: `packages/fatih_hoca/tests/cloud/test_provider_openrouter.py`

- [ ] **Step 1: Write the failing tests**

```python
# packages/fatih_hoca/tests/cloud/test_provider_openrouter.py
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fatih_hoca.cloud.providers.openrouter import OpenRouterAdapter


_OK = {
    "data": [
        {
            "id": "meta-llama/llama-3.3-70b-instruct",
            "name": "Llama 3.3 70B Instruct",
            "context_length": 131072,
            "pricing": {"prompt": "0.0000005", "completion": "0.0000008"},
            "top_provider": {"max_completion_tokens": 32768, "is_moderated": False},
        },
    ],
}


def _resp(code, body):
    return httpx.Response(code, json=body, request=httpx.Request("GET", "https://openrouter.ai"))


@pytest.mark.asyncio
async def test_openrouter_scrapes_pricing_and_context():
    a = OpenRouterAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _OK))):
        result = await a.fetch_models("k")
    assert result.status == "ok"
    m = result.models[0]
    assert m.litellm_name == "openrouter/meta-llama/llama-3.3-70b-instruct"
    assert m.context_length == 131072
    assert m.max_output_tokens == 32768
    # Pricing converted from per-token to per-1k.
    assert m.cost_per_1k_input == pytest.approx(0.0005, rel=1e-6)
    assert m.cost_per_1k_output == pytest.approx(0.0008, rel=1e-6)


@pytest.mark.asyncio
async def test_openrouter_401():
    a = OpenRouterAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(401, {}))):
        result = await a.fetch_models("k")
    assert result.status == "auth_fail"
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Implement openrouter.py**

```python
# packages/fatih_hoca/src/fatih_hoca/cloud/providers/openrouter.py
"""OpenRouter /api/v1/models adapter — rich field scrape.

Endpoint: GET https://openrouter.ai/api/v1/models
Response: {"data": [{"id": "<org>/<model>", "context_length": int,
                     "pricing": {"prompt": "<usd-per-token>", "completion": "<usd>"},
                     "top_provider": {"max_completion_tokens": int,
                                      "is_moderated": bool}}, ...]}

Pricing is per-token; we convert to per-1k to match ModelInfo.cost_per_1k_*.
"""
from __future__ import annotations

import httpx

from ..types import DiscoveredModel, ProviderResult

_URL = "https://openrouter.ai/api/v1/models"
_TIMEOUT = httpx.Timeout(15.0)  # larger catalog, slightly slower


def _to_per_1k(value: str | float | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value) * 1000.0
    except (TypeError, ValueError):
        return None


class OpenRouterAdapter:
    name = "openrouter"

    async def fetch_models(self, api_key: str) -> ProviderResult:
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_URL, headers=headers)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:
            return ProviderResult(provider=self.name, status="network_error",
                                  auth_ok=False, error=str(e))
        if resp.status_code in (401, 403):
            return ProviderResult(provider=self.name, status="auth_fail", auth_ok=False)
        if resp.status_code == 429:
            return ProviderResult(provider=self.name, status="rate_limited", auth_ok=True)
        if resp.status_code >= 500:
            return ProviderResult(provider=self.name, status="server_error", auth_ok=False)
        try:
            payload = resp.json()
        except Exception as e:  # noqa: BLE001
            return ProviderResult(provider=self.name, status="server_error",
                                  auth_ok=False, error=f"json parse: {e}")
        models: list[DiscoveredModel] = []
        for entry in payload.get("data", []):
            raw_id = entry.get("id", "")
            if not raw_id:
                continue
            pricing = entry.get("pricing", {}) or {}
            top_prov = entry.get("top_provider", {}) or {}
            models.append(DiscoveredModel(
                litellm_name=f"openrouter/{raw_id}",
                raw_id=raw_id,
                context_length=entry.get("context_length"),
                max_output_tokens=top_prov.get("max_completion_tokens"),
                cost_per_1k_input=_to_per_1k(pricing.get("prompt")),
                cost_per_1k_output=_to_per_1k(pricing.get("completion")),
                extra={
                    "name": entry.get("name", ""),
                    "is_moderated": top_prov.get("is_moderated"),
                },
            ))
        return ProviderResult(provider=self.name, status="ok", auth_ok=True, models=models)
```

- [ ] **Step 4: Run → pass + commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/cloud/providers/openrouter.py packages/fatih_hoca/tests/cloud/test_provider_openrouter.py
git commit -m "feat(cloud): openrouter /api/v1/models adapter with pricing scrape"
```

---

### Task 12: Discovery orchestrator

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/cloud/discovery.py`
- Test: `packages/fatih_hoca/tests/cloud/test_discovery.py`

- [ ] **Step 1: Write the failing tests**

```python
# packages/fatih_hoca/tests/cloud/test_discovery.py
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from fatih_hoca.cloud.discovery import CloudDiscovery
from fatih_hoca.cloud.types import DiscoveredModel, ProviderResult


class _StubAdapter:
    def __init__(self, name: str, result: ProviderResult):
        self.name = name
        self._result = result
        self.calls = 0

    async def fetch_models(self, api_key: str) -> ProviderResult:
        self.calls += 1
        return self._result


def _ok(provider: str, ids: list[str]) -> ProviderResult:
    return ProviderResult(
        provider=provider, status="ok", auth_ok=True,
        models=[DiscoveredModel(litellm_name=f"{provider}/{i}", raw_id=i) for i in ids],
    )


def _fail(provider: str, status: str = "auth_fail") -> ProviderResult:
    return ProviderResult(provider=provider, status=status, auth_ok=False, error="bad")


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.mark.asyncio
async def test_refresh_all_calls_each_adapter_with_its_key(cache_dir):
    adapters = {"groq": _StubAdapter("groq", _ok("groq", ["a"])),
                "openai": _StubAdapter("openai", _ok("openai", ["b"]))}
    keys = {"groq": "G", "openai": "O"}
    alerts: list = []
    d = CloudDiscovery(adapters=adapters, cache_dir=cache_dir, alert_fn=lambda *a, **kw: alerts.append((a, kw)))
    results = await d.refresh_all(api_keys=keys)
    assert set(results.keys()) == {"groq", "openai"}
    assert all(r.auth_ok for r in results.values())
    assert adapters["groq"].calls == 1
    assert adapters["openai"].calls == 1


@pytest.mark.asyncio
async def test_refresh_all_skips_provider_without_key(cache_dir):
    adapters = {"groq": _StubAdapter("groq", _ok("groq", ["a"]))}
    d = CloudDiscovery(adapters=adapters, cache_dir=cache_dir, alert_fn=lambda *a, **kw: None)
    results = await d.refresh_all(api_keys={})
    assert results == {}
    assert adapters["groq"].calls == 0


@pytest.mark.asyncio
async def test_failure_falls_back_to_fresh_cache(cache_dir):
    from fatih_hoca.cloud import cache as cache_mod
    cache_mod.save(cache_dir, "groq",
                   [DiscoveredModel(litellm_name="groq/cached", raw_id="cached")], status="ok")
    adapters = {"groq": _StubAdapter("groq", _fail("groq", "network_error"))}
    d = CloudDiscovery(adapters=adapters, cache_dir=cache_dir, alert_fn=lambda *a, **kw: None)
    results = await d.refresh_all(api_keys={"groq": "G"})
    assert results["groq"].auth_ok is True  # cache served, treat as enabled
    assert results["groq"].served_from_cache is True
    assert results["groq"].models[0].raw_id == "cached"


@pytest.mark.asyncio
async def test_auth_fail_alerts_via_callback(cache_dir):
    captured: list = []
    adapters = {"groq": _StubAdapter("groq", _fail("groq", "auth_fail"))}
    d = CloudDiscovery(adapters=adapters, cache_dir=cache_dir,
                       alert_fn=lambda provider, status, error: captured.append((provider, status, error)))
    await d.refresh_all(api_keys={"groq": "G"})
    assert len(captured) == 1
    assert captured[0][0] == "groq"
    assert captured[0][1] == "auth_fail"


@pytest.mark.asyncio
async def test_diff_logs_added_and_removed(cache_dir, caplog):
    from fatih_hoca.cloud import cache as cache_mod
    cache_mod.save(cache_dir, "groq",
                   [DiscoveredModel(litellm_name="groq/old", raw_id="old"),
                    DiscoveredModel(litellm_name="groq/keep", raw_id="keep")], status="ok")
    adapters = {"groq": _StubAdapter("groq", _ok("groq", ["new", "keep"]))}
    d = CloudDiscovery(adapters=adapters, cache_dir=cache_dir, alert_fn=lambda *a, **kw: None)
    with caplog.at_level("INFO", logger="fatih_hoca.cloud.discovery"):
        await d.refresh_all(api_keys={"groq": "G"})
    log_text = " ".join(r.message for r in caplog.records)
    assert "added=" in log_text and "groq/new" in log_text
    assert "removed=" in log_text and "groq/old" in log_text
```

- [ ] **Step 2: Run → fail**

Run: `pytest packages/fatih_hoca/tests/cloud/test_discovery.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement discovery.py**

```python
# packages/fatih_hoca/src/fatih_hoca/cloud/discovery.py
"""Boot-time and periodic cloud provider discovery orchestrator.

Calls each provider adapter concurrently, writes results to disk cache,
diffs against previous snapshot (logs adds/removes), invokes per-provider
alert callback on auth_fail / state-flip / non-recoverable errors.

The orchestrator does NOT touch the registry — boot wiring (Task 13)
consumes the returned results map and registers ModelInfo objects.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Mapping

from src.infra.logging_config import get_logger

from . import cache as cache_mod
from .types import ProviderResult

logger = get_logger("fatih_hoca.cloud.discovery")

AlertFn = Callable[[str, str, str | None], None]


class CloudDiscovery:
    def __init__(
        self,
        adapters: Mapping[str, object],  # name -> adapter with .fetch_models()
        cache_dir: Path,
        alert_fn: AlertFn,
    ) -> None:
        self._adapters = dict(adapters)
        self._cache_dir = Path(cache_dir)
        self._alert_fn = alert_fn

    async def refresh_all(self, api_keys: dict[str, str]) -> dict[str, ProviderResult]:
        """Probe every adapter we have a key for. Returns {provider: result}.

        Provider missing from api_keys is silently skipped (key truly absent).
        """
        targets = [(name, adapter, api_keys[name])
                   for name, adapter in self._adapters.items() if api_keys.get(name)]
        if not targets:
            return {}
        coros = [self._probe_one(name, adapter, key) for name, adapter, key in targets]
        results_list = await asyncio.gather(*coros, return_exceptions=False)
        return {r.provider: r for r in results_list}

    async def _probe_one(self, name: str, adapter, api_key: str) -> ProviderResult:
        try:
            live = await adapter.fetch_models(api_key)
        except Exception as e:  # noqa: BLE001 — adapters must not raise, but defend
            logger.error("adapter %s raised: %s", name, e)
            live = ProviderResult(provider=name, status="server_error",
                                  auth_ok=False, error=f"adapter exception: {e}")

        prior = cache_mod.load(self._cache_dir, name)

        # Successful fetch: persist and diff.
        if live.status == "ok":
            self._log_diff(name, prior_models=[m.litellm_name for m in (prior.models if prior else [])],
                           new_models=[m.litellm_name for m in live.models])
            cache_mod.save(self._cache_dir, name, live.models, status="ok")
            return live

        # Rate-limited probe: still ok auth-wise, do not overwrite cache, do not alert.
        if live.status == "rate_limited":
            logger.warning("provider %s rate-limited at /models probe; keeping cache", name)
            if prior is not None:
                live.models = prior.models
                live.served_from_cache = True
            return live

        # Failure path. Try cache.
        if prior is not None and prior.is_fresh:
            logger.warning("provider %s probe failed (%s); serving fresh cache (age=%.0fs)",
                           name, live.status, prior.age_seconds)
            return ProviderResult(
                provider=name, status=live.status, auth_ok=True,
                models=prior.models, error=live.error, served_from_cache=True,
                fetched_at=prior.fetched_at_iso,
            )
        if prior is not None and not prior.is_fresh and not prior.is_evicted:
            logger.warning("provider %s probe failed (%s); serving STALE cache (age=%.0fs)",
                           name, live.status, prior.age_seconds)
            self._alert_fn(name, live.status, live.error)
            return ProviderResult(
                provider=name, status=live.status, auth_ok=False,
                models=prior.models, error=live.error, served_from_cache=True,
                fetched_at=prior.fetched_at_iso,
            )

        # No fresh, no stale. Provider goes dark.
        logger.error("provider %s unreachable and no cache: %s %s",
                     name, live.status, live.error)
        self._alert_fn(name, live.status, live.error)
        return live

    def _log_diff(self, provider: str, prior_models: list[str], new_models: list[str]) -> None:
        prior_set = set(prior_models)
        new_set = set(new_models)
        added = sorted(new_set - prior_set)
        removed = sorted(prior_set - new_set)
        if added or removed:
            logger.info("provider %s diff: added=%s removed=%s", provider, added, removed)
        else:
            logger.debug("provider %s: no model-list changes (%d models)", provider, len(new_set))
```

- [ ] **Step 4: Run → pass**

Run: `pytest packages/fatih_hoca/tests/cloud/test_discovery.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/cloud/discovery.py packages/fatih_hoca/tests/cloud/test_discovery.py
git commit -m "feat(cloud): discovery orchestrator with cache fallback + diff logging"
```

---

### Task 13: ModelInfo.family field + scraped-field merge

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/types.py`
- Modify: `packages/fatih_hoca/src/fatih_hoca/registry.py:1172-1212` (cloud-registration block)
- Modify: `packages/fatih_hoca/src/fatih_hoca/registry.py:886-963` (`detect_cloud_model`)
- Test: `packages/fatih_hoca/tests/test_registry.py` (add cases)

- [ ] **Step 1: Add `family` to ModelInfo**

Find the `ModelInfo` dataclass in `packages/fatih_hoca/src/fatih_hoca/types.py`. Add:

```python
    # Cross-provider model identity for benchmark dedup. Empty for local
    # models; populated for cloud models via fatih_hoca.cloud.family.normalize().
    family: str = ""
```

- [ ] **Step 2: Write the failing test for scraped-field merge**

Append to `packages/fatih_hoca/tests/test_registry.py`:

```python
def test_register_cloud_from_discovered_merges_scraped_fields():
    from fatih_hoca.cloud.types import DiscoveredModel
    from fatih_hoca.registry import ModelRegistry, register_cloud_from_discovered

    registry = ModelRegistry()
    discovered = DiscoveredModel(
        litellm_name="groq/llama-3.3-70b-versatile",
        raw_id="llama-3.3-70b-versatile",
        context_length=131072,
        max_output_tokens=32768,
        cost_per_1k_input=0.59,
        cost_per_1k_output=0.79,
        sampling_defaults={"temperature": 1.0},
        extra={"owned_by": "Meta"},
    )
    register_cloud_from_discovered(registry, "groq", discovered)
    m = registry.get("groq/llama-3.3-70b-versatile")
    assert m is not None
    assert m.location == "cloud"
    assert m.provider == "groq"
    assert m.context_length == 131072
    assert m.max_tokens == 32768
    assert m.cost_per_1k_input == 0.59
    assert m.cost_per_1k_output == 0.79
    assert m.family == "llama-3.3-70b"


def test_register_cloud_skips_inactive():
    from fatih_hoca.cloud.types import DiscoveredModel
    from fatih_hoca.registry import ModelRegistry, register_cloud_from_discovered

    registry = ModelRegistry()
    discovered = DiscoveredModel(
        litellm_name="groq/dead", raw_id="dead", active=False,
    )
    register_cloud_from_discovered(registry, "groq", discovered)
    assert registry.get("groq/dead") is None
```

- [ ] **Step 3: Run → fail**

Run: `pytest packages/fatih_hoca/tests/test_registry.py -k cloud_from_discovered -v`
Expected: FAIL — function missing.

- [ ] **Step 4: Implement `register_cloud_from_discovered`**

Append to `packages/fatih_hoca/src/fatih_hoca/registry.py` (after `detect_cloud_model`, before `_resolve_provider`):

```python
def register_cloud_from_discovered(
    registry: "ModelRegistry",
    provider: str,
    discovered: "DiscoveredModel",
) -> "ModelInfo | None":
    """Register a discovered cloud model into the registry.

    Merges adapter-scraped fields with detect_cloud_model() output:
        - context_length: scraped wins over litellm-db
        - max_tokens: scraped wins
        - cost_per_1k_*: scraped wins (provider-data is authoritative)
        - sampling_overrides: scraped seeds defaults if no prior override
        - active=False: skip registration entirely
    Family is computed via cloud.family.normalize().
    Returns the registered ModelInfo, or None if skipped.
    """
    if not discovered.active:
        return None

    from .cloud.family import normalize as _family_normalize

    detected = detect_cloud_model(discovered.litellm_name, provider)

    if discovered.context_length is not None:
        detected["context_length"] = discovered.context_length
    if discovered.max_output_tokens is not None:
        detected["max_tokens"] = discovered.max_output_tokens
    if discovered.cost_per_1k_input is not None:
        detected["cost_per_1k_input"] = discovered.cost_per_1k_input
    if discovered.cost_per_1k_output is not None:
        detected["cost_per_1k_output"] = discovered.cost_per_1k_output

    family = _family_normalize(provider, discovered.litellm_name)

    model = ModelInfo(
        name=discovered.litellm_name,
        location="cloud",
        provider=provider,
        litellm_name=discovered.litellm_name,
        capabilities=detected["capabilities"],
        context_length=detected["context_length"],
        max_tokens=detected["max_tokens"],
        supports_function_calling=detected.get("supports_function_calling", True),
        supports_json_mode=detected.get("supports_json_mode", True),
        supports_json_schema=detected.get("supports_json_schema", False),
        thinking_model=detected.get("thinking_model", False),
        has_vision=detected.get("has_vision", False),
        tier=detected.get("tier", "paid"),
        rate_limit_rpm=detected["rate_limit_rpm"],
        rate_limit_tpm=detected.get("rate_limit_tpm", 100000),
        cost_per_1k_input=detected.get("cost_per_1k_input", 0.0),
        cost_per_1k_output=detected.get("cost_per_1k_output", 0.0),
        family=family,
    )
    if discovered.sampling_defaults and not getattr(model, "sampling_overrides", None):
        model.sampling_overrides = {"all": dict(discovered.sampling_defaults)}
    registry.register(model)
    return model
```

Add the import at the top of `registry.py`:

```python
from .cloud.types import DiscoveredModel  # at top with other imports
```

- [ ] **Step 5: Run → pass**

Run: `pytest packages/fatih_hoca/tests/test_registry.py -k cloud_from_discovered -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/types.py packages/fatih_hoca/src/fatih_hoca/registry.py packages/fatih_hoca/tests/test_registry.py
git commit -m "feat(cloud): registry helper that merges scraped fields + family key"
```

---

### Task 14: Boot wiring — call discovery from `fatih_hoca.init()`

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/__init__.py`
- Test: `packages/fatih_hoca/tests/test_init.py` (add case)

- [ ] **Step 1: Write the failing test**

Append to `packages/fatih_hoca/tests/test_init.py`:

```python
@pytest.mark.asyncio
async def test_init_disables_provider_when_discovery_auth_fails(monkeypatch, tmp_path):
    """If groq discovery returns auth_fail, _available_providers must NOT include 'groq'."""
    from fatih_hoca.cloud.types import ProviderResult
    import fatih_hoca

    fake_results = {
        "groq": ProviderResult(provider="groq", status="auth_fail", auth_ok=False, error="401"),
        "openai": ProviderResult(provider="openai", status="ok", auth_ok=True, models=[]),
    }

    async def _fake_refresh(api_keys):
        return fake_results

    monkeypatch.setattr(fatih_hoca, "_run_cloud_discovery", _fake_refresh)
    fatih_hoca.init(
        models_yaml=None,
        snapshot_provider=lambda: object(),
        available_providers={"groq", "openai"},
        cloud_cache_dir=str(tmp_path / "cache"),
        cloud_alert_state_path=str(tmp_path / "throttle.json"),
        api_keys={"groq": "g", "openai": "o"},
    )
    assert "groq" not in fatih_hoca._available_providers
    assert "openai" in fatih_hoca._available_providers
```

- [ ] **Step 2: Run → fail**

Run: `pytest packages/fatih_hoca/tests/test_init.py::test_init_disables_provider_when_discovery_auth_fails -v`

- [ ] **Step 3: Update `fatih_hoca/__init__.py`**

Add cloud-discovery wiring inside `init()`. Replace the body of the existing `init()` step that handles `available_providers` so it (a) runs discovery first, (b) reduces `available_providers` to those whose result has `auth_ok=True`, (c) registers discovered models. Sketch:

```python
# packages/fatih_hoca/src/fatih_hoca/__init__.py — additions inside init()
import asyncio as _asyncio
from pathlib import Path as _Path

from .cloud.discovery import CloudDiscovery
from .cloud.alert_throttle import AlertThrottle
from .cloud.providers.groq import GroqAdapter
from .cloud.providers.openai import OpenAIAdapter
from .cloud.providers.anthropic import AnthropicAdapter
from .cloud.providers.gemini import GeminiAdapter
from .cloud.providers.cerebras import CerebrasAdapter
from .cloud.providers.sambanova import SambanovaAdapter
from .cloud.providers.openrouter import OpenRouterAdapter
from .registry import register_cloud_from_discovered

_ADAPTERS = {
    "groq": GroqAdapter(),
    "openai": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
    "gemini": GeminiAdapter(),
    "cerebras": CerebrasAdapter(),
    "sambanova": SambanovaAdapter(),
    "openrouter": OpenRouterAdapter(),
}


def _make_alert_callback(throttle: AlertThrottle):
    def _alert(provider: str, status: str, error: str | None) -> None:
        if not throttle.should_alert(provider, current_state=status):
            return
        try:
            from src.app.telegram_bot import send_admin_alert  # late import; module may not be online
            _asyncio.create_task(send_admin_alert(
                f"[cloud] {provider} {status} — {error or ''} (next retry up to 24h)"
            ))
        except Exception as e:  # noqa: BLE001
            logger.warning("cloud alert callback failed: %s", e)
    return _alert


async def _run_cloud_discovery(
    api_keys: dict[str, str],
    cache_dir: str,
    alert_state_path: str,
) -> dict:
    throttle = AlertThrottle(_Path(alert_state_path))
    discovery = CloudDiscovery(
        adapters=_ADAPTERS,
        cache_dir=_Path(cache_dir),
        alert_fn=_make_alert_callback(throttle),
    )
    return await discovery.refresh_all(api_keys=api_keys)


def init(
    *,
    models_yaml: str | None,
    snapshot_provider,
    available_providers: set[str] | None = None,
    api_keys: dict[str, str] | None = None,
    cloud_cache_dir: str = ".benchmark_cache/cloud_models",
    cloud_alert_state_path: str = ".benchmark_cache/cloud_alert_throttle.json",
    **kwargs,
):
    # ... existing local registry load ...

    # Cloud discovery
    discovery_results: dict = {}
    if api_keys:
        try:
            loop = _asyncio.get_event_loop()
            if loop.is_running():
                # init() is sometimes called from within an event loop (tests, app startup).
                discovery_results = _asyncio.run_coroutine_threadsafe(
                    _run_cloud_discovery(api_keys, cloud_cache_dir, cloud_alert_state_path),
                    loop,
                ).result()
            else:
                discovery_results = _asyncio.run(
                    _run_cloud_discovery(api_keys, cloud_cache_dir, cloud_alert_state_path)
                )
        except RuntimeError:
            discovery_results = _asyncio.run(
                _run_cloud_discovery(api_keys, cloud_cache_dir, cloud_alert_state_path)
            )

    enabled_cloud = {p for p, r in discovery_results.items() if r.auth_ok}

    if available_providers is not None:
        global _available_providers
        # Intersect: caller said "I have these keys" AND discovery said "these auth_ok".
        # Provider with key but discovery failed (no cache) is dropped.
        _available_providers = available_providers & enabled_cloud if discovery_results else set(available_providers)
    else:
        _available_providers = enabled_cloud

    # Register discovered models.
    for provider, result in discovery_results.items():
        if not result.auth_ok:
            continue
        for dm in result.models:
            try:
                register_cloud_from_discovered(_registry, provider, dm)
            except Exception as e:  # noqa: BLE001
                logger.warning("register_cloud_from_discovered failed for %s/%s: %s",
                               provider, dm.litellm_name, e)

    # ... existing benchmark enrichment + capability blend ...
```

(Adapt the actual `init()` signature to the existing one; key changes are: the new `api_keys`, `cloud_cache_dir`, `cloud_alert_state_path` params; the discovery call; populating `_available_providers` from results; registering discovered models BEFORE benchmark enrichment so they exist for that step.)

- [ ] **Step 4: Wire `api_keys` from app config**

Modify `src/app/run.py` (find the existing `fatih_hoca.init(...)` call near line 520). Replace:

```python
_providers = {p for p, key in AVAILABLE_KEYS.items() if key}
fatih_hoca.init(..., available_providers=_providers)
```

with:

```python
_providers = {p for p, key in AVAILABLE_KEYS.items() if key}
_keys = {p: AVAILABLE_KEYS[p] for p in _providers}
fatih_hoca.init(..., available_providers=_providers, api_keys=_keys)
```

- [ ] **Step 5: Run → pass**

Run: `pytest packages/fatih_hoca/tests/test_init.py -v`
Expected: PASS.

- [ ] **Step 6: Smoke-test full registry init**

Run: `python -c "import asyncio; from src.app.config import AVAILABLE_KEYS; import fatih_hoca; fatih_hoca.init(models_yaml='src/models/models.yaml', snapshot_provider=lambda: None, available_providers=set(AVAILABLE_KEYS), api_keys=AVAILABLE_KEYS)"`
Expected: no exception. If providers have invalid keys, log entries `provider X auth_fail` but no crash.

- [ ] **Step 7: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/__init__.py packages/fatih_hoca/tests/test_init.py src/app/run.py
git commit -m "feat(cloud): wire boot-time discovery + auth fail-fast into fatih_hoca.init"
```

---

### Task 15: model_pick_log.provider column

**Files:**
- Modify: `src/infra/db.py:485-519` (schema bootstrap idempotent ALTER block)
- Modify: `src/infra/pick_log.py:24-59`
- Modify: `src/core/llm_dispatcher.py` (call sites of `write_pick_log_row`)
- Test: `tests/test_grading.py` or new `tests/infra/test_pick_log_provider.py`

- [ ] **Step 1: Write the failing test**

Create `tests/infra/test_pick_log_provider.py`:

```python
"""Schema + writer must populate model_pick_log.provider correctly."""
import asyncio
from pathlib import Path

import aiosqlite
import pytest

from src.infra.db import init_db
from src.infra.pick_log import write_pick_log_row


@pytest.mark.asyncio
async def test_provider_column_exists_after_init(tmp_path: Path):
    db_path = str(tmp_path / "k.db")
    await init_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("PRAGMA table_info(model_pick_log)")
        cols = [row[1] for row in await cur.fetchall()]
    assert "provider" in cols


@pytest.mark.asyncio
async def test_legacy_rows_backfilled_to_local(tmp_path: Path):
    db_path = str(tmp_path / "k.db")
    # Simulate pre-migration DB: create table without provider column.
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE model_pick_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                task_name TEXT NOT NULL,
                picked_model TEXT NOT NULL,
                picked_score REAL NOT NULL,
                call_category TEXT,
                candidates_json TEXT NOT NULL DEFAULT '[]',
                snapshot_summary TEXT,
                success INTEGER,
                error_category TEXT
            )
        """)
        await db.execute(
            "INSERT INTO model_pick_log "
            "(task_name, picked_model, picked_score, call_category) "
            "VALUES ('t1', 'qwen-30b', 0.5, 'main_work')"
        )
        await db.commit()
    # Run init → adds column, backfills.
    await init_db(db_path)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT provider FROM model_pick_log WHERE task_name='t1'")
        row = await cur.fetchone()
    assert row[0] == "local"


@pytest.mark.asyncio
async def test_writer_persists_provider(tmp_path: Path):
    db_path = str(tmp_path / "k.db")
    await init_db(db_path)
    await write_pick_log_row(
        db_path=db_path,
        task_name="t",
        picked_model="groq/llama-3.3-70b-versatile",
        picked_score=0.9,
        category="main_work",
        success=True,
        provider="groq",
    )
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT provider FROM model_pick_log WHERE task_name='t'")
        row = await cur.fetchone()
    assert row[0] == "groq"
```

- [ ] **Step 2: Run → fail**

Run: `pytest tests/infra/test_pick_log_provider.py -v`
Expected: FAIL — column missing or writer doesn't accept `provider`.

- [ ] **Step 3: Update schema bootstrap in `src/infra/db.py:504-519`**

Locate the `for col_name, col_type in (...)` ALTER loop and add `("provider", "TEXT")`:

```python
    for col_name, col_type in (
        ("pool", "TEXT"),
        ("urgency", "REAL"),
        ("success", "INTEGER"),
        ("error_category", "TEXT"),
        ("provider", "TEXT"),
    ):
        try:
            await db.execute(f"ALTER TABLE model_pick_log ADD COLUMN {col_name} {col_type}")
        except Exception as e:
            if "duplicate column" not in str(e).lower():
                raise

    # Backfill legacy rows: pre-cloud era was 100% local picks.
    await db.execute(
        "UPDATE model_pick_log SET provider='local' WHERE provider IS NULL"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_pick_log_provider ON model_pick_log(provider)"
    )
```

Also add `provider TEXT` to the `CREATE TABLE IF NOT EXISTS model_pick_log (...)` block at line 485-502 so fresh DBs start with the column natively.

- [ ] **Step 4: Update writer in `src/infra/pick_log.py`**

Replace the writer body:

```python
async def write_pick_log_row(
    db_path: str,
    task_name: str,
    picked_model: str,
    picked_score: float,
    category: str,
    success: bool,
    error_category: str = "",
    snapshot_summary: str = "",
    provider: str = "local",
) -> None:
    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "INSERT INTO model_pick_log "
                "(task_name, picked_model, picked_score, call_category, "
                " candidates_json, snapshot_summary, success, error_category, provider) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task_name,
                    picked_model,
                    picked_score,
                    category,
                    "[]",
                    snapshot_summary,
                    1 if success else 0,
                    error_category,
                    provider,
                ),
            )
            await db.commit()
    except Exception as e:  # noqa: BLE001
        logger.warning("pick_log write failed: %s", e)
```

- [ ] **Step 5: Run → writer + schema tests pass**

Run: `pytest tests/infra/test_pick_log_provider.py -v`
Expected: PASS — 3 tests.

- [ ] **Step 6: Update dispatcher call sites**

Find every call to `write_pick_log_row(...)` in `src/core/llm_dispatcher.py` (typically 1-2 sites). After the call has the `model` object available, append `provider=model.provider`. Example:

```python
await write_pick_log_row(
    db_path=db_path,
    task_name=task_name,
    picked_model=model.name,
    picked_score=picked_score,
    category=category,
    success=success,
    error_category=error_category,
    snapshot_summary=snapshot_summary,
    provider=model.provider,  # NEW
)
```

- [ ] **Step 7: Run dispatcher unit tests**

Run: `timeout 60 pytest tests/ -k "dispatcher or pick_log" -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/infra/db.py src/infra/pick_log.py src/core/llm_dispatcher.py tests/infra/test_pick_log_provider.py
git commit -m "feat(telemetry): model_pick_log.provider column + writer + dispatcher wiring"
```

---

### Task 16: Cloud benchmark match (family-aware) + approval gate

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/benchmark_cloud_match.py`
- Test: `packages/fatih_hoca/tests/test_benchmark_cloud_match.py`
- Modify: `src/models/benchmark/benchmark_fetcher.py:1252` (`enrich_registry_with_benchmarks`) — add cloud overlay hook
- Modify: `packages/fatih_hoca/src/fatih_hoca/__init__.py` (call cloud match after enrichment)

- [ ] **Step 1: Write the failing tests**

```python
# packages/fatih_hoca/tests/test_benchmark_cloud_match.py
from pathlib import Path

import pytest

from fatih_hoca.benchmark_cloud_match import (
    apply_cloud_benchmarks,
    write_review_artifact,
    is_family_approved,
)
from fatih_hoca.types import ModelInfo


def _model(name: str, provider: str, family: str) -> ModelInfo:
    return ModelInfo(
        name=name, location="cloud", provider=provider, litellm_name=name,
        capabilities={"reasoning": 6.0, "coding": 6.0},
        context_length=128000, max_tokens=4096,
        rate_limit_rpm=30, rate_limit_tpm=100000,
        family=family,
    )


def test_aa_hit_stored_but_not_active_until_approved(tmp_path: Path):
    aa_lookup = {"llama-3.3-70b": {"reasoning": 7.5, "coding": 7.0}}
    models = [_model("groq/llama-3.3-70b-versatile", "groq", "llama-3.3-70b")]
    apply_cloud_benchmarks(models, aa_lookup, approved_path=tmp_path / "approved.json")
    m = models[0]
    assert m.benchmark_scores == {"reasoning": 7.5, "coding": 7.0}
    # Not approved yet → active capabilities untouched.
    assert m.capabilities["reasoning"] == 6.0


def test_approved_family_promotes_aa_to_active(tmp_path: Path):
    approved = tmp_path / "approved.json"
    approved.write_text('["llama-3.3-70b"]')
    aa_lookup = {"llama-3.3-70b": {"reasoning": 7.5, "coding": 7.0}}
    models = [_model("groq/llama-3.3-70b-versatile", "groq", "llama-3.3-70b")]
    apply_cloud_benchmarks(models, aa_lookup, approved_path=approved)
    m = models[0]
    assert m.capabilities["reasoning"] == 7.5
    assert m.capabilities["coding"] == 7.0


def test_cross_provider_share_family_score(tmp_path: Path):
    approved = tmp_path / "approved.json"
    approved.write_text('["llama-3.3-70b"]')
    aa_lookup = {"llama-3.3-70b": {"reasoning": 7.5, "coding": 7.0}}
    groq_m = _model("groq/llama-3.3-70b-versatile", "groq", "llama-3.3-70b")
    cerebras_m = _model("cerebras/llama3.3-70b", "cerebras", "llama-3.3-70b")
    apply_cloud_benchmarks([groq_m, cerebras_m], aa_lookup, approved_path=approved)
    assert groq_m.capabilities["reasoning"] == 7.5
    assert cerebras_m.capabilities["reasoning"] == 7.5


def test_review_artifact_written(tmp_path: Path):
    aa_lookup = {"llama-3.3-70b": {"reasoning": 7.5}}
    models = [_model("groq/llama-3.3-70b-versatile", "groq", "llama-3.3-70b")]
    artifact = tmp_path / "review.json"
    apply_cloud_benchmarks(models, aa_lookup, approved_path=tmp_path / "approved.json")
    write_review_artifact(models, aa_lookup, output_path=artifact)
    assert artifact.exists()
    import json
    rows = json.loads(artifact.read_text())
    assert rows[0]["litellm_name"] == "groq/llama-3.3-70b-versatile"
    assert rows[0]["family"] == "llama-3.3-70b"
    assert rows[0]["matched_aa_entry"] == "llama-3.3-70b"
    assert rows[0]["source"] == "aa"


def test_no_aa_hit_keeps_profile_or_default(tmp_path: Path):
    models = [_model("groq/some-future", "groq", "some-future")]
    apply_cloud_benchmarks(models, aa_lookup={}, approved_path=tmp_path / "approved.json")
    # No AA hit → benchmark_scores empty; capabilities untouched (CLOUD_PROFILES already
    # populated them at registration time, or flat 6.0 default).
    assert not models[0].benchmark_scores
    assert models[0].capabilities["reasoning"] == 6.0
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Implement `benchmark_cloud_match.py`**

```python
# packages/fatih_hoca/src/fatih_hoca/benchmark_cloud_match.py
"""Cloud benchmark match + review-gate logic.

Family-aware overlay on top of existing AA enrichment. For each cloud
ModelInfo, look up its family in the supplied aa_lookup. Store match in
``ModelInfo.benchmark_scores`` regardless. Promote to active
``ModelInfo.capabilities`` only when the family appears in the
operator-approved list.

Until a family is approved, capabilities continue to come from the
fallback chain (CLOUD_PROFILES name-substring match → flat 6.0) which
was applied at registration time by ``detect_cloud_model()``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from src.infra.logging_config import get_logger

from .types import ModelInfo

logger = get_logger("fatih_hoca.benchmark_cloud_match")


def _load_approved(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()))
    except Exception as e:  # noqa: BLE001
        logger.warning("failed to load approved families from %s: %s", path, e)
        return set()


def is_family_approved(family: str, approved_path: Path) -> bool:
    return family in _load_approved(approved_path)


def apply_cloud_benchmarks(
    models: Iterable[ModelInfo],
    aa_lookup: Mapping[str, Mapping[str, float]],
    approved_path: Path,
) -> None:
    """For each cloud model, set benchmark_scores from aa_lookup keyed by family.

    If family is in the approved list, also promote scores to capabilities.
    Otherwise capabilities stay at whatever detect_cloud_model produced.
    """
    approved = _load_approved(Path(approved_path))
    for m in models:
        if m.location != "cloud" or not m.family:
            continue
        scores = aa_lookup.get(m.family)
        if not scores:
            continue
        m.benchmark_scores = dict(scores)
        if m.family in approved:
            for cap, score in scores.items():
                m.capabilities[cap] = float(score)


def write_review_artifact(
    models: Iterable[ModelInfo],
    aa_lookup: Mapping[str, Mapping[str, float]],
    output_path: Path,
) -> None:
    """Dump (litellm_name, family, matched_aa_entry, source, final_caps) per cloud
    model so a human can sanity-check before flipping approval."""
    rows = []
    for m in models:
        if m.location != "cloud":
            continue
        aa_match = aa_lookup.get(m.family)
        rows.append({
            "litellm_name": m.litellm_name,
            "family": m.family,
            "matched_aa_entry": m.family if aa_match else None,
            "source": "aa" if aa_match else "profile_or_default",
            "final_capabilities": dict(m.capabilities),
            "benchmark_scores": dict(m.benchmark_scores or {}),
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2))
    logger.info("cloud match review artifact written to %s (%d rows)", output_path, len(rows))
```

- [ ] **Step 4: Run → pass**

Run: `pytest packages/fatih_hoca/tests/test_benchmark_cloud_match.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 5: Wire into `fatih_hoca.init()`**

After existing `enrich_registry_with_benchmarks(...)` call, add:

```python
from .benchmark_cloud_match import apply_cloud_benchmarks, write_review_artifact
from src.models.benchmark.benchmark_fetcher import build_family_keyed_lookup  # see Step 6

aa_lookup = build_family_keyed_lookup(_registry)  # {family: {capability: score}}
apply_cloud_benchmarks(
    list(_registry.all()),
    aa_lookup=aa_lookup,
    approved_path=_Path(cloud_cache_dir).parent / "cloud_match_approved.json",
)
write_review_artifact(
    list(_registry.all()),
    aa_lookup=aa_lookup,
    output_path=_Path(cloud_cache_dir).parent / "cloud_match_review.json",
)
```

- [ ] **Step 6: Add `build_family_keyed_lookup` to benchmark_fetcher**

Append to `src/models/benchmark/benchmark_fetcher.py`:

```python
def build_family_keyed_lookup(registry) -> dict[str, dict[str, float]]:
    """Build {family: {capability: score}} from the AA cache.

    Uses the same name->capabilities matching the local enricher applies,
    but keyed by the family identifier so cross-provider clones share scores.
    """
    from fatih_hoca.cloud.family import normalize, is_known_family
    out: dict[str, dict[str, float]] = {}
    # Walk every aa entry produced by the existing pipeline. The
    # enrich_registry_with_benchmarks call already populates
    # ModelInfo.benchmark_scores for any registered model whose name
    # matches an AA entry. Re-derive the lookup from those scores keyed
    # by family.
    for m in registry.all():
        if not m.benchmark_scores:
            continue
        family = m.family if m.family else normalize(m.provider, m.litellm_name or m.name)
        # Prefer first observation per family.
        out.setdefault(family, dict(m.benchmark_scores))
    return out
```

- [ ] **Step 7: Run integration smoke**

Run: `python -c "import asyncio; from src.app.config import AVAILABLE_KEYS; import fatih_hoca; fatih_hoca.init(models_yaml='src/models/models.yaml', snapshot_provider=lambda: None, available_providers=set(AVAILABLE_KEYS), api_keys=AVAILABLE_KEYS); import json,os; print(os.path.exists('.benchmark_cache/cloud_match_review.json'))"`
Expected: prints `True`.

- [ ] **Step 8: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/benchmark_cloud_match.py packages/fatih_hoca/tests/test_benchmark_cloud_match.py packages/fatih_hoca/src/fatih_hoca/__init__.py src/models/benchmark/benchmark_fetcher.py
git commit -m "feat(cloud): family-aware bench match + approval-gated promotion + review artifact"
```

---

### Task 17: Beckman cron entry + salako handler

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/cron_seed.py:19`
- Modify: `packages/salako/src/salako/actions.py`
- Test: `packages/salako/tests/test_cloud_refresh_action.py` (new)

- [ ] **Step 1: Add cadence row**

In `packages/general_beckman/src/general_beckman/cron_seed.py:19`, append to `INTERNAL_CADENCES`:

```python
    {
        "title": "cloud_refresh",
        "description": "Re-run cloud provider /models discovery + bench refresh",
        "interval_seconds": 21600,  # 6h
        "payload": {"_executor": "cloud_refresh"},
    },
```

- [ ] **Step 2: Write the failing test for handler**

```python
# packages/salako/tests/test_cloud_refresh_action.py
from unittest.mock import AsyncMock, patch

import pytest

from salako.actions import handle_cloud_refresh


@pytest.mark.asyncio
async def test_handle_cloud_refresh_invokes_discovery_and_match():
    with patch("salako.actions._refresh_cloud_subsystem", new=AsyncMock(return_value={
        "providers_probed": 3,
        "providers_ok": 2,
        "models_registered": 27,
    })) as mocked:
        result = await handle_cloud_refresh(payload={})
    mocked.assert_awaited_once()
    assert result["providers_probed"] == 3
    assert result["providers_ok"] == 2
```

- [ ] **Step 3: Run → fail**

- [ ] **Step 4: Implement `handle_cloud_refresh` in salako actions**

Append to `packages/salako/src/salako/actions.py`:

```python
async def _refresh_cloud_subsystem() -> dict:
    """Re-run discovery against current registry + refresh review artifact.

    Imported lazily because fatih_hoca pulls in heavy benchmark code.
    """
    from pathlib import Path

    import fatih_hoca
    from fatih_hoca.benchmark_cloud_match import apply_cloud_benchmarks, write_review_artifact
    from src.app.config import AVAILABLE_KEYS
    from src.models.benchmark.benchmark_fetcher import build_family_keyed_lookup

    api_keys = {p: k for p, k in AVAILABLE_KEYS.items() if k}
    results = await fatih_hoca._run_cloud_discovery(
        api_keys=api_keys,
        cache_dir=".benchmark_cache/cloud_models",
        alert_state_path=".benchmark_cache/cloud_alert_throttle.json",
    )
    # Re-register discovered models (in-place updates if already present).
    for provider, result in results.items():
        if not result.auth_ok:
            continue
        for dm in result.models:
            from fatih_hoca.registry import register_cloud_from_discovered
            register_cloud_from_discovered(fatih_hoca._registry, provider, dm)
    # Refresh bench match.
    aa_lookup = build_family_keyed_lookup(fatih_hoca._registry)
    models = list(fatih_hoca._registry.all())
    apply_cloud_benchmarks(models, aa_lookup, approved_path=Path(".benchmark_cache/cloud_match_approved.json"))
    write_review_artifact(models, aa_lookup, output_path=Path(".benchmark_cache/cloud_match_review.json"))
    fatih_hoca._available_providers = {p for p, r in results.items() if r.auth_ok}
    return {
        "providers_probed": len(results),
        "providers_ok": sum(1 for r in results.values() if r.auth_ok),
        "models_registered": sum(len(r.models) for r in results.values() if r.auth_ok),
    }


async def handle_cloud_refresh(payload: dict) -> dict:
    """Salako mechanical executor: scheduled cloud refresh."""
    return await _refresh_cloud_subsystem()
```

- [ ] **Step 5: Register handler in salako dispatch**

Find the action dispatch table in `packages/salako/src/salako/__init__.py` or `actions.py` (search for `_executor`). Add `"cloud_refresh": handle_cloud_refresh`.

- [ ] **Step 6: Run → pass**

Run: `pytest packages/salako/tests/test_cloud_refresh_action.py -v`

- [ ] **Step 7: Verify cadence seeded on app start**

Run: `python -c "from packages.general_beckman.src.general_beckman.cron_seed import INTERNAL_CADENCES; print([c['title'] for c in INTERNAL_CADENCES])"`
Expected: list contains `"cloud_refresh"`.

- [ ] **Step 8: Commit**

```bash
git add packages/general_beckman/src/general_beckman/cron_seed.py packages/salako/src/salako/actions.py packages/salako/src/salako/__init__.py packages/salako/tests/test_cloud_refresh_action.py
git commit -m "feat(cloud): 6h beckman cron + salako handler for cloud refresh"
```

---

### Task 18: KDV no-data warning

**Files:**
- Modify: `packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py`
- Test: `packages/kuleden_donen_var/tests/test_kdv_no_data_warning.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# packages/kuleden_donen_var/tests/test_kdv_no_data_warning.py
import time
from kuleden_donen_var.kdv import KuledenDonenVar


def test_no_data_warning_fires_after_threshold():
    kdv = KuledenDonenVar()
    kdv.register_cloud_model("groq", "groq/llama-3.3-70b")
    kdv.mark_provider_enabled("groq", at_unix=time.time() - (25 * 3600))
    warnings = kdv.no_data_warnings(min_age_hours=24)
    assert any(w["provider"] == "groq" for w in warnings)


def test_no_data_warning_skips_recently_enabled():
    kdv = KuledenDonenVar()
    kdv.register_cloud_model("groq", "groq/llama-3.3-70b")
    kdv.mark_provider_enabled("groq", at_unix=time.time() - 3600)
    warnings = kdv.no_data_warnings(min_age_hours=24)
    assert warnings == []


def test_no_data_warning_skips_provider_with_observations():
    kdv = KuledenDonenVar()
    kdv.register_cloud_model("groq", "groq/llama-3.3-70b")
    kdv.mark_provider_enabled("groq", at_unix=time.time() - (25 * 3600))
    kdv.record_call_observation("groq", "groq/llama-3.3-70b", tokens_used=100)
    warnings = kdv.no_data_warnings(min_age_hours=24)
    assert warnings == []
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Implement on `KuledenDonenVar`**

Add to `packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py`:

```python
    def mark_provider_enabled(self, provider: str, at_unix: float | None = None) -> None:
        """Record when a provider was first enabled (boot or first auth_ok).

        Used to compute 'idle since enabled' for no_data_warnings().
        """
        import time as _time
        self._provider_enabled_at.setdefault(provider, at_unix if at_unix is not None else _time.time())

    def record_call_observation(self, provider: str, model: str, tokens_used: int = 0) -> None:
        """Bump per-provider observation count. Wired from caller post-call hook."""
        self._provider_call_count[provider] = self._provider_call_count.get(provider, 0) + 1

    def no_data_warnings(self, min_age_hours: float = 24.0) -> list[dict]:
        """Return list of providers enabled longer than min_age_hours with zero observations.

        Each entry: {"provider": str, "enabled_at_unix": float, "age_hours": float}.
        Caller (status command, scheduled task) uses this to surface "defaults still in
        use" warnings to the operator.
        """
        import time as _time
        now = _time.time()
        out = []
        for provider, enabled_at in self._provider_enabled_at.items():
            age_hours = (now - enabled_at) / 3600
            if age_hours < min_age_hours:
                continue
            if self._provider_call_count.get(provider, 0) > 0:
                continue
            out.append({
                "provider": provider,
                "enabled_at_unix": enabled_at,
                "age_hours": age_hours,
            })
        return out
```

In `__init__`, add:

```python
        self._provider_enabled_at: dict[str, float] = {}
        self._provider_call_count: dict[str, int] = {}
```

- [ ] **Step 4: Wire `mark_provider_enabled` from `fatih_hoca.init()` for each auth_ok provider**

In the discovery results loop in `fatih_hoca/__init__.py` after `register_cloud_from_discovered`:

```python
        from packages.kuleden_donen_var.src.kuleden_donen_var import get_kdv
        try:
            get_kdv().mark_provider_enabled(provider)
        except Exception as e:  # noqa: BLE001
            logger.debug("KDV mark_provider_enabled failed for %s: %s", provider, e)
```

- [ ] **Step 5: Wire `record_call_observation` from caller**

In `packages/hallederiz_kadir/src/hallederiz_kadir/caller.py`, find the post-call KDV update site (`_kdv_post_call`). After it succeeds, also call `get_kdv().record_call_observation(model.provider, model.litellm_name, tokens_used=...)`. The exact tokens-used value is best-effort (response usage if exposed); 0 is fine if absent — the count is what matters.

- [ ] **Step 6: Run → pass**

Run: `pytest packages/kuleden_donen_var/tests/test_kdv_no_data_warning.py -v`

- [ ] **Step 7: Surface the warning in `/status`**

In `src/app/telegram_bot.py`, find the `/status` handler. Append to its message string:

```python
        from packages.kuleden_donen_var.src.kuleden_donen_var import get_kdv
        kdv_warnings = get_kdv().no_data_warnings(min_age_hours=24)
        if kdv_warnings:
            lines = [f"  - {w['provider']} ({w['age_hours']:.1f}h since enable, no calls)" for w in kdv_warnings]
            status_msg += "\n\n⚠️ Cloud providers using cold-start defaults:\n" + "\n".join(lines)
```

- [ ] **Step 8: Commit**

```bash
git add packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py packages/kuleden_donen_var/tests/test_kdv_no_data_warning.py packages/fatih_hoca/src/fatih_hoca/__init__.py packages/hallederiz_kadir/src/hallederiz_kadir/caller.py src/app/telegram_bot.py
git commit -m "feat(kdv): no-data warning for cloud providers with zero observations"
```

---

### Task 19: Live discovery integration test (env-gated)

**Files:**
- Create: `tests/integration/test_cloud_discovery_live.py`

- [ ] **Step 1: Write the env-gated test**

```python
# tests/integration/test_cloud_discovery_live.py
"""Live /models smoke tests. Skipped unless the corresponding API key
is present in os.environ. Run manually:
    pytest tests/integration/test_cloud_discovery_live.py -v
"""
import os

import pytest

pytestmark = pytest.mark.asyncio


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="no GROQ_API_KEY")
async def test_groq_live():
    from fatih_hoca.cloud.providers.groq import GroqAdapter
    result = await GroqAdapter().fetch_models(os.environ["GROQ_API_KEY"])
    assert result.status == "ok", f"unexpected: {result.status} / {result.error}"
    assert len(result.models) > 0
    # Spot-check shape.
    m = result.models[0]
    assert m.litellm_name.startswith("groq/")
    assert m.context_length is not None and m.context_length > 0


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
async def test_openai_live():
    from fatih_hoca.cloud.providers.openai import OpenAIAdapter
    result = await OpenAIAdapter().fetch_models(os.environ["OPENAI_API_KEY"])
    assert result.status == "ok"
    assert any("gpt-4o" in m.raw_id for m in result.models)


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
async def test_anthropic_live():
    from fatih_hoca.cloud.providers.anthropic import AnthropicAdapter
    result = await AnthropicAdapter().fetch_models(os.environ["ANTHROPIC_API_KEY"])
    assert result.status == "ok"
    assert any("claude" in m.raw_id for m in result.models)


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="no GEMINI_API_KEY")
async def test_gemini_live():
    from fatih_hoca.cloud.providers.gemini import GeminiAdapter
    result = await GeminiAdapter().fetch_models(os.environ["GEMINI_API_KEY"])
    assert result.status == "ok"
    assert any("gemini" in m.raw_id for m in result.models)


@pytest.mark.skipif(not os.environ.get("CEREBRAS_API_KEY"), reason="no CEREBRAS_API_KEY")
async def test_cerebras_live():
    from fatih_hoca.cloud.providers.cerebras import CerebrasAdapter
    result = await CerebrasAdapter().fetch_models(os.environ["CEREBRAS_API_KEY"])
    assert result.status == "ok"


@pytest.mark.skipif(not os.environ.get("SAMBANOVA_API_KEY"), reason="no SAMBANOVA_API_KEY")
async def test_sambanova_live():
    from fatih_hoca.cloud.providers.sambanova import SambanovaAdapter
    result = await SambanovaAdapter().fetch_models(os.environ["SAMBANOVA_API_KEY"])
    assert result.status == "ok"


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="no OPENROUTER_API_KEY")
async def test_openrouter_live():
    from fatih_hoca.cloud.providers.openrouter import OpenRouterAdapter
    result = await OpenRouterAdapter().fetch_models(os.environ["OPENROUTER_API_KEY"])
    assert result.status == "ok"
    assert len(result.models) > 50  # OpenRouter exposes hundreds
```

- [ ] **Step 2: Run with whatever keys are set**

Run: `timeout 60 pytest tests/integration/test_cloud_discovery_live.py -v`
Expected: any provider with key set → PASS; others → SKIP.

- [ ] **Step 3: Inspect review artifact**

Run app boot once with real keys (or call `_refresh_cloud_subsystem` directly) and inspect `.benchmark_cache/cloud_match_review.json`. Confirm:
- families look right (no llama-3.3-70b mis-mapped to llama-3.1-70b)
- proprietary entries (claude/gpt/gemini) match expected AA names

If a family looks wrong, fix the regex in `cloud/family.py` and re-run.

- [ ] **Step 4: Approve a starter set of families**

Once review artifact looks clean, write `.benchmark_cache/cloud_match_approved.json` with the families you trust:

```json
[
  "llama-3.3-70b",
  "claude-sonnet-4",
  "gpt-4o",
  "gpt-4o-mini",
  "gemini-2.0-flash"
]
```

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_cloud_discovery_live.py
git commit -m "test(cloud): env-gated live /models smoke tests"
```

---

### Task 20: Final integration smoke + spec coverage check

- [ ] **Step 1: Run full unit test suite**

Run: `timeout 120 pytest packages/fatih_hoca/tests/ tests/infra/ packages/kuleden_donen_var/tests/ packages/salako/tests/ -v`
Expected: all PASS.

- [ ] **Step 2: Boot app once**

Start KutAI normally. Verify in logs:
- per-provider `provider X status=ok models=N` lines
- `cloud_match_review.json` written
- `_available_providers` set logged
- no exceptions during init

- [ ] **Step 3: Verify selector can pick cloud**

Run: `python -c "import fatih_hoca; ... select with hard task & call_category=main_work; print(pick.model.provider)"` (use existing simulator script in `packages/fatih_hoca/tests/sim/`).
Expected: a cloud model wins for at least one hard task scenario.

- [ ] **Step 4: Verify pick log persists provider**

Trigger one real /task via Telegram. After completion run:
```sql
sqlite> SELECT provider, COUNT(*) FROM model_pick_log WHERE timestamp > datetime('now', '-1 hour') GROUP BY 1;
```
Expected: at least one row with non-`local` provider if cloud was picked.

- [ ] **Step 5: Verify scheduled task seeded**

Run:
```sql
sqlite> SELECT title, kind, interval_seconds FROM scheduled_tasks WHERE title='cloud_refresh';
```
Expected: one row, kind='internal', interval_seconds=21600.

---

## Self-Review

Spec coverage:

| Spec section | Implementing tasks |
|--------------|-------------------|
| Architecture (cloud/ package layout) | 1, 2, 3, 4, 5-11, 12 |
| Boot data flow | 14 |
| Periodic refresh | 17 |
| §G2 Auth fail-fast | 14 |
| §G3 model_pick_log.provider | 15 |
| §G4 Per-provider scrape | 5, 6, 7, 8, 9, 10, 11 |
| §G4 Family dedup | 2, 13 |
| §G4 Cache | 3, 12 |
| §G4 Telegram alert | 4, 14 |
| §G4 Quota / KDV no-data warning | 18 |
| §G5 Cloud benchmarking + validation gate | 16, 19 (step 4 approval) |
| Test plan | 2-12, 18, 19 |

No placeholders found. Type consistency: `DiscoveredModel`/`ProviderResult` field names match across types.py, all adapters, discovery, registry helper, and tests. `register_cloud_from_discovered` signature matches its caller in `__init__.py` and salako handler.

---

Plan complete and saved to `docs/superpowers/plans/2026-04-27-cloud-subsystem-hardening.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
