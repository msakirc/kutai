# Fatih Hoca Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract model selection logic from router.py, dispatcher, registry, and capabilities into a standalone `packages/fatih_hoca/` package with a clean `select()` API, expand Nerd Herd with system snapshots, and simplify the dispatcher to an ask-load-call-retry loop.

**Architecture:** Fatih Hoca owns all model knowledge: catalog (YAML+GGUF), 15-dimension scoring, task profiles, swap budget, failure adaptation, quota planning. It queries Nerd Herd for system state via `snapshot()`. Dispatcher becomes a thin loop: ask Fatih Hoca → load via DaLLaMa → call Talking Layer → report failures. No circular deps — every arrow one-directional.

**Tech Stack:** Python 3.10, dataclasses, nerd_herd (system state), editable pip install

**Spec:** `docs/superpowers/specs/2026-04-14-fatih-hoca-design.md`
**Architecture doc:** `docs/architecture-modularization.md`

---

## File Structure

```
packages/fatih_hoca/
  pyproject.toml
  src/
    fatih_hoca/
      __init__.py            # exports: init, select, all_models, Pick, Failure,
                             #   ModelInfo, ModelRequirements
      types.py               # Pick, Failure, SwapBudget, shared types
      registry.py            # ModelInfo, GGUF scanning, YAML parsing, model catalog
      profiles.py            # FamilyProfile, FAMILY_PATTERNS, CLOUD_PROFILES,
                             #   capability estimation, quant retention
      capabilities.py        # Cap enum, TASK_PROFILES, AGENT_REQUIREMENTS,
                             #   score_model_for_task(), 15-dim dot product
      requirements.py        # ModelRequirements, CAPABILITY_TO_TASK, QuotaPlanner
      selector.py            # select(), candidate filtering (eligibility hard gates)
      ranking.py             # composite weights, swap stickiness, specialty,
                             #   sibling rebalancing, failure adaptation
  tests/
    test_types.py
    test_registry.py
    test_profiles.py
    test_capabilities.py
    test_requirements.py
    test_selector.py
    test_ranking.py
```

Files modified:
- `packages/nerd_herd/src/nerd_herd/types.py` — add SystemSnapshot, LocalModelState, CloudProviderState, etc.
- `packages/nerd_herd/src/nerd_herd/nerd_herd.py` — add `snapshot()` method
- `packages/nerd_herd/src/nerd_herd/__init__.py` — export new types
- `src/core/router.py` — becomes thin shim re-exporting from fatih_hoca
- `src/core/llm_dispatcher.py` — simplify to ask-load-call-retry loop
- `src/models/model_registry.py` — becomes thin shim
- `src/models/capabilities.py` — becomes thin shim
- `src/models/quota_planner.py` — becomes thin shim

Files deleted (eventually, after shim period):
- `src/models/gpu_scheduler.py` — dead code
- `src/models/gpu_monitor.py` — shim to nerd_herd, no longer needed
- `src/models/header_parser.py` — shim to kuleden_donen_var, no longer needed
- `src/models/rate_limiter.py` — shim to kuleden_donen_var, no longer needed

---

## Task 1: Nerd Herd — SystemSnapshot Types

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/types.py`
- Test: `packages/nerd_herd/tests/test_types.py` (create if needed)

This task adds the snapshot dataclasses that Fatih Hoca will consume. Pure data — no collectors yet.

- [ ] **Step 1: Write the failing test for SystemSnapshot**

```python
# packages/nerd_herd/tests/test_types.py
from nerd_herd.types import (
    GPUState,
    LocalModelState,
    CloudProviderState,
    CloudModelState,
    RateLimit,
    RateLimits,
    SystemSnapshot,
)


def test_system_snapshot_defaults():
    snap = SystemSnapshot()
    assert snap.vram_available_mb == 0
    assert snap.local.model_name is None
    assert snap.local.thinking_enabled is False
    assert snap.local.measured_tps == 0.0
    assert snap.local.context_length == 0
    assert snap.local.is_swapping is False
    assert snap.local.kv_cache_ratio == 0.0
    assert snap.local.vision_enabled is False
    assert snap.cloud == {}


def test_cloud_provider_state():
    provider = CloudProviderState(
        provider="anthropic",
        utilization_pct=45.0,
        consecutive_failures=2,
        last_failure_at=1713100000,
    )
    assert provider.provider == "anthropic"
    assert provider.consecutive_failures == 2
    assert provider.limits.rpm.limit is None
    assert provider.models == {}


def test_cloud_model_state():
    model = CloudModelState(
        model_id="claude-sonnet-4-20250514",
        utilization_pct=30.0,
    )
    model.limits.rpm.limit = 100
    model.limits.rpm.remaining = 80
    assert model.utilization_pct == 30.0
    assert model.limits.rpm.remaining == 80


def test_rate_limit_defaults():
    rl = RateLimit()
    assert rl.limit is None
    assert rl.remaining is None
    assert rl.reset_at is None


def test_snapshot_with_cloud():
    snap = SystemSnapshot(
        vram_available_mb=4096,
        local=LocalModelState(
            model_name="qwen3-30b",
            thinking_enabled=True,
            measured_tps=12.5,
            context_length=8192,
        ),
        cloud={
            "anthropic": CloudProviderState(
                provider="anthropic",
                utilization_pct=20.0,
            ),
        },
    )
    assert snap.local.model_name == "qwen3-30b"
    assert snap.local.thinking_enabled is True
    assert "anthropic" in snap.cloud
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/nerd_herd/tests/test_types.py -v`
Expected: FAIL — `ImportError: cannot import name 'LocalModelState'`

- [ ] **Step 3: Add snapshot dataclasses to nerd_herd types**

Add these dataclasses to the end of `packages/nerd_herd/src/nerd_herd/types.py`:

```python
@dataclass
class RateLimit:
    limit: int | None = None
    remaining: int | None = None
    reset_at: int | None = None        # absolute epoch seconds


@dataclass
class RateLimits:
    rpm: RateLimit = field(default_factory=RateLimit)
    tpm: RateLimit = field(default_factory=RateLimit)
    rpd: RateLimit = field(default_factory=RateLimit)


@dataclass
class CloudModelState:
    model_id: str = ""
    utilization_pct: float = 0.0
    limits: RateLimits = field(default_factory=RateLimits)


@dataclass
class CloudProviderState:
    provider: str = ""
    utilization_pct: float = 0.0
    consecutive_failures: int = 0
    last_failure_at: int | None = None   # epoch seconds
    limits: RateLimits = field(default_factory=RateLimits)
    models: dict[str, CloudModelState] = field(default_factory=dict)


@dataclass
class LocalModelState:
    model_name: str | None = None
    thinking_enabled: bool = False
    vision_enabled: bool = False
    measured_tps: float = 0.0
    context_length: int = 0
    is_swapping: bool = False
    kv_cache_ratio: float = 0.0


@dataclass
class SystemSnapshot:
    vram_available_mb: int = 0
    local: LocalModelState = field(default_factory=LocalModelState)
    cloud: dict[str, CloudProviderState] = field(default_factory=dict)
```

- [ ] **Step 4: Export new types from `__init__.py`**

Add to `packages/nerd_herd/src/nerd_herd/__init__.py` imports:

```python
from nerd_herd.types import (
    # ... existing imports ...
    RateLimit,
    RateLimits,
    CloudModelState,
    CloudProviderState,
    LocalModelState,
    SystemSnapshot,
)
```

And add to `__all__`:

```python
"RateLimit",
"RateLimits",
"CloudModelState",
"CloudProviderState",
"LocalModelState",
"SystemSnapshot",
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest packages/nerd_herd/tests/test_types.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 6: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/types.py packages/nerd_herd/src/nerd_herd/__init__.py packages/nerd_herd/tests/test_types.py
git commit -m "feat(nerd_herd): add SystemSnapshot types for Fatih Hoca integration"
```

---

## Task 2: Nerd Herd — snapshot() Method

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/nerd_herd.py`
- Test: `packages/nerd_herd/tests/test_snapshot.py` (create)

The `snapshot()` method assembles a `SystemSnapshot` from current GPU state plus any pushed local/cloud state. Push methods let DaLLaMa and KDV update state without Nerd Herd importing them.

- [ ] **Step 1: Write the failing test**

```python
# packages/nerd_herd/tests/test_snapshot.py
from nerd_herd.nerd_herd import NerdHerd
from nerd_herd.types import (
    SystemSnapshot,
    LocalModelState,
    CloudProviderState,
    RateLimit,
    RateLimits,
)


def test_snapshot_returns_system_snapshot():
    nh = NerdHerd(metrics_port=0)
    snap = nh.snapshot()
    assert isinstance(snap, SystemSnapshot)
    assert snap.vram_available_mb >= 0
    assert snap.local.model_name is None
    assert snap.cloud == {}


def test_snapshot_reflects_pushed_local_state():
    nh = NerdHerd(metrics_port=0)
    nh.push_local_state(LocalModelState(
        model_name="qwen3-30b",
        thinking_enabled=True,
        measured_tps=12.5,
        context_length=8192,
    ))
    snap = nh.snapshot()
    assert snap.local.model_name == "qwen3-30b"
    assert snap.local.thinking_enabled is True
    assert snap.local.measured_tps == 12.5


def test_snapshot_reflects_pushed_cloud_state():
    nh = NerdHerd(metrics_port=0)
    nh.push_cloud_state(CloudProviderState(
        provider="anthropic",
        utilization_pct=45.0,
        consecutive_failures=1,
        limits=RateLimits(rpm=RateLimit(limit=100, remaining=80)),
    ))
    snap = nh.snapshot()
    assert "anthropic" in snap.cloud
    assert snap.cloud["anthropic"].utilization_pct == 45.0
    assert snap.cloud["anthropic"].limits.rpm.remaining == 80


def test_push_local_state_clears_on_none():
    nh = NerdHerd(metrics_port=0)
    nh.push_local_state(LocalModelState(model_name="qwen3-30b"))
    nh.push_local_state(LocalModelState())  # model unloaded
    snap = nh.snapshot()
    assert snap.local.model_name is None


def test_push_cloud_state_updates_existing():
    nh = NerdHerd(metrics_port=0)
    nh.push_cloud_state(CloudProviderState(provider="anthropic", utilization_pct=10.0))
    nh.push_cloud_state(CloudProviderState(provider="anthropic", utilization_pct=90.0))
    snap = nh.snapshot()
    assert snap.cloud["anthropic"].utilization_pct == 90.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/nerd_herd/tests/test_snapshot.py -v`
Expected: FAIL — `AttributeError: 'NerdHerd' object has no attribute 'snapshot'`

- [ ] **Step 3: Add snapshot(), push_local_state(), push_cloud_state() to NerdHerd**

Add to `packages/nerd_herd/src/nerd_herd/nerd_herd.py`:

```python
# Add imports at top:
from nerd_herd.types import (
    GPUState,
    HealthStatus,
    LocalModelState,
    CloudProviderState,
    SystemSnapshot,
)

# Add to __init__:
self._local_state = LocalModelState()
self._cloud_state: dict[str, CloudProviderState] = {}

# Add methods:
def push_local_state(self, state: LocalModelState) -> None:
    """Called by DaLLaMa callbacks to update loaded model state."""
    self._local_state = state

def push_cloud_state(self, state: CloudProviderState) -> None:
    """Called by KDV callbacks to update cloud provider state."""
    self._cloud_state[state.provider] = state

def snapshot(self) -> SystemSnapshot:
    """Return a point-in-time snapshot of all system state."""
    gpu = self._gpu.gpu_state()
    return SystemSnapshot(
        vram_available_mb=self.get_vram_budget_mb() if gpu.available else 0,
        local=self._local_state,
        cloud=dict(self._cloud_state),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/nerd_herd/tests/test_snapshot.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Export snapshot-related from __init__.py (if not already done in Task 1)**

Verify `SystemSnapshot`, `LocalModelState`, `CloudProviderState` are exported.

- [ ] **Step 6: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/nerd_herd.py packages/nerd_herd/tests/test_snapshot.py
git commit -m "feat(nerd_herd): add snapshot() with push-based local/cloud state"
```

---

## Task 3: Fatih Hoca — Package Scaffold + Types

**Files:**
- Create: `packages/fatih_hoca/pyproject.toml`
- Create: `packages/fatih_hoca/src/fatih_hoca/__init__.py`
- Create: `packages/fatih_hoca/src/fatih_hoca/types.py`
- Test: `packages/fatih_hoca/tests/test_types.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"

[project]
name = "fatih_hoca"
version = "0.1.0"
description = "Model manager — scoring, selection, swap budget, failure adaptation"
requires-python = ">=3.10"
dependencies = ["nerd_herd"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create directory structure**

```bash
mkdir -p packages/fatih_hoca/src/fatih_hoca packages/fatih_hoca/tests
```

- [ ] **Step 3: Write the failing test for types**

```python
# packages/fatih_hoca/tests/test_types.py
from fatih_hoca.types import Pick, Failure, SwapBudget
import time


def test_pick_fields():
    pick = Pick(model=None, min_time_seconds=30.0)
    assert pick.min_time_seconds == 30.0


def test_failure_fields():
    f = Failure(model="qwen3-30b", reason="timeout", latency=120.0)
    assert f.model == "qwen3-30b"
    assert f.reason == "timeout"
    assert f.latency == 120.0


def test_failure_no_latency():
    f = Failure(model="groq/llama-8b", reason="rate_limit")
    assert f.latency is None


def test_swap_budget_allows_first_swap():
    sb = SwapBudget(max_swaps=3, window_seconds=300)
    assert sb.can_swap(local_only=False, priority=5) is True
    assert sb.remaining == 3


def test_swap_budget_blocks_after_max():
    sb = SwapBudget(max_swaps=2, window_seconds=300)
    sb.record_swap()
    sb.record_swap()
    assert sb.can_swap(local_only=False, priority=5) is False
    assert sb.remaining == 0


def test_swap_budget_exempt_local_only():
    sb = SwapBudget(max_swaps=1, window_seconds=300)
    sb.record_swap()
    assert sb.can_swap(local_only=False, priority=5) is False
    assert sb.can_swap(local_only=True, priority=5) is True


def test_swap_budget_exempt_high_priority():
    sb = SwapBudget(max_swaps=1, window_seconds=300)
    sb.record_swap()
    assert sb.can_swap(local_only=False, priority=9) is True
    assert sb.can_swap(local_only=False, priority=8) is False


def test_swap_budget_exhausted_property():
    sb = SwapBudget(max_swaps=1, window_seconds=300)
    assert sb.exhausted is False
    sb.record_swap()
    assert sb.exhausted is True
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest packages/fatih_hoca/tests/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fatih_hoca'`

- [ ] **Step 5: Install package editable and write types.py**

```bash
pip install -e packages/fatih_hoca
```

```python
# packages/fatih_hoca/src/fatih_hoca/types.py
"""Shared types for Fatih Hoca model selection."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Pick:
    model: object  # ModelInfo — typed loosely here, fully typed in __init__
    min_time_seconds: float


@dataclass
class Failure:
    model: str              # litellm_name that failed
    reason: str             # "timeout", "rate_limit", "context_overflow",
                            # "quality_failure", "server_error", "loading"
    latency: float | None = None


class SwapBudget:
    """Rate-limits model swaps. Max N swaps per window.

    Exemptions: local_only requests, priority >= 9.
    """

    def __init__(self, max_swaps: int = 3, window_seconds: int = 300) -> None:
        self._max = max_swaps
        self._window = window_seconds
        self._timestamps: list[float] = []

    def _prune(self) -> None:
        cutoff = time.monotonic() - self._window
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    def can_swap(self, local_only: bool, priority: int) -> bool:
        if local_only or priority >= 9:
            return True
        self._prune()
        return len(self._timestamps) < self._max

    def record_swap(self) -> None:
        self._timestamps.append(time.monotonic())

    @property
    def remaining(self) -> int:
        self._prune()
        return max(0, self._max - len(self._timestamps))

    @property
    def exhausted(self) -> bool:
        self._prune()
        return len(self._timestamps) >= self._max
```

- [ ] **Step 6: Create empty `__init__.py`**

```python
# packages/fatih_hoca/src/fatih_hoca/__init__.py
"""Fatih Hoca — model manager: scoring, selection, swap budget."""
```

- [ ] **Step 7: Run test to verify it passes**

Run: `python -m pytest packages/fatih_hoca/tests/test_types.py -v`
Expected: PASS (all 8 tests)

- [ ] **Step 8: Commit**

```bash
git add packages/fatih_hoca/
git commit -m "feat(fatih_hoca): scaffold package with Pick, Failure, SwapBudget types"
```

---

## Task 4: Fatih Hoca — Profiles (Family + Cloud)

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/profiles.py`
- Test: `packages/fatih_hoca/tests/test_profiles.py`

Move `src/models/model_profiles.py` (1557 lines) into the package. This file has zero dependencies on anything in `src/` — it's pure data + pattern matching. Copy it wholesale, then verify.

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_profiles.py
from fatih_hoca.profiles import (
    FamilyProfile,
    FAMILY_PATTERNS,
    CLOUD_PROFILES,
    detect_family,
    get_default_profile,
    get_quant_retention,
    interpolate_size_multiplier,
)


def test_detect_family_qwen():
    assert detect_family("Qwen3-30B-A3B-Q4_K_M.gguf") == "qwen3"


def test_detect_family_llama():
    assert detect_family("Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf") == "llama3"


def test_detect_family_unknown():
    result = detect_family("completely-unknown-model.gguf")
    assert result == "unknown" or result is None


def test_cloud_profiles_has_claude():
    assert "claude-sonnet-4-20250514" in CLOUD_PROFILES or any(
        "claude" in k for k in CLOUD_PROFILES
    )


def test_family_profile_fields():
    fp = get_default_profile()
    assert isinstance(fp, FamilyProfile)
    assert hasattr(fp, "base_capabilities")
    assert isinstance(fp.base_capabilities, dict)


def test_quant_retention():
    q8 = get_quant_retention("Q8_0")
    q4 = get_quant_retention("Q4_K_M")
    assert q8 > q4  # higher quant retains more capability


def test_interpolate_size_multiplier():
    # Larger models should get higher multipliers
    small = interpolate_size_multiplier(1.0, "reasoning")
    large = interpolate_size_multiplier(30.0, "reasoning")
    assert large > small
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/fatih_hoca/tests/test_profiles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fatih_hoca.profiles'`

- [ ] **Step 3: Copy model_profiles.py to profiles.py**

Copy the entire content of `src/models/model_profiles.py` to `packages/fatih_hoca/src/fatih_hoca/profiles.py`. The file has no imports from `src/` — only stdlib (`dataclasses`, `re`, `math`). Verify no `from src.` or `from .` imports exist. If any do, remove them (there shouldn't be any).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/fatih_hoca/tests/test_profiles.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/profiles.py packages/fatih_hoca/tests/test_profiles.py
git commit -m "feat(fatih_hoca): add profiles module (family patterns, cloud profiles)"
```

---

## Task 5: Fatih Hoca — Registry (ModelInfo + Catalog)

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/registry.py`
- Test: `packages/fatih_hoca/tests/test_registry.py`

Extract ModelInfo, GGUF scanning, YAML parsing, capability estimation from `src/models/model_registry.py` (2050 lines). The key change: replace `from .model_profiles import ...` with `from fatih_hoca.profiles import ...` and remove any `from .gpu_monitor` / `from .capabilities` internal imports.

**Important:** `model_registry.py` imports from `capabilities.py` (for `score_model_for_task`, `rank_models_for_task`) and from `gpu_monitor.py`. In the package version:
- Remove the `rank_models_for_task` / `score_model_for_task` usage — that belongs in selector/ranking, not registry
- GPU info comes through Nerd Herd snapshot, not direct `gpu_monitor` calls — `calculate_dynamic_context` and `calculate_gpu_layers` take raw values as params (they already do)
- The benchmark enrichment (`enrich_registry_with_benchmarks`) stays — it's registry's job to enrich ModelInfo

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_registry.py
from fatih_hoca.registry import ModelInfo, ModelRegistry


def test_model_info_is_local():
    m = ModelInfo(name="qwen3-30b", location="local", provider="local",
                  litellm_name="local/qwen3-30b")
    assert m.is_local is True


def test_model_info_is_cloud():
    m = ModelInfo(name="claude-sonnet", location="cloud", provider="anthropic",
                  litellm_name="anthropic/claude-sonnet-4-20250514")
    assert m.is_local is False


def test_model_info_is_free():
    m = ModelInfo(name="test", location="cloud", provider="groq",
                  litellm_name="groq/llama-8b",
                  cost_per_1k_input=0.0, cost_per_1k_output=0.0)
    assert m.is_free is True


def test_model_info_estimated_cost():
    m = ModelInfo(name="test", location="cloud", provider="anthropic",
                  litellm_name="anthropic/claude-sonnet-4-20250514",
                  cost_per_1k_input=0.003, cost_per_1k_output=0.015)
    cost = m.estimated_cost(1000, 1000)
    assert cost > 0


def test_model_info_score_for():
    m = ModelInfo(name="test", location="local", provider="local",
                  litellm_name="local/test",
                  capabilities={"reasoning": 7.0, "code_generation": 8.0})
    assert m.score_for("reasoning") == 7.0
    assert m.score_for("nonexistent") == 0.0


def test_registry_init_empty():
    reg = ModelRegistry()
    assert reg.all_models() == []


def test_registry_load_yaml(tmp_path):
    yaml_content = """
cloud_models:
  - name: test-cloud
    litellm_name: anthropic/test
    provider: anthropic
    context_length: 200000
    max_tokens: 8192
    capabilities:
      reasoning: 9.0
      code_generation: 8.5
    cost_per_1k_input: 0.003
    cost_per_1k_output: 0.015
"""
    yaml_file = tmp_path / "models.yaml"
    yaml_file.write_text(yaml_content)
    reg = ModelRegistry()
    models = reg.load_yaml(str(yaml_file))
    assert len(models) >= 1
    m = next(m for m in models if m.name == "test-cloud")
    assert m.provider == "anthropic"
    assert m.context_length == 200000


def test_registry_get_model():
    reg = ModelRegistry()
    m = ModelInfo(name="test", location="local", provider="local",
                  litellm_name="local/test")
    reg._models = {"test": m}
    assert reg.get("test") is m
    assert reg.get("nonexistent") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/fatih_hoca/tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fatih_hoca.registry'`

- [ ] **Step 3: Create registry.py**

Copy `src/models/model_registry.py` to `packages/fatih_hoca/src/fatih_hoca/registry.py` with these changes:

1. Replace `from .model_profiles import` → `from fatih_hoca.profiles import`
2. Remove `from .capabilities import ALL_CAPABILITIES, Cap, ...` — these will be in the same package but registry shouldn't score; keep only `ALL_CAPABILITIES` and `Cap` imports (add them after capabilities.py is created; for now use string literals or a local constant)
3. Remove `from .gpu_monitor import get_gpu_monitor` — VRAM values are passed as params to `calculate_dynamic_context` and `calculate_gpu_layers` (they already accept these as args)
4. Remove `from .benchmark.benchmark_fetcher import ...` — benchmark enrichment will be wired later; leave a stub `enrich_with_benchmarks()` that's a no-op for now
5. Remove `get_registry()` singleton — replace with `ModelRegistry` class that holds the model dict
6. Keep: `ModelInfo`, `read_gguf_metadata`, `scan_model_directory`, `estimate_capabilities`, `calculate_dynamic_context`, `calculate_gpu_layers`, `detect_vision_support`, `find_mmproj_path`, `detect_function_calling`, `detect_thinking_model`, YAML parsing
7. Keep constants: `KNOWN_PROVIDERS`, `PROVIDER_PREFIXES`, `_TOOL_CALL_FAMILIES`, `_THINKING_FAMILIES`, `_THINKING_NAME_PATTERNS`, `_FREE_TIER_DEFAULTS`

The `ModelRegistry` class wraps the model dict:

```python
class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ModelInfo] = {}

    def load_yaml(self, path: str) -> list[ModelInfo]:
        """Load cloud models from YAML catalog."""
        # ... existing YAML parsing logic ...

    def load_gguf_dir(self, model_dir: str) -> list[ModelInfo]:
        """Scan directory for GGUF files, build ModelInfo for each."""
        # ... existing scan_model_directory + enrichment logic ...

    def register(self, model: ModelInfo) -> None:
        self._models[model.name] = model

    def get(self, name: str) -> ModelInfo | None:
        return self._models.get(name)

    def all_models(self) -> list[ModelInfo]:
        return list(self._models.values())

    def by_litellm_name(self, litellm_name: str) -> ModelInfo | None:
        for m in self._models.values():
            if m.litellm_name == litellm_name:
                return m
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/fatih_hoca/tests/test_registry.py -v`
Expected: PASS (all 8 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/registry.py packages/fatih_hoca/tests/test_registry.py
git commit -m "feat(fatih_hoca): add registry module (ModelInfo, GGUF scan, YAML catalog)"
```

---

## Task 6: Fatih Hoca — Capabilities (Cap, Task Profiles, Scoring)

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/capabilities.py`
- Test: `packages/fatih_hoca/tests/test_capabilities.py`

Extract `Cap` enum, `TASK_PROFILES`, `TaskRequirements`, `score_model_for_task()` from `src/models/capabilities.py` (492 lines). This file has no external deps — pure data and math.

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_capabilities.py
from fatih_hoca.capabilities import (
    Cap,
    ALL_CAPABILITIES,
    TASK_PROFILES,
    TaskRequirements,
    score_model_for_task,
)


def test_cap_enum_has_reasoning():
    assert Cap.REASONING is not None
    assert Cap.REASONING.value == "reasoning"


def test_all_capabilities_count():
    # 14 or 15 capabilities
    assert len(ALL_CAPABILITIES) >= 14


def test_task_profiles_has_coder():
    assert "coder" in TASK_PROFILES
    profile = TASK_PROFILES["coder"]
    assert "code_generation" in profile
    assert profile["code_generation"] > 0.5


def test_task_profiles_has_shopping_advisor():
    assert "shopping_advisor" in TASK_PROFILES


def test_score_model_strong_match():
    """A strong coder model should score well for a coder task."""
    caps = {"code_generation": 9.0, "code_reasoning": 8.0, "reasoning": 7.0,
            "tool_use": 7.0, "instruction_adherence": 7.0}
    operational = {"supports_function_calling": True, "context_length": 32768,
                   "cost_per_1k_output": 0.0, "tokens_per_second": 20.0}
    reqs = TaskRequirements(task_name="coder")
    score = score_model_for_task(caps, operational, reqs)
    assert score > 5.0  # Good match


def test_score_model_hard_reject_no_function_calling():
    """Model without function calling rejected for task needing it."""
    caps = {"code_generation": 9.0}
    operational = {"supports_function_calling": False, "context_length": 32768,
                   "cost_per_1k_output": 0.0}
    reqs = TaskRequirements(task_name="coder", needs_function_calling=True)
    score = score_model_for_task(caps, operational, reqs)
    assert score == -1


def test_score_model_hard_reject_context_too_small():
    """Model with too-small context is rejected."""
    caps = {"reasoning": 8.0}
    operational = {"supports_function_calling": True, "context_length": 2048,
                   "cost_per_1k_output": 0.0}
    reqs = TaskRequirements(task_name="coder", min_context=8192)
    score = score_model_for_task(caps, operational, reqs)
    assert score == -1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/fatih_hoca/tests/test_capabilities.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fatih_hoca.capabilities'`

- [ ] **Step 3: Copy capabilities.py into the package**

Copy `src/models/capabilities.py` to `packages/fatih_hoca/src/fatih_hoca/capabilities.py`. This file has zero imports from `src/` — only stdlib. No changes needed except verifying there are no `from src.` or `from .` imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/fatih_hoca/tests/test_capabilities.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/capabilities.py packages/fatih_hoca/tests/test_capabilities.py
git commit -m "feat(fatih_hoca): add capabilities module (Cap enum, task profiles, scoring)"
```

---

## Task 7: Fatih Hoca — Requirements (ModelRequirements, AGENT_REQUIREMENTS, QuotaPlanner)

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/requirements.py`
- Test: `packages/fatih_hoca/tests/test_requirements.py`

Extract `ModelRequirements` from `src/core/router.py` (lines ~60-130), `AGENT_REQUIREMENTS` dict (line ~833), `CAPABILITY_TO_TASK` mapping, and `QuotaPlanner` from `src/models/quota_planner.py`.

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_requirements.py
from fatih_hoca.requirements import (
    ModelRequirements,
    AGENT_REQUIREMENTS,
    CAPABILITY_TO_TASK,
    QuotaPlanner,
)
from fatih_hoca.capabilities import TASK_PROFILES


def test_model_requirements_defaults():
    reqs = ModelRequirements(task="coder")
    assert reqs.task == "coder"
    assert reqs.difficulty == 5
    assert reqs.needs_function_calling is False
    assert reqs.priority == 5


def test_model_requirements_effective_task():
    reqs = ModelRequirements(task="coder", primary_capability="code_generation")
    et = reqs.effective_task
    assert isinstance(et, str)


def test_model_requirements_task_profile():
    reqs = ModelRequirements(task="coder")
    profile = reqs.task_profile
    assert isinstance(profile, dict)
    assert len(profile) > 0


def test_model_requirements_escalate():
    reqs = ModelRequirements(task="coder", difficulty=5)
    escalated = reqs.escalate()
    assert escalated.difficulty > reqs.difficulty
    assert escalated.prefer_quality is True


def test_agent_requirements_has_shopping_advisor():
    assert "shopping_advisor" in AGENT_REQUIREMENTS
    reqs = AGENT_REQUIREMENTS["shopping_advisor"]
    assert isinstance(reqs, ModelRequirements)


def test_agent_requirements_has_coder():
    assert "coder" in AGENT_REQUIREMENTS


def test_capability_to_task_mapping():
    assert isinstance(CAPABILITY_TO_TASK, dict)
    assert len(CAPABILITY_TO_TASK) > 0
    # Every mapped task should exist in TASK_PROFILES or be derivable
    for cap, task in CAPABILITY_TO_TASK.items():
        assert isinstance(cap, str)
        assert isinstance(task, str)


def test_quota_planner_initial_threshold():
    qp = QuotaPlanner()
    assert 1 <= qp.expensive_threshold <= 10


def test_quota_planner_recalculate_low_util():
    qp = QuotaPlanner()
    qp.update_paid_utilization("anthropic", 20.0, reset_in=3600)
    threshold = qp.recalculate()
    assert threshold <= 5  # low utilization → lower threshold (use paid more freely)


def test_quota_planner_recalculate_high_util():
    qp = QuotaPlanner()
    qp.update_paid_utilization("anthropic", 90.0, reset_in=300)
    threshold = qp.recalculate()
    assert threshold >= 7  # high utilization → higher threshold (reserve paid for hard tasks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/fatih_hoca/tests/test_requirements.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fatih_hoca.requirements'`

- [ ] **Step 3: Create requirements.py**

Combine into `packages/fatih_hoca/src/fatih_hoca/requirements.py`:

1. `ModelRequirements` dataclass from `src/core/router.py` (the 25+ field dataclass with `effective_task`, `task_profile`, `escalate()` etc.)
   - Change import: `from fatih_hoca.capabilities import TASK_PROFILES, ALL_CAPABILITIES`
2. `AGENT_REQUIREMENTS` dict from `src/core/router.py` line ~833
3. `CAPABILITY_TO_TASK` dict from `src/core/router.py`
4. `QuotaPlanner` class from `src/models/quota_planner.py` (229 lines, standalone — no external imports)
5. `QueueProfile` dataclass from `src/models/quota_planner.py`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/fatih_hoca/tests/test_requirements.py -v`
Expected: PASS (all 10 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/requirements.py packages/fatih_hoca/tests/test_requirements.py
git commit -m "feat(fatih_hoca): add requirements module (ModelRequirements, AGENT_REQUIREMENTS, QuotaPlanner)"
```

---

## Task 8: Fatih Hoca — Ranking (Composite Scoring + Adjustments)

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/ranking.py`
- Test: `packages/fatih_hoca/tests/test_ranking.py`

Extract the scoring pipeline from `select_model()` in `src/core/router.py`. This is the heart of model selection: 5-dimension composite scoring, weight modifiers by difficulty, swap stickiness, specialty alignment, sibling rebalancing, failure adaptation.

The ranking module takes a list of eligible candidates (already filtered by selector) and returns them ranked.

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_ranking.py
from fatih_hoca.ranking import rank_candidates, ScoredModel
from fatih_hoca.registry import ModelInfo
from fatih_hoca.requirements import ModelRequirements
from fatih_hoca.types import Failure
from nerd_herd.types import SystemSnapshot, LocalModelState


def _make_model(name: str, location: str = "local", provider: str = "local",
                capabilities: dict | None = None, tps: float = 20.0,
                cost_in: float = 0.0, cost_out: float = 0.0,
                context: int = 32768, function_calling: bool = True,
                thinking: bool = False, is_loaded: bool = False,
                **kwargs) -> ModelInfo:
    caps = capabilities or {"reasoning": 7.0, "code_generation": 7.0,
                            "tool_use": 6.0, "instruction_adherence": 6.0}
    return ModelInfo(
        name=name, location=location, provider=provider,
        litellm_name=f"{provider}/{name}" if location == "cloud" else f"local/{name}",
        capabilities=caps, tokens_per_second=tps,
        cost_per_1k_input=cost_in, cost_per_1k_output=cost_out,
        context_length=context, supports_function_calling=function_calling,
        thinking_model=thinking, is_loaded=is_loaded, **kwargs,
    )


def test_rank_prefers_loaded_model():
    loaded = _make_model("qwen3-30b", is_loaded=True)
    unloaded = _make_model("llama-8b", is_loaded=False)
    snap = SystemSnapshot(
        vram_available_mb=8000,
        local=LocalModelState(model_name="qwen3-30b", measured_tps=15.0),
    )
    reqs = ModelRequirements(task="coder", difficulty=5)
    ranked = rank_candidates([loaded, unloaded], reqs, snap, failures=[])
    assert ranked[0].model.name == "qwen3-30b"


def test_rank_prefers_cheaper_for_easy_tasks():
    expensive = _make_model("claude", location="cloud", provider="anthropic",
                            cost_in=0.003, cost_out=0.015,
                            capabilities={"reasoning": 9.0, "code_generation": 9.0})
    cheap = _make_model("groq-llama", location="cloud", provider="groq",
                        cost_in=0.0, cost_out=0.0,
                        capabilities={"reasoning": 6.0, "code_generation": 6.0})
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=2)
    ranked = rank_candidates([expensive, cheap], reqs, snap, failures=[])
    assert ranked[0].model.name == "groq-llama"


def test_rank_prefers_stronger_for_hard_tasks():
    strong = _make_model("claude", location="cloud", provider="anthropic",
                         cost_in=0.003, cost_out=0.015,
                         capabilities={"reasoning": 9.5, "code_generation": 9.0,
                                       "tool_use": 9.0, "instruction_adherence": 9.0})
    weak = _make_model("groq-llama", location="cloud", provider="groq",
                       cost_in=0.0, cost_out=0.0,
                       capabilities={"reasoning": 5.0, "code_generation": 5.0,
                                     "tool_use": 5.0, "instruction_adherence": 5.0})
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=9)
    ranked = rank_candidates([strong, weak], reqs, snap, failures=[])
    assert ranked[0].model.name == "claude"


def test_failure_adaptation_timeout_avoids_slow():
    slow = _make_model("slow-model", tps=5.0)
    fast = _make_model("fast-model", tps=30.0)
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=5)
    failures = [Failure(model="local/slow-model", reason="timeout", latency=120.0)]
    ranked = rank_candidates([slow, fast], reqs, snap, failures=failures)
    # slow-model should be penalized or excluded
    assert ranked[0].model.name == "fast-model"


def test_scored_model_has_reasons():
    m = _make_model("test")
    snap = SystemSnapshot()
    reqs = ModelRequirements(task="coder", difficulty=5)
    ranked = rank_candidates([m], reqs, snap, failures=[])
    assert len(ranked) == 1
    assert isinstance(ranked[0].reasons, list)
    assert ranked[0].score > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/fatih_hoca/tests/test_ranking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fatih_hoca.ranking'`

- [ ] **Step 3: Create ranking.py**

Extract from `select_model()` in `src/core/router.py` the scoring logic (Layer 2 and Layer 3). The function signature:

```python
# packages/fatih_hoca/src/fatih_hoca/ranking.py
"""Composite scoring, weight modifiers, swap stickiness, failure adaptation."""
from __future__ import annotations

from dataclasses import dataclass, field

from fatih_hoca.capabilities import TASK_PROFILES, score_model_for_task
from fatih_hoca.registry import ModelInfo
from fatih_hoca.requirements import ModelRequirements
from fatih_hoca.types import Failure
from nerd_herd.types import SystemSnapshot


@dataclass
class ScoredModel:
    model: ModelInfo
    score: float = 0.0
    capability_score: float = 0.0
    composite_score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def litellm_name(self) -> str:
        return self.model.litellm_name


def rank_candidates(
    candidates: list[ModelInfo],
    reqs: ModelRequirements,
    snapshot: SystemSnapshot,
    failures: list[Failure],
) -> list[ScoredModel]:
    """Score and rank eligible candidates. Returns sorted list, best first."""
    # ... extract from select_model() Layer 2 + Layer 3 ...
```

Key logic to extract from `select_model()`:
1. **Weight calculation** based on difficulty (lines ~200-230 of router.py):
   - difficulty 1-3: capability=25, cost=35, availability=20, performance=15, speed=5
   - difficulty 4-5: capability=35, cost=25, availability=20, performance=15, speed=5
   - difficulty 6-7: capability=45, cost=15, availability=20, performance=15, speed=5
   - difficulty 8-10: capability=40, cost=10, availability=15, performance=10, speed=25

2. **Prefer modifiers**: speed → speed_weight+15, quality → cap_weight+10, local → local bonus +10

3. **Per-model scoring** (5 dimensions, 0-100 each):
   - Capability Fit: task profile dot product
   - Cost Efficiency: local=100, free cloud=90, cheap=70, expensive=40-0
   - Availability: loaded=100, not loaded=50, cloud based on utilization
   - Performance History: 80 (placeholder)
   - Speed: tps-based

4. **Layer 3 adjustments**: thinking fitness, specialty alignment, swap stickiness (loaded model bonus)

5. **Sibling rebalancing**: nudge underutilized provider siblings +8%

6. **Failure adaptation**: timeout → exclude model or penalize slow; rate_limit → penalize provider; context_overflow → already filtered by selector; quality_failure → penalize model

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/fatih_hoca/tests/test_ranking.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/ranking.py packages/fatih_hoca/tests/test_ranking.py
git commit -m "feat(fatih_hoca): add ranking module (composite scoring, failure adaptation)"
```

---

## Task 9: Fatih Hoca — Selector (Eligibility Filtering + select())

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/selector.py`
- Test: `packages/fatih_hoca/tests/test_selector.py`

The selector is the main entry point. `select()` does:
1. Get system snapshot from Nerd Herd
2. Filter candidates by eligibility (hard gates from Layer 1 of `select_model()`)
3. Call `rank_candidates()` to score eligible candidates
4. Return top `Pick` or `None`

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_selector.py
from unittest.mock import MagicMock
from fatih_hoca.selector import Selector
from fatih_hoca.registry import ModelInfo, ModelRegistry
from fatih_hoca.types import Pick, Failure
from nerd_herd.types import SystemSnapshot, LocalModelState


def _make_model(name: str, location: str = "local", provider: str = "local",
                capabilities: dict | None = None, tps: float = 20.0,
                cost_in: float = 0.0, cost_out: float = 0.0,
                context: int = 32768, function_calling: bool = True,
                thinking: bool = False, is_loaded: bool = False,
                demoted: bool = False, **kwargs) -> ModelInfo:
    caps = capabilities or {"reasoning": 7.0, "code_generation": 7.0,
                            "tool_use": 6.0, "instruction_adherence": 6.0}
    return ModelInfo(
        name=name, location=location, provider=provider,
        litellm_name=f"{provider}/{name}" if location == "cloud" else f"local/{name}",
        capabilities=caps, tokens_per_second=tps,
        cost_per_1k_input=cost_in, cost_per_1k_output=cost_out,
        context_length=context, supports_function_calling=function_calling,
        thinking_model=thinking, is_loaded=is_loaded, demoted=demoted, **kwargs,
    )


def _make_selector(*models: ModelInfo) -> Selector:
    reg = ModelRegistry()
    for m in models:
        reg.register(m)
    nerd_herd = MagicMock()
    nerd_herd.snapshot.return_value = SystemSnapshot(
        vram_available_mb=8000,
        local=LocalModelState(model_name=models[0].name if models else None),
    )
    return Selector(registry=reg, nerd_herd=nerd_herd)


def test_select_returns_pick():
    m = _make_model("qwen3-30b", is_loaded=True)
    sel = _make_selector(m)
    pick = sel.select(task="coder", difficulty=5)
    assert isinstance(pick, Pick)
    assert pick.model.name == "qwen3-30b"
    assert pick.min_time_seconds > 0


def test_select_returns_none_when_no_models():
    reg = ModelRegistry()
    nerd_herd = MagicMock()
    nerd_herd.snapshot.return_value = SystemSnapshot()
    sel = Selector(registry=reg, nerd_herd=nerd_herd)
    pick = sel.select(task="coder", difficulty=5)
    assert pick is None


def test_select_excludes_demoted():
    good = _make_model("good", is_loaded=True)
    bad = _make_model("bad", demoted=True)
    sel = _make_selector(good, bad)
    pick = sel.select(task="coder", difficulty=5)
    assert pick.model.name == "good"


def test_select_filters_context_overflow():
    small_ctx = _make_model("small", context=2048, is_loaded=True)
    big_ctx = _make_model("big", context=32768)
    sel = _make_selector(small_ctx, big_ctx)
    pick = sel.select(task="coder", difficulty=5, min_context_length=8192)
    assert pick.model.name == "big"


def test_select_with_failures_avoids_failed_model():
    m1 = _make_model("model-a", is_loaded=True)
    m2 = _make_model("model-b")
    sel = _make_selector(m1, m2)
    failures = [Failure(model="local/model-a", reason="timeout", latency=120.0)]
    pick = sel.select(task="coder", difficulty=5, failures=failures)
    # model-a should be penalized, model-b preferred
    assert pick is not None


def test_select_respects_needs_function_calling():
    no_fc = _make_model("no-fc", function_calling=False, is_loaded=True)
    has_fc = _make_model("has-fc", function_calling=True)
    sel = _make_selector(no_fc, has_fc)
    pick = sel.select(task="coder", difficulty=5, needs_function_calling=True)
    assert pick.model.name == "has-fc"


def test_select_swap_budget_prefers_loaded():
    loaded = _make_model("loaded", is_loaded=True,
                         capabilities={"reasoning": 5.0, "code_generation": 5.0})
    better = _make_model("better", is_loaded=False,
                         capabilities={"reasoning": 9.0, "code_generation": 9.0})
    sel = _make_selector(loaded, better)
    # Exhaust swap budget
    sel._swap_budget.record_swap()
    sel._swap_budget.record_swap()
    sel._swap_budget.record_swap()
    pick = sel.select(task="coder", difficulty=5)
    # With budget exhausted, should strongly prefer loaded
    assert pick.model.name == "loaded"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/fatih_hoca/tests/test_selector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fatih_hoca.selector'`

- [ ] **Step 3: Create selector.py**

```python
# packages/fatih_hoca/src/fatih_hoca/selector.py
"""Model selector — eligibility filtering, Nerd Herd integration, top pick."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fatih_hoca.ranking import rank_candidates, ScoredModel
from fatih_hoca.registry import ModelInfo, ModelRegistry
from fatih_hoca.requirements import ModelRequirements, CAPABILITY_TO_TASK
from fatih_hoca.types import Pick, Failure, SwapBudget

if TYPE_CHECKING:
    from nerd_herd import NerdHerd

log = logging.getLogger("fatih_hoca.selector")


class Selector:
    def __init__(self, registry: ModelRegistry, nerd_herd: object) -> None:
        self._registry = registry
        self._nerd_herd = nerd_herd
        self._swap_budget = SwapBudget(max_swaps=3, window_seconds=300)

    def select(
        self,
        task: str = "",
        agent_type: str = "",
        difficulty: int = 5,
        needs_function_calling: bool = False,
        needs_vision: bool = False,
        needs_thinking: bool = True,
        estimated_input_tokens: int = 0,
        estimated_output_tokens: int = 0,
        min_context_length: int = 0,
        max_cost: float = 0.0,
        prefer_speed: bool = False,
        prefer_local: bool = False,
        priority: int = 5,
        local_only: bool = False,
        failures: list[Failure] | None = None,
    ) -> Pick | None:
        """Select best model for the given task. Returns Pick or None."""
        failures = failures or []
        snapshot = self._nerd_herd.snapshot()

        # Build ModelRequirements for internal use
        reqs = ModelRequirements(
            task=task,
            agent_type=agent_type,
            difficulty=difficulty,
            needs_function_calling=needs_function_calling,
            needs_vision=needs_vision,
            needs_thinking=needs_thinking,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            min_context_length=min_context_length,
            max_cost=max_cost,
            prefer_speed=prefer_speed,
            prefer_local=prefer_local,
            priority=priority,
            local_only=local_only,
        )

        # Layer 1: Eligibility — hard pass/fail gates
        all_models = self._registry.all_models()
        failed_names = {f.model for f in failures}
        candidates = []
        for m in all_models:
            reason = self._check_eligibility(m, reqs, snapshot, failed_names)
            if reason is None:
                candidates.append(m)
            else:
                log.debug("SKIP %s: %s", m.name, reason)

        if not candidates:
            log.warning("No eligible candidates for task=%s difficulty=%d", task, difficulty)
            return None

        # Layer 2+3: Score and rank
        ranked = rank_candidates(candidates, reqs, snapshot, failures)
        if not ranked:
            return None

        best = ranked[0]

        # Estimate minimum generation time
        tps = best.model.tokens_per_second or 10.0
        est_output = estimated_output_tokens or 500
        min_time = est_output / tps
        if best.model.thinking_model and needs_thinking:
            min_time *= 3  # thinking tokens overhead

        log.info("SELECT %s (score=%.1f) for task=%s d=%d [%d candidates]",
                 best.model.name, best.score, task, difficulty, len(candidates))

        return Pick(model=best.model, min_time_seconds=min_time)

    def _check_eligibility(
        self,
        model: ModelInfo,
        reqs: ModelRequirements,
        snapshot,
        failed_names: set[str],
    ) -> str | None:
        """Return rejection reason, or None if eligible."""
        if model.demoted:
            return "demoted"

        if model.litellm_name in failed_names:
            return f"failed: {model.litellm_name}"

        if reqs.needs_function_calling and not model.supports_function_calling:
            return "no function calling"

        if reqs.needs_vision and not model.has_vision:
            return "no vision"

        effective_context = reqs.min_context_length
        if effective_context == 0 and (reqs.estimated_input_tokens or reqs.estimated_output_tokens):
            effective_context = int((reqs.estimated_input_tokens + reqs.estimated_output_tokens) * 1.5)
        if effective_context > 0 and model.context_length < effective_context:
            return f"context {model.context_length} < {effective_context}"

        if reqs.max_cost > 0 and model.estimated_cost(
            reqs.estimated_input_tokens or 1000,
            reqs.estimated_output_tokens or 1000,
        ) > reqs.max_cost:
            return "too expensive"

        if reqs.local_only and not model.is_local:
            return "local_only but cloud model"

        # Swap budget: if exhausted and model is unloaded local, skip
        # (unless local_only or high priority)
        if (model.is_local and not model.is_loaded
                and not self._swap_budget.can_swap(reqs.local_only, reqs.priority)):
            return "swap budget exhausted"

        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/fatih_hoca/tests/test_selector.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/selector.py packages/fatih_hoca/tests/test_selector.py
git commit -m "feat(fatih_hoca): add selector module (eligibility filtering, select() API)"
```

---

## Task 10: Fatih Hoca — Public API (init, select, all_models)

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/__init__.py`
- Test: `packages/fatih_hoca/tests/test_init.py`

Wire up the module-level API: `init()`, `select()`, `all_models()`.

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_init.py
import fatih_hoca
from fatih_hoca import Pick, Failure, ModelInfo, ModelRequirements
from unittest.mock import MagicMock, patch
from nerd_herd.types import SystemSnapshot, LocalModelState


def test_exports_exist():
    assert hasattr(fatih_hoca, "init")
    assert hasattr(fatih_hoca, "select")
    assert hasattr(fatih_hoca, "all_models")
    assert hasattr(fatih_hoca, "Pick")
    assert hasattr(fatih_hoca, "Failure")
    assert hasattr(fatih_hoca, "ModelInfo")
    assert hasattr(fatih_hoca, "ModelRequirements")


def test_all_models_before_init():
    # Before init, should return empty or raise
    models = fatih_hoca.all_models()
    assert isinstance(models, list)


def test_init_with_yaml(tmp_path):
    yaml_content = """
cloud_models:
  - name: test-model
    litellm_name: anthropic/test
    provider: anthropic
    context_length: 200000
    max_tokens: 8192
    capabilities:
      reasoning: 9.0
    cost_per_1k_input: 0.003
    cost_per_1k_output: 0.015
"""
    yaml_file = tmp_path / "models.yaml"
    yaml_file.write_text(yaml_content)

    nerd_herd = MagicMock()
    nerd_herd.snapshot.return_value = SystemSnapshot()

    fatih_hoca.init(
        catalog_path=str(yaml_file),
        nerd_herd=nerd_herd,
    )
    models = fatih_hoca.all_models()
    assert any(m.name == "test-model" for m in models)


def test_select_after_init(tmp_path):
    yaml_content = """
cloud_models:
  - name: test-model
    litellm_name: anthropic/test
    provider: anthropic
    context_length: 200000
    max_tokens: 8192
    capabilities:
      reasoning: 9.0
      code_generation: 8.0
      tool_use: 7.0
      instruction_adherence: 7.0
    cost_per_1k_input: 0.003
    cost_per_1k_output: 0.015
    supports_function_calling: true
"""
    yaml_file = tmp_path / "models.yaml"
    yaml_file.write_text(yaml_content)

    nerd_herd = MagicMock()
    nerd_herd.snapshot.return_value = SystemSnapshot()

    fatih_hoca.init(catalog_path=str(yaml_file), nerd_herd=nerd_herd)
    pick = fatih_hoca.select(task="coder", difficulty=5)
    assert pick is not None
    assert isinstance(pick, Pick)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/fatih_hoca/tests/test_init.py -v`
Expected: FAIL — `AttributeError: module 'fatih_hoca' has no attribute 'init'`

- [ ] **Step 3: Wire up __init__.py**

```python
# packages/fatih_hoca/src/fatih_hoca/__init__.py
"""Fatih Hoca — model manager: scoring, selection, swap budget."""
from __future__ import annotations

from typing import TYPE_CHECKING

from fatih_hoca.types import Pick, Failure, SwapBudget
from fatih_hoca.registry import ModelInfo, ModelRegistry
from fatih_hoca.requirements import ModelRequirements, AGENT_REQUIREMENTS, CAPABILITY_TO_TASK
from fatih_hoca.capabilities import Cap, ALL_CAPABILITIES, TASK_PROFILES
from fatih_hoca.selector import Selector

if TYPE_CHECKING:
    from nerd_herd import NerdHerd

__all__ = [
    "init",
    "select",
    "all_models",
    "Pick",
    "Failure",
    "ModelInfo",
    "ModelRequirements",
    "AGENT_REQUIREMENTS",
    "CAPABILITY_TO_TASK",
    "Cap",
    "ALL_CAPABILITIES",
    "TASK_PROFILES",
]

_selector: Selector | None = None
_registry: ModelRegistry | None = None


def init(
    models_dir: str | None = None,
    catalog_path: str | None = None,
    nerd_herd: object = None,
) -> list[str]:
    """Initialize Fatih Hoca. Scans GGUFs, loads YAML catalog.

    Returns list of model names. select() works after this returns.
    """
    global _selector, _registry

    _registry = ModelRegistry()
    model_names: list[str] = []

    if catalog_path:
        models = _registry.load_yaml(catalog_path)
        model_names.extend(m.name for m in models)

    if models_dir:
        models = _registry.load_gguf_dir(models_dir)
        model_names.extend(m.name for m in models)

    _selector = Selector(registry=_registry, nerd_herd=nerd_herd)
    return model_names


def select(**kwargs) -> Pick | None:
    """Select best model for the given task. See Selector.select() for params."""
    if _selector is None:
        return None
    return _selector.select(**kwargs)


def all_models() -> list[ModelInfo]:
    """Return all known models."""
    if _registry is None:
        return []
    return _registry.all_models()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/fatih_hoca/tests/test_init.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Run all fatih_hoca tests**

Run: `python -m pytest packages/fatih_hoca/tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/__init__.py packages/fatih_hoca/tests/test_init.py
git commit -m "feat(fatih_hoca): wire up public API (init, select, all_models)"
```

---

## Task 11: Migration Shims — router.py, model_registry.py, capabilities.py, quota_planner.py

**Files:**
- Modify: `src/core/router.py` — replace internals with re-exports from fatih_hoca
- Modify: `src/models/model_registry.py` — re-export ModelInfo, get_registry wraps ModelRegistry
- Modify: `src/models/capabilities.py` — re-export Cap, TASK_PROFILES, score_model_for_task
- Modify: `src/models/quota_planner.py` — re-export QuotaPlanner, get_quota_planner

Every existing import path continues to work. Callers don't change yet.

- [ ] **Step 1: Write the integration test**

```python
# tests/test_fatih_hoca_shims.py
"""Verify shims preserve all existing import paths."""


def test_router_imports():
    from src.core.router import (
        ModelRequirements,
        ModelCallFailed,
        ScoredModel,
        select_model,
        CAPABILITY_TO_TASK,
        AGENT_REQUIREMENTS,
    )
    assert ModelRequirements is not None
    assert AGENT_REQUIREMENTS is not None


def test_registry_imports():
    from src.models.model_registry import ModelInfo, get_registry
    assert ModelInfo is not None
    reg = get_registry()
    assert reg is not None


def test_capabilities_imports():
    from src.models.capabilities import (
        Cap,
        ALL_CAPABILITIES,
        TASK_PROFILES,
        TaskRequirements,
        score_model_for_task,
    )
    assert Cap.REASONING is not None
    assert len(ALL_CAPABILITIES) >= 14


def test_quota_planner_imports():
    from src.models.quota_planner import QuotaPlanner, get_quota_planner
    qp = get_quota_planner()
    assert isinstance(qp, QuotaPlanner)
```

- [ ] **Step 2: Run test to verify it passes with OLD code (baseline)**

Run: `python -m pytest tests/test_fatih_hoca_shims.py -v`
Expected: PASS — these imports already work with old code

- [ ] **Step 3: Convert `src/models/capabilities.py` to shim**

Replace the entire file with:

```python
"""Capabilities — shim re-exporting from fatih_hoca."""
from fatih_hoca.capabilities import (  # noqa: F401
    Cap,
    ALL_CAPABILITIES,
    KNOWLEDGE_DIMENSIONS,
    REASONING_DIMENSIONS,
    EXECUTION_DIMENSIONS,
    TASK_PROFILES,
    TaskRequirements,
    score_model_for_task,
    rank_models_for_task,
)
```

- [ ] **Step 4: Convert `src/models/quota_planner.py` to shim**

Replace the entire file with:

```python
"""Quota planner — shim re-exporting from fatih_hoca."""
from fatih_hoca.requirements import QuotaPlanner, QueueProfile  # noqa: F401

_instance: QuotaPlanner | None = None


def get_quota_planner() -> QuotaPlanner:
    global _instance
    if _instance is None:
        _instance = QuotaPlanner()
    return _instance
```

- [ ] **Step 5: Convert `src/models/model_registry.py` to shim**

This is trickier because `model_registry.py` is 2050 lines and has the `get_registry()` singleton. The shim needs to:
1. Re-export `ModelInfo` from `fatih_hoca.registry`
2. Keep `get_registry()` returning the same singleton pattern but backed by `ModelRegistry`
3. Re-export all utility functions (`scan_model_directory`, `calculate_dynamic_context`, etc.)

```python
"""Model registry — shim re-exporting from fatih_hoca."""
from fatih_hoca.registry import (  # noqa: F401
    ModelInfo,
    ModelRegistry,
    scan_model_directory,
    calculate_dynamic_context,
    calculate_gpu_layers,
    detect_vision_support,
    find_mmproj_path,
    detect_function_calling,
    detect_thinking_model,
    estimate_capabilities,
    read_gguf_metadata,
    KNOWN_PROVIDERS,
    PROVIDER_PREFIXES,
)

_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
```

- [ ] **Step 6: Update `src/core/router.py` to delegate to fatih_hoca**

This is the most complex shim. `router.py` needs to:
1. Re-export `ModelRequirements`, `ScoredModel`, `AGENT_REQUIREMENTS`, `CAPABILITY_TO_TASK` from fatih_hoca
2. Keep `ModelCallFailed` exception
3. Keep `select_model()` but have it delegate to fatih_hoca internals
4. Keep `call_model()` (legacy shim used by hallederiz_kadir)

The key is that `select_model()` currently returns `list[ScoredModel]` while Fatih Hoca's `select()` returns `Pick | None`. During migration, `select_model()` should call ranking directly to preserve the list return type for callers that iterate candidates.

```python
"""Router — shim delegating to fatih_hoca. Preserves all import paths."""
from fatih_hoca.types import Pick, Failure  # noqa: F401
from fatih_hoca.ranking import ScoredModel  # noqa: F401
from fatih_hoca.requirements import (  # noqa: F401
    ModelRequirements,
    AGENT_REQUIREMENTS,
    CAPABILITY_TO_TASK,
)
from fatih_hoca.capabilities import ALL_CAPABILITIES, Cap, TASK_PROFILES  # noqa: F401
from fatih_hoca.registry import ModelInfo  # noqa: F401


class ModelCallFailed(RuntimeError):
    pass


def select_model(reqs: ModelRequirements) -> list[ScoredModel]:
    """Score and rank models. Returns list sorted by score descending.

    Shim: delegates to fatih_hoca ranking after eligibility filtering.
    """
    import fatih_hoca
    from fatih_hoca.ranking import rank_candidates
    from nerd_herd.types import SystemSnapshot

    # Get snapshot from the initialized selector
    if fatih_hoca._selector is None:
        return []

    snapshot = fatih_hoca._selector._nerd_herd.snapshot()
    candidates = []
    for m in fatih_hoca._registry.all_models():
        reason = fatih_hoca._selector._check_eligibility(m, reqs, snapshot, set())
        if reason is None:
            candidates.append(m)

    if not candidates:
        return []

    return rank_candidates(candidates, reqs, snapshot, failures=[])


def select_for_task(task: str, **kwargs) -> list[ScoredModel]:
    reqs = ModelRequirements(task=task, **kwargs)
    return select_model(reqs)
```

Note: `call_model()` stays in router.py during migration since hallederiz_kadir imports it. It continues to work through the dispatcher. Don't move it yet.

- [ ] **Step 7: Run shim tests**

Run: `python -m pytest tests/test_fatih_hoca_shims.py -v`
Expected: PASS

- [ ] **Step 8: Run existing test suite to verify no regressions**

Run: `python -m pytest tests/ -x --ignore=tests/integration -q`
Expected: All existing tests PASS (or same failures as before)

- [ ] **Step 9: Commit**

```bash
git add src/core/router.py src/models/model_registry.py src/models/capabilities.py src/models/quota_planner.py tests/test_fatih_hoca_shims.py
git commit -m "refactor: convert router, registry, capabilities, quota_planner to fatih_hoca shims"
```

---

## Task 12: Dispatcher Simplification

**Files:**
- Modify: `src/core/llm_dispatcher.py`

Simplify the dispatcher to the ask-load-call-retry loop from the spec. This is the biggest single change. The approach: remove dead code methods and replace the candidate loop with `fatih_hoca.select()`.

- [ ] **Step 1: Write the test for new dispatcher flow**

```python
# tests/test_dispatcher_simplified.py
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from src.core.llm_dispatcher import LLMDispatcher, CallCategory


@pytest.fixture
def dispatcher():
    d = LLMDispatcher()
    return d


@pytest.mark.asyncio
async def test_request_calls_fatih_hoca_select(dispatcher):
    """Verify dispatcher delegates model selection to fatih_hoca."""
    mock_model = MagicMock()
    mock_model.name = "test-model"
    mock_model.is_local = False
    mock_model.litellm_name = "anthropic/test"
    mock_model.thinking_model = False

    mock_pick = MagicMock()
    mock_pick.model = mock_model
    mock_pick.min_time_seconds = 30.0

    mock_result = MagicMock()
    mock_result.content = "test response"

    with patch("fatih_hoca.select", return_value=mock_pick), \
         patch("src.core.llm_dispatcher.hallederiz_kadir") as mock_hk:
        mock_hk.call = AsyncMock(return_value=mock_result)
        # This test verifies the integration point exists
        # Exact behavior depends on implementation details
```

- [ ] **Step 2: Remove dead methods from dispatcher**

Remove these methods/functions from `src/core/llm_dispatcher.py`:

| Method | Reason |
|--------|--------|
| `_select_candidates()` | Replaced by `fatih_hoca.select()` |
| `_prepare_overhead_reqs()` | Overhead uses same `select()` path |
| `_try_pinned_loaded()` | Swap stickiness handled by ranking |
| `_exclude_unloaded_local()` | Eligibility handled by selector |
| `_exclude_all_local()` | Eligibility handled by selector |
| `_should_wait_for_cold_start()` | `select()` returns None, task retries |
| `_wait_for_model_load()` | `select()` returns None, task retries |
| `ensure_gpu_utilized()` | Eliminated per spec |
| `_find_best_local_for_batch()` | Eliminated per spec |
| `_find_fastest_general_model()` | Eliminated per spec |
| `_has_pending_overhead_needs()` | Eliminated per spec |
| `_loaded_model_can_grade()` | Eliminated per spec |
| `_get_grade_exclusions()` | Eliminated per spec |
| `on_model_swap()` grade draining | Simplified — just wakes tasks |

- [ ] **Step 3: Rewrite `request()` method**

Replace the existing `request()` method with the simplified version from the spec:

```python
async def request(self, category, task, agent_type, difficulty,
                  messages, tools=None, failures=None, **kwargs):
    import fatih_hoca
    from fatih_hoca.types import Failure

    pick = fatih_hoca.select(
        task=task,
        agent_type=agent_type,
        difficulty=difficulty,
        needs_thinking=category != CallCategory.OVERHEAD,
        failures=failures or [],
        **kwargs,
    )
    if pick is None:
        raise ModelCallFailed(f"No model available for task={task} d={difficulty}")

    if pick.model.is_local:
        await self._ensure_local_model(pick.model)

    messages = self._prepare_messages(messages, pick.model)
    timeout = max(pick.min_time_seconds, self._timeout_floor(category))

    result = await hallederiz_kadir.call(
        model=pick.model,
        messages=messages,
        tools=tools,
        timeout=timeout,
    )

    if isinstance(result, CallResult):
        return self._to_response_dict(result)

    # Failed — tell Fatih Hoca what happened
    failure = Failure(
        model=pick.model.litellm_name,
        reason=result.category,
        latency=result.latency,
    )
    all_failures = (failures or []) + [failure]
    if len(all_failures) >= 5:
        raise ModelCallFailed(f"All attempts failed for task={task}")

    return await self.request(
        category, task, agent_type, difficulty,
        messages, tools,
        failures=all_failures,
        **kwargs,
    )
```

- [ ] **Step 4: Simplify `_compute_timeout` to `_timeout_floor`**

```python
def _timeout_floor(self, category: CallCategory) -> float:
    if category == CallCategory.OVERHEAD:
        return 20.0
    return 30.0
```

- [ ] **Step 5: Keep what stays**

Keep these methods unchanged:
- `_prepare_messages()` — secret redaction, thinking adaptation
- `_to_response_dict()` / `_result_to_dict()` — convert CallResult to legacy dict
- `_ensure_local_model()` — calls dallama.ensure_model
- `is_loaded_model_thinking()` — query method

- [ ] **Step 6: Remove SwapBudget from dispatcher (now in Fatih Hoca)**

Remove the `SwapBudget` class and `self.swap_budget` from dispatcher. It's now owned by the Selector.

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_llm_dispatcher.py -v`
Expected: Some tests will need updating since methods were removed. Update test mocks to match new flow.

Run: `python -m pytest tests/ -x --ignore=tests/integration -q`
Expected: PASS (with updated tests)

- [ ] **Step 8: Commit**

```bash
git add src/core/llm_dispatcher.py tests/test_llm_dispatcher.py
git commit -m "refactor(dispatcher): simplify to ask-load-call-retry loop via fatih_hoca"
```

---

## Task 13: Wire Nerd Herd Push Callbacks

**Files:**
- Modify: `src/models/local_model_manager.py` — push LocalModelState to Nerd Herd on swap
- Modify: `src/core/orchestrator.py` — wire init sequence

DaLLaMa's `on_ready` callback already fires on model load/unload. Wire it to push state to Nerd Herd. Similarly, KDV's capacity events should push CloudProviderState.

- [ ] **Step 1: Write the test**

```python
# tests/test_nerd_herd_push.py
from unittest.mock import MagicMock
from nerd_herd.types import LocalModelState, SystemSnapshot


def test_push_local_state_updates_snapshot():
    """Verify that pushing local state is reflected in snapshot."""
    from nerd_herd.nerd_herd import NerdHerd
    nh = NerdHerd(metrics_port=0)
    nh.push_local_state(LocalModelState(
        model_name="qwen3-30b",
        thinking_enabled=True,
        measured_tps=15.0,
        context_length=8192,
    ))
    snap = nh.snapshot()
    assert snap.local.model_name == "qwen3-30b"
    assert snap.local.thinking_enabled is True
```

- [ ] **Step 2: Run test to verify it passes (already implemented in Task 2)**

Run: `python -m pytest tests/test_nerd_herd_push.py -v`
Expected: PASS

- [ ] **Step 3: Wire local_model_manager _on_ready to push_local_state**

In `src/models/local_model_manager.py`, modify the `_on_ready` callback:

```python
def _on_ready(self, model_name: str, reason: str) -> None:
    # ... existing logic ...

    # Push state to Nerd Herd
    from nerd_herd.types import LocalModelState
    nerd_herd = self._get_nerd_herd()
    if nerd_herd:
        if reason in ("model_loaded", "inference_complete"):
            nerd_herd.push_local_state(LocalModelState(
                model_name=model_name,
                thinking_enabled=self._thinking_enabled,
                vision_enabled=self._vision_enabled,
                measured_tps=self.runtime_state.measured_tps if self.runtime_state else 0.0,
                context_length=self.runtime_state.context_length if self.runtime_state else 0,
                is_swapping=False,
            ))
        elif reason == "idle_unload":
            nerd_herd.push_local_state(LocalModelState())  # cleared
```

- [ ] **Step 4: Wire orchestrator init to pass nerd_herd to fatih_hoca.init()**

In `src/core/orchestrator.py`, find where the router/registry are initialized and add:

```python
import fatih_hoca

fatih_hoca.init(
    models_dir=config.MODELS_DIR,
    catalog_path=str(Path(__file__).parent.parent / "models" / "models.yaml"),
    nerd_herd=self.nerd_herd,
)
```

- [ ] **Step 5: Run integration check**

Run: `python -c "from src.core.orchestrator import Orchestrator; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add src/models/local_model_manager.py src/core/orchestrator.py tests/test_nerd_herd_push.py
git commit -m "feat: wire Nerd Herd push callbacks from DaLLaMa and orchestrator init"
```

---

## Task 14: Delete Dead Code

**Files:**
- Delete: `src/models/gpu_scheduler.py` — dead code per architecture doc
- Delete: `src/models/gpu_monitor.py` — shim to nerd_herd, no longer imported after Fatih Hoca
- Delete: `src/models/header_parser.py` — shim to kuleden_donen_var
- Delete: `src/models/rate_limiter.py` — shim to kuleden_donen_var, only imported by old router.py

- [ ] **Step 1: Verify no remaining imports**

Run grep for each file to confirm nothing imports them anymore:

```bash
grep -r "from src.models.gpu_scheduler" src/ packages/ --include="*.py"
grep -r "from src.models.gpu_monitor" src/ packages/ --include="*.py"
grep -r "from src.models.header_parser" src/ packages/ --include="*.py"
grep -r "from src.models.rate_limiter" src/ packages/ --include="*.py"
```

If any imports remain, update those files to import from the real packages (`nerd_herd`, `kuleden_donen_var`) directly.

- [ ] **Step 2: Delete the files**

```bash
rm src/models/gpu_scheduler.py
rm src/models/gpu_monitor.py
rm src/models/header_parser.py
rm src/models/rate_limiter.py
```

- [ ] **Step 3: Run tests to verify nothing breaks**

Run: `python -m pytest tests/ -x --ignore=tests/integration -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add -u src/models/
git commit -m "chore: delete dead shims (gpu_scheduler, gpu_monitor, header_parser, rate_limiter)"
```

---

## Task 15: Install + Full Test Suite

**Files:**
- Modify: `requirements.txt` — add `-e packages/fatih_hoca`

- [ ] **Step 1: Add fatih_hoca to requirements.txt**

Add this line alongside the other editable packages:

```
-e packages/fatih_hoca
```

- [ ] **Step 2: Install**

```bash
pip install -e packages/fatih_hoca
```

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -x --ignore=tests/integration -q`
Expected: ALL PASS

- [ ] **Step 4: Run integration tests (no LLM)**

Run: `python -m pytest tests/integration/ -m "not llm" -q`
Expected: PASS

- [ ] **Step 5: Smoke test imports**

```bash
python -c "
import fatih_hoca
from fatih_hoca import Pick, Failure, ModelInfo, ModelRequirements, Cap
from fatih_hoca import AGENT_REQUIREMENTS, CAPABILITY_TO_TASK, TASK_PROFILES
from src.core.router import ModelRequirements, select_model, AGENT_REQUIREMENTS
from src.models.model_registry import ModelInfo, get_registry
from src.models.capabilities import Cap, TASK_PROFILES, score_model_for_task
from src.models.quota_planner import QuotaPlanner, get_quota_planner
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt
git commit -m "chore: add fatih_hoca to requirements.txt"
```

---

## Task 16: Update Architecture Documentation

**Files:**
- Modify: `docs/architecture-modularization.md`

- [ ] **Step 1: Update Fatih Hoca entry in extracted packages table**

Change status from `Planned` to `Stable v0.1.0`:

```markdown
| **fatih_hoca** | Model manager: scoring, selection, swap budget, failure adaptation | `packages/fatih_hoca/` | Stable v0.1.0 | nerd_herd |
```

- [ ] **Step 2: Update "Extract Next" section**

Move Fatih Hoca from "Extract Next" to a new "Recently Extracted" note, or simply remove it since it's done.

Update the Nerd Herd expansion entry similarly.

- [ ] **Step 3: Update "What Was Removed" section**

Add entries for the deleted shims and the dispatcher simplification.

- [ ] **Step 4: Commit**

```bash
git add docs/architecture-modularization.md
git commit -m "docs: update architecture doc — Fatih Hoca extraction complete"
```
