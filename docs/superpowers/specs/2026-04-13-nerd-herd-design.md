# Nerd Herd — Observability Package Design

## Summary

Nerd Herd is a standalone observability package extracted from KutAI. It collects GPU/system metrics, manages VRAM budget policy, tracks service health, fetches llama-server inference metrics, pre-computes rates, and serves everything via a Prometheus-compatible `/metrics` endpoint. Grafana connects directly — no Prometheus server needed.

## Motivation

KutAI's observability is scattered across 17+ files in `src/infra/` and `src/app/api.py`. The architecture modularization doc (`docs/architecture-modularization.md`) explicitly names GPU Monitor as the next extraction target, with vision: "unified collector for GPU, inference speed (from DaLLaMa), cloud costs, rate limits, task distribution. Serves Prometheus `/metrics` for Grafana."

Current state: a Prometheus container (256MB) scrapes two endpoints (llama-server `:8080`, orchestrator `:8000`), Grafana reads from Prometheus. Nerd Herd replaces Prometheus entirely — it collects from all sources, pre-computes `rate()` values in a ring buffer, and serves a single `/metrics` endpoint that Grafana scrapes directly.

## Scope

### In scope (v1)

- **GPUCollector** — VRAM, utilization, temperature, power, external process detection (port of `gpu_monitor.py`)
- **LoadManager** — 4-mode VRAM budget policy with auto-detect loop (port of `load_manager.py`)
- **HealthRegistry** — capability flags and degraded tracking (port of `runtime_state.py`)
- **InferenceCollector** — fetches llama-server `/metrics`, re-exposes as pre-computed rate gauges
- **CollectorRegistry** — register/unregister collectors, unified collection
- **RingBuffer** — ~5 min sample history per metric, pre-computes 1m and 5m rates
- **Prometheus exposition** — HTTP server on port 9881, Prometheus text format
- **Docker-compose changes** — remove Prometheus service, point Grafana at Nerd Herd
- **Dashboard migration** — update queries from `rate(counter[window])` to direct gauge reads

### Out of scope (v1, future iterations)

- Task distribution metrics
- Model call rates / cost tracking (KutAI registers these as custom collectors in v2)
- Cloud rate limit tracking
- tps ownership (deferred decision)

### Explicitly excluded (stays in KutAI)

| Component | Why |
|-----------|-----|
| `alerting.py` | Business logic (KutAI thresholds), Telegram delivery |
| `tracing.py` | Per-task execution traces, `task_id` is KutAI domain |
| `audit.py` | Actor/action audit log, mission/task domain concepts |
| `dead_letter.py` + `dlq_analyst.py` | Tightly coupled to task/mission/checkpoint system |
| `notifications.py` | Delivery channel with shopping-specific helpers |
| `monitoring.py` | External dependency checker (URL uptime, GitHub issues) |
| `progress.py` | Mission/task milestone tracking |
| Yasar Usta process monitoring | Process lifecycle management, different abstraction |

## Package Structure

```
packages/nerd_herd/
  pyproject.toml
  src/nerd_herd/
    __init__.py          — NerdHerd facade, public API re-exports
    types.py             — GPUState, SystemState, ExternalGPUUsage, LoadMode, HealthStatus
    registry.py          — CollectorRegistry, Collector protocol
    gpu.py               — GPUCollector (pynvml/psutil)
    load.py              — LoadManager (modes, VRAM budget, auto-detect)
    health.py            — HealthRegistry (capability flags)
    inference.py         — InferenceCollector (llama-server metrics fetcher)
    ring_buffer.py       — RingBuffer for rate pre-computation
    exposition.py        — HTTP server, Prometheus text format
    platform.py          — OS-specific helpers (process tree, pid checks)
  tests/
    test_gpu.py
    test_load.py
    test_health.py
    test_inference.py
    test_ring_buffer.py
    test_exposition.py
    test_integration.py
```

## Dependencies

| Dependency | Purpose | Required |
|-----------|---------|----------|
| `yazbunu` | Structured logging | Yes (hard dep) |
| `pynvml` | NVIDIA GPU metrics | Yes (graceful degradation if no GPU) |
| `psutil` | RAM/CPU metrics | Yes |
| `prometheus_client` | Metric types, text format generation | Yes |
| `aiohttp` | HTTP server for `/metrics` + fetching llama-server | Yes |

## Public API

```python
from nerd_herd import NerdHerd, GPUState, LoadMode, HealthStatus

# Initialize
nh = NerdHerd(
    metrics_port=9881,
    llama_server_url="http://127.0.0.1:8080",
    detect_interval=30,       # GPU auto-detect poll interval
    upgrade_delay=300,        # sustained decrease before upgrade
)

# Start collection and HTTP server
await nh.start()

# GPU state
state = nh.gpu_state()                    # GPUState dataclass
state.vram_free_mb, state.temperature_c

# VRAM budget (DaLLaMa uses this)
budget_mb = nh.get_vram_budget_mb()       # raw_free * budget_fraction
fraction = nh.get_vram_budget_fraction()  # 0.0 - 1.0

# Load mode management
await nh.set_load_mode("shared", source="user")
await nh.enable_auto_management()
mode = nh.get_load_mode()                 # "full" | "heavy" | "shared" | "minimal"
allowed = nh.is_local_inference_allowed() # False when minimal

# Callbacks
nh.on_mode_change(callback)               # (old_mode, new_mode, source) -> None

# Health tracking
nh.mark_degraded("telegram")
nh.mark_healthy("telegram")
nh.is_healthy("telegram")
status = nh.get_health_status()           # HealthStatus dataclass

# Custom collectors (KutAI registers its own)
nh.register_collector("tasks", my_task_collector)

# Prometheus
lines = nh.prometheus_lines()             # raw text
# HTTP server already running on metrics_port from start()

# Shutdown
await nh.stop()
```

## Collector Protocol

```python
class Collector(Protocol):
    name: str
    
    def collect(self) -> dict[str, float | int | str]:
        """Return current metrics as flat key-value pairs."""
        ...
    
    def prometheus_metrics(self) -> list[prometheus_client.Metric]:
        """Return prometheus_client metric objects for exposition."""
        ...
```

## CollectorRegistry

```python
class CollectorRegistry:
    def register(self, name: str, collector: Collector) -> None
    def unregister(self, name: str) -> None
    def collect_all(self) -> dict[str, Any]
    def get(self, name: str) -> Collector
    def prometheus_lines(self) -> str        # all collectors' metrics as text
```

Three built-in collectors auto-registered on `NerdHerd()` init:
- `"gpu"` -> GPUCollector
- `"load"` -> LoadManager
- `"health"` -> HealthRegistry

`"inference"` registered when `llama_server_url` is provided.

## GPUCollector

Port of `src/models/gpu_monitor.py` (251 LOC).

**Dataclasses:**

- `GPUState` — available, vram_total/used/free_mb, utilization_pct, temperature_c, power_draw_w. Properties: `is_throttling` (>85C), `is_busy` (>80% util).
- `SystemState` — ram_total/available_mb, cpu_percent, gpu: GPUState. Property: `can_load_model` (RAM > 4GB headroom).
- `ExternalGPUUsage` — detected, external_vram_mb, external_process_count, our_vram_mb, total_vram_mb. Property: `external_vram_fraction`.

**Behavior:**
- 2-second cache to avoid hammering pynvml
- Graceful degradation: if pynvml init fails, `GPUState.available = False`, all values zeroed
- Process tree tracking for external usage detection via `platform.py`

**Prometheus metrics:**
- `nerd_herd_gpu_vram_used_mb`, `_free_mb`, `_total_mb` (gauge)
- `nerd_herd_gpu_utilization_pct` (gauge)
- `nerd_herd_gpu_temperature_c` (gauge)
- `nerd_herd_gpu_power_draw_w` (gauge)
- `nerd_herd_gpu_external_vram_mb`, `_external_processes` (gauge)
- `nerd_herd_system_ram_available_mb` (gauge)
- `nerd_herd_system_cpu_percent` (gauge)

## LoadManager

Port of `src/infra/load_manager.py` (278 LOC).

**Modes and VRAM budgets:**

| Mode | VRAM Budget | Description |
|------|-------------|-------------|
| full | 100% | All local capacity |
| heavy | 90% | Slight OS headroom |
| shared | 50% | Prefer cloud for heavy tasks |
| minimal | 0% | Local inference disabled |

**Key behavior:**
- `get_vram_budget_mb()` = `gpu.vram_free_mb * budget_fraction`
- `set_load_mode(mode, source="user")` — source="user" disables auto-management
- Auto-detect loop: polls GPUCollector every `detect_interval` seconds
  - Downgrade: immediate when external usage increases
  - Upgrade: only after sustained `upgrade_delay` seconds of decreased usage
- `on_mode_change(callback)` — fires `(old_mode, new_mode, source)`. KutAI wires to: persist to DB, invalidate DaLLaMa tps, notify Telegram.

**State:** In-memory only. No DB dependency. KutAI persists via callback if desired.

**Prometheus metrics:**
- `nerd_herd_load_mode` (gauge, 0=minimal, 1=shared, 2=heavy, 3=full)
- `nerd_herd_load_mode_info{mode="..."}` (gauge)
- `nerd_herd_vram_budget_fraction` (gauge)
- `nerd_herd_auto_managed` (gauge)

## HealthRegistry

Port of `src/infra/runtime_state.py` (30 LOC), slightly more structured.

**State:** Dict of capability name -> healthy/degraded with timestamps.

**HealthStatus dataclass:**
- `boot_time: str` (ISO)
- `capabilities: dict[str, bool]`
- `degraded: list[str]`

**Prometheus metrics:**
- `nerd_herd_capability_healthy{name="..."}` (gauge, 0/1) per registered capability

## InferenceCollector

Fetches llama-server's native `/metrics` endpoint and re-exposes as pre-computed rate gauges.

**Fetch interval:** Configurable, default 5s (matches current Prometheus scrape interval for llama-server).

**Source metrics parsed** (from llama-server Prometheus format):
- `llamacpp_tokens_predicted_total`
- `llamacpp_prompt_tokens_total`
- `llamacpp_tokens_predicted_seconds_total`
- `llamacpp_prompt_tokens_seconds_total`
- `llamacpp_requests_processing`
- `llamacpp_requests_pending`
- `llamacpp_kv_cache_usage_ratio`

**Exposed as pre-computed gauges:**
- `nerd_herd_inference_tokens_per_sec` (rolling 1m average from ring buffer)
- `nerd_herd_inference_prompt_tokens_per_sec` (rolling 1m average)
- `nerd_herd_inference_kv_cache_ratio` (gauge, pass-through)
- `nerd_herd_inference_requests_processing` (gauge, pass-through)
- `nerd_herd_inference_requests_pending` (gauge, pass-through)

**Graceful degradation:** If llama-server is down, all gauges report 0. No crash, no exception.

## RingBuffer

In-memory circular buffer for rate pre-computation. Each metric that needs rate computation gets a buffer.

**Design:**
- Fixed-size array of `(timestamp, value)` tuples
- Default capacity: 60 samples (5 min at 5s intervals)
- `append(timestamp, value)` — add sample, evict oldest if full
- `rate(window_seconds)` — compute `(newest - oldest_in_window) / elapsed` for counter-type metrics
- Thread-safe (used from async context, but collectors may run in executor)

**Memory:** ~60 samples * 16 bytes * ~10 metrics = ~10KB. Negligible.

## Prometheus Exposition

**HTTP server:** aiohttp on configurable port (default 9881).

**Single endpoint:** `GET /metrics` — no auth (Grafana needs open access).

**Response:** Prometheus text exposition format generated by `prometheus_client.generate_latest()` combined with custom ring-buffer rate gauges.

**Metric prefixes:**
- `nerd_herd_*` — Nerd Herd's own metrics (GPU, load, health, inference)
- Custom collectors use their own prefix (e.g. KutAI registers `kutay_*` metrics)

## Docker-Compose Changes

**Remove:**
- `prometheus` service
- `prometheus_data` volume

**Update Grafana:**
```yaml
# Datasource changes from:
url: http://prometheus:9090
# To:
url: http://host.docker.internal:9881
```

**Dashboard query migration:**

| Old (PromQL with rate) | New (direct gauge) |
|------------------------|-------------------|
| `rate(llamacpp:tokens_predicted_total[1m])` | `nerd_herd_inference_tokens_per_sec` |
| `rate(llamacpp:prompt_tokens_total[1m])` | `nerd_herd_inference_prompt_tokens_per_sec` |
| `llamacpp:kv_cache_usage_ratio` | `nerd_herd_inference_kv_cache_ratio` |
| `llamacpp:requests_processing` | `nerd_herd_inference_requests_processing` |
| `llamacpp:requests_pending` | `nerd_herd_inference_requests_pending` |
| `rate(kutay_tasks_completed_total[5m])` | `kutay_tasks_completed_rate_5m` |
| `rate(kutay_tasks_failed_total[5m])` | `kutay_tasks_failed_rate_5m` |
| `kutay_queue_depth` | `kutay_queue_depth` |
| `kutay_cost_total_usd` | `kutay_cost_total_usd` |
| `rate(kutay_model_calls_total[5m])` | `kutay_model_calls_rate_5m{model="..."}` |
| `kutay_model_healthy{model}` | `kutay_model_healthy{model}` |
| `kutay_model_swaps_total` | `kutay_model_swaps_total` |
| `kutay_model_idle_seconds` | `kutay_model_idle_seconds` |
| `kutay_model_inference_busy` | `kutay_model_inference_busy` |

## KutAI Integration

### Shim in `src/models/gpu_monitor.py`

After extraction, the existing module becomes a thin shim:

```python
from nerd_herd import NerdHerd

_nh: NerdHerd | None = None

def get_gpu_monitor():
    """Backward-compatible shim — returns GPUCollector from NerdHerd."""
    from src.app.run import get_nerd_herd
    return get_nerd_herd().registry.get("gpu")
```

### Shim in `src/infra/load_manager.py`

```python
from src.app.run import get_nerd_herd

async def get_load_mode() -> str:
    return get_nerd_herd().get_load_mode()

# ... all functions delegate to NerdHerd
```

### Shim in `src/infra/runtime_state.py`

```python
# runtime_state dict preserved for backward compat
# health operations delegate to NerdHerd's HealthRegistry
```

### NerdHerd initialization in `src/app/run.py`

```python
from nerd_herd import NerdHerd

_nerd_herd = NerdHerd(
    metrics_port=9881,
    llama_server_url="http://127.0.0.1:8080",
)

def get_nerd_herd() -> NerdHerd:
    return _nerd_herd

# In startup:
await _nerd_herd.start()

# Wire mode change callback:
_nerd_herd.on_mode_change(_handle_mode_change)

async def _handle_mode_change(old, new, source):
    # persist to DB
    # invalidate DaLLaMa measured_tps
    # notify Telegram
```

### Custom collector registration

KutAI registers its orchestrator metrics as a custom collector:

```python
nh.register_collector("orchestrator", OrchestratorCollector())
```

This collector wraps the existing `src/infra/metrics.py` counters and exposes them with `kutay_*` prefix. The ring buffer handles rate computation for counters that need it.

## Testing Strategy

- **Unit tests per collector** — mock pynvml/psutil, mock llama-server HTTP responses
- **Ring buffer tests** — verify rate accuracy against known sample sequences, edge cases (empty buffer, single sample, exactly-full buffer)
- **LoadManager tests** — mode transitions, auto-detect upgrade delay, immediate downgrade, manual override stops auto, `on_mode_change` callback fires correctly
- **Exposition tests** — verify parseable Prometheus text format, correct HELP/TYPE declarations
- **Integration test** — spin up NerdHerd, hit `/metrics` via HTTP, verify all collectors report
- **No KutAI dependency in any test**

## Migration Checklist

1. Create `packages/nerd_herd/` with package structure
2. Implement types.py, registry.py, ring_buffer.py (foundation)
3. Port GPUCollector from gpu_monitor.py
4. Port LoadManager from load_manager.py (remove DB, add callbacks)
5. Port HealthRegistry from runtime_state.py
6. Implement InferenceCollector (llama-server fetcher)
7. Implement exposition.py (HTTP server + Prometheus format)
8. Write all tests
9. Create shims in src/models/gpu_monitor.py, src/infra/load_manager.py, src/infra/runtime_state.py
10. Wire NerdHerd into src/app/run.py
11. Create KutAI OrchestratorCollector wrapping src/infra/metrics.py
12. Update src/app/api.py /metrics endpoint to delegate to NerdHerd
13. Update docker-compose.yml (remove Prometheus, update Grafana datasource)
14. Update Grafana dashboard JSON (migrate queries)
15. Add editable install to requirements.txt
16. Run full test suite
17. Update docs/architecture-modularization.md
