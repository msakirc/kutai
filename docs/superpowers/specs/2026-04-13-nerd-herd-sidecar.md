# NerdHerd Sidecar Extraction — Spec & Context

## Goal
Move NerdHerd from running inside the orchestrator to a standalone sidecar managed by Yasar Usta. One instance, always running (even when KutAI is down), no duplicates.

## Current State

### NerdHerd runs INSIDE the orchestrator
- **Global instance**: `src/app/run.py:44` — `_nerd_herd: NerdHerd | None = None`
- **Init**: `run.py:434-438` — `NerdHerd(metrics_port=9881, llama_server_url="http://127.0.0.1:8080", initial_load_mode=...)`
- **Start**: `run.py:439` — `await _nerd_herd.start()`
- **Register collector**: `run.py:445` — `register_collector("orchestrator", OrchestratorCollector())`
- **Mode change callback**: `run.py:471-473` — persists mode to DB, invalidates TPS
- **Auto-detect**: `run.py:476` — `start_auto_detect(notify_fn=_notify_gpu_change)`
- **Stop**: `run.py:493` — `await _nerd_herd.stop()`
- **Accessor**: `run.py:46-47` — `get_nerd_herd()` returns the global

### Who calls NerdHerd (via `src/infra/load_manager.py`)
All go through `get_nerd_herd()` → NerdHerd methods:
- `get_load_mode()` — telegram_bot.py:997, 3250
- `set_load_mode(mode, source)` — telegram_bot.py:3275, 4385
- `enable_auto_management()` — telegram_bot.py:3266
- `is_local_inference_allowed()` — load_manager.py:41
- `is_auto_managed()` — telegram_bot.py:3251
- `get_vram_budget_fraction()` — load_manager.py:51

### Other callers
- `src/app/api.py:246` — `nh.prometheus_lines()` for /metrics endpoint
- `src/infra/runtime_state.py:25` — `nh.mark_degraded(capability)`
- `src/models/gpu_monitor.py:20` — `nh.registry.get("gpu")` for GPU state

### NerdHerd Public API (from `packages/nerd_herd/src/nerd_herd/__init__.py`)
```python
class NerdHerd:
    def __init__(self, metrics_port=9881, llama_server_url=None, detect_interval=30, upgrade_delay=300, initial_load_mode="full", inference_poll_interval=5)
    async def start()           # start metrics server + inference collector
    async def stop()            # stop everything
    async def start_auto_detect(notify_fn=None)
    def register_collector(name, collector)
    def on_mode_change(callback)
    def get_load_mode() -> str
    def set_load_mode(mode, source=None) -> str
    def enable_auto_management()
    def is_local_inference_allowed() -> bool
    def get_vram_budget_fraction() -> float
    def get_vram_budget_mb() -> int
    def gpu_state() -> GPUState
    def get_health_status() -> HealthStatus
    def mark_degraded(capability)
    def is_healthy(capability) -> bool
    def prometheus_lines() -> str
```

### HTTP endpoints (aiohttp server on port 9881)
- `GET /health` → `{"status": "ok"}`
- `GET /metrics` → Prometheus text format

## What Needs to Change

### 1. Create `packages/nerd_herd/src/nerd_herd/__main__.py`
Standalone entry point: `python -m nerd_herd --port 9881 --llama-url http://127.0.0.1:8080`
- Starts NerdHerd with CLI args
- Writes PID file to `logs/nerd_herd.pid`
- Handles SIGINT/SIGTERM for graceful shutdown
- Runs auto-detect loop by default
- Loads initial_load_mode from DB if available, else "full"

### 2. Add NerdHerd as second sidecar in Yasar Usta

**Guard currently supports ONE sidecar:**
- `guard.py` — `self.sidecar: SidecarManager | None = None`
- `config.py` — `sidecar: SidecarConfig | None = None`

**Change to list:**
- `guard.py` — `self.sidecars: list[SidecarManager] = []`
- `config.py` — `sidecars: list[SidecarConfig] = field(default_factory=list)`
- Update all sidecar references: `ensure()`, `stop()`, `start()`, status panel, callbacks

**Add to `kutai_wrapper.py`:**
```python
sidecars=[
    SidecarConfig(name="yazbunu", command=[...], health_url="http://127.0.0.1:9880/health", ...),
    SidecarConfig(name="nerd_herd", command=[venv_python, "-m", "nerd_herd", "--port", "9881"], health_url="http://127.0.0.1:9881/health", ...),
]
```

### 3. Change orchestrator to be a CLIENT of NerdHerd (not owner)
- `src/app/run.py`: Remove NerdHerd init/start/stop. Replace `get_nerd_herd()` with HTTP client or connect to existing instance.
- `src/infra/load_manager.py`: Call NerdHerd via HTTP API instead of in-process methods.
- Simplest approach: keep the Python API but connect to the running instance's HTTP endpoints.

**OR simpler:** Keep `NerdHerd` class but add a `connect()` classmethod that creates a thin HTTP proxy instead of starting a new server. The orchestrator calls `NerdHerd.connect(port=9881)` which returns a client that delegates to HTTP.

### 4. Handle mode change notifications
Currently: `on_mode_change(callback)` runs in-process.
As sidecar: Need either:
- Polling `/health` or `/metrics` for mode changes
- WebSocket push from NerdHerd
- File-based signal (like shutdown.signal)
Simplest: orchestrator polls current mode from NerdHerd every cycle.

### 5. Handle OrchestratorCollector registration
Currently: orchestrator registers a collector that exposes task queue metrics.
As sidecar: NerdHerd can't call orchestrator methods.
Solution: Orchestrator pushes metrics to NerdHerd via HTTP POST, or NerdHerd scrapes orchestrator's own `/metrics` endpoint.

### 6. Status panel buttons
Add restart button for nerd_herd alongside yazbunu in the Yasar Usta bot's inline keyboard.

## Files to Modify
| File | Change |
|------|--------|
| `packages/nerd_herd/src/nerd_herd/__main__.py` | **CREATE** — standalone entry point |
| `packages/yasar_usta/src/yasar_usta/config.py` | `sidecar` → `sidecars: list[SidecarConfig]` |
| `packages/yasar_usta/src/yasar_usta/guard.py` | Handle list of sidecars |
| `packages/yasar_usta/src/yasar_usta/commands.py` | Inline buttons for N sidecars |
| `packages/yasar_usta/src/yasar_usta/status.py` | Show all sidecars in status panel |
| `kutai_wrapper.py` | Add nerd_herd to sidecars list |
| `src/app/run.py` | Remove NerdHerd ownership, connect as client |
| `src/infra/load_manager.py` | Switch to HTTP client or proxy |
| `src/infra/runtime_state.py` | Switch to HTTP client |
| `src/models/gpu_monitor.py` | Switch to HTTP client |
| `src/app/api.py` | Proxy to NerdHerd's /metrics or remove |

## Key Constraints
- **No duplicate instances** — orchestrator must NOT start NerdHerd if sidecar is running
- **Graceful degradation** — if NerdHerd sidecar is down, orchestrator should work (just without GPU monitoring)
- **Load mode persistence** — mode changes must survive NerdHerd restarts (DB or file)
- **Port 9881** — keep same port for Grafana compatibility
