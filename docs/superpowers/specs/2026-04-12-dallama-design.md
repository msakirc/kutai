# DaLLaMa — Design Spec

**Date:** 2026-04-12
**Status:** Draft
**Package:** `dallama`
**Purpose:** Python async llama-server process manager. Ensures the right model is loaded, tracks inference lifecycle, handles failures. Does not route, schedule, or select models.

## Context

KutAI's `src/models/local_model_manager.py` (1,193 lines) manages llama-server lifecycle but is tightly coupled to KutAI internals (DB queries, dispatcher callbacks, registry imports). Extracting it into a standalone package:

1. Creates a hard boundary — agents fixing routing bugs can't accidentally break server lifecycle code
2. Forces clean interfaces between model management and model selection
3. Produces a genuinely novel package (no Python equivalent exists for llama-server process management)

### Why not llama-server router mode?

llama-server gained router mode (`--models-dir`, `--models-max`) in late 2025 with hot model swap via `POST /models/load`. However, three hard blockers prevent adoption:

- **Per-model flags broken** ([#20851](https://github.com/ggml-org/llama.cpp/issues/20851)) — all models get same global `ctx-size`, `n-gpu-layers`. KutAI swaps between models of very different sizes.
- **mmproj can't be loaded dynamically** ([#20855](https://github.com/ggml-org/llama.cpp/discussions/20855)) — vision projector is a blocked arg in router mode. KutAI loads mmproj on-demand to save ~876MB VRAM.
- **Dynamic context window impossible** — `POST /models/load` accepts only model name, no flags. KutAI calculates context at swap time from live VRAM readings.
- **Zombie children on crash** ([#18912](https://github.com/ggml-org/llama.cpp/issues/18912)) — no cleanup, exactly what DaLLaMa's Job Objects solve.

Process restart with per-model flags remains the correct approach until these are resolved. The swap module is designed for future migration to hot-swap when llama.cpp matures (see Swap Module section).

### Thinking mode per-request

One useful finding: `thinking_budget_tokens` can be set per-request if the server starts WITHOUT `--reasoning-budget`. This is a future optimization to avoid restarts for thinking-mode toggling. Not in scope for initial extraction — noted for future work.

## Public API

Three methods, three types.

### DaLLaMa

```python
from dallama import DaLLaMa, DaLLaMaConfig, ServerConfig, ServerStatus

dallama = DaLLaMa(config=DaLLaMaConfig(
    llama_server_path="/path/to/llama-server",
    port=8080,
    idle_timeout_seconds=60,
    circuit_breaker_threshold=2,
    circuit_breaker_cooldown_seconds=300,
    on_ready=my_callback,
    get_vram_free_mb=my_vram_fn,
))

# Lifecycle
await dallama.start()   # start background tasks (watchdog, idle unloader)
await dallama.stop()    # stop server + background tasks

# Inference — DaLLaMa ensures model is loaded, tracks in-flight requests
async with dallama.infer(ServerConfig(
    model_path="/models/qwen3-30b-q4.gguf",
    model_name="qwen3-30b-q4",
    context_length=16384,
    thinking=True,
)) as session:
    # session.url = "http://127.0.0.1:8080"
    # Model is loaded, server is healthy
    response = await litellm.acompletion(
        model="openai/local-model",
        api_base=session.url,
        messages=messages,
    )

# Keep alive — reset idle timer during long tool execution between LLM calls
dallama.keep_alive()

# Status — what the dispatcher needs for routing decisions
status = dallama.status
# ServerStatus(model_name="qwen3-30b-q4", healthy=True, busy=False,
#              measured_tps=12.3, context_length=16384)
```

### ServerConfig

What the caller builds to describe the model to load. DaLLaMa does not interpret these beyond building the command line.

```python
@dataclass
class ServerConfig:
    model_path: str              # absolute path to GGUF file
    model_name: str              # human-readable identifier
    context_length: int          # ctx window to allocate
    thinking: bool = False       # --reasoning on/off
    vision_projector: str = ""   # --mmproj path, empty = no vision
    extra_flags: list[str] = field(default_factory=list)  # --no-jinja, --chat-template, etc.
```

No gpu_layers, no threads, no vram_cap. DaLLaMa decides hardware parameters internally:
- GPU layers: uses `--fit` (llama-server default). If a model needs explicit override, the caller passes it via `extra_flags`.
- Threads: auto-detected from physical CPU core count, or let llama-server decide.

### ServerStatus

What DaLLaMa exposes to callers. Minimal — only what's needed for routing decisions.

```python
@dataclass
class ServerStatus:
    model_name: str | None       # None = no model loaded
    healthy: bool                # server responding to /health
    busy: bool                   # in-flight inference active
    measured_tps: float          # from /metrics, 0 if unknown
    context_length: int          # actual loaded context
```

No hardware details. The dispatcher checks `status.model_name` to know what's loaded, `status.busy` to know if it should route elsewhere, `status.healthy` to know if local is available at all.

### DaLLaMaConfig

```python
@dataclass
class DaLLaMaConfig:
    llama_server_path: str = "llama-server"
    port: int = 8080
    host: str = "127.0.0.1"
    idle_timeout_seconds: float = 60.0
    circuit_breaker_threshold: int = 2       # consecutive failures before cooldown
    circuit_breaker_cooldown_seconds: float = 300.0
    inference_drain_timeout_seconds: float = 30.0
    health_check_interval_seconds: float = 30.0
    health_fail_threshold: int = 3           # consecutive /health failures before restart
    min_free_vram_mb: int = 4096             # refuse load if VRAM below this
    on_ready: Callable[[str | None, str], None] | None = None
    get_vram_free_mb: Callable[[], int] | None = None
```

### Callbacks

**`on_ready(model_name: str | None, reason: str)`** — called when DaLLaMa's state changes. The host wires this up to wake sleeping tasks, update status panels, etc. Reasons include: `"model_loaded"`, `"inference_complete"`, `"idle_unload"`, `"load_failed"`, `"circuit_breaker_reset"`, `"crash_recovery"`.

**`get_vram_free_mb() -> int`** — called before loading a model. DaLLaMa uses this to check if there's sufficient VRAM. If not provided, DaLLaMa skips the check and lets `--fit` handle it. The host injects this from gpu_monitor (which will become its own package later).

## Internal Architecture

Six modules, each single-responsibility, each under 300 lines.

```
packages/dallama/
├── pyproject.toml
├── src/dallama/
│   ├── __init__.py        # re-exports: DaLLaMa, DaLLaMaConfig, ServerConfig, ServerStatus
│   ├── config.py          # dataclasses: DaLLaMaConfig, ServerConfig, ServerStatus, InferenceSession
│   ├── server.py          # ServerProcess: build cmd, start/stop subprocess, poll /health
│   ├── swap.py            # SwapManager: lock, drain, circuit breaker, stop→start orchestration
│   ├── watchdog.py        # HealthWatchdog + IdleUnloader: background asyncio tasks
│   ├── metrics.py         # MetricsParser: GET /metrics → tps, kv_cache, prompt_speed
│   └── platform.py        # PlatformHelper: Job Objects, orphan cleanup, OS-specific shutdown
└── tests/
    ├── test_server.py
    ├── test_swap.py
    ├── test_watchdog.py
    ├── test_metrics.py
    └── test_platform.py
```

### Module Details

#### config.py (~80 lines)

All dataclasses. No logic beyond `__post_init__` validation (e.g., port range, path exists).

`InferenceSession` is a simple context holder returned by `infer()`:

```python
@dataclass
class InferenceSession:
    url: str          # "http://127.0.0.1:8080"
    model_name: str
```

#### server.py (~250 lines)

`ServerProcess` — manages a single llama-server subprocess.

```python
class ServerProcess:
    def build_cmd(config: ServerConfig, dallama_config: DaLLaMaConfig) -> list[str]
    async def start(config: ServerConfig) -> bool     # launch + wait for /health
    async def stop() -> None                          # graceful terminate → force kill
    async def health_check() -> bool                  # GET /health
    def is_alive() -> bool                            # process.poll()
```

Command building:
- Always: `--model`, `--port`, `--host`, `--ctx-size`, `--flash-attn auto`, `--metrics`
- Thinking: `--reasoning on` or `--reasoning off --reasoning-budget 0` (skip for `--no-jinja` models)
- Vision: `--mmproj path` if `vision_projector` is set
- Jinja: `--jinja` unless `--no-jinja` in `extra_flags`
- Extra: append `extra_flags` verbatim
- Threads: auto-detect physical cores - 2, or let llama-server default

Health wait: adaptive timeout based on model file size. Poll `/health` with increasing interval (1s → 3s). Minimum 45s, maximum 300s.

#### swap.py (~200 lines)

`SwapManager` — orchestrates model transitions. Designed for future migration to llama-server hot-swap.

```python
class SwapManager:
    async def swap(
        server: ServerProcess,
        new_config: ServerConfig,
        platform: PlatformHelper,
    ) -> bool

    def mark_inference_start() -> int      # returns generation ID
    def mark_inference_end(gen: int) -> None
    @property
    def has_inflight() -> bool
```

Swap flow:
1. Acquire `asyncio.Lock`
2. Re-check if model already loaded (resolved under lock)
3. **Circuit breaker check** — refuse if same model failed N times consecutively
4. **Drain in-flight inferences** — wait up to `inference_drain_timeout_seconds`, then force (bump generation counter so orphaned `mark_inference_end` calls are ignored)
5. `server.stop()` via platform helper
6. Sleep 2s for CUDA VRAM release
7. **VRAM check** — call `get_vram_free_mb()` if provided, refuse if below `min_free_vram_mb`
8. `server.start(new_config)`
9. **Record result** — success resets circuit breaker, failure increments counter
10. Release lock, call `on_ready`

**Future hot-swap path:** When llama.cpp fixes per-model flags, swap.py gains an alternate implementation that calls `POST /models/load` instead of stop→start. The `SwapManager` interface stays identical — callers don't know which strategy is used.

Circuit breaker state:

```python
# After circuit_breaker_threshold consecutive failures for the same model:
# - Refuse loads for circuit_breaker_cooldown_seconds
# - on_ready(None, "circuit_breaker_active")
# - When cooldown expires: on_ready(None, "circuit_breaker_reset")
```

#### watchdog.py (~150 lines)

Two background `asyncio.Task`s, started/stopped via `dallama.start()`/`dallama.stop()`.

**HealthWatchdog:**
- Every `health_check_interval_seconds`, checks:
  - Process alive? (crash detection via `poll()`)
  - `/health` responsive? (hang detection via HTTP)
- Skip checks during active swap (`swap_in_progress` flag) or idle unload
- 3 consecutive `/health` failures → restart same model via `SwapManager.swap()`
- Process exit (crash) → immediate restart via `SwapManager.swap()`
- Respects circuit breaker — if restart also fails, doesn't loop

**IdleUnloader:**
- Every 30s, checks idle time (time since last `infer()` exit or `keep_alive()` call)
- If idle > `idle_timeout_seconds` AND no in-flight inferences → `server.stop()`
- Calls `on_ready(None, "idle_unload")`

#### metrics.py (~100 lines)

`MetricsParser` — fetches and parses llama-server's Prometheus-format `/metrics` endpoint.

```python
class MetricsParser:
    async def fetch(api_base: str) -> MetricsSnapshot

@dataclass
class MetricsSnapshot:
    generation_tokens_per_second: float
    prompt_tokens_per_second: float
    kv_cache_usage_percent: float
    requests_processing: int
    requests_pending: int
    prompt_tokens_total: int
    generation_tokens_total: int
```

Called periodically by the watchdog to update `ServerStatus.measured_tps`. Also callable on-demand for diagnostics.

Handles metric name normalization (llama.cpp uses colons or underscores depending on version: `llamacpp:foo` vs `llamacpp_foo`).

#### platform.py (~150 lines)

`PlatformHelper` — OS-specific process management.

```python
class PlatformHelper:
    def create_process(cmd: list[str], stderr_path: str) -> subprocess.Popen
    async def graceful_stop(process: Popen, timeout: float = 10) -> None
    def kill_orphans(executable_name: str = "llama-server") -> None
```

**Windows:**
- `CREATE_NO_WINDOW` flag on `Popen`
- Job Object with `KILL_ON_JOB_CLOSE` — child processes die when parent dies, even on crash
- Assign llama-server to Job Object after spawn
- Graceful stop via `CTRL_BREAK_EVENT`, fallback to `kill()`
- Orphan cleanup via `taskkill /F /IM llama-server.exe`

**Linux/Mac:**
- Standard `SIGTERM` → wait → `SIGKILL`
- Orphan cleanup via `pkill -f llama-server`

### DaLLaMa Main Class (~150 lines)

Composes the modules. Thin orchestration only.

```python
class DaLLaMa:
    def __init__(self, config: DaLLaMaConfig):
        self._config = config
        self._platform = PlatformHelper()
        self._server = ServerProcess(config, self._platform)
        self._swap = SwapManager(config)
        self._metrics = MetricsParser()
        self._watchdog: Task | None = None
        self._idle_unloader: Task | None = None
        self._current_config: ServerConfig | None = None

    async def start(self):
        """Start background tasks."""
        self._platform.kill_orphans()  # clean slate
        self._watchdog = create_task(self._run_watchdog())
        self._idle_unloader = create_task(self._run_idle_unloader())

    async def stop(self):
        """Stop server and background tasks."""
        # cancel background tasks
        # stop server
        # cleanup

    @asynccontextmanager
    async def infer(self, config: ServerConfig):
        """Ensure model is loaded, yield session, track lifecycle."""
        needs_swap = (
            self._current_config is None
            or config.model_name != self._current_config.model_name
            or config.thinking != self._current_config.thinking
            or config.vision_projector != self._current_config.vision_projector
        )
        if needs_swap:
            success = await self._swap.swap(self._server, config, self._platform)
            if not success:
                raise DaLLaMaLoadError(config.model_name)
            self._current_config = config

        gen = self._swap.mark_inference_start()
        try:
            yield InferenceSession(
                url=f"http://{self._config.host}:{self._config.port}",
                model_name=config.model_name,
            )
        finally:
            self._swap.mark_inference_end(gen)
            self._idle_timer_reset()

    def keep_alive(self):
        """Reset idle timer without starting inference."""
        self._idle_timer_reset()

    @property
    def status(self) -> ServerStatus:
        return ServerStatus(
            model_name=self._current_config.model_name if self._current_config else None,
            healthy=self._server.is_alive() and ...,
            busy=self._swap.has_inflight,
            measured_tps=self._last_metrics.generation_tokens_per_second,
            context_length=self._current_config.context_length if self._current_config else 0,
        )
```

## Dependencies

| Dependency | Why | Required? |
|---|---|---|
| `httpx` | Async HTTP for `/health` and `/metrics` | Yes |

No pynvml, no psutil, no litellm, no pydantic. VRAM info comes via injected `get_vram_free_mb` callback. CPU core detection uses `os.cpu_count()` or lets llama-server default.

## KutAI Integration

### Shim

`src/models/local_model_manager.py` becomes a thin shim:

```python
from dallama import DaLLaMa, DaLLaMaConfig, ServerConfig, ServerStatus

# Build config from KutAI's .env and registry
_dallama = DaLLaMa(DaLLaMaConfig(
    llama_server_path=os.environ.get("LLAMA_SERVER_PATH", "llama-server"),
    port=int(os.environ.get("LLAMA_SERVER_PORT", "8080")),
    on_ready=_on_dallama_ready,
    get_vram_free_mb=lambda: get_gpu_monitor().get_state().gpu.vram_free_mb,
))

def get_local_manager() -> DaLLaMa:
    return _dallama
```

### What Gets Removed from KutAI

- `src/models/local_model_manager.py` — 1,193 lines → ~30 line shim
- `src/models/gpu_scheduler.py` — 233 lines → deleted entirely (dispatcher doesn't send work if DaLLaMa is busy)
- All `accelerate_retries` callbacks from model management → replaced by single `on_ready`

### What Stays in KutAI

- `src/models/gpu_monitor.py` — stays until monitoring package extraction
- `src/models/model_registry.py` — unchanged, builds `ServerConfig` from `ModelInfo`
- `src/models/capabilities.py` — unchanged
- `src/core/llm_dispatcher.py` — owns swap budget, routes local/cloud, checks `dallama.status.busy`
- `src/core/router.py` — model selection, litellm calls

### Dispatcher Changes

The dispatcher's swap budget check becomes:

```python
# Before requesting DaLLaMa:
if dallama.status.model_name != needed_model:
    if not swap_budget.can_swap(priority=task.priority):
        # Route to cloud instead
        return await self._cloud_route(...)

# DaLLaMa handles the rest
async with dallama.infer(server_config) as session:
    response = await call_model(api_base=session.url, ...)
```

## Testing Strategy

- **Unit tests per module** — mock subprocess, mock httpx for /health and /metrics
- **Integration test** — if llama-server is available (env var `LLAMA_SERVER_PATH`), run a real load/infer/unload cycle with a small model
- **KutAI shim tests** — verify backward compatibility: same imports, same behavior
- **Swap drain test** — simulate in-flight inference during swap, verify drain-then-proceed
- **Circuit breaker test** — simulate N consecutive failures, verify refusal + cooldown + reset
- **Watchdog test** — simulate process crash, verify auto-restart
- **Platform test** — verify orphan cleanup, Job Object creation (Windows-only)

## What's NOT in Scope

- GPU scheduling / priority queue — removed, dispatcher doesn't send work if busy
- Model selection / scoring — registry + capabilities, separate concern
- Cloud routing — dispatcher, separate concern
- Swap budget — dispatcher, separate concern
- Rate limiting — cloud operator, future package
- Continuous GPU monitoring — future monitoring package
- Retry logic — dispatcher / future retry package
- Per-request thinking toggle optimization — future work (requires starting without `--reasoning-budget`)
- Migration to llama-server hot-swap — future work, swap.py designed for it
