# Dynamic Context, Model Exclusion, Grafana Monitoring — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Dynamically size llama-server context based on available RAM, route error recovery to a different model than the one that failed, and add a Prometheus+Grafana monitoring stack.

**Architecture:** Three independent changes. Task 1 modifies the model swap path to recalculate context+gpu_layers from live system state. Task 2 adds `exclude_models` to error recovery context by querying the conversation log for the failed task's model. Task 3 adds a `/metrics` Prometheus endpoint and docker services for Prometheus+Grafana with a pre-provisioned dashboard.

**Tech Stack:** Python (asyncio, httpx, psutil, pynvml), Docker Compose, Prometheus, Grafana

---

## Task 1: Dynamic Context Length at Swap Time

**Files:**
- Modify: `src/models/model_registry.py` (add `calculate_dynamic_context()`, update registration)
- Modify: `src/models/local_model_manager.py:159-226` (recalculate at swap time)

### Step 1: Add `calculate_dynamic_context()` to model_registry.py

Add this function right after the existing `calculate_gpu_layers()` (after line 345):

```python
def calculate_dynamic_context(
    file_size_mb: float,
    n_layers: int,
    gpu_layers: int,
    available_ram_mb: int,
    available_vram_mb: int,
    family_key: str | None = None,
    max_context: int | None = None,
) -> int:
    """
    Calculate the largest safe context length given current memory.

    Called at swap time (after stopping old server) so memory readings
    reflect the true free state.

    KV cache cost model:
        kv_per_token = n_layers * 0.5 KB  (conservative estimate for Q4/Q8 KV)
        total_kv = kv_per_token * context_length

    Memory budget:
        offloaded_weight = file_size_mb * (gpu_layers / n_layers)  → lives in VRAM
        ram_weight = file_size_mb - offloaded_weight                → lives in RAM
        ram_budget = (available_ram - ram_weight - 4096) * 0.80     → 4GB OS reserve
        vram_budget = (available_vram - offloaded_weight - 500) * 0.80  → 500MB CUDA overhead
        kv_budget = ram_budget + vram_budget                        → KV can split across both
    """
    if n_layers <= 0 or file_size_mb <= 0:
        return 4096

    # Weight distribution
    gpu_frac = gpu_layers / n_layers if n_layers > 0 else 0
    vram_weights_mb = file_size_mb * gpu_frac
    ram_weights_mb = file_size_mb * (1 - gpu_frac)

    # Available memory after loading weights
    OS_RESERVE_MB = 4096
    CUDA_OVERHEAD_MB = 500
    SAFETY = 0.80

    ram_for_kv = max(0, (available_ram_mb - ram_weights_mb - OS_RESERVE_MB)) * SAFETY
    vram_for_kv = max(0, (available_vram_mb - vram_weights_mb - CUDA_OVERHEAD_MB)) * SAFETY
    kv_budget_mb = ram_for_kv + vram_for_kv

    # KV cache cost: n_layers * 0.5 MB per 1024 tokens
    kv_per_1k_tokens = n_layers * 0.5  # MB
    if kv_per_1k_tokens <= 0:
        return 4096

    max_ctx_from_memory = int((kv_budget_mb / kv_per_1k_tokens) * 1024)

    # Cap at family default or explicit max
    if max_context is None and family_key and family_key in FAMILY_PROFILES:
        max_context = FAMILY_PROFILES[family_key].context_default
    if max_context is None:
        max_context = 131072

    # Floor at 4096, round down to nearest 1024
    ctx = min(max_ctx_from_memory, max_context)
    ctx = max(ctx, 4096)
    ctx = (ctx // 1024) * 1024

    return ctx
```

Also add the FAMILY_PROFILES import at the top of the function or ensure it's accessible (it's already imported via `from .model_profiles import FAMILY_PROFILES` at line 42).

### Step 2: Update `_swap_model()` to recalculate context and gpu_layers

In `src/models/local_model_manager.py`, modify `_swap_model()` — after `await self._stop_server()` (line 200) and before `await self._start_server(model_info)` (line 203), insert the dynamic recalculation:

```python
            # Stop existing server
            await self._stop_server()

            # ── Recalculate context + gpu_layers with live memory state ──
            # Server is stopped, so readings reflect true free memory.
            gpu_monitor.invalidate_cache()  # force fresh reading
            fresh_state = gpu_monitor.get_state()

            from .model_registry import calculate_dynamic_context, calculate_gpu_layers

            # Check for user override in models.yaml
            registry_overrides = registry.get_overrides(model_name)

            if "context_length" not in registry_overrides:
                new_ctx = calculate_dynamic_context(
                    file_size_mb=model_info.file_size_mb,
                    n_layers=model_info.total_layers,
                    gpu_layers=model_info.gpu_layers,
                    available_ram_mb=fresh_state.ram_available_mb,
                    available_vram_mb=fresh_state.gpu.vram_free_mb,
                    family_key=model_info.family,
                )
                if new_ctx != model_info.context_length:
                    logger.info(
                        f"📐 Dynamic context: {model_info.context_length} → {new_ctx} "
                        f"(RAM free: {fresh_state.ram_available_mb}MB, "
                        f"VRAM free: {fresh_state.gpu.vram_free_mb}MB)"
                    )
                    model_info.context_length = new_ctx

            if "gpu_layers" not in registry_overrides:
                new_layers = calculate_gpu_layers(
                    file_size_mb=model_info.file_size_mb,
                    n_layers=model_info.total_layers,
                    available_vram_mb=fresh_state.gpu.vram_free_mb,
                    context_length=model_info.context_length,
                )
                if new_layers != model_info.gpu_layers:
                    logger.info(
                        f"📐 Dynamic GPU layers: {model_info.gpu_layers} → {new_layers}"
                    )
                    model_info.gpu_layers = new_layers

            # Start new server
            success = await self._start_server(model_info)
```

### Step 3: Add `invalidate_cache()` to GPUMonitor

In `src/models/gpu_monitor.py`, add a method to force a fresh reading:

```python
    def invalidate_cache(self) -> None:
        """Force next get_state() to poll fresh values."""
        self._last_poll = 0.0
```

### Step 4: Add `get_overrides()` to registry

In `src/models/model_registry.py`, add a helper method to `ModelRegistry` that returns the user's raw overrides dict for a model name (from models.yaml). This lets the swap code know whether context_length/gpu_layers were explicitly set by the user:

```python
    def get_overrides(self, model_name: str) -> dict:
        """Return user overrides from models.yaml for a given model name."""
        return self._overrides.get(model_name, {})
```

Check where `_overrides` is stored — find the models.yaml parsing code and ensure the raw overrides dict is kept on the registry instance.

### Step 5: Remove the hardcoded DEFAULT_LOCAL_CTX

In `src/models/model_registry.py` around line 909-918, replace the static default with a call to the new function. This is for boot-time registration (swap-time recalculation will override it, but boot-time should also be reasonable):

```python
            # Context length — dynamic based on available memory
            if "context_length" in model_overrides:
                context_length = model_overrides["context_length"]
            else:
                context_length = calculate_dynamic_context(
                    file_size_mb=raw["file_size_mb"],
                    n_layers=raw["n_layers"],
                    gpu_layers=0,  # not yet calculated
                    available_ram_mb=state.ram_available_mb if state else 16384,
                    available_vram_mb=available_vram,
                    family_key=raw["family_key"],
                )
```

### Step 6: Commit

```bash
git add src/models/model_registry.py src/models/local_model_manager.py src/models/gpu_monitor.py
git commit -m "feat: calculate context length dynamically based on available RAM/VRAM at swap time"
```

---

## Task 2: Error Recovery Excludes the Failed Model

**Files:**
- Modify: `src/core/orchestrator.py:1403-1472` (look up failed model, pass exclude_models)
- Modify: `src/agents/base.py:1586-1594` (read exclude_models from context)
- Modify: `src/infra/db.py` (add helper to get last model used for a task)

### Step 1: Add DB helper to get the last model used for a task

In `src/infra/db.py`, add after `log_conversation()`:

```python
async def get_last_model_for_task(task_id: int) -> str | None:
    """Get the last model used for a task from conversation log."""
    db = await get_db()
    cursor = await db.execute(
        """SELECT model_used FROM conversations
           WHERE task_id = ? AND model_used IS NOT NULL
           ORDER BY id DESC LIMIT 1""",
        (task_id,),
    )
    row = await cursor.fetchone()
    return row[0] if row else None
```

### Step 2: Update `_spawn_error_recovery()` to exclude the failed model

In `src/core/orchestrator.py`, modify `_spawn_error_recovery()`. After the `is_timeout` line (around line 1446), add model lookup and pass `exclude_models` in the context:

```python
        is_timeout = "timed out" in error_str.lower() or "timeout" in error_str.lower()

        # Find which model the failed task used — exclude it from recovery routing
        failed_model = None
        try:
            from ..infra.db import get_last_model_for_task
            failed_model = await get_last_model_for_task(task_id)
            if failed_model:
                logger.info(
                    f"[Task #{task_id}] Error recovery will exclude model: {failed_model}"
                )
        except Exception:
            pass

        try:
            recovery_task_id = await add_task(
                title=stable_title,
                description=recovery_description,
                goal_id=goal_id,
                parent_task_id=task_id,
                agent_type="error_recovery",
                tier="medium",
                priority=max(failed_task.get("priority", 5), 7),
                context={
                    "failed_task_id": task_id,
                    "failed_agent_type": agent_type,
                    "error": error_str,
                    "original_title": title,
                    "prefer_speed": is_timeout,
                    "exclude_models": [failed_model] if failed_model else [],
                },
            )
```

### Step 3: Ensure `_build_model_requirements()` reads `exclude_models` from context

In `src/agents/base.py`, around line 1591-1594, verify or add:

```python
        if task_ctx.get("prefer_speed"):
            reqs.prefer_speed = True
            reqs.prefer_local = False

        # ── Model exclusion ──
        exclude = task_ctx.get("exclude_models", [])
        if exclude:
            reqs.exclude_models = exclude
```

This code already exists at line 1592-1594 (the `exclude_models` block). Just verify it's there and runs after the `prefer_speed` block we added earlier.

### Step 4: Commit

```bash
git add src/core/orchestrator.py src/agents/base.py src/infra/db.py
git commit -m "feat: error recovery excludes the model that caused the failure"
```

---

## Task 3: Prometheus + Grafana Docker Stack

**Files:**
- Modify: `src/app/api.py` (add `/metrics` Prometheus endpoint)
- Modify: `sandbox/docker-compose.yml` (add prometheus + grafana services)
- Create: `monitoring/prometheus.yml`
- Create: `monitoring/grafana/provisioning/datasources/prometheus.yml`
- Create: `monitoring/grafana/provisioning/dashboards/dashboard.yml`
- Create: `monitoring/grafana/dashboards/kutay-overview.json`

### Step 1: Add `/metrics` Prometheus endpoint to api.py

In `src/app/api.py`, add after the `/llm` endpoint (before the `/projects` section):

```python
    # ── Prometheus Metrics ────────────────────────────────────────────────

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics():
        """Prometheus-compatible metrics endpoint. No auth — Prometheus needs open access."""
        from src.infra.metrics import get_all_counters
        from src.models.local_model_manager import get_local_manager

        lines = []

        # ── Orchestrator counters ──
        counters = get_all_counters()
        tasks_ok = int(counters.get("tasks_completed", 0))
        tasks_fail = int(counters.get("tasks_failed", 0))
        queue = int(counters.get("queue_depth", 0))
        cost = counters.get("cost_total", 0.0)

        lines.append(f"# HELP kutay_tasks_completed_total Total tasks completed")
        lines.append(f"# TYPE kutay_tasks_completed_total counter")
        lines.append(f"kutay_tasks_completed_total {tasks_ok}")

        lines.append(f"# HELP kutay_tasks_failed_total Total tasks failed")
        lines.append(f"# TYPE kutay_tasks_failed_total counter")
        lines.append(f"kutay_tasks_failed_total {tasks_fail}")

        lines.append(f"# HELP kutay_queue_depth Current task queue depth")
        lines.append(f"# TYPE kutay_queue_depth gauge")
        lines.append(f"kutay_queue_depth {queue}")

        lines.append(f"# HELP kutay_cost_total_usd Total inference cost in USD")
        lines.append(f"# TYPE kutay_cost_total_usd counter")
        lines.append(f"kutay_cost_total_usd {cost:.6f}")

        # Per-model call counts and tokens
        model_calls = {k.split(":", 1)[1]: int(v)
                       for k, v in counters.items() if k.startswith("model_calls:")}
        if model_calls:
            lines.append(f"# HELP kutay_model_calls_total Model call count by model")
            lines.append(f"# TYPE kutay_model_calls_total counter")
            for model, count in model_calls.items():
                safe = model.replace('"', '\\"')
                lines.append(f'kutay_model_calls_total{{model="{safe}"}} {count}')

        model_tokens = {k.split(":", 1)[1]: int(v)
                        for k, v in counters.items() if k.startswith("tokens:")}
        if model_tokens:
            lines.append(f"# HELP kutay_tokens_total Token count by model")
            lines.append(f"# TYPE kutay_tokens_total counter")
            for model, tokens in model_tokens.items():
                safe = model.replace('"', '\\"')
                lines.append(f'kutay_tokens_total{{model="{safe}"}} {tokens}')

        # ── Local model manager status ──
        try:
            mgr = get_local_manager()
            status = mgr.get_status()
            loaded = status.get("loaded_model") or ""
            healthy = 1 if status.get("healthy") else 0
            swaps = status.get("total_swaps", 0)
            idle = status.get("idle_seconds", 0)
            busy = 1 if status.get("inference_busy") else 0

            lines.append(f"# HELP kutay_model_healthy Is the local model healthy")
            lines.append(f"# TYPE kutay_model_healthy gauge")
            lines.append(f'kutay_model_healthy{{model="{loaded}"}} {healthy}')

            lines.append(f"# HELP kutay_model_swaps_total Total model swaps")
            lines.append(f"# TYPE kutay_model_swaps_total counter")
            lines.append(f"kutay_model_swaps_total {swaps}")

            lines.append(f"# HELP kutay_model_idle_seconds Seconds since last inference")
            lines.append(f"# TYPE kutay_model_idle_seconds gauge")
            lines.append(f"kutay_model_idle_seconds {idle:.1f}")

            lines.append(f"# HELP kutay_model_inference_busy Is inference currently running")
            lines.append(f"# TYPE kutay_model_inference_busy gauge")
            lines.append(f"kutay_model_inference_busy {busy}")
        except Exception:
            pass

        lines.append("")
        return "\n".join(lines)
```

Also add the import at the top of `create_app()` or inline:
```python
from starlette.responses import PlainTextResponse
```

### Step 2: Create `monitoring/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "llama-server"
    metrics_path: /metrics
    static_configs:
      - targets: ["host.docker.internal:8080"]
    scrape_interval: 5s

  - job_name: "kutay-orchestrator"
    metrics_path: /metrics
    static_configs:
      - targets: ["host.docker.internal:8000"]
    scrape_interval: 15s
```

### Step 3: Create Grafana provisioning files

**`monitoring/grafana/provisioning/datasources/prometheus.yml`:**
```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
```

**`monitoring/grafana/provisioning/dashboards/dashboard.yml`:**
```yaml
apiVersion: 1
providers:
  - name: "default"
    orgId: 1
    folder: ""
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    options:
      path: /var/lib/grafana/dashboards
      foldersFromFilesStructure: false
```

### Step 4: Create the Grafana dashboard JSON

Create `monitoring/grafana/dashboards/kutay-overview.json` — a pre-built dashboard with 4 rows:

1. **LLM Performance** — generation tok/s, prompt tok/s, KV cache usage %
2. **System Resources** — VRAM free, RAM available, GPU utilization, temperature
3. **Orchestrator** — tasks completed/failed rate, queue depth, cost accumulation
4. **Model Management** — current model, swap count, idle time, inference busy

Panels use `llamacpp:*` metrics from llama-server and `kutay_*` metrics from the orchestrator.

(Full JSON is large — generate it programmatically with standard Grafana panel definitions, time series for rates, stat panels for gauges.)

### Step 5: Add services to docker-compose.yml

In `sandbox/docker-compose.yml`, add:

```yaml
  prometheus:
    image: prom/prometheus:latest
    container_name: kutay-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ../monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    extra_hosts:
      - "host.docker.internal:host-gateway"
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: "0.5"

  grafana:
    image: grafana/grafana:latest
    container_name: kutay-grafana
    restart: unless-stopped
    ports:
      - "3001:3000"
    volumes:
      - ../monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ../monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: "0.5"
```

And add named volumes at the bottom:
```yaml
volumes:
  prometheus_data:
  grafana_data:
```

### Step 6: Commit

```bash
git add src/app/api.py sandbox/docker-compose.yml monitoring/
git commit -m "feat: add Prometheus metrics endpoint and Grafana monitoring stack"
```

---

## Execution Order

Tasks 1, 2, and 3 are fully independent — they touch different files. They can be executed in parallel or in any order.
