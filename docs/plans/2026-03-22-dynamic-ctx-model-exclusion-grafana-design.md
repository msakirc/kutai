# Design: Dynamic Context, Model Exclusion, Grafana Monitoring

**Date:** 2026-03-22
**Status:** Approved

## 1. Dynamic Context Length Based on Available RAM

### Problem
Context length was hardcoded (16384), wasting memory on small prompts and OOM-ing on large models. The GGUF metadata reports the theoretical max (e.g., 262144) which is unusable on partial-offload setups.

### Design
Calculate context length **at swap time** (not boot), right after stopping the old server and before starting the new one. At that moment, we know exact free RAM + VRAM.

**New function** `calculate_dynamic_context()` in `model_registry.py`:
- Inputs: `file_size_mb`, `n_layers`, `available_ram_mb`, `available_vram_mb`, `gpu_layers`
- KV cache cost: `(n_layers * 0.5 MB) per 1024 tokens` (existing formula)
- Memory budget: `(available_ram + available_vram - model_weights - OS_RESERVE) * 0.80`
- Floor: 4096, Cap: family profile `context_default`
- User override in `models.yaml` always wins

**Integration point:** `LocalModelManager._swap_model()` calls this after `_stop_server()`, patches `model.context_length` and `model.gpu_layers` before `_start_server()`.

## 2. Error Recovery Uses a Different Model

### Problem
When a task fails (especially on timeout), error recovery routes to the same model that caused the failure — guaranteeing another failure.

### Design
Pass `exclude_models: [failed_model_name]` in the error recovery task context. The existing `_build_model_requirements()` already reads `exclude_models` from context and the router already filters them out.

**Model name source:** Read from the failed task's checkpoint (`used_model` field) or from the conversation log in DB. Fall back to empty list if unavailable.

**Changes:**
- `orchestrator._spawn_error_recovery()` — look up last used model, add to context
- `base.py._build_model_requirements()` — already reads `exclude_models` from context (no change needed)

## 3. Prometheus + Grafana Docker Stack

### Problem
No way to visually monitor llama-server performance or orchestrator health over time.

### Design

**New `/metrics` endpoint** in `api.py` — Prometheus text format. Exposes:
- `kutay_tasks_completed_total`, `kutay_tasks_failed_total`
- `kutay_queue_depth`, `kutay_cost_total`
- `kutay_model_calls_total{model="..."}`, `kutay_tokens_total{model="..."}`
- `kutay_model_swap_total`, `kutay_model_loaded`, `kutay_model_idle_seconds`

No `prometheus_client` dependency — plain string formatting.

**Docker services** added to `sandbox/docker-compose.yml`:
- Prometheus on port 9090, scraping llama-server:8080 + orchestrator API
- Grafana on port 3001 (3000 taken by vane), pre-provisioned datasource + dashboard

**Config files:**
```
monitoring/
  prometheus.yml
  grafana/
    provisioning/
      datasources/prometheus.yml
      dashboards/dashboard.yml
    dashboards/
      kutay-overview.json
```

**Dashboard panels:**
- LLM: generation tok/s, prompt tok/s, KV cache usage %
- System: VRAM/RAM usage, GPU temp, utilization
- Orchestrator: tasks completed/failed, queue depth, costs, model calls
- Model: current model name, swap count, idle time
