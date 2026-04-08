# KutAI Inference Performance X-Ray

> Model speed analysis, routing decisions, and Perplexica latency breakdown.
> Based on measurements from 2026-04-05 session + live benchmarks.
> All benchmarks run with `--fit` (llama-server auto-allocates GPU layers), `--flash-attn auto`, `--threads 9`, `--batch-size 2048`, `--ubatch-size 512`.

---

## GPU Environment

- **GPU**: NVIDIA GeForce RTX 3070 Ti, 8GB VRAM (8192 MiB total)
- **CPU**: 9 inference threads (physical cores - 2)
- **llama-server**: v8398+, `--fit` (default-on), flash-attn auto, batch 2048, ubatch 512
- **CRITICAL**: Do NOT pass `--n-gpu-layers` — it overrides `--fit` and causes VRAM thrashing for models that don't fully fit. Only pass it when models.yaml has an explicit `gpu_layers` override.

---

## Full Benchmark Results

### Benchmark v3 (Driver 595.97, llama.cpp v8668, 2026-04-05) — CURRENT

Tested via `LocalModelManager.ensure_model()` with `--fit` auto-allocation, `enable_thinking=False`, clean server restart per model. Context set by `calculate_dynamic_context()`.

| Model | Size | Type | Gen tok/s | Prompt tok/s | VRAM | Load | Thinking | Notes |
|-------|------|------|----------|-------------|------|------|----------|-------|
| **GigaChat3.1-Lightning** | 6.1GB | MoE 64E/4A | **107.4** | 295.7 | 7.5GB | 8s | NO | Fastest model, Russian-focused |
| **nerdsking-python-7B** | 5.1GB | Dense | **63.7** | 396.9 | 7.1GB | 6s | NO | Code-only specialist |
| **gpt-oss-20b** | 11.3GB | MoE 32E/4A | **37.4** | 101.7 | 7.5GB | 14s | YES (always-on) | OpenAI open-source, strong agentic |
| **Qwen3.5-35B-A3B (MoE)** | 21GB | MoE | **28.4** | 65.4 | 7.2GB | 23s | NO | Best quality+speed combo |
| **Qwen3.5-9B-UD** | 5.6GB | Dense | **27.6** | 83.9 | 7.3GB | 8s | NO | Good prompt throughput, has vision (mmproj) |
| **Qwen3-Coder-30B-A3B (MoE)** | 17GB | MoE | **27.4** | 65.1 | 7.5GB | 17s | NO | Best for code tasks |
| **GLM-4.7-Flash-UD** | 17GB | MoE | **24.6** | 50.1 | 7.5GB | 20s | NO | v8668 fixed thinking disable |
| **Gemma-4-26B-A4B (MoE)** | 13GB | MoE 26B/4B | **23.2** | 47.4 | 7.4GB | 20s | NO | Vision (mmproj), Google's best open model |
| **Apriel-15B-Thinker** | 8.7GB | Dense | **6.7** | 27.5 | 7.3GB | 11s | INLINE | Vision (mmproj), always-on CoT in content |
| **Qwen3.5-27B** | 16GB | Dense | **2.9** | 8.2 | 6.9GB | 20s | YES | Still thinks despite disable flag, hard-filtered for >1000 output tokens |

**Thinking control**: llama.cpp v8668+ uses `--reasoning off --reasoning-budget 0` (the old `--chat-template-kwargs {"enable_thinking": false}` is deprecated and ignored). Works for all thinking models **except** Qwen3.5-27B (still produces thinking tokens despite flag). Always-on (cannot disable): gpt-oss (reasoning_content), Apriel (inline in content via `--no-jinja --chat-template chatml`), Qwen3.5-27B.

**New models (2026-04-05)**: GigaChat3.1-Lightning, gpt-oss-20b, Apriel-15B-Thinker Q4_K_L + mmproj, Gemma-4-26B-A4B + mmproj.
**Removed**: gemma-3-27b-heretic (replaced by Gemma 4).

### --fit vs --n-gpu-layers 99

`--fit` (default-on in llama.cpp v8000+) auto-calculates optimal GPU layer allocation based on actual free VRAM, reserving ~1GB headroom. Passing `--n-gpu-layers 99` **overrides** `--fit` and force-loads all layers, causing VRAM thrashing when the model doesn't fit:

| Model | --n-gpu-layers 99 | --fit (auto) | Improvement |
|-------|-------------------|-------------|-------------|
| Apriel-15B (8.7GB) | 3.7 tok/s (thrashing) | **6.8 tok/s** | **1.8x** |
| Qwen3.5-9B (5.6GB) | 25.4 tok/s | **25.9 tok/s** | ~same |

Rule: **never pass `--n-gpu-layers` unless models.yaml specifies explicit `gpu_layers` override.**

### Speed Tiers

| Tier | Models | Gen tok/s | Use Case |
|------|--------|-----------|----------|
| **Ultra** (>80 tok/s) | GigaChat3.1-Lightning, nerdsking-7B | 89-116 | Classification, simple Q&A, fast overhead |
| **Fast** (>20 tok/s) | gpt-oss-20b, Qwen3.5-35B MoE, Coder-30B, GLM, 9B, Gemma-4 | 21-36 | Main work, complex tasks, code, vision |
| **Slow** (<10 tok/s) | Apriel-15B, Qwen3.5-27B | 3-7 | Vision tasks (Apriel), background batch (27B) |

---

## Perplexica (Vane) Performance Analysis

### End-to-End Test: "Barış Alper Yılmaz assists 2025-2026"

| Phase | Time | Component | Details |
|-------|------|-----------|---------|
| SearXNG web search | ~6-10s | DuckDuckGo + Wikipedia + others | Inside Docker, httpx, 6s timeout per engine |
| Transformers embedding | ~5-10s | Xenova/all-MiniLM-L6-v2 on CPU | Inside Docker, runs locally |
| **LLM synthesis** | **~50-55s** | Qwen3.5-9B at 41 tok/s | Large context (search results), ~700 output tokens |
| **Total** | **73.8s** | | Status 200, 3550 char answer, 14 sources |

### Simple query: "hello world"

| Phase | Time |
|-------|------|
| Total | 40.9s |

### Bottleneck: LLM Synthesis (70-75% of total time)

The LLM synthesis phase dominates because:
1. Vane sends ALL search results (~10-15 pages of snippets) as context
2. The LLM must read, analyze, and synthesize a comprehensive answer
3. Even at 41 tok/s (fastest model), generating 700 tokens takes ~17s
4. But context processing (prompt eval) for thousands of tokens of search results adds 30-40s

### Why SearXNG Sometimes Fails

SearXNG inside the Vane container has a 6-second timeout per search engine. When engines are slow:
- DuckDuckGo: intermittent 6s timeouts (rate limiting or network)
- Wikipedia: ConnectTimeout to `en.wikipedia.org` (DNS resolution OK but connection slow)
- Result: Vane gets partial or no search results, LLM generates empty/poor answer

### Acceleration Strategy

**Recommended: Direct SearXNG → Agent pipeline (bypass Vane LLM)**

Current: `web_search → Vane → SearXNG → embed → LLM synthesis → return`
Proposed: `web_search → SearXNG directly → return raw results → KutAI agent synthesizes`

This eliminates the redundant LLM synthesis inside Vane. The KutAI agent already has an LLM context and will process the results anyway. Estimated savings: **50-55s per search**.

SearXNG is accessible inside the container at port 8080. Direct API: `http://localhost:3000/api/searxng` or through Docker networking.

### Perplexica + Model Interaction

Perplexica/Vane needs llama-server running to synthesize. Model considerations:

| Scenario | Impact |
|----------|--------|
| Fast model loaded (Qwen3.5-9B, 41 tok/s) | 40-75s total, usable |
| Slow model loaded (27B dense, 1.3 tok/s) | Would take 500s+, will timeout |
| Thinking model loaded (GLM, reasoning tokens) | Extra latency from wasted tokens |
| No model loaded | Vane returns 500 error, falls back to DuckDuckGo |

**Rule**: Perplexica should only be attempted when a fast non-thinking model is loaded on llama-server.

---

## Router Model Selection Issues

### Problem 1: Stickiness Bonus Too Strong (1.40x)

The loaded model gets a 1.40x score multiplier. For Qwen3.5-35B-A3B (score 36.1):
- With stickiness: `36.1 * 1.40 = 50.5`
- Qwen3.5-9B without stickiness: ~34

Result: the slow 35B model wins even for `prefer_speed=True` tasks.

### Problem 2: No True Speed Coefficient

The router scores models on capability, cost, availability, performance history, and speed — but the "speed" dimension uses provider tier (cloud fast/slow), not measured tok/s. Local models all get the same speed score regardless of whether they generate at 1 tok/s or 41 tok/s.

### Problem 3: Classification Triggers Swaps

Classification calls (difficulty 3, ~50 tokens output) should use whatever model is loaded. Instead, the router sometimes selects a different model, triggering a 15-30s swap for a 2-second task.

### Problem 4: Dense >16GB Models Unusable

Qwen3.5-27B (1.3 tok/s) and gemma-3-27b (1.0 tok/s) are effectively unusable on 8GB VRAM. They should be auto-disabled or given very low priority when VRAM < model size.

---

## Required Fixes (Implementation Plan)

### Fix 1: True Speed Coefficient for Models

**Problem**: Router has no awareness of actual measured tok/s.
**Fix**: Store measured gen tok/s in ModelInfo after each inference. Use it as a multiplier in the speed dimension of scoring.

Existing field: `ModelInfo.tokens_per_second` (currently 0.0 for all).
Existing measurement: `router.py` logs `llm performance | speed='X tok/s'` after each call.

Missing link: feed the measured speed back into ModelInfo so the scorer uses it.

### Fix 2: Speed Coefficient Affecting Model Selection

**Problem**: `prefer_speed=True` doesn't actually prefer faster models.
**Fix**: When `prefer_speed=True`, multiply the measured tok/s into the availability or speed score dimension. A 40 tok/s model should score 10x higher on speed than a 4 tok/s model.

### Fix 3: Perplexica Avoids Thinking/Slow Models

**Problem**: Perplexica LLM synthesis is unusable with slow or thinking models.
**Fix**: In `web_search.py:_search_perplexica()`, check the loaded model's measured speed. If < 10 tok/s or model is a known thinking model, skip Perplexica and go straight to DuckDuckGo.

### Fix 4: Agent-Model Map for Web Search

**Problem**: Agents that use web_search (researcher, shopping_advisor, assistant, executor) may get paired with models that can't serve Perplexica.
**Fix**: In AGENT_REQUIREMENTS, agents with web_search in their tools should have `prefer_speed=True` to favor fast models. The shopping_advisor already has this; extend to researcher and assistant.

### Fix 5: Web Search Triggers Model Consideration

**Problem**: When a slow model is loaded and Perplexica fails, the agent falls back to DuckDuckGo (poor results). The system should proactively prefer fast models for search-heavy tasks.
**Fix**: If the task profile includes web_search capability and the loaded model is slow, the dispatcher should consider a swap to a fast model before the agent starts.

### Fix 6: Use `--fit` Instead of `--n-gpu-layers 99` (REVISED 2026-04-05)

**Problem**: `--n-gpu-layers 99` overrides `--fit` and force-loads all layers. For models that exceed VRAM (Apriel 8.7GB on 8GB GPU), this causes VRAM thrashing (3.7 tok/s vs 6.8 tok/s with proper allocation).
**Fix**: Do NOT pass `--n-gpu-layers` at all. llama-server's `--fit` (default-on since v8000+) auto-calculates optimal GPU layers based on actual free VRAM. Only pass explicit `--n-gpu-layers` when `models.yaml` specifies a `gpu_layers` override.

### Fix 7: Disable/Deprioritize Unusable Models

**Problem**: Dense models >16GB (Qwen3.5-27B, gemma-3-27b) run at 1 tok/s on 8GB VRAM.
**Fix**: Auto-demote models where `file_size_mb > available_vram * 1.5` — they'll be >50% CPU-bound. Or add a minimum speed threshold (e.g., 3 tok/s) and demote models that fall below after first measurement.

---

## File Reference

| File | Relevance |
|------|-----------|
| `src/models/model_registry.py:452` | `calculate_gpu_layers()` — conservative VRAM formula |
| `src/models/model_registry.py:_THINKING_FAMILIES` | GLM classified as thinking model |
| `src/models/local_model_manager.py:518` | llama-server launch command |
| `src/core/router.py` | Scoring pipeline, stickiness multiplier (1.40x) |
| `src/core/llm_dispatcher.py` | Swap budget, MAIN_WORK vs OVERHEAD routing |
| `src/tools/web_search.py` | Perplexica integration, timeout (15s) |
| `docs/shopping-intelligence-xray.md` | Shopping system architecture |
| `docs/orchestrator-xray.md` | Orchestrator architecture, scoring weights |
| `scripts/benchmark_all.py` | Benchmark script for all models |
