# KutAI Inference Performance X-Ray

> Model speed analysis, routing decisions, and Perplexica latency breakdown.
> Based on measurements from 2026-03-28 session logs + live benchmarks.
> All benchmarks run with `--n-gpu-layers 99` (auto-fit), `--flash-attn auto`, `--threads 9`, `--batch-size 2048`, `--ubatch-size 512`.

---

## GPU Environment

- **GPU**: NVIDIA 8GB VRAM (8192 MiB total)
- **CPU**: 9 inference threads (physical cores - 2)
- **llama-server**: flash-attn auto, batch 2048, ubatch 512

---

## Full Benchmark Results (2026-03-28)

All models tested with `--n-gpu-layers 99` (llama-server auto-clamps to VRAM capacity).

| Model | Size | Gen tok/s | Prompt tok/s | VRAM | Load | Thinking | Status |
|-------|------|----------|-------------|------|------|----------|--------|
| **nerdsking-python-7B** | 5.1GB | **82.6** | 2662 | 6.3GB | 6s | NO | Full GPU, fast but code-only |
| **Qwen3.5-9B-UD-Q4_K_XL** | 5.6GB | **53.0** | 432 | 7.6GB | 7s | YES* | Full GPU, best general-purpose speed |
| **Qwen3-Coder-30B-A3B (MoE)** | 17GB | **7.1** | 76 | 7.6GB | 50s | NO | Partial GPU, good for code |
| **GLM-4.7-Flash-UD-Q4_K_XL** | 17GB | **5.6** | 58 | 7.6GB | 49s | YES (forced) | Partial GPU, forced thinking wastes tokens |
| **Qwen3.5-35B-A3B-UD (MoE)** | 21GB | **4.4** | 47 | 7.6GB | 62s | YES* | Partial GPU, highest quality |
| **Qwen3.5-27B.Q4_K_M** | 16GB | **0.6** | 5 | 7.7GB | 49s | YES | Mostly CPU, medium test timed out (120s) |
| **gemma-3-27b-heretic** | 14GB | **1.0** | 7 | 7.6GB | 26s | NO | Mostly CPU, medium test timed out |
| **Apriel-15B-Thinker** | 12GB | **FAIL** | -- | -- | >90s | -- | Failed to load (OOM or format issue) |

*Thinking models: Qwen3.5-9B and 35B have thinking capability but it can be disabled via `--chat-template-kwargs`. GLM-4.7-Flash ignores the disable flag.

**Benchmark methodology**: All models tested with `--n-gpu-layers 99` (auto-fit to VRAM). Short test: 1-sentence prompt, 50 output tokens. Medium test: ~100 token prompt, 200 output tokens. Medium gen tok/s used as the definitive speed metric.

**Orchestrator vs benchmark**: The orchestrator's conservative `calculate_gpu_layers()` gave GLM only 16-17 layers (1.2 tok/s). With auto-fit (`--n-gpu-layers 99`), GLM reaches 5.6 tok/s — a **4.7x improvement** from the same model with better params.

### Speed Tiers

| Tier | Models | Gen tok/s | Use Case |
|------|--------|-----------|----------|
| **Fast** (>30 tok/s) | Qwen3.5-9B, nerdsking-7B | 41-85 | Classification, simple Q&A, shopping search, Perplexica synthesis |
| **Medium** (5-10 tok/s) | Qwen3-Coder-30B, GLM-4.7, Qwen3.5-35B MoE | 4.5-7.4 | Complex tasks where quality matters, code generation |
| **Slow** (<2 tok/s) | Qwen3.5-27B, gemma-3-27b | 1.0-1.3 | Unusable for interactive tasks, only background batch work |

---

## GLM-4.7-Flash Performance Analysis

### Orchestrator vs Optimal Configuration

| Config | GPU Layers | Ctx | Gen tok/s | Improvement |
|--------|-----------|-----|-----------|-------------|
| **Orchestrator** (calculated) | 16-17 | 8192 | **1.2** | baseline |
| **Manual** (`--n-gpu-layers 40`) | ~40 | 4096 | **9.0** | **7.5x faster** |
| **Auto-max** (`--n-gpu-layers 99`) | auto | 4096 | **6.0** | **5x faster** |

### Root Cause of Slow Speed

1. **Conservative GPU layer calculation**: `calculate_gpu_layers()` uses `(available_vram - 300) * 0.90` safety margin, computing only 16-17 layers for GLM (17GB model). With 75% of layers on CPU, generation drops to 1.2 tok/s.

2. **Forced thinking**: GLM-4.7-Flash is in `_THINKING_FAMILIES` and generates `reasoning_content` even with `enable_thinking: false`. These wasted tokens reduce effective content speed to near 0 for short outputs.

3. **Why not 30 tok/s**: Even with maximum GPU offload (40 layers, 7.7GB VRAM), GLM only achieves 6-9 tok/s. The 30 tok/s previously observed was likely Qwen3.5-9B (which genuinely achieves 41 tok/s). GLM at 29.9B params with partial GPU offload cannot reach 30 tok/s on 8GB VRAM.

### Fix Impact

Passing `--n-gpu-layers 99` instead of the calculated value gives llama-server freedom to fit as many layers as possible. This is the single biggest performance improvement available: **1.2 → 6.0+ tok/s** (5x) for all partially-offloaded models.

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

### Fix 6: Pass `--n-gpu-layers 99` Instead of Calculated Value

**Problem**: Conservative GPU layer calculation leaves most layers on CPU for large models.
**Fix**: Pass `--n-gpu-layers 99` and let llama-server auto-fit. llama-server already handles VRAM limits gracefully. Only override with calculated value if `models.yaml` specifies explicit `gpu_layers`.

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
