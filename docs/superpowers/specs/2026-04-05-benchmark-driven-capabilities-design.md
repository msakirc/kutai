# Benchmark-Driven Model Capabilities

**Date:** 2026-04-05
**Status:** Approved
**Branch:** feat/unified-task-lifecycle

## Problem

1. **8/12 benchmark fetcher URLs return 404** — Artificial Analysis, HF Leaderboard, LiveCodeBench, BFCL, LMSys Arena, BigCodeBench all broken. Only Aider and OpenRouter still work.
2. **Family profile capability ratings are outdated** — scores were set manually and haven't kept up with actual benchmark data.
3. **No thinking-on vs thinking-off differentiation** — thinking-capable models (Qwen3.5, Gemma 4, gpt-oss, Apriel Thinker, GLM, QwQ) score identically regardless of thinking mode, but capabilities differ significantly.
4. **Family-based scaling is imprecise** — size interpolation + quant retention math adds unnecessary error for the ~10 curated GGUF models actually on disk.

## Design

### Core Principle

**Benchmark APIs are the primary source of capability scores.** Family profiles are fallback only for models with no benchmark data.

### Capability Flow (Priority Order)

```
1. Benchmark cache (fetched from 7+ APIs, stored locally)
   ↓ (miss)
2. models.yaml capability overrides (manual per-model)
   ↓ (miss)
3. Family profile + size/quant scaling (fallback for unknown models)
```

### Mode Variants: Dual/Multi ModelInfo Entries

Any server flag that requires a restart to toggle = separate ModelInfo entry. Currently two such flags:

1. **Thinking** (`--reasoning on/off`) — changes reasoning capabilities
2. **Vision** (`--mmproj <path>`) — enables vision, drains ~900MB VRAM

**Variant naming convention:** `ModelName[-thinking][-vision]`

**Example — Gemma-4-26B-A4B (thinking + vision capable):**
- `Gemma-4-26B-A4B` — base, fastest, no VRAM overhead
- `Gemma-4-26B-A4B-thinking` — reasoning tasks, slower
- `Gemma-4-26B-A4B-vision` — image understanding, +900MB VRAM
- `Gemma-4-26B-A4B-thinking-vision` — both (rare, heaviest)

**Example — Qwen3.5-35B-A3B (thinking only, no vision):**
- `Qwen3.5-35B-A3B` — base
- `Qwen3.5-35B-A3B-thinking` — reasoning tasks

**Example — GigaChat3.1-Lightning (neither):**
- `GigaChat3.1-Lightning` — single entry, no variants

**Capability differences per mode:**

Thinking ON vs OFF:
- **Reasoning, planning, analysis, code_reasoning**: +0.5 to +1.5
- **Instruction adherence, structured_output**: -0.5 to -1.0
- **Conversation**: -0.5
- **Code generation**: +0.0 to +0.5 (better logic but worse format compliance)

Vision ON vs OFF:
- **Vision**: 0.0 → actual score (only capability that changes)
- All other capabilities identical

Benchmark sources that test both thinking modes (Aider, LiveCodeBench) provide separate scores. For sources that don't distinguish, the base score maps to thinking-off, and a delta function estimates thinking-on scores. Vision scores come from vision-specific benchmarks or family profiles.

### Scorer Integration

No changes to `score_model_for_task()`. The router picks the best model from the registry — it now has mode variants as separate candidates. Vision variants only win when `needs_vision=True`. Thinking variants compete naturally based on capability fit.

### Swap Logic

The `LocalModelManager` maps all variants of the same model to the same GGUF path. Mode transitions are **lightweight restarts** (kill + relaunch with different flags), not full model swaps:
- Same GGUF, no re-read from disk needed beyond what llama-server does at startup
- Does NOT count against the swap budget
- If already loaded in the correct mode, no action needed

## File Changes

### 1. `src/models/benchmark/benchmark_fetcher.py`

**Source Inventory (8 sources → covers both local and cloud models):**

| # | Source | URL | Auth | Format | Models | Covers |
|---|--------|-----|------|--------|--------|--------|
| 1 | **Artificial Analysis** | `GET https://artificialanalysis.ai/api/v2/data/llms/models` | `x-api-key` header | JSON | 459 | cloud + local |
| 2 | **Chatbot Arena ELO** | `https://huggingface.co/api/datasets/mathewhe/chatbot-arena-elo/parquet/default/train/0.parquet` | none | Parquet | 218 | cloud + local |
| 3 | **Open LLM Leaderboard** | `https://huggingface.co/api/datasets/open-llm-leaderboard/contents/parquet/default/train/0.parquet` | none | Parquet | 4576 | local only |
| 4 | **LiveCodeBench** | `https://livecodebench.github.io/performances_generation.json` | none | JSON | 28 | cloud + local |
| 5 | **BFCL** | `https://gorilla.cs.berkeley.edu/data_overall.csv` | none | CSV | 109 | cloud + local |
| 6 | **Aider Polyglot** | `https://raw.githubusercontent.com/Aider-AI/aider/main/aider/website/_data/polyglot_leaderboard.yml` | none | YAML | 69 | cloud + local |
| 7 | **BigCodeBench** | `https://huggingface.co/api/datasets/bigcode/bigcodebench-results/parquet/default/train/0.parquet` | none | Parquet | 202 | cloud + local |
| 8 | **OpenRouter** | `https://openrouter.ai/api/v1/models` | none | JSON | 500+ | cloud + local |

**Removed:** Old HF Leaderboard (results dataset → 500), old LMSys Arena URLs (all 404).
**Added:** Chatbot Arena ELO (HF Parquet), replaces old LMSys Arena fetcher.

**Artificial Analysis API details:**
- API key from env: `ARTIFICIAL_ANALYSIS_API_KEY`
- Rate limit: 1,000 requests/day
- Fields: `artificial_analysis_intelligence_index`, `artificial_analysis_coding_index`, `artificial_analysis_math_index`, `mmlu_pro`, `gpqa`, `hle`, `livecodebench`, `scicode`, `math_500`, `aime`, `ifbench`, `lcr`, `terminalbench_hard`, `tau2`
- Also provides: `pricing`, `median_output_tokens_per_second`, `model_creator.name`, `slug`
- Attribution required: link to https://artificialanalysis.ai/

**Capability mapping per source:**

| Source | Capabilities Covered |
|--------|---------------------|
| Artificial Analysis | reasoning, domain_knowledge, code_generation, code_reasoning, analysis, instruction_adherence (richest — 15 benchmark scores) |
| Chatbot Arena ELO | conversation, prose_quality (human preference signal) |
| Open LLM Leaderboard | reasoning, instruction_adherence, analysis, domain_knowledge (IFEval, BBH, MATH, GPQA, MUSR, MMLU-PRO) |
| LiveCodeBench | code_generation, code_reasoning |
| BFCL | tool_use, structured_output |
| Aider | code_generation, code_reasoning, instruction_adherence |
| BigCodeBench | code_generation, instruction_adherence |
| OpenRouter | context_utilization (context_length tiers) |

**Cloud model enrichment:**
- Sources 1, 2, 4, 5, 6 all cover cloud models (GPT-4o, Claude, Gemini, etc.)
- Cloud models in `CLOUD_PROFILES` get the same benchmark enrichment as local models
- Benchmark data takes priority over hardcoded cloud profiles when available

**Parquet handling:**
- Sources 2, 3, 7 now use HuggingFace Parquet format instead of JSON
- Use `pandas` or `pyarrow` to read (both already in typical Python env)
- Fallback: download raw bytes + parse with minimal parquet reader if pandas unavailable

**Thinking-on/off data:**
- Where sources test both modes, store separate entries
- Aider polyglot has both "ModelName" and "ModelName (thinking)" entries
- AA may have separate entries for thinking variants (e.g., o3 vs o3-mini)
- Cache format: `{model_name: {"thinking_off": {caps}, "thinking_on": {caps}}}`

**Model name matching:**
- Update `_MODEL_ALIASES` for all 10 current GGUF models + cloud models
- AA uses slugs (e.g., `llama-3-3-instruct-70b`), Arena uses HF-style names
- Better fuzzy matching: strip quant suffixes, normalize separators, match by slug/creator

### 2. `src/models/model_profiles.py`

**No structural changes.** Family profiles stay as-is for fallback. Update any obviously wrong ratings for families of models we actually have on disk:

- `gigachat`: Verify ratings against benchmarks
- `gpt_oss`: Verify ratings against benchmarks
- `gemma4`: Verify ratings against benchmarks (currently rated very high — 9.0 reasoning for a 4B-active MoE seems generous)
- `apriel_thinker`: Verify ratings against benchmarks

### 3. `src/models/model_registry.py`

**Multi-variant ModelInfo registration:**
- When a GGUF is discovered, register entries for each valid mode combination:
  - Base: `ModelName` — `thinking_model=False`, `has_vision=False`
  - If thinking-capable: `ModelName-thinking` — `thinking_model=True`
  - If has mmproj: `ModelName-vision` — `has_vision=True`, vision capability populated
  - If both: `ModelName-thinking-vision` — both enabled
- All variants share the same `path` to the physical GGUF file
- New ModelInfo fields:
  - `is_variant: bool` — True for any non-base entry
  - `base_model_name: str` — links variant back to base (for swap logic)
  - `variant_flags: set[str]` — e.g. `{"thinking"}`, `{"vision"}`, `{"thinking", "vision"}`

**`estimate_capabilities()` changes:**
- Check benchmark cache first (via `enrich_registry_with_benchmarks`)
- Benchmark data keyed by `{model: {thinking_off: {caps}, thinking_on: {caps}}}`
- Fall back to family profile + scaling only when no benchmark data
- For thinking variants without benchmark data, apply delta function to base scores

**Non-thinking models (GigaChat, nerdsking):**
- Single ModelInfo entry as before, no changes

### 4. `src/models/capabilities.py`

**No changes.** `score_model_for_task()` already works — it receives a model's capabilities dict and scores it. The dual-entry approach means it naturally gets the right scores.

### 5. `src/models/benchmark/benchmark_cli.py`

**Enhanced `enrich` command:**
- Show which models got benchmark data vs fell back to family profiles
- Show thinking-on vs thinking-off data availability
- Add `--force-refresh` flag to bypass cache TTL

### 6. `src/core/router.py`

**Variant-aware routing:**
- Router sees all variants as separate candidates — no special scoring logic needed
- **Filtering**: `needs_vision=True` → exclude non-vision variants. `needs_thinking=True` → exclude non-thinking variants. If unset, all compete.
- **Swap logic**: When winner is a different variant of the already-loaded model, it's a lightweight restart (same GGUF, toggle flags). Does NOT count against swap budget.

### 7. `src/models/local_model_manager.py`

**Mode-aware loading:**
- `load_model()` accepts the ModelInfo which has `thinking_model` and `has_vision` flags
- If same GGUF is loaded but mode flags differ (thinking and/or vision), do a lightweight restart (kill + relaunch with different `--reasoning`/`--mmproj` flags) instead of full unload+load cycle
- This is faster than a full swap (same GGUF, llama-server just needs different startup flags)

## Models on Disk & Registry Entries

| GGUF File | Type | Quant | Thinking | Vision | Registry Entries |
|-----------|------|-------|----------|--------|-----------------|
| Qwen3.5-35B-A3B | MoE 256E/8A | Q4_K_XL | Yes | No | base, thinking (2) |
| Qwen3.5-27B | Dense | Q4_K_M | Yes | No | base, thinking (2) |
| Qwen3.5-9B | Dense | Q4_K_XL | Yes | Yes (mmproj) | base, thinking, vision, thinking-vision (4) |
| Qwen3-Coder-30B-A3B | MoE 128E/8A | Q4_K_XL | Yes | No | base, thinking (2) |
| GLM-4.7-Flash | MoE 64E/4A | Q4_K_XL | Yes | No | base, thinking (2) |
| Gemma-4-26B-A4B | MoE 128E/8A | IQ4_NL | Yes | Yes (mmproj) | base, thinking, vision, thinking-vision (4) |
| gpt-oss-20b | MoE 32E/4A | Q4_K_XL | Yes | No | base, thinking (2) |
| Apriel-1.6-15b-Thinker | Dense | Q4_K_L | Yes | Yes (mmproj) | base, thinking, vision, thinking-vision (4) |
| GigaChat3.1-Lightning | MoE 64E/4A | Q4_K_M | No | No | base (1) |
| nerdsking-python-coder-7B | Dense | Q5_K_M | No | No | base (1) |

**Total: 10 GGUFs → 24 registry entries** (10 base + 8 thinking + 3 vision + 3 thinking-vision)

## Out of Scope

- Task profiles (weight vectors) — unchanged
- Cloud model profiles — unchanged
- Router scoring formula/weights — unchanged
- Speed benchmarking — already done per inference-performance-xray.md
- models.yaml per-model overrides — stay as-is, still work
