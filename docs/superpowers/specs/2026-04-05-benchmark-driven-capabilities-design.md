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

### Thinking-On vs Thinking-Off: Dual ModelInfo Entries

Thinking-capable models get **two separate ModelInfo entries** in the registry, pointing to the same physical GGUF:

- `Qwen3.5-35B-A3B` — thinking OFF, faster, better format compliance
- `Qwen3.5-35B-A3B-thinking` — thinking ON, stronger reasoning, slower

Each entry has its own capability scores. The router selects between them like any other model — the thinking variant naturally wins for reasoning-heavy tasks, the non-thinking variant wins for speed/format tasks.

General capability differences (thinking ON vs OFF):
- **Reasoning, planning, analysis, code_reasoning**: +0.5 to +1.5
- **Instruction adherence, structured_output**: -0.5 to -1.0
- **Conversation**: -0.5
- **Code generation**: +0.0 to +0.5 (better logic but worse format compliance)

Benchmark sources that test both modes (Aider, LiveCodeBench) provide separate scores. For sources that don't distinguish, the base score maps to thinking-off, and a delta function estimates thinking-on scores.

### Scorer Integration

No changes to `score_model_for_task()`. The router already picks the best model from the registry — it now has thinking and non-thinking variants as separate candidates. The `LocalModelManager` maps both variants to the same GGUF path, toggling `--reasoning on/off` at load time. If the model is already loaded in the correct mode, no swap needed. If loaded in wrong mode, it's a lightweight restart (same file, different flag).

## File Changes

### 1. `src/models/benchmark/benchmark_fetcher.py`

**URL Updates:**

| Source | Old URL | New URL |
|--------|---------|---------|
| Artificial Analysis | `artificialanalysis.ai/api/text/v1/leaderboard` (404) | Use authenticated API with `ARTIFICIAL_ANALYSIS_API_KEY` from .env |
| HF Leaderboard | `open-llm-leaderboard/results` dataset (500) | `open-llm-leaderboard/contents` via datasets-server |
| LiveCodeBench | `livecodebench.github.io/assets/data/results.json` (404) | `livecodebench.github.io/performances_generation.json` |
| BFCL | `gorilla/.../leaderboard_output.json` (404) | `gorilla.cs.berkeley.edu/data_overall.csv` (CSV format) |
| LMSys Arena | `lmsys/chatbot-arena-leaderboard` HF paths (404) | `lmarena-ai/leaderboard-dataset` via datasets-server |
| BigCodeBench | `bigcodebench/.../results.json` (404) | `bigcode/bigcodebench-results` via datasets-server |
| Aider | Same URLs | Still working, no changes |
| OpenRouter | Same URL | Still working, no changes |

**New response parsing:**
- BFCL: CSV parser instead of JSON
- LiveCodeBench: New JSON schema with `performances` array and `models` array
- HF Leaderboard: `contents` dataset has different column names (`IFEval`, `BBH`, `MATH Lvl 5`, `GPQA`, `MUSR`, `MMLU-PRO`)
- LMSys Arena: New dataset structure with `rating` field, multiple configs (text, coding, vision, etc.)
- BigCodeBench: HF dataset with `complete`, `instruct`, `model`, `size` fields

**Thinking-on/off data:**
- Where sources test both modes, store separate entries
- Aider polyglot leaderboard often has both "ModelName" and "ModelName (thinking)" entries
- Cache format: `{model_name: {"thinking_off": {caps}, "thinking_on": {caps}}}`

**Model name matching:**
- Update `_MODEL_ALIASES` for all 10 current GGUF models
- Better fuzzy matching: strip quant suffixes, normalize separators

### 2. `src/models/model_profiles.py`

**No structural changes.** Family profiles stay as-is for fallback. Update any obviously wrong ratings for families of models we actually have on disk:

- `gigachat`: Verify ratings against benchmarks
- `gpt_oss`: Verify ratings against benchmarks
- `gemma4`: Verify ratings against benchmarks (currently rated very high — 9.0 reasoning for a 4B-active MoE seems generous)
- `apriel_thinker`: Verify ratings against benchmarks

### 3. `src/models/model_registry.py`

**Dual ModelInfo registration for thinking models:**
- When a thinking-capable GGUF is discovered, register TWO ModelInfo entries:
  - `ModelName` — `thinking_model=False`, base capabilities
  - `ModelName-thinking` — `thinking_model=True`, thinking capabilities
- Both entries share the same `path` to the physical GGUF file
- `is_thinking_variant: bool` field added to ModelInfo to identify the thinking entry
- `base_model_name: str` field links thinking variant back to its base (for swap logic)

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

**Dual-entry awareness:**
- Router sees thinking and non-thinking variants as separate candidates — no special logic needed for scoring
- **Swap logic**: When the winner is a thinking variant of the already-loaded model (or vice versa), it's a lightweight restart (same GGUF, toggle `--reasoning` flag), NOT a full model swap. This should not count against the swap budget.
- Filter: if `needs_thinking=True` in requirements, exclude non-thinking variants. If `needs_thinking=False`, exclude thinking variants. If unset, both compete.

### 7. `src/models/local_model_manager.py`

**Thinking mode toggle:**
- `load_model()` accepts the ModelInfo which has `thinking_model` flag
- If same GGUF is loaded but thinking mode differs, do a lightweight restart (kill + relaunch with different `--reasoning` flag) instead of full unload+load cycle
- This is faster than a full swap (no GGUF re-read from disk)

## Models on Disk (Reference)

| Model | Type | Quant | Thinking | Vision | mmproj |
|-------|------|-------|----------|--------|--------|
| Qwen3.5-35B-A3B | MoE 256E/8A | Q4_K_XL | Yes | No | — |
| Qwen3.5-27B | Dense | Q4_K_M | Yes | No | — |
| Qwen3.5-9B | Dense | Q4_K_XL | Yes | No | mmproj-F16 |
| Qwen3-Coder-30B-A3B | MoE 128E/8A | Q4_K_XL | Yes | No | — |
| GLM-4.7-Flash | MoE 64E/4A | Q4_K_XL | Yes | No | — |
| Gemma-4-26B-A4B | MoE 128E/8A | IQ4_NL | Yes | Yes | mmproj-F16 |
| gpt-oss-20b | MoE 32E/4A | Q4_K_XL | Yes | No | — |
| Apriel-1.6-15b-Thinker | Dense | Q4_K_L | Yes | Yes | mmproj-f16 |
| GigaChat3.1-Lightning | MoE 64E/4A | Q4_K_M | No | No | — |
| nerdsking-python-coder-7B | Dense | Q5_K_M | No | No | — |

## Out of Scope

- Task profiles (weight vectors) — unchanged
- Cloud model profiles — unchanged
- Router scoring formula/weights — unchanged
- Speed benchmarking — already done per inference-performance-xray.md
- models.yaml per-model overrides — stay as-is, still work
