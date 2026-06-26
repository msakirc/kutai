# Design: VRAM-aware local model load sizing

**Date:** 2026-05-31
**Status:** Draft for review
**Author:** session (founder-directed)
**Supersedes:** the `calculate_dynamic_context` + `BASELINE_LOCAL_CTX` floor + swap-time `-ngl` OOM-fallback machinery, and the orphaned load-mode enforcement in `src/core/router.py`.

---

## TL;DR

Local model loading currently sizes **context from VRAM** (wrong axis) and treats
**GPU layers as a static build-time constant** (never recomputed per call). The
user-set resource policy ("yük modu": full/heavy/shared/minimal) is disconnected
from both the live selector and the loader. The result: every local load runs at
a *de facto* fixed 16384 context, which OOMs the GPU under transient VRAM pressure
(mission_79, 2026-05-30) yet still forces reloads for genuinely large tasks.

Invert it to the correct model:

> **context = the call's actual need (deterministic) → GPU layers = fit(need-ctx, live VRAM, yük-modu).**

`--fit` stays primary/default; an explicit `-ngl` is passed **only** when the load
mode caps VRAM below what `--fit` would greedily grab.

---

## Goals

1. Context window per load is driven by the **call's real need**, not by live VRAM.
2. GPU-layer count is computed **per load** from (need-ctx, live free VRAM, load mode, model features), or delegated to `--fit` when no cap applies.
3. The **yük modu** resource budget actually reaches **both** model selection and model loading.
4. One VRAM-accounting source of truth (no 0.5-vs-1.0 KV drift).
5. Retire the band-aids (`calculate_dynamic_context`, `BASELINE_LOCAL_CTX`, swap-time OOM `-ngl` fallback, the uncommitted `vram_context_ceiling` stopgap).

## Non-goals

- Changing cloud model selection/quota logic.
- Changing the circuit breaker (it stays, as a genuine-failure backstop).
- Changing the yük-modu UI or the auto-detect loop in `nerd_herd/load.py` (already correct; just consume it).
- Multi-GPU.

---

## Current state (evidence)

### The inversion

`src/models/local_model_manager.py::LocalModelManager.swap()`:
1. `calculate_dynamic_context(... available_vram_mb=projected_vram_free ...)` → derives ctx **from VRAM** (`packages/fatih_hoca/src/fatih_hoca/registry.py:440`). KV=**1.0** MB/layer/1k, reserve 1300/1800.
2. `min_context` bump (the call's real need, demoted to a floor).
3. `BASELINE_LOCAL_CTX=16384` floor (intake #73) — VRAM-blind; bumps ctx back up unconditionally.

`gpu_layers` is `info.gpu_layers`, a **static value computed once at registry-build**
(`registry.py:1582`, `calculate_gpu_layers(..., context_length=8192)`, KV=**0.5**),
never recomputed per call with live VRAM. The per-load layer decision is punted to
`--fit`; on OOM, `swap.py` retries with a stale build-time `-ngl` (`packages/dallama/src/dallama/swap.py:281-302`).

### Runtime proof (kutai.jsonl, 2026-05-29 → 05-30)

- The dynamic calc returns 4096–12288 even with **5–7 GB VRAM free**, then the floor bumps to 16384 on essentially **every** load → ctx is *de facto* constant 16384.
- OOM under transient spike: `cudaMalloc 4854 MiB → out of memory` at ~4.2 GB free → circuit breaker 300s → with cloud exhausted, `researcher` DLQ-storm.
- Real per-call needs (`min_context` bumps): `4412 ×6, 7240, 7331, 7360, 8533, 8566 ×6, 9313, 10207 ×8, 22155, 22474, 24428, 26382, 28335`.
- Reloads ("to expand context"): only 5, all upward from ≥16384 (`16384→18247`, `22155→`, …). None from 4096. Smallest genuine need = **4412**.

### The yük-modu disconnect

- Policy is fully defined: `packages/nerd_herd/src/nerd_herd/load.py` — `VRAM_BUDGETS = {full:1.0, heavy:0.9, shared:0.5, minimal:0.0}`, `is_local_inference_allowed = mode != "minimal"`, plus an auto-detect loop that downgrades on external GPU usage.
- Enforcement lives only in `src/core/router.py:233-247` and calls the **sync** shims `is_local_inference_allowed()` / `get_vram_budget_fraction()` which **always return `True` / `1.0`** (`src/infra/load_manager.py:43,67`). The async versions query NerdHerd; the router calls the sync ones → enforcement is a no-op.
- `fatih_hoca.select()` (the live selection path) has **zero** load-mode references.
- The loader never receives the budget at all.

→ yük modu is **doubly severed**: a no-op at selection, absent at load.

---

## Target architecture

### A. need-ctx (deterministic, primary)

The dispatcher already computes the requirement
(`src/core/llm_dispatcher.py:386-388`):

```python
_min_ctx = min_context
if _min_ctx <= 0 and (estimated_input_tokens or estimated_output_tokens):
    _min_ctx = int((estimated_input_tokens + estimated_output_tokens) * 1.3) + 512
```

Promote it to the authoritative load context:

```python
need_ctx = clamp(ceil_2048(_min_ctx if _min_ctx > 0 else MIN_CTX),
                 low=MIN_CTX, high=model.trained_window)
```

- `MIN_CTX = 8192` (evidence: smallest genuine need 4412; 8192 covers the whole bottom cluster with margin; estimate-less overhead calls are <4k so never reload). Env override `LLAMA_MIN_CTX`.
- `ceil_2048` = round up to a 2048 multiple (llama-server pre-allocates full-window KV).
- `model.trained_window` = registry ceiling (never exceed the trained window).

`swap()` loads at `need_ctx` **exactly**. No `calculate_dynamic_context`, no baseline floor. The existing `loaded_ctx_insufficient` reload guard (`llm_dispatcher.py:582`) stays — it forces a reload when a later call needs more than the resident window.

### B. VRAM budget from yük modu (resolved once, async, at the source)

The **dispatcher** resolves load-mode state once per request (it is async and is
the single orchestration point), then threads it into both selection and loading:

```python
local_allowed = await is_local_inference_allowed_async()      # minimal → False
budget_frac   = await get_vram_budget_fraction_async()        # full 1.0 / heavy 0.9 / shared 0.5
```

Effective VRAM ceiling for a load:

```python
budget_mb = min(int(vram_total_mb * budget_frac), live_free_mb)
```

(`fraction × total` is the *policy* ceiling — "leave headroom for the desktop";
`live_free_mb` is the *physical* ceiling; use the min.)

This **replaces** the sync-stub path in `router.py`. The sync `get_vram_budget_fraction()` / `is_local_inference_allowed()` shims are removed (or kept only as explicitly-labelled non-policy defaults for non-async callers, if any remain).

### C. Layer fit (unified calculator, mode-aware)

One calculator, correct constants. Evolve the existing `calculate_gpu_layers`
(`registry.py:564`) in place — fix KV to the real rate, add feature-reserve kwargs,
keep the name and its registry-build caller:

```python
def calculate_gpu_layers(file_size_mb, n_layers, context_length, available_vram_mb,
                         *, thinking=False, vision=False) -> int:
    reserve = BASE_RESERVE_MB                       # 1300
    if thinking: reserve += THINKING_RESERVE_MB     # +500
    if vision:   reserve += VISION_RESERVE_MB       # +N (mmproj/clip)
    usable = available_vram_mb - reserve
    weight_per_layer = file_size_mb / n_layers
    kv_per_layer     = (context_length / 1024) * KV_PER_LAYER_PER_1K_MB   # 1.0, unified
    layers = int(usable / (weight_per_layer + kv_per_layer))
    return clamp(layers, 0, n_layers)
```

Constants (`KV_PER_LAYER_PER_1K_MB`, `BASE_RESERVE_MB`, `THINKING_RESERVE_MB`,
`VISION_RESERVE_MB`) hoisted to module level in `registry.py`, single source of truth.

Decision at load time (in `LocalModelManager.swap()`):

```python
fit_layers = calculate_gpu_layers(..., context_length=need_ctx,
                            available_vram_mb=budget_mb, thinking=..., vision=...)
if budget_frac >= 1.0 and budget_mb >= live_free_mb:
    pass                       # full mode + no cap → --fit, no -ngl (greedy, as today)
elif fit_layers <= 0:
    # budget can't hold even 1 layer at need_ctx → don't load locally
    raise/▸ skip → selection should have excluded this; treat as availability
else:
    extra_flags += ["--n-gpu-layers", str(fit_layers)]   # proactive cap below --fit
```

This is the user's rule verbatim: **`--fit` primary and default; pass `-ngl` only
when load mode forces less VRAM than `--fit` would take.**

### D. Wiring

- **Selection (`fatih_hoca.select()` / its eligibility filter):** accept the
  dispatcher-resolved `(local_allowed, budget_frac, gpu_state)`. If `not local_allowed`
  → exclude local. If `budget_frac < 1.0` → exclude local models that can't fit
  `≥1` layer (or a min viable fraction) within `budget_mb` at the call's `need_ctx`;
  apply the existing local-preference penalty. Port this from the dead router.py block.
- **Loading (`LocalModelManager.swap()`):** receives `need_ctx` + `budget_mb` + feature
  flags; sets `ServerConfig.context_length = need_ctx` and conditionally injects `-ngl`
  per §C. dallama's `swap.py` stays a pure executor (receives a `ServerConfig`, runs it).
- **Retire** `src/core/router.py` load-mode enforcement (sync-stub path).

### Retirements (delete)

| Item | Location | Why |
|------|----------|-----|
| `calculate_dynamic_context` | `registry.py:440` | ctx no longer derived from VRAM |
| `BASELINE_LOCAL_CTX` + floor | `local_model_manager.py` | need-ctx makes it obsolete |
| `vram_context_ceiling` + `_floored_baseline_ctx` (stopgap) | `registry.py` / `local_model_manager.py` | uncommitted; superseded |
| OOM `-ngl` fallback + `_stderr_shows_oom` | `dallama/swap.py:281-329` | replaced by proactive cap; circuit breaker remains |
| sync `get_vram_budget_fraction` / `is_local_inference_allowed` consumption | `router.py:233-247,317` | replaced by async resolution in dispatcher |

---

## Data flow

```
Beckman → dispatcher.request(min_context, est_in/out, needs_thinking, needs_vision, ...)
  ├─ resolve load mode (async): local_allowed, budget_frac
  ├─ need_ctx = clamp(ceil2048(min_ctx or estimate or MIN_CTX), MIN_CTX, model_window)
  ├─ fatih_hoca.select(reqs, local_allowed, budget_frac, gpu_state)   # mode-aware eligibility
  └─ ensure_local_model(model, need_ctx, budget_mb, thinking, vision)
        └─ swap(): ServerConfig(ctx=need_ctx, extra_flags=[-ngl fit_layers] if capped)
              └─ dallama.swap → llama-server (--fit by default, -ngl only when capped)
```

## Load-mode semantics

| Mode | budget_frac | Local? | Loader behaviour |
|------|-------------|--------|------------------|
| full | 1.0 | yes | `--fit` greedy (no `-ngl`) |
| heavy | 0.9 | yes | `-ngl` capping VRAM ≤ 90% × total |
| shared | 0.5 | yes | `-ngl` capping VRAM ≤ 50% × total; selection prefers cloud for big models |
| minimal | 0.0 | **no** | local excluded at selection → cloud only |

Auto-detect (`nerd_herd/load.py`) keeps downgrading mode on external GPU usage; now
that downgrade actually changes how models load.

## Error handling

- A genuine OOM is still possible (estimate wrong, spike between calc and alloc). The
  circuit breaker stays as the backstop, but should trip **far** less: ctx is right-sized
  and `-ngl` is proactively capped, so `--fit`'s greedy grab no longer races desktop spikes.
- "Can't fit ≥1 layer within budget" is an **availability** condition (founder principle:
  WAIT, don't DLQ) — rides the existing `loading`/`availability` transient ladder.
- `loaded_ctx_insufficient` reload guard retained for upward context growth.

## Testing

- `gpu_layers_for`: tight VRAM → fewer layers; ample → all layers; thinking/vision raise reserve → fewer layers; budget cap reduces layers vs free.
- `need_ctx`: uses min_context when present; estimate fallback; MIN_CTX floor when both absent; clamps to model window; rounds to 2048.
- `swap()` integration (unit on the policy, mocked dallama): full mode → no `-ngl`; shared mode → `-ngl == calculate_gpu_layers(budget_mb)`; minimal handled at selection.
- Selection eligibility: minimal excludes local; shared excludes a model too big for budget; full admits.
- Regression: replay the mission_79 spike numbers (need_ctx ~8192, free ~4.2 GB) → asserts a fitting `-ngl`, no OOM path.
- Constant unification: one KV rate; no 0.5 remaining.

## Risks / open questions

- **R1 `gpu_layers_for` accuracy.** If it under-offloads, models run slower (CPU); if it over-offloads, OOM. Mitigation: reserves tuned from observed values (1300 base, +500 thinking); circuit breaker backstop; validate against the speed cache after rollout.
- **R2 fatih_hoca.select signature.** Threading mode state may touch several call sites; verify select() is the sole live selection path before deleting router.py enforcement (audit call sites, not docstrings).
- **R3 budget basis.** Chosen `min(frac×total, live_free)`. `nerd_herd.get_vram_budget_mb` currently uses `free×frac`; reconcile to one definition.
- **R4 vision reserve** (`VISION_RESERVE_MB`) needs a measured value; start conservative.
```
