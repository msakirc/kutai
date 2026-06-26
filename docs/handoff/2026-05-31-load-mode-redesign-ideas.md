# Handoff — Load mode / local-load resource management redesign (IDEAS)

**Date:** 2026-05-31
**Type:** Design ideas / direction. NOT a spec to implement verbatim — read, then write a fresh spec via brainstorming.
**Trigger:** mission_79 researcher OOM-storm (see sibling handoff `2026-05-31-mission-debug-continuation.md`). Root-causing it exposed that the whole local-load resource model is inverted and that "yük modu" is disconnected.

---

## How we got here (the arc)

1. `researcher prior_art_search` spewed `All models failed … Failed to load Qwen3.5-9B-UD-Q4_K_XL-thinking (demoted 5 min)` / `No model candidates available`.
2. Root cause: transient external VRAM spike → 9B OOM at a **floored ctx=16384** → circuit breaker 300s → cloud exhausted → starvation. (Details in the debug handoff.)
3. First fix attempted: make the `BASELINE_LOCAL_CTX=16384` floor VRAM-aware (shipped as an **uncommitted stopgap** — see debug handoff).
4. Founder pushback #1: *"ctx should be calculated dynamically per call + vram state; gpu layers dynamically; vram already considered; worked for months."* → the stopgap patches the wrong axis.
5. Founder pushback #2: *"determine the exact context NEEDED, then calculate layers to fit (or rely on --fit / pass a custom number), considering vram + load mode."* → **the correct model is the inverse of the current one.**
6. Founder pushback #3 (the crown insight): *"is limiting vram at load mode useful or reverse? user apps rely on RAM more than GPU; limiting vram clamps ram even more."* → the VRAM-cap mechanism is backwards for the common case.
7. Founder asked for the "moon shot," then scoped to **need-ctx only** for now, then parked the whole thing to **this handoff** (no implementation this session).

---

## The core principle (what the redesign must encode)

> **context = the call's real need (deterministic per call) → GPU layers = fit(need-ctx, live VRAM, mode). `--fit` is primary/default; pass `-ngl` only when something forces a smaller footprint than `--fit` would grab.**

Current code does the opposite on **both** axes (see spec evidence section): ctx is *derived from VRAM* and bumped to a fixed 16384 floor; gpu_layers is a *static build-time constant*. Net effect: every local load runs at a de-facto fixed 16384 ctx, which OOMs under VRAM pressure yet still reloads for genuinely large tasks.

### need-ctx (the agreed, simplest fix)

The requirement is ALREADY computed in the dispatcher (`src/core/llm_dispatcher.py:386-388`, `_min_ctx = min_context or (est_in+est_out)*1.3+512`). Promote it to BE the load ctx:

```
need_ctx = clamp(ceil_2048(min_context or estimate or MIN_CTX), MIN_CTX, model.trained_window)
```

- `MIN_CTX = 8192` — **evidence-backed** (kutai.jsonl 05-29/30): smallest genuine task need = 4412; the bottom cluster is 4412–10207; 8192 covers it; the only reloads ever seen were upward from ≥16384 (18k→28k tasks, which carry their own `min_context`). 4096 has zero margin (4412 > 4096). Env `LLAMA_MIN_CTX`.
- Load at need_ctx exactly. Retire `calculate_dynamic_context` + the `BASELINE_LOCAL_CTX` floor + the uncommitted `vram_context_ceiling`/`_floored_baseline_ctx` stopgap.
- With small ctx, `--fit` fits the 9B comfortably even under the spike → OOM gone, without touching layer logic. **This alone is the fire fix.**

---

## The crown insight: VRAM-cap is the WRONG mechanism for "back off"

| Load shape | VRAM | CPU/RAM | Speed |
|---|---|---|---|
| Full GPU offload | high | **low** | fast |
| CPU offload (capped `-ngl`) | low | **high** (weights+KV in RAM, compute on CPU) | slow |
| Cloud | 0 | 0 | net |

Desktop lag comes from **CPU+RAM** contention (swapping, stutter), not idle VRAM. For a non-gamer, **VRAM is the spare resource.** So capping KutAI's VRAM pushes the model onto CPU/RAM — the exact resources that make the desktop feel slow — while idling VRAM nobody uses. **Backwards.**

VRAM-capping only helps under **genuine GPU contention** (games, video render, CUDA apps). And even then, the better yield is **cloud or unload**, not CPU-offload (which then fights the game for CPU/RAM too).

→ Genuinely useful "yield to user" levers: **prefer cloud → keep local ON GPU with need-ctx → unload when idle.** CPU-offload (`-ngl` cap) only as a last resort: local is the *only* option AND model doesn't fully fit.

---

## The disconnect (facts, verified this session)

- Policy is fully defined: `packages/nerd_herd/src/nerd_herd/load.py` — `VRAM_BUDGETS={full:1.0, heavy:0.9, shared:0.5, minimal:0.0}`, `is_local_inference_allowed = mode!="minimal"`, plus an auto-detect loop that downgrades on `external_vram_fraction`.
- Telegram UI works: `🖥 Yük Modu` → Full/Heavy/Shared/Minimal/Otomatik (`src/app/telegram_bot.py:169`, handlers `load_full|heavy|shared|minimal`).
- **Severed at selection:** enforcement lives only in `src/core/router.py:233-247,317` and calls the **sync** shims `is_local_inference_allowed()` / `get_vram_budget_fraction()` which **always return `True`/`1.0`** (`src/infra/load_manager.py:43,67`). The async versions query NerdHerd; the router calls the sync ones → **no-op.**
- `fatih_hoca.select()` (the live selection path) has **zero** load-mode references.
- **Severed at load:** the loader (`LocalModelManager.swap`) never receives the budget at all.
- Auto-detect only senses **external VRAM** — there is NO RAM-pressure or user-presence signal today. That's the structural blind spot behind the crown insight.

---

## Moon shot: Resource Governor (the end state)

Replace the static "VRAM-%" lever with a **placement governor**. Sense more, cap less.

**Signals:**
- VRAM free + external GPU usage ✓ (have it)
- **RAM pressure** (psutil.virtual_memory — easy, missing)
- **User presence**: input-idle (`GetLastInputInfo` via ctypes) + fullscreen/game detection (missing — highest-leverage new signal)
- Cloud availability ✓ (KDV)

**Policy (placement, not capping):**
| User state | Action |
|---|---|
| Away / idle (most missions run overnight) | **Full send** — max local GPU, big models |
| Active, light (browse/work) | keep local **on GPU** + need-ctx + throttle concurrency + cloud for bursts. **No CPU-offload.** |
| Active, GPU-heavy (game/render) | **cloud**; if cloud dead → **unload + defer** local-only tasks (WAIT, founder principle) |
| Active, RAM-heavy | keep on GPU, smaller model, no CPU-offload |

**Mechanisms:** selection bias (cloud↔local) + need-ctx + concurrency throttle + idle-unload/resume + small VRAM safety margin (~0.5–1 GB so spikes never OOM). `-ngl` cap only last-resort.

**Manual buttons → presets** over the governor: Full = ignore user; Otomatik = governor decides; Heavy/Shared = cloud-bias strength; Minimal = cloud-only/pause. Load mode is NOT killed — its *mechanism* changes from VRAM-cap to placement.

---

## Decisions locked this session

- **ctx = need**, not VRAM-derived. MIN_CTX = 8192.
- Load-mode mechanism should change from **VRAM-cap → placement** (cloud-bias + keep-on-GPU + unload). VRAM-cap / CPU-offload is wrong for RAM-heavy desktop use.
- Governor is the end state; needs RAM + presence sensing (new).
- Stopgap (VRAM-aware floor) → revert when the real redesign lands (it's uncommitted; see debug handoff for the keep-vs-revert decision still pending).

## Open questions for the next designer

- **Staging:** ship need-ctx alone first (kills the OOM, removes the inversion), then the governor? (Founder leaned minimal — need-ctx only — then parked all of it here.)
- Is `fatih_hoca.select()` the *sole* live selection path? **Audit call sites, not docstrings** before deleting router.py enforcement.
- Budget basis to reconcile (only relevant if any VRAM-budget survives): `min(frac×total, live_free)` vs nerd_herd's `free×frac`.
- Presence detection on Windows: `GetLastInputInfo` (idle), foreground-fullscreen / nvidia-smi utilization (game), psutil (RAM). Feasible; pick the cheapest reliable signals.
- Where should need_ctx be computed/threaded — dispatcher (has the estimates) passing into swap as the target ctx (not a floor)?

---

## Spec disposition — KEEP (do not kill)

`docs/superpowers/specs/2026-05-31-vram-aware-load-sizing-design.md` stays. It is **useful**:
- Its **current-state / evidence** section (the 5 inversions, runtime-log proof, the sync-stub disconnect, the need-ctx data) is accurate and expensive to re-derive — reuse it.
- Its **need-ctx** design (§A) matches the agreed fix.

But it is **superseded in direction**:
- Its framing is "full redesign, ship now" — stale (founder parked it).
- Its §B/§C/§D (load-mode → VRAM-budget → `-ngl` cap) encode the **wrong mechanism** (the VRAM-cap the crown insight refutes). Treat §B/§C/§D as historical, NOT the plan.

→ Mine the spec for evidence + need-ctx; take the load-mode *direction* from THIS handoff. The eventual implementation spec should be written fresh (brainstorming) once the staging question is answered.
