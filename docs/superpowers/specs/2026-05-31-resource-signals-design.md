# Design — Desktop-Aware Resource Signals (the "Resource Governor" moon shot, reframed)

**Date:** 2026-05-31
**Status:** Approved design, ready for implementation plan.
**Supersedes direction of:** `docs/handoff/2026-05-31-p2-resource-governor-handoff.md` and `docs/handoff/2026-05-31-load-mode-redesign-ideas.md` (both proposed a standalone "governor" — rejected; see §2).
**Prereq shipped:** P1 need-ctx (commit `2330f1ab`) — OOM fire is out.
**Mines for evidence only:** `docs/superpowers/specs/2026-05-31-vram-aware-load-sizing-design.md` (its current-state/evidence section; its §B/§C/§D VRAM-cap mechanism is the WRONG lever — ignore).

## 0. Verification status (verified against code 2026-05-31)

Every load-bearing claim below was checked against the source, not docstrings:

- **`pressure_for` IS called for local candidates.** `ranking.py::_apply_utilization_layer` (138-159) calls `snapshot.pressure_for(sm.model, …)` for **every** scored model unconditionally. New local-only signals will fire. ✓ (this is the linchpin — design dies without it)
- **Admission gate** = `threshold = max(-1.0, -0.5 - 0.5*urgency)`; admit iff `urgency >= threshold and urgency > -1.0` (`selector.py:326,335`). A `−10` sentinel clamps the scalar to `−1.0` → fails `> -1.0` → blocked even at max urgency. Hard-veto mechanism confirmed. ✓
- **Local pressure ≈ S9 only today.** Local model → `provider=""` → empty matrix → S1-S8,S10,S11 return 0; only S9 fires. New signals are the first real local-sensing expansion. ✓
- **Fatih has the data for need-ctx.** `select()` returns `Pick(...)` at `selector.py:426`; `reqs.effective_context_needed` + `best.model.context_length` available there. Dispatcher recursion calls `select()` directly, so a `need_ctx` Pick field reaches both admission + retry. ✓
- **Load mode is fully severed** — `_check_eligibility` has NO VRAM-budget filter (only `vram_available_mb==0 → no_vram_available`, `selector.py:628`). The `ranking.py:309` comment claiming "selector already filtered VRAM budget" is **false/stale**. Nothing enforces budget anywhere. ✓
- **`snapshot()` is sync, once per admission** (`nerd_herd.py:140`, called at `selector.py:90`). RAM (`SystemState` already has `ram_available_mb`, ~1ms) + presence (`GetLastInputInfo`, ~1ms) are cheap per-snapshot. External-GPU (`gpu.py::detect_external_gpu_usage`, pynvml ~10-20ms) is **already throttled to a 30s `_auto_detect_loop`** — read its cached result, do NOT re-detect per snapshot. ✓
- **`load_mode` is NOT on `SystemSnapshot`** today (it lives in `LoadManager._mode`, `load.py:69`). Must add a `load_mode` field, populate in `snapshot()`, so `pressure_for` (a `SystemSnapshot` method) reads `self.load_mode`. ✓

---

## 1. The problem

KutAI is an autonomous agent running missions (often overnight) on a **desktop the founder also uses**. Two goods compete:

- **AI throughput** — missions finish fast.
- **Desktop responsiveness** — don't make the machine lag while the founder works/plays.

They conflict only when the founder is present **and** the work contends for the same resource. The machine is idle ~16h/day. The old framing — *"how much VRAM may KutAI take"* (a static VRAM-% cap, `VRAM_BUDGETS`) — is the wrong question and a **backwards lever**: capping VRAM pushes weights/KV onto CPU+RAM, which is exactly what makes a desktop lag, while idling spare VRAM nobody uses. For a non-gamer, **VRAM is the spare resource.**

Worse, the cap is **severed**: `src/infra/load_manager.py` sync stubs `is_local_inference_allowed()` / `get_vram_budget_fraction()` hard-return `True` / `1.0`; the live selection path (`fatih_hoca` ranking) is load-mode-blind by an explicit comment. The whole "yük modu" UI drives nothing.

## 2. Why there is NO governor (the key decision)

Every resource decision already has an owner:

| Decision | Owner | State |
|---|---|---|
| sense GPU/VRAM/inference/external-GPU | **Nerd Herd** | exists |
| cloud ↔ local placement (placement **is** selection) | **Fatih Hoca** `ranking.py` | exists |
| which model / swap-or-not | **Fatih Hoca** | exists |
| hold-vs-admit (defer-or-run) | **Beckman** `next_task` | exists |
| load mechanics (ctx, `--fit`, layers, unload) | **DaLLaMa** | exists |

A standalone "governor" would govern nothing — it would re-implement selection (Fatih) and admission (Beckman). **Rejected.**

The machine **already does placement and deferral** through one mechanism: a per-model **pressure scalar** (Nerd Herd, 11 signals + 3 modifiers, folded worst-wins) consumed identically by Fatih (`score *= 1 + K·pressure`) and by Beckman's admission gate (`pressure ≥ max(−1, −0.5 − 0.5·urgency)`; below threshold → hold/WAIT).

**The actual gap:** the 11 signals see only cloud quota / rate-limits / token budgets / reliability / queue. The local arm (**S9**) sees only in-flight-busy and idle/warm. **Nothing senses the desktop** — no user-presence, no RAM-pressure signal. The moon shot is therefore **new signals feeding the existing engine, not a new brain.**

### Grounding finding

For a local model `provider=""` → empty `RateLimitMatrix` → **S1,S2,S3,S4,S5,S7,S10,S11 all return 0.0**. Local-pool pressure today **is S9 alone**. The new desktop signals are the first real expansion of local sensing; they slot into `combine.py`'s `OTHER_BUCKET` alongside S9 (weight 1.0, worst-of-negatives).

## 3. Design

### 3.1 Two new signals (Nerd Herd, local-pool only)

Codebase discipline = **one signal, one contract** (S1-stock and S9-timing were deliberately split 2026-05-03; overloading a signal is an anti-pattern here). So the desktop sensing is NOT folded into S9. Two new signals, each guarded `if not getattr(model, "is_local", False): return 0.0` (cloud has zero desktop impact), each negative-only (a "yield the machine" signal):

- **S12 — user-presence** (the human-attention axis)
  Inputs: input-idle seconds (`GetLastInputInfo` via ctypes) + foreground-fullscreen / game detection.
  - User actively present, foreground app is **fullscreen/game** → **hard veto** `−10.0` sentinel (see §3.3).
  - User present, normal desktop use → graded negative, e.g. `−0.3 … −0.6` scaled on recency of last input.
  - User idle/away (idle > threshold, e.g. 300s) → `0.0` (no penalty; full local send).

- **S13 — machine-contention** (the machine-busy axis; can fire while the user is away, e.g. an overnight render)
  Inputs (both already collected — no new sensor): RAM (`SystemState.ram_available_mb`, already gathered) + external-GPU fraction (`ExternalGPUUsage.external_vram_fraction`, already maintained by the 30s `_auto_detect_loop`). S13 reads the **cached** external-GPU value from the snapshot — it does NOT call `detect_external_gpu_usage` itself (pynvml ~10-20ms too costly per admission).
  - external-GPU heavy (another CUDA/render/game process owns the GPU) → **hard veto** `−10.0`.
  - RAM pressure high (loading another model / CPU-offload would thrash) → graded negative scaled on `% used` above a threshold (e.g. 80% → 0, 95% → −1.0).
  - otherwise → `0.0`.

**Net new sensing: only user-presence (S12).** RAM + external-GPU already collected; the work is plumbing them onto `SystemSnapshot` and into S13.

Both sit in `OTHER_BUCKET`. Worst-of-negatives means the strongest yield signal dominates (if the user is gaming, that veto wins regardless of RAM headroom) — correct.

### 3.2 Load mode = new modifier M4 (the wiring that was missing)

Load mode stops being a VRAM-% and becomes a **per-signal weight on S12/S13**, mirroring `M3_difficulty_weights`. New `M4_load_mode_weights(mode) -> dict[str,float]`, applied in `pressure_for` before the fold:

| Mode (Telegram button) | M4 effect on S12/S13 |
|---|---|
| **Full** (ignore user) | weight `0.0` → desktop signals silenced → behaves like today |
| **Otomatik** (governor decides) | weight `1.0` → signals as-is |
| **Heavy / Shared** (cloud-bias strength) | weight `>1.0` (e.g. 1.5 / 2.0) → amplified desktop penalty |
| **Minimal** (cloud-only / pause local) | hard local veto — short-circuit (see §3.3) |

Flat pool-level scalar rejected: too blunt to express "ignore presence but keep the external-GPU hard-veto." Per-signal weight is the right granularity and matches the existing modifier pattern. **This is how load mode finally reaches live selection** — through the pressure engine, replacing the severed `router.py` path.

The current load mode is held in `nerd_herd/load.py` (`LoadManager._mode`) but is **not** on `SystemSnapshot`. Add a `load_mode: str` field to `SystemSnapshot`, populate it in `snapshot()` from `get_load_mode()`, so `pressure_for` (a `SystemSnapshot` method) reads `self.load_mode` and applies M4.

**Bonus — the auto-detect loop becomes useful for free.** `load.py::_auto_detect_loop` (30s) already senses external-GPU and downgrades the mode (`suggest_mode_for_external_usage`). Today that mode change drives nothing (selection is mode-blind). Once M4 wires mode → pressure, this *existing* loop becomes a live external-GPU→cloud-bias path with zero new code. Keep it.

### 3.3 Hard veto mechanism (reuse, don't invent)

S9 already establishes the pattern: a `−10.0` sentinel survives any M3/M4 weight in `[0.5, 2.0]`, so the weighted value stays ≤ `−5.0`, `combine._clamp` floors the scalar at `−1.0`, and Beckman's strict `pressure > −1.0` gate blocks admission **even for max-urgency tasks**. Used for: gaming/fullscreen (S12), external-GPU-heavy (S13), and **Minimal** mode. Effect chains to founder principle: local hard-vetoed → Fatih picks cloud → if cloud unavailable, **no candidate → Beckman holds/WAITs** (does not DLQ).

### 3.4 need-ctx ownership (founder objection #1: dispatcher computes ctx — wrong place)

Fatih already computes `estimates.per_call_tokens / total_tokens / iterations` for S2/S3. **need-ctx becomes a Fatih output, returned with the pick**, and threaded to DaLLaMa as the load **target** (not a floor). The dispatcher stops computing context size. Formula unchanged from P1:
`need_ctx = clamp(ceil_2048(min_context or estimate or MIN_CTX), MIN_CTX, model.context_length)`, `MIN_CTX = 8192` (env `LLAMA_MIN_CTX`).

### 3.5 DaLLaMa load mechanics

`need_ctx` as the load target; `--fit` owns GPU-layer fitting (do **not** pass `-ngl` unless `models.yaml` pins `gpu_layers`); ~0.5–1 GB VRAM safety margin so spikes never OOM; idle-unload (already exists) as the resume-on-demand mechanism. CPU-offload (`-ngl` cap) only as explicit last resort: local is the only option **and** the model doesn't fully fit.

## 4. Data flow

```
nerd_herd.snapshot()
  ├─ existing: vram, local model state, cloud matrices (KDV), in_flight, queue
  └─ NEW sensor fields on SystemSnapshot:
       user_idle_s, foreground_fullscreen, ram_pressure_frac, external_gpu_frac, load_mode

SystemSnapshot.pressure_for(model, ...):
   sig = { S1..S11,  S12=s12_presence(model, snap),  S13=s13_contention(model, snap) }
   weights = M3_difficulty_weights(...) ⊗ M4_load_mode_weights(load_mode)   # M4 scales S12/S13
   ... existing M1/M2 ...
   breakdown = combine_signals(sig, weights)   # S12,S13 ∈ OTHER_BUCKET, worst-of-negatives

Fatih ranking:  score *= 1 + K·scalar           → cloud wins when local pressure drops  (placement, free)
Beckman gate:   admit iff scalar ≥ thr(urgency) → low-urgency holds/WAITs under pressure (deferral, free)
Fatih pick:     also returns need_ctx           → DaLLaMa loads at need_ctx, --fit, margin
```

## 5. What gets deleted (clean slate)

Audit call sites (not docstrings) first, then remove:

- `src/infra/load_manager.py` sync stubs `is_local_inference_allowed()` / `get_vram_budget_fraction()` (no-op `True`/`1.0`). **VERIFIED safe** — only callers are the dead `router.py` lines below.
- **`VRAM_BUDGETS` — do NOT blanket-delete (audit surprise).** It is LIVE: `load.py::_auto_detect_loop` + `exposition.py` (Prometheus) read it. The **budget-fraction-as-VRAM-cap semantics** die (no more capping VRAM); the **mode-transition thresholds** inside `suggest_mode_for_external_usage` and the auto-detect loop **stay** (that's the external-GPU sensor we keep, §3.2). Plan must separate the two uses surgically, not `rm` the dict.
- `src/core/router.py:233-247,317` load-mode enforcement (**VERIFIED dead** — `select_model`/`select_for_task` have zero live callers; root-debt-map confirmed).
- the load-mode-blind comment + dead VRAM-penalty branch in `fatih_hoca/ranking.py:309-310`.
- dispatcher context-size computation (moves to Fatih per §3.4).
- P1 dead symbols already flagged for removal: `calculate_dynamic_context`, `vram_context_ceiling`, `BASELINE_LOCAL_CTX`, `_floored_baseline_ctx` and their re-exports + tests (`tests/test_local_ctx_floor.py`, ctx tests in `fatih_hoca/tests/test_registry.py`).

## 6. Testing

- **S12/S13 unit tests** — cloud model → 0.0; local idle/away → 0.0; local + present-normal → graded negative; local + fullscreen → −10; local + external-GPU-heavy → −10; local + high-RAM → graded negative. Mirror existing `signals/` test style.
- **M4 unit tests** — Full silences (weight 0), Otomatik passthrough (1.0), Heavy/Shared amplify (>1.0), Minimal hard-vetoes.
- **Fold integration** — S12/S13 worst-wins inside OTHER_BUCKET; a −10 sentinel pegs final scalar at −1.0 through M3×M4 weight extremes `[0.5,2.0]`.
- **Admission integration (Beckman)** — high local pressure + low urgency → task held; cloud candidate admitted instead; cloud-dead + local-veto → no candidate → WAIT (not DLQ).
- **Selection integration (Fatih)** — present-user shifts pick from local to cloud; away-user keeps local.
- **need-ctx** — regression of the P1 values (4412→8192 / 0→8192 / 18000→18432 / 40000-cap→ceiling), now sourced from Fatih.
- **Presence sensor** — `GetLastInputInfo` / fullscreen / psutil collectors tested behind a fake (no real hardware in CI).
- Run with timeouts per project rule; re-run `packages/fatih_hoca/tests/sim/run_scenarios.py` + `run_swap_storm_check.py` after weight changes.

## 7. Open items / risks

- **Presence detection cost** — `GetLastInputInfo` is a cheap WinAPI call; fullscreen detect via foreground-window + monitor-rect compare, or nvidia-smi utilization as a coarse proxy. Sample on the existing snapshot cadence, not a hot loop.
- **S12/S13 thresholds** (idle-away cutoff, RAM %, graded-negative slopes) are starting guesses — tune against real `kutai.jsonl` once live, like the S1/S9 thresholds were.
- **M3 ⊗ M4 composition** — both produce per-signal weight dicts; multiply them (M4 only touches S12/S13, M3 doesn't, so no collision). Confirm the multiply, not overwrite, in `pressure_for`.
- **Snapshot freshness vs cost (resolved by audit)** — presence (`GetLastInputInfo` ~1ms) + RAM (~1ms) read fresh per-snapshot. External-GPU (pynvml ~10-20ms) is **NOT** per-snapshot — read the 30s `_auto_detect_loop` cached value. Don't add `detect_external_gpu_usage` to the per-admission `snapshot()` hot path.
- **Minimal-mode mechanism** — there is no `cloud_only` today (only `local_only` rejecting cloud, `selector.py:474`). Minimal = a NEW local hard-veto. Cleanest as an eligibility reason (`load_mode_minimal`) in `_check_eligibility` for local models, OR an M4 sentinel — plan picks. Eligibility-reason is structurally cleaner (local simply ineligible) and gives a clear diag.
- **need-ctx threading** — confirm the pick→DaLLaMa path carries `need_ctx` through the task/context DB round-trip if admission and load are separated in time. `select()` returns it on `Pick`; dispatcher reads `pick.need_ctx` (replaces the `_min_ctx` heuristic at `llm_dispatcher.py:386-388`, kept only as fallback). Line ref refreshed post 2026-06-05 de-accretion: the heuristic is now at `llm_dispatcher.py:362-364` (was 386-388); `pick` is in scope there (`model = pick.model`), so `pick.need_ctx` substitutes cleanly. Re-verified 2026-06-09.
```
