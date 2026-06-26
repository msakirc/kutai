# Handoff — mission debug continuation (live errors)

**Date:** 2026-05-31
**Predecessor:** `docs/handoff/2026-05-30-mission79-dlq-fixes-handoff.md` (read it — its open items are still open).
**HEAD:** `a72cc1ae` (the 2026-05-30 handoff commit). KutAI **was restarted this session** by the founder, so the running build = current `main` (the 13 mission_79 fixes ARE live).

---

## Tonight's errors (resolved → root-caused)

Telegram showed, repeatedly, after a manual "retry all" (5/5):
```
🟠 [ERROR] models.local_model_manager: Failed to load Qwen3.5-9B-UD-Q4_K_XL-thinking (demoted for 5 min)
❌ 1.0 prior_art_search  All models failed for 'researcher': Failed to load local model Qwen3.5-9B-UD-Q4_K_XL-thinking
❌ 1.0 prior_art_search  All models failed for 'researcher': No model candidates available
```

### Root cause (definitive, evidence in dallama.jsonl @ 2026-05-30T20:36–20:39 UTC = 23:36 local)

1. researcher picked local 9B (thinking). llama-server OOM'd on load:
   `ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4854.94 MiB … cudaMalloc failed: out of memory`
2. The OOM `-ngl 27` fallback retried → still OOM (4309 MiB). 2 swap failures →
   `Circuit breaker tripped: … failed 2 times — refusing loads for 300s`
3. For 5 min the 9B was refused; cloud was exhausted (gemini daily / cerebras+groq rate-limited per prior session) → **zero candidates** → the storm.

**It self-resolved.** GPU was **5761 MiB free** when checked (2257/8192 used) — a 9B needs ~5 GB, loads fine. At 23:36 a transient desktop GPU app (Epic/GOG/Spotify/Firefox all live) held ~4–5 GB; the spike passed; the 300s breaker expired ~20:42 UTC.

### The REAL underlying bug (why a 9B OOM'd in the first place)

`BASELINE_LOCAL_CTX=16384` floor (`src/models/local_model_manager.py`, intake #73, added 2026-05-22) is **VRAM-blind**: the dynamic calc correctly sized ctx down to ~4096 under the spike, the floor bumped it back to 16384, and at 16384 the 9B's weights+KV (4309 MiB even at `-ngl 27`) didn't fit in ~4.2 GB free → OOM.

```
ctx=16384 (floored):  27 layers 3267 + KV 432 + compute ~600 = 4309 MiB → >4.2 free → OOM
ctx=4096  (need):     27 layers 3267 + KV 108 + compute ~600 = ~3975 MiB → fits
```

The floor is literally the difference between OOM and load. Full architectural analysis + the fix direction: **`docs/handoff/2026-05-31-load-mode-redesign-ideas.md`** (the proper fix is need-driven ctx; the load-mode VRAM-cap is a separate, wrong-mechanism concern).

---

## ⚠️ Uncommitted stopgap in the working tree — DECISION PENDING

A VRAM-aware version of the baseline floor was written + tested this session but **NOT committed** and is **NOT in the running build** (the live build still has the VRAM-blind floor → the OOM CAN recur until a real fix ships).

Files (all green: 372 fatih_hoca + 7 new tests passed):
- `packages/fatih_hoca/src/fatih_hoca/registry.py` (M) — added `vram_context_ceiling()`
- `src/models/model_registry.py` (M) — export shim
- `src/models/local_model_manager.py` (M) — `_floored_baseline_ctx()` + wired into `swap()`
- `packages/fatih_hoca/tests/test_registry.py` (M) — 3 tests
- `tests/test_local_ctx_floor.py` (??) — 4 tests

Founder's call was **"revert as part of redesign"** — i.e. the proper need-ctx redesign supersedes it. Two clean options:

**Commit as interim OOM safety** (recommended IF a restart is likely before the redesign ships — it's strictly better than the live VRAM-blind floor):
```
git add packages/fatih_hoca/src/fatih_hoca/registry.py src/models/model_registry.py \
        src/models/local_model_manager.py packages/fatih_hoca/tests/test_registry.py \
        tests/test_local_ctx_floor.py
git commit -m "fix(dallama): VRAM-aware baseline ctx floor (interim) — stop 16384 floor OOMing local loads under transient VRAM pressure"
```

**Revert** (if going straight to the need-ctx redesign, no interim restart):
```
git checkout -- packages/fatih_hoca/src/fatih_hoca/registry.py src/models/model_registry.py \
                src/models/local_model_manager.py packages/fatih_hoca/tests/test_registry.py
rm tests/test_local_ctx_floor.py
```

Either way, the **real** fix = need-ctx (see redesign-ideas handoff). The stopgap only makes the existing (wrong-axis) floor not-OOM; it does not fix the inversion.

---

## Still-open items from the 2026-05-30 handoff (NOT closed this session)

1. **HIGHEST — grade-reject branch ignores availability.** `apply.py::_apply_posthook_verdict_locked` grade-reject (~4348-4470) hardcodes `category="quality"` → an availability-caused grade-child failure still quality-fast-DLQs. Delicate (runs under `_source_verdict_guard`). Needs TDD.
2. **`accelerate_retries`/`capacity_restored` early-wake — verify e2e.** Wiring exists (kdv→router→db.accelerate_retries) but unverified that deferred rows actually wake when capacity returns.
3. Genuine model-quality failures (writer under-spec, compliance_overlay `required_documents` as string) — NOT bugs; ride retries.
4. interview_script regen dormant on a VALID artifact (design Q for founder).
5. Pre-existing test fails (baseline, not introduced): `tests/test_grading.py` (add_skill not called) + `test_reversibility_registry.py` (`publish_preview_pages` missing from VERB_REVERSIBILITY).

---

## Environment / gotchas (carried from prior handoff, still true)

- Live DB: `C:\Users\sakir\ai\kutai\kutai.db` (NOT `./data`). `tasks` keyed by int `id`. Probe: `scripts/_probe_task.py <id>`.
- Logs: repo `logs/*.jsonl`. dallama.jsonl = llama-server lifecycle (the OOM stderr lives here). kutai.jsonl = `models.local_model_manager` ctx-sizing logs.
- llama-server on **port 8081** (mitmproxy squats 8080). Binary `C:\Users\sakir\ai\llama.cpp\llama-server.exe`, models `C:\Users\sakir\ai\models\`.
- venv python: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe`. Run pytest with `-p no:warnings`, always with a timeout.
- Never taskkill llama-server. Restart KutAI via Telegram, not Claude Code.
- GPU: single 8 GB, shared with desktop. Transient external VRAM spikes are real and WILL recur — that's the whole motivation for the load-mode redesign handoff.

---

## Suggested next steps

1. Decide the stopgap (commit-interim vs revert) per the founder's "revert in redesign" intent + restart likelihood.
2. Run a FRESH graded mission (not the poisoned #79). Watch: 9B loads without OOM, availability tasks back off (not DLQ), no researcher storm.
3. If the OOM recurs before the redesign: set `LLAMA_BASELINE_CTX=8192` (or `4096`) in `.env` + restart as an immediate mitigation.
4. Pick up the load-mode redesign via `2026-05-31-load-mode-redesign-ideas.md` (answer the staging question, write a fresh spec).
5. Then open item #1 (grade-reject availability routing) — the last availability gap.
