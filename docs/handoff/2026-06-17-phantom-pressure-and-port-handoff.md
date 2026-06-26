# Handoff — 2026-06-17: Live "No model candidates" outage (S10 phantom + stale llama port) + FC gate

## TL;DR

A live total stall (0 selector picks; every analyst/researcher task → `All models failed: No model candidates available`) was traced to **two independent root causes**, both fixed. A third, earlier fix in the same session (FC eligibility gate) is also covered here. **Nothing is pushed; all restart-gated.** The port fix requires a **full wrapper relaunch**, not just `/restart`.

| Fix | Commit | Merged to main | Status |
|-----|--------|----------------|--------|
| FC gate accepts json_schema (un-veto Apriel) + non-chat GGUF guard | `cf669ad6` (`98b00b1a`+`0161a2c5`) | yes | LIVE-verified (`no_function_calling` gone post-restart) |
| kdv S10: 429/quota ≠ reliability failure (kill phantom -1.0) | `e3cc46b3` | yes (`d82d562c`) | restart-gated |
| llama_endpoint: `.env`-authoritative port (fix stale 8081) | `f63df5ab` | yes (`d82d562c`) | restart-gated, **needs wrapper relaunch** |

All on `main`, **NOT pushed**. Current `main` HEAD: `d82d562c`.

---

## Root cause 1 — CLOUD phantom -1.0 pressure (S10, not S1)

**Symptom:** `selector: all candidates below pressure threshold task=analyst ... scalars=[gemini/*=-1.00, groq/*=-1.00, cerebras/*=-1.00]` — every free cloud model pinned at exactly -1.00. With local off (minimal) → `select=None` → 0 picks across the fleet (last 4000 log lines at 14:44 UTC: 0 picks, 98 pressure-empties).

**Proven phantom, not genuine exhaustion:** `groq/meta-llama/llama-4-scout-17b-16e-instruct` had `rpd_remaining=1000/1000`, 0 rate-limit hits, yet logged -1.00. A full-quota model cannot legitimately be -1.0. All providers `status=active`.

**Why the prior fix missed it:** `01dba42f` ("unknown rate-limit remaining → full") patched **S1** (`_rl` in `kuleden_donen_var/nerd_herd_adapter.py`). Empirical repro showed S1 = **0.0** with the real state — the -1.0 comes from **S10 (reliability)**.

**Mechanism:** `KuledenDonenVar.record_failure` (`packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py`) fed **every** non-auth failure — including `rate_limit`/`rate_limited`/`daily_exhausted` — into the rolling reliability outcome window (`_record_outcome`). 429/quota is **capacity** (recoverable; already gated by `record_429` rpm-cooldown + `mark_daily_exhausted` daily-veto earlier in the same method), not model quality. The backlog bursting onto free cloud → genuine 429s → `recent_success_rate` craters → `s10_failure` → -1.0; and `provider_prior_rate` (aggregates `_outcomes` across **all** provider siblings) drags healthy full-quota models down too. `combine_signals` passes the worst-of-negatives (OTHER_BUCKET, weight 1.0) straight to `.scalar`.

**Fix (`e3cc46b3`):** exclude `("auth_failure","rate_limit","rate_limited","daily_exhausted")` from `_record_outcome`. Only genuine quality failures (`server_error`/`timeout`/malformed) drive S10. A perpetually-429 model is still kept out of selection by the daily/rpm vetoes (S10 was never the keep-out mechanism). Review-verified: `classify_error` funnels all capacity errors into `rate_limited`/`daily_exhausted`, so the exclusion tuple is complete.

---

## Root cause 2 — LOCAL stale llama-server port (Expo:8081 collision)

**Symptom (after switching load mode to `full`):** `dallama.server: Port 8081 already in use — aborting start` → `Failed to load model: Qwen3.5-9B-...-thinking` → circuit breaker → all local refused → empty local pool.

**Findings:**
- `.env` has `LLAMA_SERVER_PORT=8090` (fixed 2026-06-16). But DaLLaMa launched on **8081**.
- Port 8081 is held by PID 40772 = the **Expo / React-Native dev server** (`node ... expo ... run:android` for the Bilinç mobile project) — unrelated, legitimately occupying 8081.
- Fresh `resolve_llama_port()` returns 8090 (code is correct). The **running process** got 8081.

**Cause:** the Yaşar Usta wrapper is long-lived and was launched **before** the `.env→8090` edit, carrying a stale `LLAMA_SERVER_PORT=8081` in its process environment. The orchestrator is its child → inherits 8081. `resolve_llama_port` was **env-wins** (read `.env` only when the env var was unset, via `_ensure_dotenv_loaded`), and `load_dotenv` does **not** override an existing env var — so `.env=8090` never took effect. A `/restart` only re-spawns the orchestrator (inherits the wrapper's env); the wrapper itself never relaunched.

**Fix (`f63df5ab`):** new `_dotenv_port()` reads the `.env` file directly (`dotenv_values(find_dotenv(usecwd=True))`, no `os.environ` mutation). `resolve_llama_port` now prefers `.env` on a mismatch with the inherited env and warns loudly (pointing at the stale-wrapper cause). Falls back to the process env when `.env` lacks the key; still raises when neither source provides a valid integer port (the 2026-06-14 wrong-port-orphan guard is preserved). Removed the now-unused `_ensure_dotenv_loaded` (no other importers).

---

## Note: the two reported tasks were NOT local_only

`#459160` (competitive_positioning) and `#459220` (design_tokens) have `resolve_local_only=False` (verified against the live DB). An earlier hypothesis (sensitivity → `local_only` → cloud forbidden) was **wrong for these tasks** — the `54×local_only` warnings belonged to a different set of analyst tasks. Their failure was purely the S10 phantom (cloud) + minimal/port (local).

---

## Verification done

- TDD RED→GREEN for both fixes.
- `181` kuleden_donen_var tests + `9` llama_endpoint tests pass.
- `run_scenarios.py` all PASS; `run_swap_storm_check.py` ≤0.5% (no churn).
- Opus adversarial review: **SAFE TO MERGE** (both). FC-fix review also SAFE.
- Merged-main import smoke: `resolve_llama_port()` → `8090`; kdv imports clean.

---

## OUTSTANDING — user actions

1. **Full wrapper relaunch** — stop and relaunch `kutai_wrapper.py` (Yaşar Usta) from a clean shell with no `LLAMA_SERVER_PORT` exported. A plain `/restart` will NOT clear the stale 8081 (orchestrator inherits the wrapper env). This is required for the port fix to take effect and to load both code fixes.
2. **`git push`** — carries `d82d562c` (and the FC-fix `cf669ad6` + other unpushed main commits).
3. **Verify after relaunch:**
   - `netstat` shows llama-server on **8090**; Expo still on 8081.
   - Local loads succeed (Apriel/Qwen); `selector pick:` lines appear for analyst/researcher.
   - Cloud no longer uniformly -1.0; `model_pick_log` shows cloud picks again.
   - Apriel picked for analyst/researcher (FC fix).

---

## Follow-up (review finding B1 — medium, non-blocking)

Two **other** env-wins port readers still exist and can resurrect the stale-port split-brain while a stale wrapper runs:
- `kutai_wrapper.py:110` `_reconcile_stray_llama` — reads `os.environ.get("LLAMA_SERVER_PORT")` (env-wins) and passes it as `keep_port` to `kill_stray_servers`. With a stale wrapper holding 8081 while llama correctly runs on 8090, it could **kill the healthy 8090 server as a "stray."**
- `packages/nerd_herd/src/nerd_herd/__main__.py` `_resolve_llama_url` — env-wins; the sidecar inherits the stale env and **probes the wrong port** (false "leak / no model" alarm).

Both only bite while a stale long-lived wrapper is running (the exact condition a clean relaunch clears). Recommended hardening: make both call/mirror `resolve_llama_port`'s `.env`-authoritative logic, or have the wrapper overwrite its own `os.environ["LLAMA_SERVER_PORT"]` from `.env` at boot so children inherit the corrected value. This fully eliminates the stale-env class rather than fixing one of three sites.

---

## Key files

- `packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py` — `record_failure` S10 exclusion (~line 594).
- `packages/kuleden_donen_var/tests/test_kdv.py` — +5 tests; `test_recent_success_rate_includes_quota_failures` → `_excludes_`.
- `src/infra/llama_endpoint.py` — `_dotenv_port`, `_warn_port_mismatch`, rewritten `resolve_llama_port`.
- `tests/infra/test_llama_endpoint.py` — rewritten for `.env`-authoritative contract.
- (FC fix) `packages/fatih_hoca/src/fatih_hoca/selector.py` (~553), `capabilities.py` (~449), `registry.py` (`is_non_chat_model_name`).

## Gotcha for next session

A long-lived process + inherited stale env means a **"restart-gated" `.env` fix is inert until the TOP process relaunches** — not the child. When an env-derived config change "doesn't take" after a restart, check whether the value is inherited from a parent process rather than re-read from `.env`.
