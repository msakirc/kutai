# Handoff — gemini "All models failed: Daily limit exhausted" re-pick / re-admit

**Date:** 2026-06-16
**Status:** core fix SHIPPED (committed `bb86b00e`, NOT pushed, restart-gated) + LIVE-VERIFIED working in a ~3h window. One structural hole remains (owner declined to build it now — see §5).
**Memory:** `project_daily_exhausted_selector_blind_20260615.md`, `feedback_test_serialization_boundary.md`.

---

## 1. Symptom

Telegram, recurring across days:
```
❌ [1.x] <step>  All models failed for 'analyst': Daily limit exhausted for gemini/gemini-flash-latest
❌ ...            All models failed for 'analyst': litellm.RateLimitError: GeminiException - 429
❌ ...            All models failed for 'analyst': No model candidates available
```
User's framing (correct, and the invariant to hold): **a task must not be admitted onto a model that 429'd minutes ago.**

## 2. Root cause (SHIPPED fix)

Learn-side ≠ decide-side stores.
- **Learn (correct):** a real 429 → `kdv.on_response` → `mark_daily_exhausted` (`packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py:548/573`) → `model_limits[id]` rpd/tpd=0, reset_at=now+floor(1h). `pre_call` correctly refuses thereafter.
- **Decide (was blind):** `Selector.select` reads `self._nerd_herd.snapshot()` (`packages/fatih_hoca/src/fatih_hoca/selector.py:91`). That nerd_herd is the **sidecar `NerdHerdClient`** (`src/app/run.py:654`). Its cached `snapshot.cloud` does NOT carry KDV's per-model `daily_exhausted`/`rpm_cooldown`: `client._overlay_local` mirrors only swap/queue/image (`packages/nerd_herd/src/nerd_herd/client.py:304-352`); `configure_in_flight_push` (`src/infra/rate_limiter.py:82`) writes the in-process **singleton** `_cloud_state`, not the client cache. `NerdHerdClient` has no `push_cloud_state`.
- Net: a daily-dead model still passed eligibility (`selector.py:616` `mstate=None` → gate skipped) and ranked un-penalized (`ranking.py:420` avail_score never zeroed) → admitted → `pre_call` refused post-admission → `ModelCallFailed` → ~15-min Beckman re-pend → repeat. The selector saw the SAME stale snapshot every cycle, so it re-picked the dead model indefinitely.

**Fix (`bb86b00e`):** in `selector.py`, immediately after `snapshot = self._nerd_herd.snapshot()`, rebuild `snapshot.cloud` from the live in-process KDV — `build_cloud_provider_state(get_kdv(), prov)` for each `kdv._providers` key. Mirrors the existing in_flight in-process-truth overlay right above it (lines ~105-131). Fail-open on ImportError; only overwrites when the rebuild yields data. Covers every selector entry path (Beckman admission + husam retry; all go through `fatih_hoca.select → _selector.select`).

Files:
- `packages/fatih_hoca/src/fatih_hoca/selector.py` (overlay block)
- `tests/test_selector_cloud_overlay_boundary.py` (NEW, root test — src-importable; injects exhaustion via the KDV seam NOT `nh.snapshot()`, the exact boundary that was broken; main test proven RED with fix disabled, GREEN with).

## 3. Evidence it works (LIVE)

Verified against prod DB (`C:\Users\sakir\ai\kutai\kutai.db`, UTC). Restart booted ~21:40 UTC 06-15 (pick gap 21:38:23→21:42:15). PC died ~00:59 UTC 06-16 (power loss), then OFF ~12h — **do not read the post-00:59 silence as success; that was downtime.** Valid uptime window = ~21:40→00:59 UTC (~3h).

- `admission_violations` site=`daily_exhausted_at_call` for `gemini/gemini-2.5-flash`: **OLD afternoon (12:00–21:40) = 42** → **FIXED (21:40–01:00) = 0**. This count is the exact fingerprint: old code re-selected the dead model so `pre_call` refused it 42×; fixed code excludes it at selection so `pre_call` never sees it.
- `model_pick_log`: 2.5-flash FAIL `daily_exhausted` at 21:42:43 + 21:43:36 (1 min apart = concurrent first-touch burst), then **picked 0 times** for the rest of the run. Traffic moved to gemma-4-31b (×113 ok), local Qwen, openrouter.
- Re-admit scan: **0** models re-picked >3min after a 429 in the whole post-fix window.
- gemini 429 classification post-fix: 3 `daily_exhausted`, 0 `rate_limited` (no misclassification observed).

## 4. Why the user STILL saw errors after the fix (NOT the re-pick loop)

The post-restart errors at 21:42–21:43 were a **different, un-fixed mode**, not re-admit:
- **First-touch live 429s** (`litellm.RateLimitError`, the live-call shape — NOT the `pre_call` "Daily limit exhausted" string). Gemini sends **no rpd headers**, so KutAI can only learn a model is dead by eating one real 429. 2.5-flash literally succeeded at 21:42:26–30, then exhausted mid-burst at 21:42:43. The selector overlay can only stop *re-picking after* the mark; it cannot prevent the first 429.
- **Concurrent fan-out race:** multiple analyst tasks admitted onto 2.5-flash in the same ~15s while it was healthy; 2 landed after quota ran out before the mark propagated.
- **`No model candidates available`:** genuine simultaneous drain of the gemini free pool.

These are unavoidable-by-the-overlay. They are NOT the "429'd 15 min ago, re-admitted" bug — that one is fixed.

## 5. Surviving structural hole (owner declined to build NOW — next session candidate)

The only path back to "re-admitted a model that 429'd minutes ago" post-fix is **misclassification**: the overlay can only suppress what KDV marked. If a gemini daily-429 is ever tagged `rate_limited` instead of `daily_exhausted` (gemini sends no `x-ratelimit-reset` → `is_rpm_cooldown` stays False → **no durable cooldown installed**), the model stays eligible and gets re-admitted on the next pend → re-429. Not observed post-fix (3/3 classified daily), but structurally possible; classifier patterns live in `packages/hallederiz_kadir/src/hallederiz_kadir/retry.py:108-131` (`perday`/`perdayperproject`/`requests per day`/`(rpd)`/`tokens per day`).

**Recommended fix (classification-independent, makes the invariant真 hold) — NOT YET BUILT:**
Defense-in-depth bench in KDV: **any** 429 on a model installs a respected admission cooldown (even without a reset header), and a 2nd 429 within a window escalates the bench (exponential), independent of daily-vs-rpm parsing. Surface it through the same `snapshot.cloud` overlay (already wired). Then a model that 429'd can never be re-admitted until proven recovered, even if the body parse fails. Likely sites: `rate_limiter.record_429` (install a floor), `is_rpm_cooldown`/a new `recently_failed` flag, `nerd_herd_adapter.build_cloud_provider_state` (surface it), `selector._check_eligibility` (already reads the flags).

**Owner ruling: do NOT ship band-aids.** Explicitly REJECTED this session as not addressing the invariant: (a) alias-group daily suppression, (b) concurrent-fanout daily guard, (c) 00:00-UTC cooldown pin, (d) suppress-Telegram-on-retryable-availability. The 00:00-UTC pin (`kdv.py:565`, `_MIN_DAILY_COOLDOWN=3600`) remains a minor latent improvement but is not the fix.

## 6. Outstanding for USER

1. **`git push`** `bb86b00e` to origin (currently main-only; live via restart since editable installs point at the working tree).
2. Fix is already live (was running 21:40–00:59 UTC before power loss). After next boot it stays live. Re-verify next daily boundary (~00:00 UTC) that `daily_exhausted_at_call` stays ~0 and no model is re-picked >3min after a 429.
3. Decide whether to build §5 (defense-in-depth bench) — the only thing that closes the re-admit invariant under classification failure.

## 7. Useful queries (prod DB, UTC)

```sql
-- re-pick of a dead model (should be ~0 post-fix):
SELECT timestamp,picked_model,error_category FROM model_pick_log
WHERE success=0 AND error_category IN ('daily_exhausted','rate_limited')
ORDER BY timestamp DESC LIMIT 30;
-- selector admitting a known-dead model (the fingerprint; should be 0):
SELECT model,COUNT(*) FROM admission_violations
WHERE site='daily_exhausted_at_call' AND timestamp>'<boot_ts>' GROUP BY model;
```
Note: logs (`logs/*.jsonl`) are LOCAL time (UTC+3); DB is UTC.
