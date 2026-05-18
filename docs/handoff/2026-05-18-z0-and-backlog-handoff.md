# Handoff — Z0 mission-preflight + backlog (§2 + §3 remaining)

**Date:** 2026-05-18
**Supersedes:** §2 + §3 of `docs/handoff/2026-05-17-wiring-audit-z0-handoff.md`.
**Context:** The 2026-05-17 handoff's §1 wiring sweep is **done** — Z7 §1.A all
8 unwired features wired, §1.C/§1.D closed (17 commits on `main`, each with
host-path tests; see memory `project_wiring_sweep_20260518`). What remains is
§2 (finish Z0) and §3 (the left-behind backlog). This handoff is the accurate,
re-verified state of both.

---

## 1. Z0 — mission preflight (§2)

Z0 was Tier 0 and is **still partial**. A parallel session worked it in the
`worktree-z0-mission-preflight` branch — **18 commits, not yet merged to
`main`**. STEP 0 for any Z0 work: merge or rebase that branch first, then
continue from there (do not re-implement what it already did).

### Already shipped — on `worktree-z0-mission-preflight` (not main)

`1efb40d5`→`1733f723`, 18 commits, ~3300 LOC, 34 files. Covers:

- **Lifecycle:** `/pause_mission` `/resume_mission` `/kill_mission` with
  snapshot; `missions.lifecycle_state` + admission gate on it; lifecycle
  event emitter + DLQ-cascade auto-pause.
- **Budget:** `missions` ceiling columns; `spent_usd` tracking; 50/75/90%
  threshold notifies to the mission Telegram thread; dispatcher passes
  `remaining_budget_usd`, pauses the mission on budget failure; Fatih Hoca
  `remaining_budget_usd` filter + `SelectionFailure`.
- **Safety:** new `packages/safety_guard/` — collision detectors + hardcoded
  blocklist + executor `pre_action` hook (Allow/Wait/Block), wired into
  mr_roboto mechanical dispatch; dangerous i2p_v3 steps tagged
  `reversibility=none` + `locked`.
- **Thread:** `TelegramInterface.provision_mission_thread` + pinned status
  formatter; ceiling question at mission start + lifecycle button callbacks.

That is the lifecycle / budget / safety half of Z0.

### NOT shipped — the work

The **founder-profile / preflight-intake half** is untouched. Verified absent
2026-05-18: no `founder_profiles` table, no preflight wizard, no vault.

| Phase | Missing | Notes |
|---|---|---|
| A — Founder profile | `founder_profiles` table (identity, voice samples, prior missions, time-zone, notification prefs) + cross-mission inheritance | the cross-mission memory the system claims as moat M2 |
| E — Readiness verdict | mechanical check that **gates mission start** (OK / warnings / blockers); blockers stop start, warnings need ack | mission start is currently ungated by readiness |
| B — Preflight wizard | step-by-step Telegram intake (tier → cost → north-star → compliance → readiness → thread → kill-switch); skippable fast-path | `/preflight` today is a CLI, not a wizard |
| F/G/J | north-star capture at intake; compliance fingerprint pre-intake; idle-policy (`wait`/`proceed_with_default`/`pause` per action) | downstream zones (Z9 north-star, Z6 compliance) assume these exist |
| C — Vault | per-founder encrypted credential store + passphrase + recovery | distinct from the existing general `credential_store.py`; largest, can trail |

**Recommended order:** A (profile schema — unblocks everything) → E (readiness
gate — the honest "is this mission ready" check) → B (wizard wraps A+E) →
F/G/J (small, fold into the wizard) → C (vault — largest, trails). Each phase
per `superpowers:writing-plans` + TDD.

**Do NOT tag `z0-complete` until A/E/B/F/G/J land.** (C may trail behind the
tag if explicitly noted.)

---

## 2. Backlog (§3)

### 2a. Pre-existing test failures

Re-verified 2026-05-18. Status since the 2026-05-17 handoff:

- **CLOSED** — `test_grounding_posthook.py` uncollectable syntax error
  (`class TestMr. Roboto`) fixed in `a31a5f33`; `test_reversibility_registry`
  (`capture_hint`/`source_scout`/`yalayut_discovery` missing) fixed in
  `b79fd222`, `yalayut_demand` added in `17979594`.
- **STILL OPEN:**
  - `test_code_review_posthook::test_apply_code_review_pass_with_other_pending_keeps_ungraded`
    — patches `update_task` on the wrong import path.
  - `test_z4_t3_visual_review_posthook::test_visual_review_in_simple_blocker_verdict_kinds`
    — brittle: scans apply.py source text in a 6-line window; real code spans
    18. Routing is correct; the verifier is wrong.
  - `test_admission_cache` (2) — admission cache, unrelated.
  - `test_beckman_posthooks::test_apply_request_posthook_grade...` — test
    passes a JSON-string `source_ctx` where production passes a dict.
  - `test_migration_ungraded_to_posthooks` (2) — `posthook_migration` not
    exported from `general_beckman`.
- **Environment, not code** — `fastapi` + `sentence_transformers` are not
  installed in the dev env, so all webhook-route and embedding tests fail
  everywhere. Either install them or accept the skips; **not regressions**.

### 2b. Unbuilt scoped work

- **`gorsel_ustasi`** — image-gen provider package. Scoped Z1→Z2, built by
  neither. Verified absent 2026-05-18 (`packages/gorsel_ustasi` does not
  exist). Z1 emits placeholder images only. Needs: a provider-abstraction
  package (per `project_z1_strategic_locks_20260509` — image provider
  abstraction was a strategic lock).
- **Web preview hosting** (C10 / F1) — the `emit_preview_url` verb exists; the
  cloudflared / local-port / GitHub-Pages host + viewer was deferred to Z2 and
  never built.
- **Z8 on-call verbs real impl** — `restart_service` / `scale_up` etc. fail
  loud (honest) but are stubs; need vendor cloud-API wiring when accounts
  exist. (`/force_action` to trigger them manually shipped `4a2343d1`.)

### 2c. Two unconnected feedback loops

Both write paths confirmed to exist (2026-05-18):
- Z9 reinforce loop — `record_verdict.py` writes `model_pick_log` rows with
  `call_category='reinforce'`; `fatih_hoca/grading.py::reinforce_bonus` reads.
- Z10 calibration loop — `confidence_outcomes` → `confidence_reliability_scores`
  → prompt builder (`general_beckman/cron.py`).

They were each "the learning loop" in their own zone doc, were never unified,
and read different tables. **Decision needed:** merge them into one loop, or
document them as deliberately separate. No code is broken — this is a
coherence call.

### 2d. Completion tags

Verified 2026-05-18 — tags present: `z1`,`z2`,`z3`,`z4`,`z7`,`z8`,`z9`.
**Missing: `z0`, `z5`, `z6`, `z10`.** If tagging retroactively:
Z5 ≈ `13d7af25`, Z6 ≈ `2af62429`, Z10 ≈ `9011325`. Z0 must NOT be tagged
until §1 of this doc lands.

### 2e. History blemish (cosmetic — no fix urgency)

The 2026-05-17 Z7 audit-log test fix was `--amend`-ed into a parallel
session's `dc1b2b52 docs(z5)` commit instead of the Z7 fix commit `e90e71c6`.
`main` is functionally correct (test passes); `e90e71c6` in isolation has the
stale test. Local history only, nothing pushed. Clean up only during an
unrelated history rewrite.

---

## 3. Operational note — parallel-agent fleet

This repo is worked by a large parallel-agent fleet (~40 `worktree-agent-*`
worktrees + feature branches) that **shares the main working directory and git
HEAD**. A parallel session can switch your branch under you. Before any ad-hoc
work: `git branch -v` + `git status` to see what other sessions have in
flight. For Z0 specifically — `worktree-z0-mission-preflight` is the live
in-progress branch; coordinate, do not duplicate.

---

## 4. Suggested sequencing

1. **Merge `worktree-z0-mission-preflight` to `main`** (after confirming it is
   complete + green).
2. **Z0 §1** — phases A → E → B → F/G/J → C, in order, TDD per phase.
3. **§2d tags** — tag Z5/Z6/Z10 retroactively now (independent, 2 min); tag Z0
   only after step 2.
4. **§2b unbuilt** — `gorsel_ustasi` then web-preview hosting (each a small
   scoped package/feature).
5. **§2c** — the founder decides: merge the two feedback loops or document the
   split.
6. **A real prototype-tier mission, end to end** (§4.4 of the prior handoff) —
   only meaningful after Z0 lands; the true reliability proof.
