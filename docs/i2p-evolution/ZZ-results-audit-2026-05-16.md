# i2p Evolution — Results Audit (intended vs built)

**Date:** 2026-05-16
**Scope:** 11 zone-improvement sessions (Z0–Z10) run against the
`docs/i2p-evolution/` zone docs scaffolded 2026-05-08.
**Method:** per-zone comparison of the zone doc's intended scope vs git
history, completion tags, handoff docs, and code spot-checks. Four parallel
audit agents, one synthesis.

> Caveat shared by every zone: **no real i2p mission was ever run through the
> new machinery.** Every "COMPLETE" rests on unit tests, and many of those
> tests mock the integration seam. Green ≠ working.

---

## 1. Zone-by-zone scorecard

| Zone | Doc | Tag | Genuine completion | Headline problem |
|---|---|---|---|---|
| Z0 Preflight | z0-mission-preflight | none | **~35–40%** | Founder profile / wizard / vault / readiness gate never built |
| Z1 Pre-code | 01-pre-code | z1-complete-05-12 | 39/40 items | 8 mechanical post-hooks were **dead at tag time** (fixed same day) |
| Z2 Foundation | 02-build-foundation-v2 | z2-complete-05-12 | all tiers | semgrep gates **no-op on Windows** (the dev/prod box) |
| Z3 Review density | 03-build-review-density-v2 | z3-complete-05-12 | all tiers (cleanest) | multi-file expansion **latent — off by default, never run** |
| Z4 Visual review | 04-build-visual-review-v2 | z4-complete-05-16 | all tiers | 1 failing verifier test; needs host Playwright |
| Z5 Mobile | 05-build-mobile-track-v2 | **none** | all tiers | broke 2 Z6 tests; 2 founder decisions pending |
| Z6 Real-world bridge | 06-real-world-bridge-v2 | **none** | all 10 gaps (structurally) | "Stripe sandbox" = real api.stripe.com; B3 review gate publishes unreviewed |
| Z7 Humanish | 07-humanish-layers-v3 | z7-complete-05-16 | 21/21 patterns | B9 audit log never called; A11 handlers unregistered; "test-green but hollow" |
| Z8 Operations | 08-operations-v2 | z8-complete | all tiers but monitoring_kit | **6 on-call verbs are stubs** — agent can't restart/rollback/scale |
| Z9 Growth | 09-growth-v2 | z9-complete-05-15 | all tiers | shipped a **dead signal→backlog pipeline**, fixed post-tag |
| Z10 Cross-cutting | 10-cross-cutting | **none** | all 7 phases | egress = prompt-whitelist not iptables; two unconnected feedback loops |

---

## 2. Did the gaps close?

**Roughly: ~60–70% of the agent-closable gaps genuinely closed.** The rest
fall into four failure modes that recur across zones.

### 2.1 Strong — Tier 1 "make the code zone reliable" (Z1/Z2/Z3)

This is where the work is real. Registry pattern, auto-wired post-hooks,
25 QA-modality kinds, recipe library, cross-mission `mission_lessons`,
ADR drift gate, layer-aware tooling, self-critique guard — all shipped with
code that exists and tests that pass. The original frame named Tier 1 the
prerequisite for everything; it is the most credible delivery.

Caveats even here: multi-file expansion (Z3 T1B/T2C) is **off by default and
has never executed in a mission**; `MULTI_FILE_RULES` covers exactly one
stack combo; `gorsel_ustasi` image-gen package — scoped Z1→Z2 — was built by
**neither** zone; web-preview hosting (C10/F1) likewise orphaned.

### 2.2 The four recurring failure modes

**A. Dead wiring — executor exists, nothing invokes it.**
- Z1: 7 of 8 mechanical post-hooks (compliance gate, prior-art coverage,
  cross-mission dedup) were declared in `i2p_v3.json` but silently filtered
  out of `POST_HOOK_REGISTRY` — no-ops until commit `171ffefd`, landed the
  same day as the `z1-complete` tag.
- Z9: the entire T3 signal→backlog pipeline shipped with `classify_signals`
  and `score_backlog` registered but nothing scheduling them. Tagged
  `z9-complete` *before* the fix; tag re-pointed afterward.
- Z7: `wrap_external_verb` / `log_external_send` (B9 audit log) is called by
  zero external-publish verbs; `audit_completeness_check` post-hook is a
  registered stub. A11 mention handlers never call `register_handler`.

**B. Stub-returns-success — the verb is reachable but does nothing.**
- Z8: `restart_service`, `rollback_to_last_green`, `scale_up/down`,
  `drain_traffic`, `rotate_failed_key` all route to `_stub_handler` →
  `{"status":"ok","stub":True}`. The on-call agent looks fully wired
  (whitelist + cooldowns enforced) but cannot perform a single real
  operation. Tests assert the whitelist, never the effect.

**C. Scaffolding mistaken for integration — needs real accounts to function.**
- Z6: vendor adapters, Stripe recipe, Apple/Google adapters all shipped, but
  the "sandbox" Stripe test hits `api.stripe.com` for real (or fails the
  admission gate) — there is no local mock mode in the executor.
- Z7: all four email providers require founder-provisioned API keys; every
  lifecycle/outreach/changelog/crisis email silently fails until the founder
  completes the Z6 `founder_action` credential flow.
- This is *partly* by design (agent prepares, founder owns legal/financial
  identity) — but the line between "working" and "stub awaiting credentials"
  is not surfaced in the completion claims.

**D. Trust-critical gate silently bypassed.**
- Z6/Z7 B3: `apply.py` reads `source_ctx.get("draft")`, but
  `incident_draft_update.py` returns the draft in its result dict and never
  writes it to context. The `if not draft: skip` branch fires every time —
  **incident status updates publish to customers without founder review.**
  The doc flags this exact flow as "Trust-critical." The 2026-05-16 Z7
  fix-pass named it Critical #3 and did **not** fix it.

### 2.3 Z0 — the foundation that didn't land

Z0 was Tier 0: it must land first and feed every other zone. In practice it
shipped last (`fa100f9a feat(z0,minimal)`, 2026-05-12, *after* Z1 closed) and
minimal. Shipped: `ambition_tier` + `cost_ceiling_usd` + attention-budget
columns, kill/pause/resume commands, per-mission thread, budget alerts.
**Not shipped:** founder profile (cross-mission identity/voice/prefs),
preflight wizard, credential vault + recovery, north-star capture, compliance
fingerprint at intake, idle-policy, the readiness verdict that gates mission
start. The doc's `## Updates` log still shows only the 2026-05-08 scaffold
entry — the zone reads as untouched. "Z0 outputs feed every zone" is
currently fiction; downstream zones lean on three DB columns, not a contract.

---

## 3. New gaps that surfaced

1. **Green tests over mocked seams.** The dominant new risk. Z7's own
   fix-pass handoff calls the result "test-green but partially hollow" and
   told the next session to de-mock tautological tests — done only partially.
   Z1/Z9 dead wiring proves the pattern: a zone can pass its whole suite and
   still be a no-op in a live mission.
2. **Cross-zone collisions.** Z5 split i2p step 14.8 into 5 sub-steps and
   broke 2 Z6 invariant tests (`test_z6_polish_phase14_mobile`,
   `test_z6_t6a_reversibility`). Still failing; blocked on 2 founder
   decisions (D1/D2 in the 2026-05-17 handoff).
3. **Two unconnected feedback loops.** Z9's reinforce loop writes
   `model_pick_log.call_category='reinforce'`; Z10's calibration loop writes
   `confidence_outcomes`. They were each "the learning loop" in their zone
   docs but never unified — Fatih Hoca selection and confidence calibration
   read different tables.
4. **Tag hygiene.** Z0, Z5, Z6, Z10 have no completion tag. Scope of each is
   unbounded for future diffing.
5. **Platform reality.** semgrep (`pattern_lint`, `design_system_check`,
   layer filter) soft-skips on Windows — the gates silently pass on the
   actual dev box. iOS device capture is macOS-CI-only.

---

## 4. Distance to a real "idea → product"

Measured against the original frame's five-tier sequencing:

- **Tier 1 — code zone reliable.** ~75% there. Z1/Z2/Z3 are real. Remaining:
  multi-file expansion is latent, `gorsel_ustasi` unbuilt, and — critically —
  nothing has run end-to-end. The frame said "an agent that fakes completion
  poisons every later signal"; the dead-wiring and stub patterns are exactly
  that failure, shipped at smaller scale.
- **Tier 2 — bridge to real-world ops.** Scaffolding only. Z6 adapters and
  Z8 webhook/lifecycle spine exist, but the bridge cannot be crossed: on-call
  verbs are stubs, monitoring_kit recipes don't exist, every vendor path
  needs founder credentials that no flow has actually exercised.
- **Tier 3 — iteration loop.** Z9 built it, shipped it dead, fixed it
  un-retested. By design the reinforce loop produces 0.0 until months of
  confirmed verdicts accumulate — the "learning" is invisible at MVP.
- **Tier 4 — humanish.** Z7 shipped 16 tables / 11 jobs / 21 patterns, but
  the trust-critical incident-comms gate is broken and the audit log is
  decorative.
- **Tier 5 — money + legal.** Honestly framed. Stripe + compliance templates
  ship as founder-reviewed drafts; no pretense of replacing counsel.

**Bottom line.** The system moved from "concept → compiling code" to
"concept → compiling code, plus a broad scaffold for everything downstream
that has mostly never executed." The irreducible founder-territory items
(legal personhood, credentials, taste, relationships) are correctly left to
the human. But the *agent-closable* gaps the analysis promised are only
~two-thirds genuinely closed — the rest are stubs, latent code, dead wiring,
or trust gates that silently skip. The gulf to a stranger paying for, using,
and returning to the product is still wide, and the most load-bearing
estimate — "is the code zone actually reliable?" — cannot be answered until a
real mission runs.

---

## 5. Recommended next moves

1. **Run one real prototype-tier mission end-to-end.** Highest-value action.
   It will flush every dead-wiring / stub / latent path the unit tests hid.
2. **Wiring-audit sweep.** Grep every registered executor / post-hook / cron
   for an actual caller or scheduler; the Z1 and Z9 misses prove this class
   of bug is systemic, not isolated.
3. **Fix B3 incident-review gate** — it is a live trust-critical defect
   (customer-facing publish without review), already named twice, still open.
4. **Finish Z0 properly** — founder profile, preflight wizard, vault,
   readiness gate. Everything downstream assumes a contract that isn't there.
5. **De-mock the tautological tests** — Z7's handoff already ordered this;
   extend it to Z1/Z8/Z9 integration seams.
6. **Make stubs honest** — Z8 on-call verbs should fail loud
   ("not implemented — needs vendor cloud API") not return `status: ok`.
7. **Tag Z0/Z5/Z6/Z10**, unify the Z9/Z10 feedback loops, resolve the Z5↔Z6
   Phase-14 test collision (D1/D2).

## Updates

- 2026-05-16 — audit created from 4 parallel zone-comparison agents.
