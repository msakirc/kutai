# Handoff — Wiring-audit sweep + Z0 + consolidated left-behind work

**Date:** 2026-05-17
**Context:** The i2p results audit (`docs/i2p-evolution/ZZ-results-audit-2026-05-16.md`)
found **dead wiring** — an executor/post-hook/cron/verb that exists, is registered, and
passes its unit tests, but has *no production trigger* — to be the dominant failure mode
across the 11 zone sessions. Two independent audits since (Z7 5-agent sweep →
`2026-05-17-z7-unwired-features.md`; yalayut Phase-4 review → `2026-05-16-yalayut-demand-signals-gap.md`)
confirm the pattern is systemic, not isolated. This handoff consolidates: the remaining
wiring debt, the unfinished Z0 zone, and every other item left behind.

Nothing here is a regression. It is all "shipped but not reachable" or "never scoped."

---

## 1. The wiring-audit sweep

### Why a sweep

Unit tests mock the integration seam, so a feature can be 100% green and still be a
no-op in a live mission. Three audits so far each found dead wiring by the same method;
a full sweep across all zones will find more. Until it runs, no "COMPLETE" claim can be
trusted and a real end-to-end mission run will trip over these one by one.

### Method (apply per zone)

For every registered **post-hook kind**, **mr_roboto verb/executor**, **cron job**, and
**agent**:

1. Grep for a real caller / scheduler / `i2p_v3.json` step that invokes it.
2. If `auto_wire_triggers=[]` on a PostHookSpec → confirm an explicit workflow step
   declares it; otherwise it never fires.
3. If a cron executor → confirm it is in `cron_seed.INTERNAL_CADENCES`.
4. If a founder_action emits an `expected_output_schema` flag → confirm something
   *consumes* that flag.
5. If a Telegram command is the entry point → confirm it is not a stub that only
   replies text.
6. For each, write a test that exercises the **host path**, not the verb in isolation.

### Known orphans (start the sweep with these — already found)

#### A. Z7 humanish-layers — 8 unwired features (from `2026-05-17-z7-unwired-features.md`)

| # | Feature | Broken link | Wiring needed |
|---|---|---|---|
| 1 | C9 changelog/announcement blast | only caller `changelog/publish` verb is never enqueued; step 11.4 produces artifact, never publishes | `/changelog publish` cmd OR i2p mechanical step OR machine-actionable `changelog_freshness` founder_action |
| 2 | FAQ flywheel write | `_apply_faq_approval` has zero non-test callers; `_faq_approval_pending` flag consumed by nothing → `support_docs_*` collections stay empty forever | founder-action approval handler routing approved FAQ into `_apply_faq_approval` |
| 3 | A6 deliverability check | `outreach_deliverability_check` has `auto_wire_triggers=[]`, no step, no cron — pause never written | cron seed OR non-empty triggers OR explicit step |
| 4 | A7 cold-outreach send | `/outreach upload` is a pure stub — no list storage, no approval flow, no task | make `/outreach upload` persist list + create approval card + dispatch `outreach/send` |
| 5 | A7 reply handling | no ESP inbound-reply webhook → `outreach/handle_reply` never called; `template_id='follow_up'` is an inert free-form string | ESP reply webhook route; template registry or prompt branch |
| 6 | A11 mention monitor | no `/mention_monitor` cmd, not in `cron_seed.INTERNAL_CADENCES`; digest step's `skip_when` is an unparseable bare token; no `mention_digest` template | command + cron seed; fix `skip_when` shape; add template |
| 7 | A9 investor bullets | scheduled, but `missions.product_id` is never populated and `metric_emit`/`review_density_metric` growth_events are never written → all queries return empty | **decisions:** who sets `product_id`; who emits metric events |
| 8 | briefing_compose recovered-lessons | dominant `mission_lessons` writers (`apply.py:490`, `mission_lessons.py:253`, `record_verdict.py:162`) omit `mission_id` in `source_ref`; posthook is `auto_wire_triggers=[]` with no step | make writers include `mission_id`; declare posthook on a real step |

Note: A11 mention *handler registration* (the registry side) was fixed this round
(`e90e71c6`). Item 6 above is the *scheduling* side — still open.

#### B. Yalayut demand-signal subsystem — 6 of 7 signal types unwired (from `2026-05-16-yalayut-demand-signals-gap.md`)

`record_signal()` fires for only `founder` (via `/yalayut discover`). The other six —
`planning_miss`, `step_entry_miss`, `tool_call`, `hint_miss`, `dlq`, `repeat_pattern` —
are constants with no caller. And nothing periodically drains `pending_signals()`, so the
autonomous on-demand discovery loop is functionally dead. Two work units (fire the six
sites; add an autonomous drain) — the handoff has the full plan sketch + acceptance.

#### C. Core / orchestrator — verified live bugs

- **Orchestrator lane bug (P1).** `src/core/orchestrator.py:99` and `:121` enqueue
  yalayut discovery / source-scout tasks with `lane="mechanical"`. `lanes.py` defines
  only `LANE_ONESHOT="oneshot"` and `LANE_ONGOING="ongoing"` — there is **no
  `"mechanical"` lane**. The pump selects `oneshot`/`ongoing`; these tasks are silently
  orphaned. Yalayut autonomous discovery never runs. **Fix:** enqueue with
  `lane="oneshot"` (or the correct lane). Verified present 2026-05-17.
- **`demo/distribute` orphan verb.** Dispatchable but not a step in `i2p_v3.json`
  phase 13 — the other 5 `demo/*` verbs are. Add the step or drop the verb.

#### D. Other zones — known dead/latent wiring

- **Z3 multi-file expansion** — `multi_file_expansion` dial off by default, never run in
  a mission; `MULTI_FILE_RULES` covers exactly one stack (`fastapi+nextjs`). The whole
  T2 expansion machinery + `integration_replay` is latent.
- **Z9 `metric_emit` post-hook** — never built as a post-hook kind (only a
  `growth_events.kind` string). Ties to Z7 item 7.
- **Z9 `record_hypothesis`** — wired only at i2p step `7.0y`; ad-hoc / non-i2p Phase-8+
  missions record no hypothesis despite the founder-approved "all Phase 8+" scope.
- **Z8 `monitoring_kit_{fastapi,nextjs,django}_v1` recipes** — never authored; i2p step
  13.3 `monitoring_setup` stays `[NEEDS-REAL-TOOLS]`.
- **Z8 on-call verbs** — now fail loud (`4b9d675c`, this round) instead of lying, but
  still not real; need vendor cloud-API wiring when accounts exist.
- **Z8 `/force_action`** command + weekly ticket-FAQ cluster job — never shipped.

---

## 2. Z0 — finish mission preflight

Z0 was Tier 0 — it must land first and feed every other zone — but only shipped
~35–40%. Doc: `docs/i2p-evolution/z0-mission-preflight.md` (its `## Updates` log still
shows only the 2026-05-08 scaffold; update it when work lands).

**Shipped (`fa100f9a` + earlier):** `ambition_tier`, `cost_ceiling_usd`,
`founder_attention_budget_minutes` columns; `/pause_mission` `/kill_mission`
`/resume_mission`; per-mission Telegram thread; budget threshold alerts; `/preflight`
read/write command.

**Not shipped — the work:**

| Phase | Missing | Notes |
|---|---|---|
| A — Founder profile | `founder_profiles` table (identity, voice samples, prior missions, time-zone, notification prefs); cross-mission inheritance | the cross-mission memory the whole system claims as moat M2 |
| B — Preflight wizard | step-by-step Telegram intake flow (tier → cost → north-star → compliance → readiness → thread → kill-switch); skippable fast-path | `/preflight` today is a CLI, not a wizard |
| C — Vault | per-founder encrypted credential store + passphrase + recovery flow | distinct from the existing general `credential_store.py` |
| E — Readiness verdict | mechanical check that **gates mission start** (OK / warnings / blockers); blockers stop start, warnings need ack | currently mission start is ungated |
| F/G/J | north-star capture at intake; compliance fingerprint pre-intake; idle-policy (`wait`/`proceed_with_default`/`pause` per action) | downstream zones (Z9 north-star, Z6 compliance) assume these exist |

**Recommended order:** A (profile schema — unblocks everything) → E (readiness gate — the
honest "is this mission ready" check) → B (wizard wraps A+E) → F/G/J (small, fold into
the wizard) → C (vault — largest, can trail). Each phase per `superpowers:writing-plans`
+ TDD. Cross-reference: every downstream consumer doc gets a one-line "Z0 inputs:" note.

---

## 3. Everything else left behind

### 3a. Pre-existing test failures (carry-overs — flagged by the fix agents)

- **`tests/test_grounding_posthook.py` — uncollectable.** Literal syntax error:
  `class TestMr. RobotoCheckGroundingVerb:` (space in identifier — a bad find-replace of
  "MrRoboto"→"Mr. Roboto"), 3 classes affected. The whole file fails collection, so the
  **G-grounding post-hook tests have not run at all**. ~1-line-per-class fix; high value
  (grounding is a load-bearing feature).
- `test_code_review_posthook::test_apply_code_review_pass_with_other_pending_keeps_ungraded`
  — patches `update_task` on the wrong import path.
- `test_z4_t3_visual_review_posthook::test_visual_review_in_simple_blocker_verdict_kinds`
  — brittle: scans apply.py source text in a 6-line window; the real code spans 18. The
  routing is correct; the verifier is wrong.
- `test_reversibility_registry` — verbs `capture_hint`, `source_scout`,
  `yalayut_discovery` not registered in `VERB_REVERSIBILITY`.
- `test_admission_cache` (2) — admission cache, unrelated.
- `test_beckman_posthooks::test_apply_request_posthook_grade...` — test passes a
  JSON-string `source_ctx` where production always passes a dict.

### 3b. Unbuilt scoped work

- **`gorsel_ustasi`** image-gen provider package — scoped Z1→Z2, built by neither. Z1
  emits placeholder images only.
- **Web preview hosting surface** (C10 / F1) — `emit_preview_url` verb exists; the
  cloudflared/local-port/GitHub-Pages host + viewer was deferred to Z2, never built.
- **Z8 monitoring_kit recipes** — see §1.D.

### 3c. Two unconnected feedback loops

Z9's reinforce loop writes `model_pick_log` rows with `call_category='reinforce'`
(read by `fatih_hoca/grading.py::reinforce_bonus`). Z10's calibration loop writes
`confidence_outcomes` (→ `confidence_reliability_scores` → prompt builder). Each was
"the learning loop" in its own zone doc; they were never unified and read different
tables. Decide whether to merge them or document them as deliberately separate. Also
verify the `model_pick_log.call_category='reinforce'` **write** path actually exists —
the audit could only confirm the read side.

### 3d. Completion tags

No git tag for **Z0, Z5, Z6, Z10**. If tagging retroactively: Z5 ≈ `13d7af25`,
Z6 ≈ `2af62429`, Z10 ≈ `9011325`. Z0 should NOT get a `complete` tag until §2 is done.

### 3e. History blemish (cosmetic, no fix urgency)

The 2026-05-17 fix round's Z7 audit-log test fix was `git commit --amend`-ed into the
parallel session's `dc1b2b52 docs(z5): ...` commit instead of the Z7 fix commit
`e90e71c6` — a `docs(z5)` commit landed between the cherry-pick and the amend. Main is
functionally correct (test passes); `e90e71c6` in isolation has the stale test. Local
history only, nothing pushed. Clean up only if doing a history rewrite anyway.

---

## 4. Suggested sequencing

1. **Quick wins first** — orchestrator lane bug (§1.C, P1, ~1 line), grounding-test
   syntax error (§3a, unblocks dead tests). Both tiny, both high-value.
2. **Wiring-audit sweep** — work the §1 orphan list, then run the §1 method across the
   zones not yet swept. Each item: wire + a host-path test.
3. **Z0** — §2, in the recommended phase order.
4. **A real prototype-tier mission, end to end** — only meaningful after 1–3; it is the
   only thing that proves the code zone is actually reliable and will flush whatever the
   sweep missed.
5. Backlog: §3b unbuilt work, §3c loop unification, §3d tags.

The brutal-truth items from the original frame (legal personhood, brand taste,
relationships, credentials) stay founder territory — not on this list by design.
