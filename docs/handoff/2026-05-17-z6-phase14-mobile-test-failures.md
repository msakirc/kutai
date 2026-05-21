# Handoff — 3 failing Z6 Phase-14 mobile tests

> **RESOLVED 2026-05-21.** D1 (option b) + D2 (option a) were implemented in
> `7e5ecc6d`: `14.8.screenshots` got `vendor_interaction: false` (excluded from
> the vendor heuristic); `14.8.review_status` + `14.8.review_status_play` got
> `read_only: true` (exempt from the irreversibility invariant). Completed in
> `9efea092`, which resolved 14.8's residual `read_only`+`irreversible`+`locked`
> contradiction → `full`, moving the founder-gate lock onto the real uploads
> (`14.8.submit` / `14.8.submit_play`). All 3 tests green. Kept for history.

**Date:** 2026-05-17
**Status:** open — needs 2 design decisions, then a small fix
**Scope:** `tests/workflows/` — pre-existing failures, NOT caused by the yalayut Phase 2 work (confirmed via `git stash` diff during that session). Surfaced while reviewing yalayut P2.

## TL;DR

The Z5 mobile-CI work split the single Phase-14 app-store step `14.8` into
four sub-steps (`14.8`, `14.8.screenshots`, `14.8.submit`,
`14.8.review_status`). The Z6 invariant tests still encode the *old* world
("the only mobile-vendor step in Phase 14 is 14.8" — literally in the test
docstring). Three Z6 assertions now fail. Two of the three are not just a
missing tag — they expose a heuristic false-positive and a too-strict
invariant. Each needs a decision before the fix.

## The 3 failures

Run: `timeout 120 .venv/Scripts/python.exe -m pytest tests/workflows/test_z6_polish_phase14_mobile.py tests/workflows/test_z6_t6a_reversibility.py -q`

```
FAILED tests/workflows/test_z6_polish_phase14_mobile.py::test_phase14_mobile_vendor_steps_tagged
FAILED tests/workflows/test_z6_polish_phase14_mobile.py::test_phase14_steps_dependent_on_app_store_submission_inherit
FAILED tests/workflows/test_z6_t6a_reversibility.py::test_needs_real_tools_steps_are_irreversible
```

Exact errors:

```
AssertionError: Phase-14 mobile-vendor step 14.8.screenshots
  (app_store_screenshots_capture) missing real_tool_kind
  — test_z6_polish_phase14_mobile.py:67

AssertionError: downstream mobile-vendor step 14.8.screenshots must be tagged
  — test_z6_polish_phase14_mobile.py:124

AssertionError: needs_real_tools steps not irreversible: ['14.8', '14.8.review_status']
  — test_z6_t6a_reversibility.py:59
```

## Current state — the four `14.8.*` steps in `src/workflows/i2p/i2p_v3.json`

| id | name | agent | real_tool_kind | reversibility | needs_real_tools |
|---|---|---|---|---|---|
| `14.8` | app_store_submission | executor | `apple_appstore\|google_play` | **full** | true |
| `14.8.screenshots` | app_store_screenshots_capture | mechanical | **null** | full | null |
| `14.8.submit` | app_store_submit_binary | mechanical | `apple_appstore\|google_play` | irreversible | true |
| `14.8.review_status` | app_store_review_status_poll | mechanical | `apple_appstore\|google_play` | **full** | true |

`14.8.submit` is correctly tagged — use it as the reference shape. The bolded
cells are what the tests trip on.

## Root cause — per failure

### Failures 1 & 2 — `14.8.screenshots` false-positives the vendor heuristic

`_looks_mobile_vendor()` (test helper) flags a step when the keyword
`"app store"` / `"play store"` / `"testflight"` appears in `name + instruction`.
`14.8.screenshots` trips it because its **instruction** says
*"capture per-device/per-locale **App Store** / **Play store** screenshots"*.

But the step is **genuinely local** — its payload is
`{"action": "capture_screenshots", "capture_mode": "device"}`, and the
instruction is *"reuse `capture_screenshots` ... Playwright device descriptors
against the Expo Web export + adb screencap"*. It never calls Apple/Google.
The actual vendor upload is `14.8.submit`. The test's own docstring says local
build steps "must NOT be tagged — they don't need real tools" — `14.8.screenshots`
is exactly that case. The keyword heuristic just can't tell.

So this is NOT "add `real_tool_kind` to `14.8.screenshots`" — that would be
wrong (it would make the Z6 admission gate block a local step when creds are
absent).

### Failure 3 — the "needs_real_tools ⇒ irreversible" invariant is too strict

`test_needs_real_tools_steps_are_irreversible` asserts every step with
`needs_real_tools` true has `reversibility == "irreversible"`. It catches:

- `14.8.review_status` — `app_store_review_status_poll`. **Read-only.** Polling
  review status calls the vendor API (so it needs real credentials → real
  tools) but changes nothing → genuinely reversible. The invariant conflates
  "needs credentials" with "has an irreversible side-effect".
- `14.8` — `app_store_submission`, `agent: executor`. This looks like an
  orchestration parent; the actual irreversible binary upload is `14.8.submit`
  (correctly tagged irreversible). If `14.8` does not itself perform the
  irreversible act, either its `reversibility` or its `needs_real_tools` is
  mistagged.

## Decisions needed

**D1 — how should the vendor-step test distinguish local-but-app-store-flavoured
steps from real vendor calls?** Options:
- (a) Tighten `_looks_mobile_vendor` to also require a non-local signal —
  e.g. `payload.action` not in a local-action allowlist
  (`capture_screenshots`, build actions), or `needs_real_tools` intent. Risk:
  the test is meant to *catch* untagged vendor steps, so it can't simply trust
  the tag it's verifying.
- (b) Add an explicit step field, e.g. `vendor_interaction: false`, on
  genuinely-local steps; the test excludes those. Most honest — makes intent
  declared, not inferred.
- (c) Reword `14.8.screenshots`'s instruction so the keyword doesn't appear
  (fragile — the next mobile step re-breaks it).
- Recommended: **(b)** — explicit marker.

**D2 — should read-only real-tool steps be exempt from the irreversible
invariant?** A status poll needs creds but is idempotent. Options:
- (a) Add a `read_only: true` (or `reversibility: "full"` + a `real_tool_read`
  marker) and exempt those in `test_needs_real_tools_steps_are_irreversible`.
- (b) Keep the invariant; decide `14.8.review_status` should be `irreversible`
  anyway (semantically wrong for a poll).
- Recommended: **(a)** — the invariant should be "real-tool steps with a
  *write* side-effect are irreversible". Also re-check `14.8`'s tags against
  whether it performs the act or just orchestrates `14.8.submit`.

## Fix sketch (after D1/D2)

1. `src/workflows/i2p/i2p_v3.json` — add the chosen markers to `14.8.screenshots`
   (local) and `14.8.review_status` (read-only); re-check `14.8`'s
   `needs_real_tools` / `reversibility`.
2. `tests/workflows/test_z6_polish_phase14_mobile.py` — `_looks_mobile_vendor`
   or the per-test filter honours the D1 marker.
3. `tests/workflows/test_z6_t6a_reversibility.py::test_needs_real_tools_steps_are_irreversible`
   — honour the D2 read-only exemption.
4. Validate JSON: `python -c "import json,io; json.load(io.open('src/workflows/i2p/i2p_v3.json',encoding='utf-8')); print('ok')"`
5. Green check: `timeout 150 .venv/Scripts/python.exe -m pytest tests/workflows/ -q` → expect `92 passed`.

## Notes / context

- The Z5 mobile work is in flight — uncommitted `recipes/mobile_ci/`,
  `recipes/mobile_deep_links/`, `recipes/mobile_push/`,
  `packages/mr_roboto/src/mr_roboto/maestro_run.py`,
  `packages/mr_roboto/tests/test_z5_t3b_ci.py`. The `14.8.*` sub-steps were
  added by recent z5 commits (`9bd3b8cf feat(z5): gen_mobile_ci + fastlane
  verbs`). Whoever owns Z5 mobile is the right person for D1/D2 — they know
  whether `14.8.screenshots` / `14.8.review_status` are local or vendor.
- Files: workflow `src/workflows/i2p/i2p_v3.json`; tests
  `tests/workflows/test_z6_polish_phase14_mobile.py`,
  `tests/workflows/test_z6_t6a_reversibility.py`.
- `i2p_v3.json` is UTF-8 — read it with `io.open(..., encoding='utf-8')`,
  the cp1252 default chokes on it.
