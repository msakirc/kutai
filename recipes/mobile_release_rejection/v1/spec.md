# mobile_release_rejection recipe v1 — App Store / Play rejection playbook

## What this recipe is

A **diagnosis-and-fix playbook** for the most common reasons an Expo /
React Native app gets rejected from the Apple App Store or Google Play
review. It is consumed when step `14.8.review_status` (Z5 T5) reports a
`rejected` verdict — the playbook maps the rejection reason to a concrete
fix, drafts a reviewer response, and (where the fix is code) re-runs the
relevant `14.8` submit chain.

Unlike a `feature` recipe, it ships **no production code templates** — its
value is the rejection knowledge in `lessons.md` (seeded into the
`mission_lessons` table, domain `mobile_release_rejection`) plus two small
operational templates: a reviewer-response draft and a demo-account
descriptor.

## Rejection reasons covered

Each reason below has a `diagnosis` (how to recognise it from the
rejection notice) and a `fix` (what to change before resubmitting).

### 1. Guideline 4.3 — Spam / "duplicate app"

- **Diagnosis:** Apple cites *Guideline 4.3(a) Design — Spam*. Triggered
  by apps that look like a re-skinned template, thin "web wrapper" apps,
  or apps too similar to others from the same account.
- **Fix:** Add genuine native value — at least one capability that needs
  the device (push, camera, offline store, biometrics). Remove generic
  template copy. If it is a web wrapper, justify the native surface or
  ship real native screens. Differentiate metadata and screenshots from
  any sibling apps. This is often a *positioning* fix, not a code fix —
  escalate to the founder with a drafted re-positioning.

### 2. Privacy-label / Data-Safety mismatch

- **Diagnosis:** Apple App Privacy "nutrition label" or Google Play Data
  Safety form does not match observed runtime behaviour — reviewer saw
  the app collect data the form did not declare (or vice versa). Apple
  Guideline 5.1.1 / 5.1.2.
- **Fix:** Re-derive the privacy labels from the *actual* SDKs and network
  calls (analytics, crash reporting, ad SDKs, auth providers all collect
  data). Regenerate `privacy_nutrition_labels.json` in step `14.8`. Do not
  over-declare either — a label claiming tracking the app does not do also
  fails. Match the form to reality exactly.

### 3. Crash on review

- **Diagnosis:** Reviewer reports the app crashes on launch or on a
  specific action. Often a release-build-only crash (Hermes/proguard
  stripped a symbol, a dev-only shim missing, a `__DEV__` guard).
- **Fix:** Reproduce against a **release** build, not the dev/metro build
  (`expo run:ios --configuration Release` / `assembleRelease`). Check
  ProGuard/R8 keep rules, Hermes byte-code, missing env vars in the
  release scheme. Add the crash to the Maestro smoke flow so it cannot
  regress. Wire a crash reporter (Sentry / Crashlytics) so the next
  rejection comes with a stack trace.

### 4. Missing demo account / login wall

- **Diagnosis:** Apple Guideline 2.1 — the app is behind a login the
  reviewer cannot pass. Reviewer asks for credentials or a demo mode.
- **Fix:** Provision a stable demo account (see
  `templates/demo_account.json.template`) and put the credentials in the
  App Store Connect *App Review Information* notes — never in the binary.
  Keep the account seeded with representative data and never let it
  expire or get rate-limited. For Play, fill the *App access* section.

### 5. IPv6 / network reachability

- **Diagnosis:** Apple Guideline 2.1 — Apple reviews on an IPv6-only,
  NAT64/DNS64 network. App fails because it hard-codes IPv4 literals,
  uses a non-IPv6 backend, or assumes a specific DNS path.
- **Fix:** Never embed IPv4 address literals — use hostnames. Confirm the
  backend / CDN is dual-stack. Test on a NAT64 network (macOS can create
  one: *Sharing → Internet Sharing* with the "Create NAT64 Network"
  option). Make sure WebSocket / long-poll endpoints also resolve over
  IPv6.

### 6. Metadata issues

- **Diagnosis:** Apple Guideline 2.3 (accurate metadata) or Google Play
  policy — screenshots show content not in the app, description claims a
  feature that is absent, keywords stuffed, placeholder text shipped,
  wrong category, or a broken support / privacy-policy URL.
- **Fix:** Regenerate `store_metadata.json` in step `14.8` so every claim
  is true of the shipped build. Re-capture screenshots from the real app
  via `14.8.screenshots`. Verify the support URL and privacy-policy URL
  return 200. Drop keyword stuffing. Pick the most accurate category.

### 7. Background / permission usage not justified

- **Diagnosis:** Apple Guideline 5.1.1 / 2.5.4 or Play sensitive-permission
  policy — the app requests location / background / camera / contacts
  without a clear in-context reason, or the iOS `NS*UsageDescription`
  strings are vague.
- **Fix:** Remove any permission the app does not use. Write specific,
  user-facing `NSCameraUsageDescription` / `NSLocationWhenInUseUsage...`
  strings (Expo: the `app.json` `ios.infoPlist` / `android.permissions`
  blocks). Request permissions in context, just before the feature needs
  them — not at launch.

### 8. In-app-purchase / external-payment issues

- **Diagnosis:** Apple Guideline 3.1.1 — the app sells digital goods or
  unlocks features through a non-Apple payment path, or links out to an
  external purchase flow.
- **Fix:** Digital goods and feature unlocks must use Apple In-App
  Purchase / Google Play Billing. Physical goods and services may use an
  external processor. If the app genuinely sells only physical goods,
  state that clearly in the review notes. Otherwise integrate
  `expo-in-app-purchases` / RevenueCat.

## How the playbook is used

1. `14.8.review_status` polls the store and reports `rejected` with a
   reason string / guideline citation.
2. The mission classifies the rejection against the 8 reasons above.
3. For a **code fix** (3, 5, 7): patch the app, re-run the relevant part
   of the `14.8` submit chain.
4. For a **metadata / privacy fix** (2, 6): regenerate the `14.8`
   artifacts and re-run `14.8.submit`.
5. For a **positioning / payment / login fix** (1, 4, 8): draft a
   reviewer response from `templates/rejection_response.md.template` and
   escalate to the founder — these need human judgement before
   resubmission.

## Templates

| Template | Purpose |
|---|---|
| `rejection_response.md.template` | Reviewer-response draft: acknowledges the rejection, states the fix, requests re-review. |
| `demo_account.json.template` | Demo-account descriptor for the App Review Information notes (reason 4). |

## Post-hooks

`imports_check`, `test_run` — when the rejection produced a code fix, the
re-instantiated project must still import and pass its test suite before
resubmission.
