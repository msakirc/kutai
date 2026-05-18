# mobile_release_rejection Recipe v1 — Known Lessons

App Store / Google Play rejection pitfalls. Seeded into `mission_lessons`
(domain `mobile_release_rejection`) on recipe instantiation, so a later
mission's `14.8` submit chain inherits them before it ever hits review.

- **Guideline 4.3 "spam" is a positioning problem, not a bug**: Apple
  rejects apps under *4.3(a) Design — Spam* when they read as a re-skinned
  template or a thin web wrapper. There is no code patch — the app needs a
  genuine native capability (push, camera, offline store, biometrics) and
  metadata/screenshots that differentiate it from sibling apps. Treat a
  4.3 rejection as a founder escalation with a drafted re-positioning, not
  an auto-resubmit.

- **Privacy labels must match runtime exactly — both directions fail**:
  the Apple App Privacy "nutrition label" and Google Play Data Safety form
  are checked against what the reviewer observes. Declaring data the app
  does not collect fails just as hard as omitting data it does. Derive the
  labels from the actual SDK list (every analytics, crash, ad and auth SDK
  collects data) — never hand-write them.

- **"Crash on review" is almost always a release-build-only crash**: the
  metro/dev build hides it. Hermes byte-code, ProGuard/R8 symbol
  stripping, and env vars missing from the release scheme only bite
  `--configuration Release` / `assembleRelease`. Always reproduce against
  a release build before claiming the crash is unreproducible.

- **A login wall with no demo account is an automatic 2.1 rejection**:
  reviewers cannot create accounts or pass SMS/OTP gates. Provision a
  stable demo account, seed it with representative data, and put the
  credentials in App Store Connect *App Review Information* / Play *App
  access* — never inside the binary. Make sure the account never expires
  or rate-limits mid-review.

- **Apple reviews on an IPv6-only NAT64 network**: hard-coded IPv4 address
  literals in API base URLs or WebSocket endpoints, or a backend/CDN that
  is not dual-stack, fail review even though they work on the dev box.
  Use hostnames only and confirm the whole request path resolves over
  IPv6. macOS can create a NAT64 test network via Internet Sharing.

- **Screenshots and description are a contract — Guideline 2.3**:
  screenshots showing content the app does not have, a description
  claiming an absent feature, placeholder copy, or a broken support /
  privacy-policy URL all reject under accurate-metadata rules.
  Re-generate `store_metadata.json` and re-capture screenshots from the
  *shipped* build whenever the build changes.

- **Every requested permission needs an in-context justification**: iOS
  rejects vague `NS*UsageDescription` strings (Guideline 5.1.1) and Play
  rejects sensitive permissions without a clear use. Remove unused
  permissions, write specific user-facing usage strings, and request the
  permission just before the feature needs it — not at launch.

- **Digital goods must use IAP / Play Billing — Guideline 3.1.1**:
  selling digital content or unlocking features through an external
  payment processor, or linking out to an external purchase page, is a
  hard rejection. Physical goods and services may use an external
  processor; if so, say it plainly in the review notes.

- **Fix the reason, then guard against its return**: when the fix is code
  (crash, IPv6, permissions), add a regression guard — a Maestro smoke
  step for the crashing flow, a release-build CI job, or a test asserting
  no IPv4 literals. The next rejection should be a new reason, never a
  repeat of one already paid for.

- **Resubmission resets the review queue — batch fixes**: each
  resubmission goes to the back of the review queue (often 24-48h). Do not
  resubmit for one fix at a time; address every plausible issue the
  reviewer raised (and any sibling issues in the same area) in a single
  resubmission to avoid serial multi-day round-trips.

- **The reviewer-response wording matters**: a terse or defensive
  *Resolution Center* reply slows re-review. Acknowledge the specific
  guideline, state exactly what changed, and — for metadata/positioning
  rejections — explain why the app now complies. Draft it from the
  `rejection_response` template and have the founder approve the tone.
