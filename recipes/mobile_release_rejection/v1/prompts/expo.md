# Expo — mobile_release_rejection/v1 instantiation notes

When this playbook fires after a `14.8.review_status` rejection on an
Expo / React Native app:

- **Classify before you fix.** Read the rejection notice and match it to
  one of the 8 reasons in `spec.md`. Apple cites a *Guideline* number
  (e.g. `4.3`, `2.1`, `5.1.1`); Google Play cites a *policy* name. The
  guideline number is the fastest route to the right fix template — do
  not guess.

- **Reproduce against a release build.** A "crash on review" (reason 3)
  almost never reproduces in the metro/dev build. Always reproduce with
  `npx expo run:ios --configuration Release` or
  `./gradlew assembleRelease` + install — Hermes byte-code, ProGuard/R8
  stripping and missing release-scheme env vars only bite the release
  build.

- **Privacy labels are derived, not declared.** For reason 2, do not hand-
  write the privacy labels. Walk the actual dependency tree: every
  analytics SDK, crash reporter, ad SDK and auth provider collects data.
  Regenerate `privacy_nutrition_labels.json` in step `14.8` so the Apple
  App Privacy label and Play Data Safety form match runtime behaviour
  *exactly* — under-declaring and over-declaring both fail review.

- **Demo credentials go in review notes, never the binary.** For reason 4
  put the demo account into App Store Connect *App Review Information* /
  Play *App access*. Shipping credentials inside the app is itself a
  rejection (and a security hole). Seed the account with representative
  data and make sure it never expires or rate-limits during review.

- **`NS*UsageDescription` strings live in `app.json`.** Expo manages the
  iOS `Info.plist` via the `ios.infoPlist` block and Android permissions
  via `android.permissions` in `app.json` / `app.config.js`. For reason 7,
  edit those — do not hand-edit the generated `ios/`/`android/` dirs
  (`expo prebuild` regenerates them).

- **IPv6: hostnames only.** For reason 5, grep the codebase for IPv4
  address literals in API base URLs, WebSocket endpoints and any
  hard-coded config. Apple reviews on NAT64/DNS64 — anything that
  resolves only over IPv4 fails. Confirm the backend and CDN are
  dual-stack.

- **Some rejections are founder territory.** Reasons 1 (spam /
  positioning), 8 (payment model) and the *response wording* for any
  rejection need human judgement. Draft the reviewer response from
  `rejection_response.md.template`, fill the diagnosis + fix, and
  escalate to the founder — do not auto-resubmit a positioning fix.

- **Add a regression guard.** Whenever the fix is code (reasons 3, 5, 7),
  add a check that the failure cannot recur silently: a Maestro smoke
  step for the crashing flow, a release-build CI job, or a test that
  asserts no IPv4 literals. The next rejection should be a *new* reason,
  never a repeat.
