# mobile_ci Recipe v1 — Known Lessons

Pitfalls captured from free-first GitHub Actions mobile CI. Seeded into
`mission_lessons` (domain `mobile_ci`) on recipe instantiation.

- **GitHub free-tier macOS minutes run out fast — 10x multiplier**: macOS
  runner minutes are billed at **10x** against the monthly free allowance
  (2,000 min on the Free plan → only ~200 macOS minutes). A single iOS
  build can take 15-25 min, so ~8-13 iOS builds exhaust the month. Gate
  the iOS job behind a path filter, cache aggressively, and never run it
  on every draft push. The Android (`ubuntu-latest`) job is 1x — cheap.

- **macOS-runner caching is mandatory, not optional**: without an
  `actions/cache` step for `ios/Pods` (and `~/.cocoapods`), every iOS
  build re-downloads and re-compiles all CocoaPods — adding 5-10 of the
  expensive 10x-billed minutes. Key the cache on `hashFiles('ios/Podfile.lock')`.

- **`fastlane match` fails until the CI keychain is unlocked**: a fresh
  macOS runner has a locked login keychain. `match` / `build_app` then
  fail with "could not install certificate" or hang on a GUI prompt.
  Call Fastlane's `setup_ci` action (or `security unlock-keychain`) at the
  start of every iOS lane — it creates a temporary unlocked keychain.

- **Code-signing secrets go in GitHub Secrets, never the repo**: the
  `match` repo password (`MATCH_PASSWORD`), the App Store Connect API key,
  and the Android keystore must be GitHub Secrets. Committing a `.p12`,
  a provisioning profile, or `keystore.jks` to the repo is a credential
  leak that lets anyone sign builds as you. The workflow decodes them at
  build time and they never touch disk in plaintext in the repo.

- **`xcodebuild` needs the prebuilt `ios/` dir — run `expo prebuild` first**:
  an Expo project has no native `ios/` directory until `expo prebuild`
  generates it from `app.json`. `xcodebuild -workspace ios/App.xcworkspace`
  fails with "does not exist" if the prebuild step is missing or runs
  after the build step. Order matters: checkout → npm ci → prebuild →
  build.

- **Android release keystore: base64 in a secret, decoded at build time**:
  store the keystore as `base64 keystore.jks | pbcopy` → a GitHub Secret
  (`ANDROID_KEYSTORE_BASE64`). The workflow does
  `echo "$ANDROID_KEYSTORE_BASE64" | base64 --decode > android/app/release.keystore`.
  A release `assembleRelease` without a keystore produces an unsigned APK
  that the Play Store rejects on upload.

- **`pilot` and `supply` are irreversible — gate them**: `fastlane pilot`
  uploads to TestFlight and `fastlane supply` uploads to the Play internal
  track. Testers and the stores ingest the binary immediately; you cannot
  un-upload it (only expire/withdraw). Never wire `pilot`/`supply` into an
  automatic on-every-push lane — gate them behind a manual
  `workflow_dispatch` trigger or a confirmation step.

- **`workflow_dispatch` is required for manual release runs**: without a
  `workflow_dispatch:` trigger in the `on:` block, the only way to run the
  release lanes is to push to `main`, which couples every release to a
  commit. Always include `workflow_dispatch` so a release can be triggered
  deliberately from the Actions tab.

- **`expo prebuild` is not idempotent across Expo SDK upgrades**: a
  prebuild generated on Expo SDK 50 and committed will drift from SDK 51's
  expected native config. Treat `ios/` and `android/` as build artifacts
  — `.gitignore` them and let CI regenerate every run. Committing them
  causes "pod install" mismatches that only reproduce on the runner.

- **Node version mismatch breaks `npm ci` caching**: the runner's
  `setup-node` version must match the `engines.node` in `package.json`
  and the version Expo expects. A mismatch makes `npm ci` rebuild native
  deps and silently invalidates the npm cache every run. Pin
  `NODE_VERSION` (default 20) consistently across both jobs.

- **`gradlew` needs `--no-daemon` on CI**: the Gradle daemon lingers
  between steps on a runner and can hold a stale classpath, causing
  flaky "cannot find symbol" failures. Always pass `--no-daemon` for CI
  builds — the daemon's warm-start benefit is irrelevant for a one-shot
  runner.
