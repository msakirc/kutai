# Expo — mobile_ci/v1 instantiation notes

When wiring the free-first mobile CI recipe into an Expo project:

- **`expo prebuild` is mandatory before any native build.** The generated
  workflow runs `npx expo prebuild --platform <ios|android>` first — this
  materialises the native `ios/` and `android/` directories from
  `app.json` / `app.config.js`. `xcodebuild` and `./gradlew` operate on
  those generated dirs; without prebuild they do not exist. Do not commit
  the generated `ios/` / `android/` dirs — let CI regenerate them.

- **macOS runners cost 10x.** GitHub bills `macos-latest` minutes at a
  10x multiplier against the free monthly allowance. Keep the iOS job
  minimal: cache `ios/Pods`, skip simulator-only steps, and gate the iOS
  job behind a path filter if the repo also has non-app changes. The
  Android job on `ubuntu-latest` is 1x — cheap, run it freely.

- **Code signing comes from GitHub Secrets, never the repo.** The
  Fastfile `match` lane reads `MATCH_PASSWORD` from a secret; the
  App Store Connect API key arrives as `APP_STORE_CONNECT_KEY`; the
  Android keystore is base64-encoded into `ANDROID_KEYSTORE_BASE64` and
  decoded at build time. Committing any of these is a credential leak.

- **`fastlane match` needs the keychain unlocked on CI.** On a fresh
  macOS runner the login keychain is locked. Before the `match` / `build`
  lanes, unlock it (`security unlock-keychain`) or `match` fails with an
  opaque "could not install certificate" error. The `setup_ci` Fastlane
  action handles this — call it at the top of the iOS lanes.

- **The `<<XCODE_SCHEME>>` must match the Expo app name.** `expo prebuild`
  derives the Xcode scheme from the `name` field in `app.json`. If
  `XCODE_SCHEME` here disagrees, `xcodebuild -scheme` fails with
  "scheme not found". Set `XCODE_SCHEME` to the sanitised app name.

- **Lane reversibility is real.** `build` and `match` are safe to re-run.
  `pilot` (TestFlight) and `supply` (Play internal) push a binary onto a
  store track that testers and the store ingest — those are irreversible.
  The mr_roboto `fastlane` verb tags them accordingly; expect a
  confirmation gate before `pilot` / `supply` in a real mission.

- **Android needs a JDK and a keystore.** The Android job sets up JDK 17
  (`<<JAVA_VERSION>>`) and decodes the release keystore before
  `./gradlew assembleRelease`. A debug build skips the keystore; a
  release build that omits it produces an unsigned APK the Play Store
  rejects.
