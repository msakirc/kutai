# mobile_ci recipe v1 — free-first mobile CI

## What this recipe implements

A complete **free-first** mobile build + release pipeline for an Expo /
React Native app, using GitHub Actions runners instead of paid cloud
build services (EAS Build/Submit). Per the founder decisions of
2026-05-16 (`docs/i2p-evolution/05-build-mobile-track-v2.md`), this is the
**default** mobile build path; `eas_build` / `eas_submit` are demoted to
an optional fallback.

It ships three artifacts into the mission workspace:

1. **`.github/workflows/mobile.yml`** — a GitHub Actions workflow with two
   jobs:
   - an **iOS job** on a `macos-latest` runner: `expo prebuild --platform
     ios` → `xcodebuild` / Fastlane;
   - an **Android job** on a free `ubuntu-latest` runner: `expo prebuild
     --platform android` → `./gradlew assembleRelease`.
2. **`fastlane/Fastfile`** — Fastlane lanes: `build`, `match` (code
   signing), `pilot` (TestFlight upload), `supply` (Play internal upload).
3. **`fastlane/Appfile`** — app identity (bundle id, Apple team id,
   Android package name) shared by every lane.

## Why free-first

iOS builds need macOS-local Xcode tooling that cannot run on the Windows
dev box. The two cloud paths are EAS Build (paid past a small free tier)
or **GitHub Actions macOS runners** (free tier covers the workload, more
setup). The founder picked GitHub Actions. macOS minutes bill at a **10x
multiplier** on the free tier — the workflow keeps the iOS job lean and
caches CocoaPods aggressively.

## How the recipe drives the mr_roboto verbs

- `gen_mobile_ci` generates the workflow file directly (it can also be
  instantiated from this recipe's `mobile.yml.template`).
- `fastlane` runs an individual lane. Its reversibility is per-lane:
  `build` / `match` are local + re-runnable (`full`); `pilot` / `supply`
  push a binary to a real store track (`irreversible`).

## Parameters

| Param | Default | Meaning |
|---|---|---|
| `BUNDLE_ID` | `com.example.app` | iOS bundle id / Android package name |
| `APPLE_TEAM_ID` | `ABCDE12345` | Apple Developer team id |
| `XCODE_SCHEME` | `App` | Xcode scheme name for `xcodebuild` |
| `NODE_VERSION` | `20` | Node version for the runners |
| `JAVA_VERSION` | `17` | JDK version for the Android job |

## Secrets the workflow expects

Set these in the generated repo's GitHub Secrets — never commit them:
`MATCH_PASSWORD`, `APP_STORE_CONNECT_KEY`, `ANDROID_KEYSTORE_BASE64`,
`ANDROID_KEYSTORE_PASSWORD`, `PLAY_SERVICE_ACCOUNT_JSON`.

## Post-hooks

`imports_check` — confirms the instantiated config is syntactically sound.
