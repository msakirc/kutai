# Z2 — Mobile track

## Frame

i2p today is web-shaped. tech_stack_decision implicitly assumes
NextJS / React; phase-7 frontend_scaffold writes `frontend/package.json`;
visual review (when wired) targets browser viewports. Real products
increasingly need mobile — native iOS+Android, cross-platform via
React Native / Flutter / Expo, or both. Largest single scope expansion.

Built last because depends on visual review (subproject), recipes (memory
+ recipe library from foundation), and operations (mobile distribution
has its own ops shape).

## Current state

- Phase 4 has `tech_stack_research_and_selection` (4.2) but it implicitly leans web.
- All phase-7 frontend_* steps are web-shaped (Vite/Next, Vitest/Jest).
- All template feat.7 frontend_components emit `.tsx` for React; no Expo / RN / Flutter shape.
- No mobile tooling adapters in mr_roboto.
- No mobile visual review (depends on [04-build-visual-review.md](04-build-visual-review.md) device-screenshot mode).
- No mobile-specific recipes.
- No app-store submission flow.

## Gaps

### Fixable by automation

**A. Stack branching at phase 4**
- New conditional group: `target_platform ∈ {web, mobile_native, mobile_cross_platform, both}`.
- Each branch reroutes phase-7 frontend_* steps + phase-8 feature template frontend steps.
- "both" runs web + mobile in parallel; integration reviewer checks API contract drift across both surfaces.

**B. Mobile recipes (extend recipe library from 02)**
- Auth: Sign in with Apple + Google + email; session token storage in keychain/keystore.
- Navigation: stack/tab navigators (native) or expo-router (cross-platform).
- Persistence: SQLite (native) / WatermelonDB / Realm.
- Push notifications: FCM + APNs (cross-platform via Expo, or per-platform).
- Deep linking: universal links + app links.
- Offline-first: sync engine (PowerSync / WatermelonDB / custom).

**C. Mobile tooling adapters (mr_roboto)**
- `expo_cli` — `expo prebuild`, `expo run:ios`, `expo run:android`, `expo build` (EAS Build).
- `ios_simulator` — `xcrun simctl` boot/shutdown/install/launch.
- `android_emulator` — `adb` boot/install/launch.
- `xcode_build` / `gradle_build` — native CI builds.

**D. Mobile e2e**
- Detox (native + RN).
- Maestro (cross-platform; YAML flows; cheaper to author).
- Recipe-driven smoke flows (sign in, complete onboarding, perform core action, sign out).

**E. Mobile visual review (extends 04)**
- Device screenshots (real or emulated) at multiple devices/orientations.
- Per-platform reference baselines (iOS native vs Android Material 3 vs cross-platform shared).
- Same vision-diff harness as 04 with device-shaped capture.

**F. App store submission flow**
- Generate screenshots + metadata + privacy nutrition labels per platform.
- TestFlight + Play internal-track uploads.
- Submit-for-review templates with rejection-handling playbook.
- Versioning + bundle-id management.
- Deeply intertwined with [06-real-world-bridge.md](06-real-world-bridge.md) — Apple/Google developer accounts are real-world identity work.

**G. Per-platform build/test/sign/distribute**
- Build matrices: iOS (Debug/Release × simulator/device × universal/x86/arm).
- Code signing: certificate + provisioning profile management; Fastlane match patterns.
- Distribution: TestFlight / Play internal / Firebase App Distribution / direct .ipa/.apk.

### Founder territory
- Apple Developer Program enrollment (KYC, $99/yr).
- Google Play Console enrollment ($25 one-time, KYC).
- Bundle ID + package name choices (mostly irreversible once published).
- App Store Connect tax forms.
- Privacy nutrition labels approval.
- App store review responses.

## Proposed direction

### Phase A — Branch infrastructure
- Phase 4 conditional group `target_platform` added to i2p_v3.json.
- Phase 7 frontend_* steps branched: web variant (existing), mobile_cross_platform variant (Expo recipe-set), mobile_native variant (Swift+Kotlin recipe-set).
- Feature template frontend_* steps branched same way.

### Phase B — Mobile recipes (~6 recipes)
- mobile_auth, mobile_nav, mobile_persistence, mobile_push, mobile_deep_links, mobile_offline_sync.
- Each ships scaffold templates + tests + lessons.md per stack (Expo / RN / Flutter / native).

### Phase C — Mr. Roboto adapters
- Wrappers for expo_cli, xcrun simctl, adb, fastlane, EAS Build.
- Structured output (JSON) so post-hook gates can act on build/test results.

### Phase D — Mobile QA
- detox + maestro adapters.
- Mobile visual review extends [04-build-visual-review.md](04-build-visual-review.md).
- Performance: Reactotron / Flipper integration for RN; Instruments / Android Profiler for native.

### Phase E — Distribution
- Screenshot generator (per-device, per-locale).
- Metadata + nutrition labels generator from spec.
- TestFlight / Play internal upload via Fastlane / Gradle.
- App-review checklist: privacy policy URL, support URL, content rating, in-app purchase flag, encryption export status.
- Rejection-handling playbook (most-common rejection reasons + fix templates).

## Human-in-loop pattern

| Step | Agent does | Founder does | Reversibility |
|---|---|---|---|
| target_platform pick | proposes from spec | confirms or overrides | high after sprint 0 |
| Stack pick (Expo/RN/Flutter/native) | recommends per platform + recipe inventory | picks | high after sprint 0 |
| Apple/Google enrollment | reminds + checklists | enrolls (legal entity, KYC, fees) | one-way |
| Bundle ID / package name | proposes from spec | picks | one-way after publish |
| Code signing setup | guides through Fastlane match | runs match init, controls keys | full |
| App store metadata | drafts | edits + approves | full pre-submit |
| Submit for review | uploads + initiates | clicks submit | reversible (withdraw) |
| Review-rejection response | drafts response based on rejection reason | reviews + sends | full |

## Dependencies

- **Inbound:** [02-build-foundation.md](02-build-foundation.md) — recipe library + mechanical gates. [03-build-review-density.md](03-build-review-density.md) — multi-pass reviews + multi-file expansion (mobile features have similar shape). [04-build-visual-review.md](04-build-visual-review.md) — visual-review harness extended to device mode. [06-real-world-bridge.md](06-real-world-bridge.md) — developer account onboarding pairs naturally; KYC + payment + identity are shared concerns.
- **Outbound:** [08-operations.md](08-operations.md) — mobile-specific monitoring (crash reporting per platform via Sentry mobile SDKs, Firebase Crashlytics).

## Open questions

- **Default cross-platform pick.** Expo (best DX, RN-based, hosted services nice) vs Flutter (single codebase, Dart, performance). (Default Expo unless spec demands platform-specific UI fidelity.)
- **Native recipe coverage.** Cover Swift+Kotlin natively or treat as advanced/manual? (Cross-platform first; native as a follow-up wave once cross-platform proves out.)
- **EAS vs self-hosted CI.** EAS is convenient but external dep + cost. Self-hosted (GitHub Actions + macOS runners) cheaper but more setup. (EAS for v1; pluggable so self-hosted comes later.)
- **Code signing automation depth.** Fastlane match works but requires GitHub repo + passphrase. (Use match; passphrase in vault from cross-cutting.)
- **App store screenshot generation.** Detox snapshots vs Fastlane snapshot vs design-tool exports. (Detox for accuracy; Fastlane for variants.)
- **Push notification provider.** Expo's hosted push vs raw APNs/FCM. (Expo hosted v1.)
- **Beta testing flow.** TestFlight + Play internal handle this natively; what's the agent's role? (Manage uploads + tester invitations; reads tester feedback; surfaces to founder.)

## Agent task brief

When picking up this doc:
1. Read 00-README + dependencies (02 + 03 + 04 + 06).
2. Phase 4 i2p_v3.json: design + add target_platform conditional group + branched phase-7 step variants. Test the branching with a fixture mission.
3. Phase B: pick recipe set, draft 6 mobile recipes following the recipe schema from 02.
4. Phase C: scaffold mr_roboto adapter verbs; structured output schema; tests against sample fixtures.
5. Phase D: detox + maestro adapters; mobile visual review extension.
6. Phase E: TestFlight + Play internal upload flows; metadata + screenshots + rejection-handling.
7. Resolve open questions or escalate.
8. Cross-reference outbound to [08-operations.md](08-operations.md).
9. Add `## Updates` entry.

## Updates

- 2026-05-08 — initial doc; absorbs Wave 8 + theme T12 from `docs/plans/2026-05-07-i2p-capability-expansion.md`.
