# Z5 — Mobile track (v2)

**Supersedes** [05-build-mobile-track.md](05-build-mobile-track.md) (v1, 2026-05-08).
v1 was a pre-Z6 surface sketch. This v2 is the deep-dive: re-audited against the
current codebase, corrects 6 stale claims, reshapes the gap map around a
constraint v1 never named (host OS), and ends with a 5-tier batched
implementation plan.

**Audited 2026-05-16** against: `src/workflows/i2p/i2p_v3.json`,
`packages/mr_roboto/`, `src/infra/recipes.py` + `recipes/`,
`packages/coulson/.../reflection.py`, `src/integrations/adapters/`,
`src/founder_actions/`, `src/security/credential_store.py`.

---

## Founder decisions (2026-05-16)

Locked at kick-off by the founder (these were v1 "open questions" — Z5 must
not default them silently):

1. **Framework: Expo / React Native.** Cross-platform only for v1. Native
   Swift+Kotlin recipes deferred to a follow-up wave.
2. **Build infra: GitHub Actions macOS runners**, not EAS cloud. iOS builds
   on a free GH-hosted macOS runner via `xcodebuild` + Fastlane; Android on a
   Linux runner via `gradlew` (or locally on the Windows box). Generated
   GitHub Actions workflows are the primary CI surface.
3. **Cost: free-first.** EAS Build/Submit is acceptable only as an optional
   fallback, never the default — more setup is fine if it stays free. The
   `eas_build`/`eas_submit` adapters shipped in T3 remain, demoted to fallback.

These decisions reshape T3 (add a GH Actions / Fastlane build path) and T5
(submit via Fastlane `pilot`/`supply`, not `eas submit`).

## Stale-claim corrections (what v1 got wrong)

| # | v1 claim | Reality (2026-05-16) |
|---|---|---|
| S1 | "No mobile tooling adapters; no mobile visual review" | Partially **anticipated**. `capture_screenshots` ships an explicit `capture_mode="device"` Z5 placeholder that soft-skips today. `STACK_BLOCKS` (coulson reflection) already has a full `expo` block. Z5 *fills hooks*, not greenfield. |
| S2 | "App-store submission deeply intertwined with 06-real-world-bridge" — framed as future work | **Z6 already shipped the hard half.** `src/integrations/adapters/apple_jwt.py` (ES256 App Store Connect JWT), `google_sa.py` (OAuth service-account), `credential_schemas/apple_appstore.json` + `google_play.json` with enrollment notes, and `z6_admission.py` which **auto-emits a `founder_action(kind=vendor_enroll)`** when a step needs `apple_appstore`/`google_play` and no credential is stored. Distribution = *use* `vendor_call`, not build auth. |
| S3 | "New conditional group `target_platform`" | The conditional-group mechanism already exists (`metadata.conditional_groups`, with `fallback_steps` for alternate definitions). A `mobile_app_submission` group already exists. **No schema work needed** — but see S4. |
| S4 | (silent) | **`mobile_app_submission` is mis-wired.** Its `if_true: ["14.10"]` points at a step that is now `arm_analytics_digest_cron` — Z9 reused the `14.10` id. The group is dangling. Z5 must repoint it. |
| S5 | "branch ∈ {web, mobile_native, mobile_cross_platform, both}" — 4-way | **Host OS = Windows (`win32`).** `xcrun simctl`, Xcode, iOS Simulator are macOS-only and cannot run on this box. `mobile_native` (Swift+Kotlin) is therefore doubly out of scope for v1: un-buildable locally *and* it doubles the recipe surface. v1 ships **Expo cross-platform only**. Effective values: `web` (existing) / `mobile` / `both`. |
| S6 | "EAS for v1; pluggable so self-hosted comes later" | iOS cannot build on the Windows dev box. Two cloud paths exist: EAS Build (paid past a small free tier) or **GitHub Actions macOS runners** (free tier for the workload, more setup). Founder picked the free-first GH Actions path (see Founder decisions). Android (`adb`, `gradle`) runs locally on Windows. |

---

## Current state (re-audited, precise)

**Workflow JSON** — `i2p_v3.json`: 17 phases, 275 steps.
- `platform_requirements` artifact (produced ~step 3.x) already has fields
  `primary_platform, browser_support, mobile_support, screen_sizes, os_versions,
  device_types`. `condition_check` helper `platforms_include('ios'|'android')`
  reads it. So platform *detection* is partly modelled; platform *branching of
  the build* is not.
- Step `4.2` `tech_stack_research_and_selection` reads `platform_requirements`,
  emits `tech_stack_decision`. No `target_platform` field today.
- Phase-7 frontend: `7.5 frontend_scaffold` (+ `7.5.git_commit`) — web-shaped,
  produces `frontend/package.json`, reads `tech_stack_decision`.
- Feature template (`feature_implementation_template`, embedded lines
  ~12073–12592): `feat.7 frontend_state`, `feat.8 frontend_components`,
  `feat.9 frontend_pages`, `feat.10 frontend_tests` — all emit `.tsx` under
  `frontend/src/{types,api,state,components,app}/`, Next.js App Router shape.
  Template steps gate on a `condition` string, not conditional groups.
- `conditional_groups` use `group_id, condition_artifact, condition_check,
  if_true, if_false, fallback_steps[]` — `fallback_steps` carries full step
  defs for the alternate path. **This is the branching primitive Z5 reuses.**

**Recipes** — `src/infra/recipes.py` + `recipes/` (24 recipes today).
- `Recipe` schema: `name, version, requires{tech_stack[],features[]},
  post_hooks[], templates{}, prompts{stack→md}, lessons_domain,
  dependencies{}, param_defaults{}, entry_points{}`.
- `match_recipe(stack, recipes)` parses stacks on `+` → exact 1.0 / superset
  0.7 / 50%-overlap 0.4. **A mobile recipe just needs `tech_stack` tokens like
  `fastapi+sqlite+expo`. Zero schema change.**
- Templates use `RECIPE_PARAM:KEY=default` markers + `<<KEY>>` substitution.
- `lessons.md` per recipe → seeded into `mission_lessons` table on
  instantiation (`lessons_domain` is the key).
- `instantiate_picked_recipes` mr_roboto verb already exists.

**Post-hook registry** — `general_beckman/posthooks.py`, 45 kinds, data-driven
`POST_HOOK_REGISTRY` dict. New kind = registry entry + handler +
`apply.py` dispatch. `auto_wire_triggers` may be a glob list or a dial-aware
callable; the expander auto-attaches hooks by matching `produces` globs.

**mr_roboto** — 123 verbs in one `_run_dispatch` `if action ==` chain in
`__init__.py`. `run_cmd.py` is the subprocess primitive: asyncio,
`timeout_s`, **returns `{ok:False, error:"executable not found"}` on missing
exe** (free soft-skip), tail-decoded stdout/stderr. `run_semgrep.py` is the
canonical "wrap a CLI, return structured JSON, gate on exit code" example.
`Action(status, result, error, reversibility)` is the return type;
verdict-style verbs return `status="completed"` with the verdict inside
`result`.

**Visual review (Z4)** — `capture_screenshots.py`: Playwright headless
Chromium, 4 breakpoints × 2 color modes, determinism injections (frozen
`Date`, seeded `Math.random`, animations off). `capture_mode="device"`
**already stubbed for Z5** — soft-skips with reason. `visual_review.py`:
vision-model DIFF/AUDIT, per-mission + cross-mission baselines keyed by
`token_hash`, severity rules → `verdict`.

**Z6 real-world bridge** — `apple_appstore` adapter actions today:
`list_apps, list_builds, list_app_store_versions, submit_for_review,
list_review_states`. `google_play`: `list_apps, list_release_tracks,
upload_apk_metadata, list_review_status`. **Gap: no binary-upload action** —
TestFlight/Play-internal upload needs Transporter/`altool` (macOS) or **EAS
Submit** (cloud). EAS Submit is the answer. `founder_actions` +
`credential_store` (Fernet vault) fully handle enrollment + key paste.

---

## Reshaped gap map

### Closeable by automation

**G1 — Platform branching rails.** `4.2` emits `target_platform`; a new
`frontend_platform` conditional group reroutes `7.5` → Expo variant; feature
template gets platform-aware frontend variants. Repoint the dead
`mobile_app_submission` group. *Reuses existing primitives; no JSON schema
change.*

**G2 — Expo recipes.** 3 core (`mobile_auth`, `mobile_nav`,
`mobile_persistence`) + 3 stretch (`mobile_push`, `mobile_deep_links`,
`mobile_offline_sync`). Each: `recipe.yaml` with `tech_stack:
[fastapi+sqlite+expo, fastapi+postgres+expo]`, `prompts/expo.md`, templates,
`tests/`, `lessons.md`. *Recipe infra 100% reused.*

**G3 — mr_roboto mobile adapters.** `expo_cli` (prebuild/export/doctor),
`eas_build` (cloud iOS+Android), `eas_submit` (TestFlight + Play internal),
`android_build` (local `gradle`/`adb`, Windows-ok). All thin `run_cmd`
wrappers → structured JSON; iOS routes through EAS only; missing-CLI
soft-skips for free.

**G4 — Mobile QA + device visual review.** Implement `capture_mode="device"`
as: (a) **Playwright device descriptors** (`iPhone 14`, `Pixel 7` presets ship
with Playwright, run headless on Windows) against the **Expo Web** export —
the realistic v1; (b) `adb exec-out screencap` for real Android; (c) `xcrun
simctl io` deferred to a future macOS runner. Add `maestro` adapter
(YAML flows, cross-platform, cheaper to author than Detox).

**G5 — Distribution.** New real submission step (replacing the dangling
`14.10` wiring) that: generates per-device/per-locale screenshots (reuses G4
capture), generates metadata + privacy-nutrition-labels from the spec, calls
`eas_submit`, and uses `vendor_call(apple_appstore|google_play)` for
review-status polling. Enrollment founder-actions already auto-emit via
`z6_admission`. Ship a `mobile_release_rejection` playbook recipe.

### Founder territory (unchanged from v1, confirmed)

Apple Developer Program enrollment ($99/yr, KYC) · Google Play Console
($25, KYC) · bundle-id / package-name choice (one-way post-publish) · App
Store Connect tax forms · privacy-nutrition-label approval · review-rejection
final responses · code-signing key custody.

These are already surfaced as `founder_action` cards by Z6 — Z5 verifies the
wiring fires for mobile steps, it does not re-build it.

---

## Implementation plan — 5 tiers

**Canonical-first rule (per `feedback_canonical_first_for_tier3plus`):** T1
lands to `main` before T2+ agents are dispatched — T2–T5 all depend on the
`target_platform` field and the Expo variant ids.

### T1 — Platform foundation + branching rails *(do first, land to main)*

1. Add `target_platform ∈ {web, mobile, both}` to the `platform_requirements`
   artifact schema (`required_fields`, `_schema_version` bump) and extend
   step `4.2` instruction to emit it from `mobile_support`.
2. Fix `mobile_app_submission` conditional group: repoint `if_true` off the
   dead `14.10` onto the new submission step id from T5 (placeholder
   `14.x_mobile_submit` reserved now, defined in T5).
3. Add `frontend_platform` conditional group: `condition_check`
   `target_platform in ('mobile','both')`; `if_true` → Expo scaffold step
   `7.5m`; `fallback_steps` carries the full `7.5m` def (Expo Router scaffold,
   produces `app/package.json` + `app.json`).
4. Feature-template frontend variants: add platform-conditioned variants of
   `feat.7–10` (Expo: `app/`, `StyleSheet`/NativeWind, `expo-router`). Template
   steps can't use conditional groups → expander selects the variant by
   `mission.target_platform`. Add the selection in `workflows/engine/expander.py`.
5. Fixture mission test: `target_platform=mobile` routes to `7.5m`;
   `=web` routes to `7.5`; `=both` runs both.

*Effort: ~1.5d. Acceptance: branching test green; `python -c "import json;
json.load(open('src/workflows/i2p/i2p_v3.json'))"` passes; no orphan step ids.*

### T2 — Expo recipes (3 core) *(parallel-safe: 3 agents)*

One agent per recipe: `mobile_auth` (Sign in with Apple + Google + email,
`expo-secure-store` keychain/keystore), `mobile_nav` (`expo-router` stack+tab),
`mobile_persistence` (`expo-sqlite` + Drizzle). Each follows the `recipe.yaml`
schema, `tech_stack: [fastapi+sqlite+expo, fastapi+postgres+expo]`,
`prompts/expo.md`, `tests/`, `lessons.md` (`lessons_domain` set),
`post_hooks: [imports_check, test_run, pattern_lint]`.

*Effort: ~2d. Acceptance: `match_recipe("fastapi+sqlite+expo", …)` returns each
at 1.0; `instantiate_recipe` round-trips into a temp dir; lessons seed into
`mission_lessons`.*

### T3 — mr_roboto mobile adapters ✅ *(shipped 2026-05-16)*

`expo_cli`, `android_build`, `eas_build`, `eas_submit` — `run_cmd`-based,
structured JSON, `VERB_REVERSIBILITY` (`eas_submit` → `irreversible`, builds
→ `full`), 24 tests. `eas_*` are now the **fallback** path per the founder
decision.

### T3b — GitHub Actions mobile CI + Fastlane *(addendum, free-first path)*

Added after the founder picked GH Actions over EAS:
1. `gen_mobile_ci` mr_roboto verb — generates `.github/workflows/mobile.yml`:
   a macOS-runner job (`xcodebuild` + Fastlane for iOS) and a Linux/macOS job
   (`gradlew` for Android), keyed to the Expo `prebuild` output.
2. `fastlane` mr_roboto verb — wraps `fastlane` lanes (`build`, `match` for
   signing, `pilot` for TestFlight, `supply` for Play internal). `run_cmd`-
   based, soft-skip when absent. Reversibility: `match`/`build` → `full`,
   `pilot`/`supply` → `irreversible`.
3. `mobile_ci` recipe — ships the Fastlane `Fastfile` + `Appfile` templates,
   the workflow YAML, and `lessons.md` (free-tier minute limits, macOS-runner
   caching, `match` keychain on CI, code-signing secrets via GH Secrets).

*Effort: ~1.5d. Acceptance: generated workflow YAML is valid; `fastlane`
verb dispatches + soft-skips; `pilot`/`supply` reversibility = `irreversible`.*

### T4 — Mobile QA + device visual review *(parallel-safe: 2 agents)*

Agent A: implement `capture_mode="device"` in `capture_screenshots.py` —
Playwright device descriptors against Expo Web export; `adb screencap` arm;
`xcrun` arm raises a clear "macOS runner required" soft-skip. Extend
`visual_review` baselines to device frames.
Agent B: `maestro` adapter + `mobile_smoke` post-hook kind (registry entry +
handler) running a recipe-driven YAML flow (sign in → onboard → core action →
sign out). Stretch: recipes 4–6 (`mobile_push`, `mobile_deep_links`,
`mobile_offline_sync`).

*Effort: ~2.5d. Acceptance: device capture produces frames on Windows via
Playwright presets; `mobile_smoke` gates on Maestro exit code.*

### T5 — Distribution

1. New step `14.x_mobile_submit` (real id assigned at impl): generates
   per-device/per-locale screenshots (reuses T4 capture), metadata +
   privacy-nutrition-labels from spec artifacts, triggers the generated GH
   Actions workflow which runs `fastlane pilot` (TestFlight) / `fastlane
   supply` (Play internal); `eas_submit` is the fallback. Polls review status
   via `vendor_call(apple_appstore|google_play)`.
2. Repoint `mobile_app_submission.if_true` onto it (closes the T1 placeholder).
3. `mobile_release_rejection` playbook recipe — common rejection reasons +
   fix templates, modelled on the `incident_playbook_*` recipes.
4. Verify `z6_admission` emits the `vendor_enroll` founder-action when the
   submit step runs without `apple_appstore`/`google_play` credentials.

*Effort: ~2d. Acceptance: submit step dry-runs end-to-end with a fake
`eas_submit`; missing credential → founder-action card; rejection recipe
matches + instantiates.*

---

## Human-in-loop pattern

| Step | Agent does | Founder does | Reversibility |
|---|---|---|---|
| `target_platform` pick | proposes from `platform_requirements` | confirms/overrides | high pre-sprint-0 |
| Expo vs (future) native | recommends Expo; native escalated, not auto | n/a for v1 | — |
| Apple/Google enrollment | `z6_admission` auto-emits `founder_action` card | enrolls (KYC, fees) | one-way |
| Bundle id / package name | proposes from spec | picks | one-way post-publish |
| Code-signing keys | guides EAS credentials flow | controls keys, pastes to vault | full |
| Store metadata + screenshots | generates draft | edits + approves | full pre-submit |
| Submit for review | `eas_submit` uploads | clicks submit / acks cost | reversible (withdraw) |
| Rejection response | drafts from rejection reason + playbook | reviews + sends | full |

## Open questions — resolved

Framework / build-infra / cost were founder calls — see **Founder decisions
(2026-05-16)** above. Remaining agent-level defaults:

- **iOS device screenshots:** Playwright device descriptors against the Expo
  Web export for v1 (runs on Windows); real-device iOS screenshots run on the
  GH Actions macOS runner as a follow-up.
- **Push provider:** Expo hosted push for v1 (recipe `mobile_push`, stretch).
- **Beta testing:** agent manages Fastlane `pilot` uploads + tester invites;
  reads feedback, surfaces to founder.

## Dependencies

- **Inbound:** Z2 (recipe library + post-hook registry) ✅, Z3 (multi-pass
  review) ✅, Z4 (visual-review harness + `capture_mode` hook) ✅, Z6
  (Apple/Google adapters + vault + `founder_actions`) ✅. All shipped — Z5 is
  unblocked.
- **Outbound:** Z8/operations — mobile crash reporting (Sentry mobile SDK /
  Firebase Crashlytics) is a follow-up, noted not built.

## Updates

- 2026-05-08 — v1 initial doc (pre-Z6 surface sketch).
- 2026-05-16 — v2: full re-audit; 6 stale claims corrected (notably the
  Windows host-OS constraint and the Z6 distribution overlap); gap map
  reshaped; 5-tier batched plan added. Supersedes v1.
- 2026-05-16 — founder decisions locked (Expo / GitHub Actions macOS runners /
  free-first). T3 marked shipped; T3b GH-Actions+Fastlane addendum added;
  T5 submit path switched from `eas_submit` to Fastlane `pilot`/`supply`.
  T1+T2+T3 shipped to `main` (9 commits).
