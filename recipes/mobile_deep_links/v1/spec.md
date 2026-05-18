# mobile_deep_links Recipe v1 — Specification

## Scope

Deep linking for an Expo / React Native app: custom URL scheme plus
verified universal links (iOS) and app links (Android). Ships the
`expo-router` linking configuration, the `app.json` / `app.config.ts`
scheme + associated-domains + intent-filter wiring, and the
`apple-app-site-association` body to host on the link domain.

## When to Pick This Recipe

**Use** `mobile_deep_links/v1` when:
- The app is built with Expo and routes with `expo-router`.
- Stack is `fastapi+sqlite+expo`, `fastapi+postgres+expo`, or `expo`.
- You need shareable `https://` links that open the app directly, plus a
  custom scheme for in-app / dev linking.

## What It Generates

| File | Role |
|------|------|
| `linking.template.ts` | `expo-router` linking config: prefixes, screen map, `buildShareLink` / `buildSchemeLink` / `parseDeepLink` |
| `app.config.template.ts` | Expo config fragment: `scheme`, `ios.associatedDomains`, `android.intentFilters` |
| `well_known/apple-app-site-association.template.json` | AASA body to host at `/.well-known/` on the link domain |
| `flows/deep_links_smoke.flow.yaml` | Maestro smoke flow (sign in → onboard → open deep link → sign out) |
| `tests/linking_smoke.template.ts` | Link-builder + prefix smoke tests |

## Mobile QA

`flows/deep_links_smoke.flow.yaml` is the input to the `mobile_smoke`
post-hook: it signs in, onboards, opens `myapp://item/42` via Maestro's
`openLink`, asserts the item screen rendered, and signs out. Declare
`post_hooks: ["mobile_smoke"]` + `maestro_flows:
["flows/deep_links_smoke.flow.yaml"]` on the mobile build step.

## Parameters

| Param | Default | Meaning |
|-------|---------|---------|
| `APP_SCHEME` | `myapp` | Custom URL scheme |
| `ASSOCIATED_DOMAIN` | `links.example.com` | Domain hosting AASA + assetlinks |
| `ANDROID_PACKAGE` | `com.example.app` | Android package name |
| `IOS_BUNDLE_ID` | `com.example.app` | iOS bundle identifier |
| `APPLE_TEAM_ID` | `ABCDE12345` | Apple Developer team id (AASA `appID` prefix) |
| `DEEP_LINK_PREFIX_PATH` | `/app` | Path prefix verified links live under |

## Out of Scope

- Hosting the `.well-known/` files (a backend/CDN task — recipe only ships
  the AASA body).
- `assetlinks.json` generation (needs the release signing fingerprint —
  produced at build time).
- Deferred deep linking / install attribution.
