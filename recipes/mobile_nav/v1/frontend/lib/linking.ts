/**
 * Deep-link configuration.
 *
 * With Expo Router, the route tree IS the linking config — `app/` file paths
 * map directly to URL paths, so you rarely need an explicit `linking` object.
 * This file documents the resulting URL surface and exposes the prefixes for
 * any code that needs them (analytics, share-link generation, tests).
 *
 * The custom scheme (`<<APP_SCHEME>>://`) comes from `app.json`'s `scheme`
 * field. Universal/App Links (`https://<<DEEP_LINK_HOST>>.example.com/...`)
 * require the `associatedDomains` (iOS) + `intentFilters` (Android) entries
 * in `app.json` AND a hosted `apple-app-site-association` /
 * `assetlinks.json` file — those are a deployment concern, not code.
 *
 * RECIPE_PARAM markers:
 *   // RECIPE_PARAM:APP_SCHEME=myapp
 *   // RECIPE_PARAM:DEEP_LINK_HOST=app
 */

// RECIPE_PARAM:APP_SCHEME=myapp
const APP_SCHEME = "<<APP_SCHEME>>";
// RECIPE_PARAM:DEEP_LINK_HOST=app
const DEEP_LINK_HOST = "<<DEEP_LINK_HOST>>";

/** URL prefixes that resolve into this app. */
export const linkingPrefixes = [
  `${APP_SCHEME}://`,
  `https://${DEEP_LINK_HOST}.example.com`,
] as const;

/**
 * Reference map of deep-link path -> screen file.
 * Expo Router derives this automatically; kept here for documentation and
 * for tests that assert the public URL surface.
 */
export const deepLinkMap = {
  "/": "app/(tabs)/index.tsx",
  "/explore": "app/(tabs)/explore.tsx",
  "/profile": "app/(tabs)/profile.tsx",
  "/sign-in": "app/(auth)/sign-in.tsx",
  "/detail/:id": "app/detail/[id].tsx",
} as const;
