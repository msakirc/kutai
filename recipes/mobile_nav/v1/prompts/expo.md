# Expo — mobile_nav/v1 instantiation notes

Recipe-specific guidance for wiring file-based navigation with **Expo Router
v3** (Expo SDK 51). Complements the generic `expo` block in
`STACK_BLOCKS` — read both.

## Why Expo Router, not React Navigation directly

Expo Router is a thin file-based layer ON TOP of React Navigation. You get the
same `@react-navigation` stack/tab navigators, but routes are derived from the
`app/` directory instead of hand-registered in a `createStackNavigator` call.
Benefits this recipe relies on: deep links map to files for free, typed routes
are generated, and `(group)` folders organise without a config object. Do NOT
add `@react-navigation/native` navigators by hand alongside Expo Router — you
end up with two competing navigation trees.

## File-based routing rules

- **`app/` is the routing root.** Every `.tsx` file under `app/` is a route.
  The project entry is `expo-router/entry` (set in `package.json` `main`).
- **`_layout.tsx` per directory.** A directory that owns a navigator MUST have
  a `_layout.tsx`. `app/_layout.tsx` is the root navigator; `(tabs)/_layout.tsx`
  is the bottom-tab navigator nested inside it.
- **`index.tsx` = directory root.** `(tabs)/index.tsx` is `/`, not
  `/(tabs)/index`.
- **Dynamic segments** use `[param].tsx` → `/detail/:id`. Spread/catch-all uses
  `[...name].tsx` — this recipe uses `[...missing].tsx` for the 404 screen.

## Route groups `(name)`

Parenthesised folders — `(tabs)`, `(auth)` — are **organisational only**. They
add NO path segment. `(auth)/sign-in.tsx` is reachable at `/sign-in`. Use them
to give a section its own `_layout.tsx` (tab bar vs full-screen stack) without
polluting the URL. Never reference `(tabs)` / `(auth)` in a user-facing path
string or a deep link.

## `<Stack.Screen options>` and `<Tabs.Screen options>`

- Configure a route from its parent `_layout.tsx`: `<Stack.Screen name="..."
  options={{ title, headerShown, presentation }} />`.
- Or configure a route from INSIDE itself by rendering `<Stack.Screen
  options={{...}} />` in the component body — useful for dynamic titles built
  from `useLocalSearchParams` (see `app/detail/[id].tsx`).
- `<Tabs.Screen name>` must match a sibling filename; a mismatch silently
  renders nothing.

## Navigation APIs

- **`useRouter()`** — imperative: `router.push(href)`, `router.replace(href)`,
  `router.back()`. Use after async work or in callbacks.
- **`<Link href>`** — declarative: renders an accessible pressable. Prefer for
  static destinations; supports `asChild` to wrap a custom component.
- **`useLocalSearchParams()`** — read `[param]` values, scoped to the current
  screen. Prefer over `useGlobalSearchParams()` (which updates on ANY route
  change and causes extra re-renders).
- React Native has no SSR, so `Link` and `router.push` behave identically at
  runtime — `Link` is purely an ergonomics/accessibility choice, not a
  hydration one.

## Typed routes

Set `experiments.typedRoutes: true` in `app.json`. Expo generates
`.expo/types/router.d.ts` so `href` strings and `[param]` shapes are
compile-time checked. Run `npx expo customize tsconfig.json` once so the
generated types are picked up. The `lib/routes.ts` helper centralises dynamic
routes so a renamed `[id]` segment is a one-line fix.

## Auth-gated route group

The `(auth)` group holds the sign-in flow. The gate logic lives ONCE in
`app/_layout.tsx` (`useProtectedRoute`): it reads `useSegments()`, and if the
user is unauthenticated and not already in `(auth)` it calls
`router.replace('/(auth)/sign-in')`. Critical: run the redirect inside a
`useEffect`, never during render — navigating mid-render throws "Attempted to
navigate before mounting the Root Layout". Gate on an `isReady` flag so the
sign-in screen doesn't flash before the persisted session loads.

## Deep linking

The route tree IS the linking config — no `linking` object needed. Set
`scheme` in `app.json` for the custom scheme (`myapp://detail/42`). Universal
Links / App Links additionally need `ios.associatedDomains` +
`android.intentFilters` in `app.json` AND a hosted
`apple-app-site-association` / `assetlinks.json` — those are deployment files,
not code. `lib/linking.ts` documents the resulting URL surface.
