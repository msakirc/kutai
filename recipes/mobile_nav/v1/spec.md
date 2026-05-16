# mobile_nav Recipe (v1)

## Scope

File-based navigation for an Expo / React Native app using **Expo Router v3**
(Expo SDK 51). Ships a complete navigation skeleton:

- A **root stack navigator** (`app/_layout.tsx`).
- A **bottom-tab navigator** nested inside the stack (`app/(tabs)/_layout.tsx`)
  with three example tabs: Home, Explore, Profile.
- A **dynamic route** (`app/detail/[id].tsx`) demonstrating typed params.
- A **404 catch-all** (`app/[...missing].tsx`).
- An **auth-gated route group** (`app/(auth)/`) — unauthenticated users are
  redirected to a sign-in screen by a single gate in the root layout.
- **Typed route helpers** (`lib/routes.ts`) and a documented **deep-link
  config** (`app/app.json` + `lib/linking.ts`).

This is a **frontend-only** recipe. It has no backend template — navigation is
pure client-side. Pair it with `mobile_auth` for real credential handling and
`mobile_persistence` for local storage.

## When to pick

- Any Expo app that needs more than a single screen.
- Apps with a bottom-tab main surface plus a separate sign-in flow.
- Stack ∈ `fastapi+sqlite+expo`, `fastapi+postgres+expo`, or bare `expo`.

## When NOT to pick

- A single-screen utility app (no navigator needed).
- A web-only project → use the web frontend recipes (Next.js App Router).
- A bare React Native project not using Expo — Expo Router requires the Expo
  runtime and `expo-router/entry`.
- Apps that need a drawer + tabs + stack with deeply custom transitions —
  start from this recipe but expect to extend the `_layout.tsx` files.

## Shape

```
app/
  _layout.tsx            Root Stack + auth gate (useProtectedRoute)
  (tabs)/
    _layout.tsx          Bottom-tab navigator
    index.tsx            Home tab        -> /
    explore.tsx          Explore tab     -> /explore
    profile.tsx          Profile tab     -> /profile
  (auth)/
    _layout.tsx          Auth Stack (no tab bar)
    sign-in.tsx          Sign-in screen  -> /sign-in
  detail/
    [id].tsx             Dynamic route   -> /detail/:id
  [...missing].tsx       404 catch-all
  app.json               Expo config: scheme, typedRoutes, deep-link domains
lib/
  routes.ts              Typed route helpers (rename-safe dynamic routes)
  auth.ts                Session context stub (replace with mobile_auth)
  linking.ts             Deep-link prefixes + URL surface map
package.json             expo-router + safe-area + screens deps
```

## Navigation model

- **Root stack** owns three children: the `(tabs)` group, the `(auth)` group
  (presented modally), and the `[...missing]` catch-all.
- **Tabs** are nested inside the stack — the tab bar shows only while a
  `(tabs)` route is active; the sign-in screen and detail screen render
  full-screen above it.
- **Route groups** `(tabs)` / `(auth)` add no path segment. `(tabs)/index.tsx`
  is `/`; `(auth)/sign-in.tsx` is `/sign-in`.

## Auth gate

All redirect logic is in `app/_layout.tsx::useProtectedRoute`:

1. Wait for `isReady` (persisted session loaded) — avoids a sign-in flash on
   cold start.
2. Read `useSegments()`; `inAuthGroup = segments[0] === "(auth)"`.
3. Unauthenticated + not in `(auth)` → `router.replace(SIGN_IN_ROUTE)`.
4. Authenticated + in `(auth)` → `router.replace('/(tabs)')`.

Screens never navigate to `/sign-in` themselves — centralising the gate avoids
double-redirect races.

## Deep linking

- Custom scheme from `app.json` `scheme` → `myapp://detail/42`.
- Universal/App Links need `ios.associatedDomains` + `android.intentFilters`
  (present in `app.json`) plus hosted `apple-app-site-association` /
  `assetlinks.json` files (deployment, not code).
- The route tree is the linking config — no explicit `linking` object.

## Typed routes

`experiments.typedRoutes: true` in `app.json` generates route type
declarations. `lib/routes.ts` wraps dynamic routes in helper functions so a
renamed `[id]` segment is fixed in one place.

## RECIPE_PARAM markers

| Marker | Default | Description |
|--------|---------|-------------|
| `APP_NAME` | `MyApp` | App display name (app.json, sign-in screen) |
| `APP_SCHEME` | `myapp` | Custom URL scheme + package slug |
| `SIGN_IN_ROUTE` | `/(auth)/sign-in` | Route the auth gate redirects to |
| `HOME_TAB_TITLE` | `Home` | Title of the first tab |
| `EXPLORE_TAB_TITLE` | `Explore` | Title of the second tab |
| `PROFILE_TAB_TITLE` | `Profile` | Title of the third tab |
| `INITIAL_TAB` | `index` | `name` of the tab shown first |
| `DEEP_LINK_HOST` | `app` | Subdomain for universal-link domains |

## Dependencies

**Frontend (npm)**:
- `expo`, `expo-router` — file-based navigation runtime
- `react-native-safe-area-context` — required peer dep of Expo Router
- `react-native-screens` — required peer dep (native screen primitives)

Backend: none — this is a frontend-only recipe.

## Post-hooks

`imports_check`, `test_run`, `pattern_lint` — run after instantiation against
the generated `app/` tree.

## Known non-goals (v1)

- No drawer navigator (extend `_layout.tsx` if needed).
- No real authentication — `lib/auth.ts` is a stub; use `mobile_auth`.
- No screen-transition / shared-element animation customisation.
- No nested-tab or master-detail layouts beyond the single example.
- No persisted navigation state across cold starts.
