# mobile_nav Recipe v1 — Known Lessons

Expo Router pitfalls captured from prior implementations. Seeded into
`mission_lessons` (domain `mobile_nav`) on recipe instantiation.

- **`app/` must be the routing root.** Expo Router only discovers routes under
  the `app/` directory, and the project `main` in `package.json` must be
  `expo-router/entry`. Put screens under `src/screens/` and nothing renders —
  no error, just a blank app.

- **Every navigator directory needs a `_layout.tsx`.** A directory under `app/`
  with screen files but no `_layout.tsx` has no navigator — its routes exist
  but cannot be presented. `app/_layout.tsx` (root) and `(tabs)/_layout.tsx`
  are both mandatory.

- **Route groups `(name)` add NO path segment.** `(tabs)/index.tsx` is `/`,
  not `/(tabs)/index`; `(auth)/sign-in.tsx` is `/sign-in`. Writing
  `router.push('/tabs/profile')` 404s — the group folder is invisible to the
  URL. Only reference the group in `router.replace('/(tabs)')`-style internal
  navigation where the group root itself is the target.

- **Never navigate during render.** Calling `router.replace()` in a component
  body (or directly in a `useSegments`-driven branch) throws "Attempted to
  navigate before mounting the Root Layout". Always redirect inside a
  `useEffect`. The recipe's `useProtectedRoute` does this correctly.

- **Gate on an `isReady` flag.** If the auth gate redirects before the
  persisted session has loaded, every cold start flashes the sign-in screen
  for a frame. Hold the redirect until session state is hydrated.

- **`index.tsx` is the directory root, not a literal `/index` path.** Both
  `app/index.tsx` and `app/(tabs)/index.tsx` resolve to `/`. Two `index` files
  resolving to the same path is a routing conflict — keep only one root index.

- **Typed routes need `experiments.typedRoutes: true` in `app.json`.** Without
  it, `href` strings are plain `string` and typos compile fine. After enabling,
  run `npx expo start` once to regenerate `.expo/types/router.d.ts`; stale
  generated types cause false type errors.

- **`<Tabs.Screen name>` must match a sibling filename.** A `name` with no
  matching file silently renders an empty tab; a file with no `Tabs.Screen`
  entry still appears as a tab using its filename. Keep the `_layout.tsx`
  `Tabs.Screen` list and the actual files in sync.

- **Use `useLocalSearchParams`, not `useGlobalSearchParams`.** Local params are
  scoped to the current screen and stable; global params update on ANY route
  change, causing unexpected re-renders and stale-closure bugs in screens that
  aren't even focused.

- **Add a `[...missing].tsx` catch-all.** Without an explicit catch-all route,
  an unknown deep link (or a typo'd `href`) renders a blank screen with no
  feedback. The spread filename `[...name].tsx` matches any unmatched path.

- **`Link` vs `router.push` is ergonomics, not behaviour.** React Native has no
  SSR, so both navigate identically at runtime. Use `<Link>` for static
  destinations (accessible pressable, `asChild` support) and `router.push` for
  imperative navigation after async work. Do not expect `Link` to "prefetch"
  like Next.js — that is a web-only concept.

- **Deep links: custom scheme is free, universal links are not.** Setting
  `scheme` in `app.json` makes `myapp://...` work immediately. App Links /
  Universal Links additionally require `associatedDomains` (iOS) +
  `intentFilters` (Android) in `app.json` AND a server-hosted
  `apple-app-site-association` / `assetlinks.json` — forgetting the hosted
  files means the link opens the browser instead of the app.

- **`react-native-screens` and `react-native-safe-area-context` are required
  peers.** Expo Router will not render correctly without them installed and at
  Expo-SDK-pinned versions. Install via `npx expo install` (not raw `npm
  install`) so versions match the SDK.
