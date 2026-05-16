/**
 * Typed route helpers.
 *
 * Centralises every navigable destination so screens never hand-write path
 * strings. Each helper returns an `Href`-shaped object that `router.push` /
 * `router.replace` / `<Link href>` accept.
 *
 * Why a helper file AND `experiments.typedRoutes`?
 *  - `typedRoutes` (app.json) gives compile-time checking of literal `href`
 *    strings and the generated `.expo/types/router.d.ts` declarations.
 *  - This file gives a single rename-safe surface for DYNAMIC routes — when a
 *    `[id]` segment is renamed you fix one function, not every call site.
 *
 * Keep the pathnames in sync with the files under `app/`. The `pathname`
 * values use the route GROUP-stripped form (`(tabs)` / `(auth)` add no
 * segment, so they never appear here).
 */

/** Minimal Href shape — mirrors expo-router's `Href` without importing it,
 *  so this file type-checks even before `expo-router` is installed. */
export type RouteHref =
  | string
  | { pathname: string; params?: Record<string, string> };

export const routes = {
  /** Home tab — `/`. */
  home: (): RouteHref => "/",

  /** Explore tab — `/explore`. */
  explore: (): RouteHref => "/explore",

  /** Profile tab — `/profile`. */
  profile: (): RouteHref => "/profile",

  /** Sign-in screen inside the `(auth)` group — `/sign-in`. */
  signIn: (): RouteHref => "/sign-in",

  /** Dynamic detail route — `/detail/:id`. */
  detail: (id: string): RouteHref => ({
    pathname: "/detail/[id]",
    params: { id },
  }),
} as const;

export type RouteName = keyof typeof routes;
