/**
 * Root layout — Expo Router file-based navigation.
 *
 * This file MUST live at `app/_layout.tsx`. Expo Router treats `app/` as the
 * routing root; `_layout.tsx` here is the top of the navigation tree and wraps
 * every route in the project. Without it there is no navigator at all.
 *
 * Tree shape produced by this recipe:
 *
 *   app/
 *     _layout.tsx            <- this file (root Stack)
 *     (auth)/
 *       _layout.tsx          <- auth route group (Stack)
 *       sign-in.tsx          <- /sign-in  (group ()  adds no path segment)
 *     (tabs)/
 *       _layout.tsx          <- bottom-tab navigator
 *       index.tsx            <- / (home tab)
 *       explore.tsx          <- /explore
 *       profile.tsx          <- /profile
 *     [...missing].tsx       <- 404 catch-all
 *
 * RECIPE_PARAM markers:
 *   // RECIPE_PARAM:APP_NAME=MyApp
 *   // RECIPE_PARAM:SIGN_IN_ROUTE=/(auth)/sign-in
 *
 * The substitution engine fills the parameter tokens before this file lands
 * in the mission workspace. RECIPE_PARAM comment lines are left intact on
 * purpose — they document the knobs and are valid TS comments.
 */
import { useEffect } from "react";
import { Stack, useRouter, useSegments } from "expo-router";
import { SafeAreaProvider } from "react-native-safe-area-context";

import { useSession } from "../lib/auth";

// RECIPE_PARAM:SIGN_IN_ROUTE=/(auth)/sign-in
const SIGN_IN_ROUTE = "<<SIGN_IN_ROUTE>>";

/**
 * useProtectedRoute — auth gate.
 *
 * Redirects unauthenticated users into the `(auth)` group, and authenticated
 * users out of it. Runs as an effect (NOT during render) because navigating
 * mid-render throws "Attempted to navigate before mounting the Root Layout".
 */
function useProtectedRoute(isSignedIn: boolean, isReady: boolean) {
  const segments = useSegments();
  const router = useRouter();

  useEffect(() => {
    if (!isReady) return; // wait until the layout has mounted + session loaded

    const inAuthGroup = segments[0] === "(auth)";

    if (!isSignedIn && !inAuthGroup) {
      // Not signed in and trying to view a protected route -> bounce to sign-in.
      router.replace(SIGN_IN_ROUTE as never);
    } else if (isSignedIn && inAuthGroup) {
      // Already signed in but sitting on the sign-in screen -> go to the app.
      router.replace("/(tabs)" as never);
    }
  }, [isSignedIn, isReady, segments, router]);
}

export default function RootLayout() {
  const { isSignedIn, isReady } = useSession();

  useProtectedRoute(isSignedIn, isReady);

  return (
    <SafeAreaProvider>
      <Stack screenOptions={{ headerShown: false }}>
        {/* Route GROUPS — parentheses are organisational only, they add NO
            path segment. `(tabs)` is reachable at `/`, not `/tabs`. */}
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen
          name="(auth)"
          options={{ headerShown: false, presentation: "modal" }}
        />
        <Stack.Screen
          name="[...missing]"
          options={{ title: "Not Found" }}
        />
      </Stack>
    </SafeAreaProvider>
  );
}
