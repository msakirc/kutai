/**
 * Auth route group layout — `app/(auth)/_layout.tsx`.
 *
 * The `(auth)` parenthesised group keeps the sign-in flow visually separate
 * from the tab app WITHOUT adding a `/auth` path segment — `sign-in.tsx`
 * here is reachable at `/sign-in`.
 *
 * This layout is a plain Stack with no tab bar, so the auth screens render
 * full-screen. The actual "is the user allowed here" decision lives in the
 * root layout's `useProtectedRoute` gate — this file only defines the
 * navigator shape.
 */
import { Stack } from "expo-router";

export default function AuthLayout() {
  return (
    <Stack screenOptions={{ headerShown: false }}>
      <Stack.Screen name="sign-in" options={{ title: "Sign In" }} />
    </Stack>
  );
}
