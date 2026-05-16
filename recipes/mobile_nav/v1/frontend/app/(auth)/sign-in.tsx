/**
 * Sign-in screen — route `/sign-in` (inside the `(auth)` group).
 *
 * Unauthenticated users are redirected here by the root layout's auth gate.
 * On successful sign-in we just flip session state; the gate then redirects
 * INTO the tab app. This screen never calls `router.push('/(tabs)')` itself —
 * keeping all redirect logic in one place avoids race conditions.
 *
 * RECIPE_PARAM markers:
 *   // RECIPE_PARAM:APP_NAME=MyApp
 */
import { useState } from "react";
import { View, Text, Pressable, ActivityIndicator, StyleSheet } from "react-native";

import { useSession } from "../../lib/auth";

// RECIPE_PARAM:APP_NAME=MyApp
const APP_NAME = "<<APP_NAME>>";

export default function SignInScreen() {
  const { signIn } = useSession();
  const [busy, setBusy] = useState(false);

  async function handleSignIn() {
    setBusy(true);
    try {
      // Replace with a real credential flow (mobile_auth recipe wires this).
      await signIn("demo-session-token");
    } finally {
      setBusy(false);
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Welcome to {APP_NAME}</Text>
      <Text style={styles.body}>Sign in to continue.</Text>

      <Pressable style={styles.button} onPress={handleSignIn} disabled={busy}>
        {busy ? (
          <ActivityIndicator color="#ffffff" />
        ) : (
          <Text style={styles.buttonText}>Sign in</Text>
        )}
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: "center", justifyContent: "center", padding: 24, gap: 12 },
  title: { fontSize: 22, fontWeight: "600" },
  body: { fontSize: 14, color: "#475569" },
  button: {
    backgroundColor: "#2563eb",
    paddingVertical: 12,
    paddingHorizontal: 28,
    borderRadius: 8,
    minWidth: 140,
    alignItems: "center",
  },
  buttonText: { color: "#ffffff", fontWeight: "500" },
});
