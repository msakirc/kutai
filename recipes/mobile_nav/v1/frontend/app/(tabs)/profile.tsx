/**
 * Profile tab — route `/profile`.
 *
 * Shows the sign-out action. Calling `signOut()` flips the session state;
 * the auth gate in `app/_layout.tsx` then redirects to the sign-in route
 * automatically — screens never navigate to `/sign-in` themselves.
 */
import { View, Text, Pressable, StyleSheet } from "react-native";

import { useSession } from "../../lib/auth";

export default function ProfileScreen() {
  const { signOut } = useSession();

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Profile</Text>
      <Text style={styles.body}>
        Signing out flips session state; the root layout's auth gate handles
        the redirect to the sign-in screen.
      </Text>

      <Pressable style={styles.button} onPress={() => signOut()}>
        <Text style={styles.buttonText}>Sign out</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: "center", justifyContent: "center", padding: 24, gap: 12 },
  title: { fontSize: 24, fontWeight: "600" },
  body: { fontSize: 14, color: "#475569", textAlign: "center" },
  button: { backgroundColor: "#dc2626", paddingVertical: 10, paddingHorizontal: 18, borderRadius: 8 },
  buttonText: { color: "#ffffff", fontWeight: "500" },
});
