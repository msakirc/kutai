/**
 * Home tab — route `/`.
 *
 * `(tabs)/index.tsx` is the app entry route. `index` is special: it maps to
 * the directory root, so this is `/`, not `/(tabs)/index`.
 *
 * Demonstrates typed navigation via the `routes` helper and `useRouter`.
 */
import { View, Text, Pressable, StyleSheet } from "react-native";
import { useRouter } from "expo-router";

import { routes } from "../../lib/routes";

export default function HomeScreen() {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Home</Text>
      <Text style={styles.body}>
        This is the home tab. Navigation below uses the typed `routes` helper.
      </Text>

      {/* router.push with a typed route object — see lib/routes.ts */}
      <Pressable
        style={styles.button}
        onPress={() => router.push(routes.detail("42"))}
      >
        <Text style={styles.buttonText}>Open detail 42</Text>
      </Pressable>

      <Pressable
        style={styles.button}
        onPress={() => router.push(routes.explore())}
      >
        <Text style={styles.buttonText}>Go to Explore</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: "center", justifyContent: "center", padding: 24, gap: 12 },
  title: { fontSize: 24, fontWeight: "600" },
  body: { fontSize: 14, color: "#475569", textAlign: "center" },
  button: { backgroundColor: "#2563eb", paddingVertical: 10, paddingHorizontal: 18, borderRadius: 8 },
  buttonText: { color: "#ffffff", fontWeight: "500" },
});
