/**
 * 404 catch-all — `app/[...missing].tsx`.
 *
 * The `[...name]` spread filename matches ANY unmatched path. Expo Router
 * needs an explicit catch-all to render a not-found screen; without one an
 * unknown deep link renders a blank screen.
 *
 * `unstable_settings` is intentionally NOT used here — the catch-all is a
 * normal route.
 */
import { View, Text, StyleSheet } from "react-native";
import { Link, Stack } from "expo-router";

export default function NotFoundScreen() {
  return (
    <View style={styles.container}>
      <Stack.Screen options={{ title: "Not Found" }} />
      <Text style={styles.title}>This screen doesn’t exist.</Text>
      <Link href="/" style={styles.link}>
        Go to the home screen
      </Link>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: "center", justifyContent: "center", padding: 24, gap: 12 },
  title: { fontSize: 18, fontWeight: "600" },
  link: { fontSize: 16, color: "#2563eb", fontWeight: "500" },
});
