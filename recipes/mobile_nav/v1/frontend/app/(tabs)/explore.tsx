/**
 * Explore tab — route `/explore`.
 *
 * Demonstrates the `<Link>` component (declarative navigation). `Link` is
 * preferred for static destinations because it renders an accessible
 * pressable and supports `prefetch`/`asChild`; `router.push` is for
 * imperative navigation (after an async action, a callback, etc.).
 */
import { View, Text, StyleSheet } from "react-native";
import { Link } from "expo-router";

export default function ExploreScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Explore</Text>

      {/* Declarative navigation. `href` is type-checked when
          experiments.typedRoutes is enabled in app.json. */}
      <Link href="/profile" style={styles.link}>
        Go to Profile
      </Link>

      {/* Dynamic route link — the [id] segment is filled by `params`. */}
      <Link
        href={{ pathname: "/detail/[id]", params: { id: "7" } }}
        style={styles.link}
      >
        Open detail 7
      </Link>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: "center", justifyContent: "center", padding: 24, gap: 16 },
  title: { fontSize: 24, fontWeight: "600" },
  link: { fontSize: 16, color: "#2563eb", fontWeight: "500" },
});
