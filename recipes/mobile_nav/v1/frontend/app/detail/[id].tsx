/**
 * Dynamic route — `app/detail/[id].tsx` → `/detail/:id`.
 *
 * The `[id]` filename is the dynamic segment. Read it with
 * `useLocalSearchParams` (NOT `useGlobalSearchParams` — local params are
 * scoped to THIS screen and don't leak/update when other screens change).
 *
 * `<Stack.Screen options>` here sets the per-route title dynamically from the
 * param. Each route may also configure its own header without a `_layout`
 * entry by rendering `<Stack.Screen>` inline.
 */
import { View, Text, StyleSheet } from "react-native";
import { Stack, useLocalSearchParams } from "expo-router";

export default function DetailScreen() {
  // Typed params: with experiments.typedRoutes the param shape is inferred.
  const { id } = useLocalSearchParams<{ id: string }>();

  return (
    <View style={styles.container}>
      {/* Inline per-route screen options — dynamic title from the param. */}
      <Stack.Screen options={{ title: `Detail ${id}` }} />

      <Text style={styles.title}>Detail screen</Text>
      <Text style={styles.body}>Loaded id from the route param: {id}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: "center", justifyContent: "center", padding: 24, gap: 8 },
  title: { fontSize: 22, fontWeight: "600" },
  body: { fontSize: 14, color: "#475569" },
});
