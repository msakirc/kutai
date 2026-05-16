/**
 * mobile_persistence v1 — Expo data-layer entry point.
 *
 * RECIPE_PARAM markers:
 *   // RECIPE_PARAM:ENTITY=note
 *   // RECIPE_PARAM:ENTITY_PLURAL=notes
 *
 * This file is the screen-level glue that ties the `db/` modules together. It
 * exports:
 *   - `PersistenceProvider`  — wraps the app in a react-query QueryClient and
 *                              runs migrations before rendering children.
 *   - `NoteListScreen`       — a live-query list screen (push-based refresh).
 *   - `useNoteActions`       — convenience hook bundling the mutations.
 *
 * The real `db/` modules (schema / client / queries / hooks) ship in the
 * `db/` template directory and are instantiated alongside this file. During
 * instantiation the planner places `db/` under `app/db/` and this file under
 * `app/(persistence)/` or wherever the feature screen lives.
 *
 * Why on-device SQLite (expo-sqlite) over AsyncStorage:
 *   AsyncStorage is an unindexed key/value blob store — every "query" is a
 *   full deserialize + JS-side filter. For structured, relational, or
 *   list-heavy data (anything you sort, paginate, or join) SQLite gives real
 *   indexes, transactions, and typed queries via Drizzle. Use AsyncStorage
 *   only for tiny flat settings; use this recipe for entities.
 */
import React, { useEffect, useState } from "react";
import { View, Text, FlatList, Pressable, StyleSheet } from "react-native";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { runMigrations } from "./db/client";
import { useLiveNotes, useCreateNote, useUpdateNote, useDeleteNote } from "./db/hooks";
import type { Note, NewNote } from "./db/schema";

// A single QueryClient instance for the whole app — never recreate per render.
const queryClient = new QueryClient();

// ---------------------------------------------------------------------------
// Provider — runs migrations, then mounts children
// ---------------------------------------------------------------------------

/**
 * PersistenceProvider — mount this at the app root (e.g. in `app/_layout.tsx`).
 *
 * It awaits `runMigrations()` before rendering children. Rendering a screen
 * before migrations finish is the #1 cause of "no such table" crashes — the
 * gate below makes that impossible.
 */
export function PersistenceProvider({ children }: { children: React.ReactNode }) {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    runMigrations()
      .then(() => setReady(true))
      .catch((e) => setError(e instanceof Error ? e : new Error(String(e))));
  }, []);

  if (error) {
    return (
      <View style={styles.center}>
        <Text style={styles.error}>Database init failed: {error.message}</Text>
      </View>
    );
  }

  if (!ready) {
    return (
      <View style={styles.center}>
        <Text>Preparing local data…</Text>
      </View>
    );
  }

  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

// ---------------------------------------------------------------------------
// Actions hook
// ---------------------------------------------------------------------------

/** Bundles the create/update/delete mutations into one object for screens. */
export function useNoteActions() {
  const create = useCreateNote();
  const update = useUpdateNote();
  const remove = useDeleteNote();

  return {
    create: (input: NewNote) => create.mutateAsync(input),
    toggleDone: (note: Note) =>
      update.mutateAsync({ id: note.id, patch: { done: !note.done } }),
    remove: (id: number) => remove.mutateAsync(id),
    isMutating: create.isPending || update.isPending || remove.isPending,
  };
}

// ---------------------------------------------------------------------------
// List screen — live query
// ---------------------------------------------------------------------------

/**
 * NoteListScreen — renders every row via `useLiveNotes`. Because the live
 * query is push-based, inserting/updating/deleting elsewhere in the app
 * re-renders this list with no manual refetch.
 */
export function NoteListScreen() {
  const { data } = useLiveNotes();
  const actions = useNoteActions();

  return (
    <View style={styles.container}>
      <Text style={styles.heading}>Notes</Text>
      <FlatList
        data={data}
        keyExtractor={(item) => String(item.id)}
        ListEmptyComponent={<Text style={styles.empty}>No notes yet.</Text>}
        renderItem={({ item }) => (
          <Pressable
            style={styles.row}
            onPress={() => actions.toggleDone(item)}
            onLongPress={() => actions.remove(item.id)}
          >
            <Text style={[styles.title, item.done && styles.titleDone]}>
              {item.title}
            </Text>
            {item.body ? <Text style={styles.body}>{item.body}</Text> : null}
          </Pressable>
        )}
      />
      <Pressable
        style={styles.addBtn}
        disabled={actions.isMutating}
        onPress={() =>
          actions.create({ title: `Note ${Date.now() % 10000}`, body: "" })
        }
      >
        <Text style={styles.addBtnText}>Add note</Text>
      </Pressable>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles — StyleSheet.create per the expo stack reminders (no raw CSS)
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: { flex: 1, padding: 16 },
  center: { flex: 1, alignItems: "center", justifyContent: "center", padding: 24 },
  heading: { fontSize: 22, fontWeight: "600", marginBottom: 12 },
  row: { paddingVertical: 12, borderBottomWidth: StyleSheet.hairlineWidth },
  title: { fontSize: 16, fontWeight: "500" },
  titleDone: { textDecorationLine: "line-through", opacity: 0.5 },
  body: { fontSize: 14, color: "#666", marginTop: 2 },
  empty: { color: "#999", paddingVertical: 24, textAlign: "center" },
  error: { color: "#b00020", textAlign: "center" },
  addBtn: {
    marginTop: 16,
    backgroundColor: "#2563eb",
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: "center",
  },
  addBtnText: { color: "#fff", fontWeight: "600" },
});
