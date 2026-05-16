/**
 * mobile_persistence v1 — data-layer smoke tests.
 *
 * NOTE — execution context:
 *   Runs POST-instantiation against the instantiated recipe in the mission
 *   workspace. Recipe sources carry a `.template` suffix and are NOT
 *   importable from the recipe tree directly.
 *
 *   These tests exercise the `db/` query layer (insert / read / update /
 *   delete round-trip + the read-through cache). They run under jest with
 *   the `jest-expo` preset, which provides an in-memory expo-sqlite shim.
 *   Migrations are applied once in `beforeAll` — the same gate the app uses.
 *
 * RECIPE_PARAM:ENTITY=note
 *
 * Run (post-instantiation):
 *   npx jest data_layer
 */
import { runMigrations, __resetForTests } from "../db/client";
import {
  listNotes,
  listNotesCached,
  getNote,
  createNote,
  updateNote,
  deleteNote,
  invalidateCache,
} from "../db/queries";

beforeAll(async () => {
  // Migrations MUST run before the first query — mirrors PersistenceProvider.
  await runMigrations();
});

beforeEach(() => {
  __resetForTests();
  invalidateCache();
});

describe("mobile_persistence data layer", () => {
  test("insert then read round-trips", async () => {
    const created = await createNote({ title: "First", body: "hello" });
    expect(created.id).toBeGreaterThan(0);
    expect(created.title).toBe("First");

    const fetched = await getNote(created.id);
    expect(fetched).not.toBeNull();
    expect(fetched?.body).toBe("hello");
  });

  test("listNotes returns rows newest-first", async () => {
    await createNote({ title: "old", body: "" });
    await createNote({ title: "new", body: "" });
    const rows = await listNotes();
    expect(rows).toHaveLength(2);
    expect(rows[0].title).toBe("new");
  });

  test("update mutates the row and bumps updatedAt", async () => {
    const created = await createNote({ title: "draft", body: "" });
    const updated = await updateNote(created.id, { title: "final", done: true });
    expect(updated?.title).toBe("final");
    expect(updated?.done).toBe(true);
    expect(updated!.updatedAt).toBeGreaterThanOrEqual(created.updatedAt);
  });

  test("delete removes the row", async () => {
    const created = await createNote({ title: "temp", body: "" });
    const ok = await deleteNote(created.id);
    expect(ok).toBe(true);
    expect(await getNote(created.id)).toBeNull();
  });

  test("read-through cache serves a snapshot and invalidates on mutation", async () => {
    await createNote({ title: "one", body: "" });
    const first = await listNotesCached();
    expect(first).toHaveLength(1);

    // A mutation must invalidate the cache — next cached read sees the change.
    await createNote({ title: "two", body: "" });
    const second = await listNotesCached();
    expect(second).toHaveLength(2);
  });

  test("getNote returns null for an unknown id", async () => {
    expect(await getNote(999999)).toBeNull();
  });
});
