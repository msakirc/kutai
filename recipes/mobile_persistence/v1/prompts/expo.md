# Expo — mobile_persistence/v1 instantiation notes

Recipe-specific guidance for wiring `expo-sqlite` + Drizzle into an Expo
project. Complements (does not repeat) the general `expo` STACK_BLOCK
reminders in `packages/coulson/.../reflection.py` — Expo Router, `StyleSheet`,
EAS Build, `Platform.OS` guards already covered there.

## expo-sqlite: sync handle, async queries

- Open the database with **`openDatabaseSync`**, not `openDatabaseAsync`.
  Drizzle's Expo driver (`drizzle-orm/expo-sqlite`) is built on the
  synchronous `SQLiteDatabase` handle — passing an async handle throws at
  `drizzle()` construction time. The file-open is cheap; the sync call is
  correct. Query *execution* through Drizzle is still `await`-ed.
- Pass **`{ enableChangeListener: true }`** to `openDatabaseSync`. This is
  what powers `useLiveQuery`. Without it, live queries mount fine but never
  refresh — a silent, confusing bug.

## Drizzle's `drizzle-kit generate` flow

- The schema in `db/schema.ts` is the single source of truth. Never hand-edit
  generated migration SQL.
- `db/drizzle.config.ts` uses `dialect: "sqlite"` + `driver: "expo"`. The
  `expo` driver makes `drizzle-kit` emit a `migrations/migrations.js` bundle
  (a JS module inlining the SQL) alongside the raw `.sql` — required because
  Metro cannot read `.sql` files at runtime.
- Workflow after any schema change: edit `schema.ts` → run
  `npx drizzle-kit generate` → commit the new `migrations/NNNN_*.sql`,
  the refreshed `migrations/migrations.js`, and `migrations/meta/_journal.json`.
- **Never run `drizzle-kit push`** for an Expo app — there is no dev DB server;
  the SQLite file lives on each device. Always go `generate` + on-device
  migrator.

## Run migrations on app start — before the first query

- Call `runMigrations()` (from `db/client.ts`) from a top-level effect, and
  gate the UI on its completion. `PersistenceProvider` in `frontend.template.tsx`
  does exactly this: it shows a "Preparing local data…" view until
  `migrate()` resolves, then mounts children.
- Rendering any screen that queries a table before migrations finish raises
  `no such table: notes`. The provider gate makes that impossible — keep it.
- The Expo migrator tracks applied migrations in its own
  `__drizzle_migrations` table, so `migrate()` is safe to call on every cold
  start; it only applies what is pending.

## Why on-device SQLite over AsyncStorage

- AsyncStorage is an **unindexed key/value blob store**. Every "query" is a
  full JSON deserialize + JS-side filter — O(n) on the whole dataset, no
  sorting, no joins, no transactions.
- Use SQLite (this recipe) for **structured data**: anything you list, sort,
  paginate, filter, or relate. You get real indexes, typed queries via
  Drizzle, and atomic transactions.
- Use AsyncStorage only for **tiny flat settings** (a theme flag, an
  onboarding-seen boolean). Mixing the two is fine — pick per data shape.

## The `useLiveQuery` hook

- `useLiveQuery(db.select()...)` from `drizzle-orm/expo-sqlite` is push-based:
  the component re-renders automatically whenever the underlying rows change,
  with no manual `invalidateQueries` or refetch.
- It depends on `enableChangeListener: true` on the DB handle (see above).
- Prefer `useLiveQuery` for always-on list screens. Prefer the react-query
  hooks (`useNotesQuery` + mutations) when you need explicit loading/error
  state, optimistic updates, or a later server-sync layer.
- `useLiveQuery` returns `{ data, error, updatedAt }` — `data` is `[]` until
  the first read resolves, never `undefined`, so no loading guard is needed.

## Read-through cache

- `db/queries.ts` ships a tiny in-memory read-through cache
  (`listNotesCached`) for **non-React call sites** (background tasks, sync
  jobs). Every mutation calls `invalidateCache()`.
- For UI code, the react-query layer in `db/hooks.ts` supersedes it —
  `staleTime` mirrors `CACHE_STALE_MS`. Do not stack both for the same screen.

## RECIPE_PARAM markers

| Marker | Default | Description |
|--------|---------|-------------|
| `ENTITY` | `note` | Singular entity name |
| `ENTITY_PLURAL` | `notes` | Plural — table + query-key naming |
| `TABLE_NAME` | `notes` | SQLite table name |
| `DB_FILE_NAME` | `app.db` | On-device SQLite file name |
| `CACHE_STALE_MS` | `30000` | Read-through cache / react-query staleTime window |
