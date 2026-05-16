# mobile_persistence Recipe v1 — Known Lessons

Pitfalls captured from prior `expo-sqlite` + Drizzle implementations. Seeded
into `mission_lessons` (domain `mobile_persistence`) on recipe instantiation.

- **`openDatabaseSync` vs `openDatabaseAsync`**: Drizzle's Expo driver (`drizzle-orm/expo-sqlite`) is built on the *synchronous* `SQLiteDatabase` handle. Passing the result of `openDatabaseAsync` into `drizzle()` throws at construction time. Always open with `openDatabaseSync`. The file-open is cheap; query execution through Drizzle stays async.

- **Migrations must run before the first query**: Rendering any screen that touches a table before `migrate()` resolves raises `no such table: <table>`. Gate the UI on `runMigrations()` — the `PersistenceProvider` in `frontend.template.tsx` does this. Never query a table from a module-level call or an un-gated effect.

- **Drizzle's Expo driver needs `enableChangeListener: true`**: Pass `{ enableChangeListener: true }` to `openDatabaseSync`, or `useLiveQuery` mounts fine but never refreshes on data changes. It fails silently — no error, the list just goes stale. This is the #1 "live query doesn't update" cause.

- **SQLite file lives in the app sandbox and is wiped on uninstall**: `expo-sqlite` stores the `.db` file under the app's private documents directory. Uninstalling the app deletes it; there is no automatic cloud backup. Treat on-device SQLite as a cache/working-set, not durable storage — sync anything that must survive to a server, or export it explicitly.

- **No cross-platform web fallback by default**: `expo-sqlite` is native-only. On `expo start --web` it throws or no-ops. If the same codebase ships to web, branch on `Platform.OS` and use a different store (IndexedDB / a web-SQLite WASM build) for web — do not assume this recipe works on web.

- **Never run `drizzle-kit push` for an Expo app**: `push` targets a live dev DB server. There is none — the SQLite file lives on each device. Always use `drizzle-kit generate` to emit migrations, committed and applied on-device by the Expo migrator. `push` will either error or silently do the wrong thing.

- **Commit the generated `migrations/migrations.js` bundle**: Metro cannot read raw `.sql` files at runtime, so `drizzle-kit` (driver `expo`) inlines the SQL into a JS module. That `migrations.js` (and `meta/_journal.json`) must be committed and kept in sync with the `.sql` files — a stale or missing bundle means migrations silently don't apply.

- **Store timestamps as INTEGER epoch-millis, not ISO strings**: SQLite has no native datetime type. Use `integer` columns with `unixepoch() * 1000` defaults. ISO-8601 text strings sort lexically (mostly OK) but break arithmetic and comparisons against numeric times, and mixing the two formats in one column corrupts ordering. The schema template uses epoch-millis — keep it.

- **`useLiveQuery` returns `data: []`, never `undefined`**: Unlike react-query's `useQuery`, `useLiveQuery` from `drizzle-orm/expo-sqlite` initialises `data` to an empty array and fills it on first read. Do not write `if (!data) return <Loading/>` — it will never trigger. Render the empty state from the array length instead.

- **`drizzle-kit generate` requires the schema path to be relative to project root**: In `drizzle.config.ts`, `schema: "./db/schema.ts"` and `out: "./migrations"` are resolved from the directory you run `drizzle-kit` in. Running it from a subfolder produces migrations in the wrong place or fails to find the schema. Always run `npx drizzle-kit generate` from the project root.

- **`.returning()` is required to read back inserted/updated rows**: `expo-sqlite` does not auto-populate the result of an `insert`/`update`. Chain `.returning()` on the Drizzle query to get the row (with its generated `id`) back. Without it you get an empty result and have to issue a second `select`.

- **One `QueryClient` instance for the whole app**: Creating a `new QueryClient()` inside a component re-creates the cache on every render, wiping all cached queries. Instantiate it once at module scope (as `frontend.template.tsx` does) and pass it to a single root `QueryClientProvider`.

- **Don't stack the read-through cache and react-query for the same screen**: `listNotesCached()` and `useNotesQuery` both cache. Using both for one list means two caches with independent invalidation — a mutation invalidates one but not the other and the UI shows stale data. UI code uses the hooks; background/non-React code uses `listNotesCached()`.
