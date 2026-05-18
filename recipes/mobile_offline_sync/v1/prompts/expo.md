# Expo — mobile_offline_sync/v1 instantiation notes

When wiring this recipe into an Expo / React Native project:

- **expo-sqlite is the durable layer**: the mutation queue persists to an
  on-disk SQLite DB so a queued mutation survives an app kill / restart.
  An in-memory queue (or AsyncStorage holding a JSON blob) loses writes on
  crash — do not "simplify" it that way.
- **Use the async expo-sqlite API**: `openDatabaseAsync` / `execAsync` /
  `runAsync` / `getAllAsync`. The legacy `openDatabase` callback API is
  deprecated. `runAsync` returns `lastInsertRowId` — that is the queue id.
- **`networkMode:"offlineFirst"`**: set on both queries and mutations in
  the QueryClient. Without it, react-query queries hang in `loading` when
  offline instead of serving the cached value, and mutations never fire so
  the sync engine never gets a chance to queue them.
- **Replay must be ordered**: drain the queue by `id ASC` (insertion
  order). Out-of-order replay corrupts state when mutation B depends on
  mutation A (e.g. create-then-update). The sync engine does this.
- **Mutations must be idempotent server-side**: a mutation can be replayed
  after the client already got a (lost) success — design endpoints to
  tolerate a repeat. A client-generated UUID as the row key + an upsert is
  the standard fix.
- **Bump-attempts, then give up**: a permanently-failing mutation (e.g.
  the server rejects it 4xx) must not retry forever. `bumpAttempts` +
  `MAX_REPLAY_ATTEMPTS` caps it; exhausted rows are skipped by
  `pendingMutations`. Surface exhausted rows to the user — silently
  dropping them loses data.
- **`expo-network` connectivity is best-effort**: `isInternetReachable`
  can be `null` briefly and a "connected" Wi-Fi can still be a captive
  portal. Treat the network signal as a *hint to try draining*, not a
  guarantee — `drainQueue` re-checks each call and replay failures re-queue.
- **Drain on mount AND on reconnect**: `useOfflineSync` drains once at
  startup (for mutations queued in a prior session) and again on every
  reconnect event. Missing the mount drain strands prior-session writes.
- **Optimistic cache updates**: when `submitMutation` returns
  `{queued:true}`, apply an optimistic update to the react-query cache so
  the UI reflects the change immediately; reconcile on the next successful
  drain.
