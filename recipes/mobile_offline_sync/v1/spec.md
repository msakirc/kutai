# mobile_offline_sync Recipe v1 — Specification

## Scope

An offline-first sync layer for an Expo / React Native app: mutations made
while offline are queued in a persistent `expo-sqlite` store and replayed
in order on reconnect. `@tanstack/react-query` provides offline-first
caching for reads; `expo-network` drives connectivity detection.

## When to Pick This Recipe

**Use** `mobile_offline_sync/v1` when:
- The app is built with Expo and must remain usable without connectivity.
- Stack is `fastapi+sqlite+expo`, `fastapi+postgres+expo`, or `expo`.
- Writes must not be lost when the device is offline — they should queue
  durably and replay automatically.

**Do not use** when:
- The app is read-only offline (no offline writes) — plain react-query
  caching is enough.
- You need full CRDT / conflict-resolution merge semantics — this recipe
  is last-write-wins replay, not a CRDT.

## What It Generates

| File | Role |
|------|------|
| `mutation_queue.template.ts` | `expo-sqlite` persistent mutation queue: enqueue / pending / dequeue / bump-attempts |
| `sync_engine.template.ts` | `isOnline`, `submitMutation`, `drainQueue` — queue + ordered replay |
| `query_client.template.ts` | offline-first `QueryClient` + `useOfflineSync` reconnect hook |
| `tests/sync_smoke.template.ts` | constant + API-surface smoke tests |

## Data Flow

1. A mutation is submitted. Online → hit the API directly. Offline (or the
   call fails) → append to the SQLite queue + apply an optimistic cache
   update.
2. `expo-network` detects connectivity returning.
3. `drainQueue` replays queued rows oldest-first; 2xx → delete the row,
   failure → bump the attempt counter.
4. A row that exhausts `MAX_REPLAY_ATTEMPTS` is skipped and surfaced.

## Parameters

| Param | Default | Meaning |
|-------|---------|---------|
| `QUEUE_TABLE` | `mutation_queue` | SQLite table name for the queue |
| `DB_NAME` | `app.db` | expo-sqlite database file name |
| `MAX_REPLAY_ATTEMPTS` | `5` | Give-up bound per queued mutation |
| `SYNC_BATCH_SIZE` | `25` | Rows drained per `drainQueue` call |
| `STALE_TIME_MS` | `30000` | react-query `staleTime` |

## Out of Scope

- CRDT / three-way-merge conflict resolution (this is ordered last-write
  replay).
- Background sync while the app is killed (needs `expo-task-manager` /
  `expo-background-fetch` — a follow-up).
- Bidirectional sync of *server*-side changes (this recipe replays client
  writes; reads rely on react-query refetch).
