/**
 * Offline-first sync engine — replay queued mutations on reconnect.
 *
 * RECIPE_PARAM markers:
 *   // RECIPE_PARAM:SYNC_BATCH_SIZE=25
 *
 * The template engine substitutes `KEY` tokens and inline literals
 * before this file is written. RECIPE_PARAM markers stay intact.
 *
 * Flow: a mutation made while offline is appended to the SQLite queue and
 * an optimistic update is applied to the react-query cache. expo-network
 * tells us when connectivity returns; `drainQueue` then replays the queue
 * in order against the live API, removing rows that succeed and bumping
 * the attempt counter on rows that fail.
 */
import * as Network from "expo-network";
import {
  enqueueMutation,
  pendingMutations,
  dequeueMutation,
  bumpAttempts,
  queueDepth,
} from "./mutation_queue";

// Inline-substituted numeric literal — see _substitute_inline_params.
export const SYNC_BATCH_SIZE = 25; // RECIPE_PARAM:SYNC_BATCH_SIZE=25

export type SyncResult = {
  replayed: number;
  failed: number;
  remaining: number;
};

/** True when the device currently has a usable internet connection. */
export async function isOnline(): Promise<boolean> {
  const state = await Network.getNetworkStateAsync();
  return Boolean(state.isConnected && state.isInternetReachable);
}

/**
 * Submit a mutation. When online, hits the API directly. When offline,
 * appends to the persistent queue and returns `{queued:true}` so the UI
 * can apply an optimistic update.
 */
export async function submitMutation(
  apiBase: string,
  endpoint: string,
  method: string,
  payload: unknown,
): Promise<{ queued: boolean; ok: boolean }> {
  if (await isOnline()) {
    const ok = await replayOne(apiBase, endpoint, method, payload);
    if (ok) return { queued: false, ok: true };
    // Online but the call failed — queue it so the next drain retries.
  }
  await enqueueMutation(endpoint, method, payload);
  return { queued: true, ok: false };
}

/** Replay a single mutation against the API. Returns true on a 2xx. */
async function replayOne(
  apiBase: string,
  endpoint: string,
  method: string,
  payload: unknown,
): Promise<boolean> {
  try {
    const res = await fetch(`${apiBase}${endpoint}`, {
      method,
      headers: { "Content-Type": "application/json" },
      body: method === "GET" ? undefined : JSON.stringify(payload),
    });
    return res.ok;
  } catch {
    // Network blip mid-replay — treat as a failure; the row stays queued.
    return false;
  }
}

/**
 * Drain the offline mutation queue. Replays up to SYNC_BATCH_SIZE rows in
 * insertion order. A successful row is removed; a failed row has its
 * attempt counter bumped (and is eventually skipped once it exhausts
 * MAX_REPLAY_ATTEMPTS — see mutation_queue).
 *
 * Idempotent and safe to call repeatedly; a no-op when offline.
 */
export async function drainQueue(apiBase: string): Promise<SyncResult> {
  if (!(await isOnline())) {
    return { replayed: 0, failed: 0, remaining: await queueDepth() };
  }

  const batch = await pendingMutations(SYNC_BATCH_SIZE);
  let replayed = 0;
  let failed = 0;

  for (const m of batch) {
    let payload: unknown = {};
    try {
      payload = JSON.parse(m.body);
    } catch {
      payload = {};
    }
    const ok = await replayOne(apiBase, m.endpoint, m.method, payload);
    if (ok) {
      await dequeueMutation(m.id);
      replayed += 1;
    } else {
      await bumpAttempts(m.id);
      failed += 1;
    }
  }

  return { replayed, failed, remaining: await queueDepth() };
}
