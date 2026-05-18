/**
 * Offline mutation queue — expo-sqlite persistent store.
 *
 * RECIPE_PARAM markers:
 *   // RECIPE_PARAM:QUEUE_TABLE=mutation_queue
 *   // RECIPE_PARAM:DB_NAME=app.db
 *   // RECIPE_PARAM:MAX_REPLAY_ATTEMPTS=5
 *
 * The template engine replaces `KEY` tokens before this file is written.
 * RECIPE_PARAM comment markers are left intact so the file stays
 * self-documenting and ast/ts-parses both before and after substitution.
 *
 * Mutations made while offline are appended here. The sync engine drains
 * the queue in insertion order on reconnect. Each row survives an app
 * restart because expo-sqlite persists to disk.
 */
import * as SQLite from "expo-sqlite";

const DB_NAME = "<<DB_NAME>>";
const QUEUE_TABLE = "<<QUEUE_TABLE>>";

// Inline-substituted numeric literal — see _substitute_inline_params.
export const MAX_REPLAY_ATTEMPTS = 5; // RECIPE_PARAM:MAX_REPLAY_ATTEMPTS=5

export type QueuedMutation = {
  id: number;
  endpoint: string;
  method: string;
  body: string; // JSON-serialised payload
  created_at: number;
  attempts: number;
};

let _db: SQLite.SQLiteDatabase | null = null;

/** Open (once) the on-disk SQLite DB. expo-sqlite is sync-open + async-exec. */
export async function getDb(): Promise<SQLite.SQLiteDatabase> {
  if (_db) return _db;
  _db = await SQLite.openDatabaseAsync(DB_NAME);
  await _db.execAsync(
    `CREATE TABLE IF NOT EXISTS ${QUEUE_TABLE} (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       endpoint TEXT NOT NULL,
       method TEXT NOT NULL,
       body TEXT NOT NULL,
       created_at INTEGER NOT NULL,
       attempts INTEGER NOT NULL DEFAULT 0
     );`,
  );
  return _db;
}

/** Append a mutation to the queue. Returns the new row id. */
export async function enqueueMutation(
  endpoint: string,
  method: string,
  payload: unknown,
): Promise<number> {
  const db = await getDb();
  const result = await db.runAsync(
    `INSERT INTO ${QUEUE_TABLE} (endpoint, method, body, created_at, attempts)
     VALUES (?, ?, ?, ?, 0)`,
    endpoint,
    method.toUpperCase(),
    JSON.stringify(payload),
    Date.now(),
  );
  return result.lastInsertRowId;
}

/** Read pending mutations in insertion order, oldest first. */
export async function pendingMutations(limit: number): Promise<QueuedMutation[]> {
  const db = await getDb();
  return db.getAllAsync<QueuedMutation>(
    `SELECT id, endpoint, method, body, created_at, attempts
       FROM ${QUEUE_TABLE}
      WHERE attempts < ?
      ORDER BY id ASC
      LIMIT ?`,
    MAX_REPLAY_ATTEMPTS,
    limit,
  );
}

/** Remove a mutation after a successful replay. */
export async function dequeueMutation(id: number): Promise<void> {
  const db = await getDb();
  await db.runAsync(`DELETE FROM ${QUEUE_TABLE} WHERE id = ?`, id);
}

/** Bump the attempt counter after a failed replay (for backoff / give-up). */
export async function bumpAttempts(id: number): Promise<void> {
  const db = await getDb();
  await db.runAsync(
    `UPDATE ${QUEUE_TABLE} SET attempts = attempts + 1 WHERE id = ?`,
    id,
  );
}

/** Count of mutations still waiting (and not exhausted). */
export async function queueDepth(): Promise<number> {
  const db = await getDb();
  const row = await db.getFirstAsync<{ n: number }>(
    `SELECT COUNT(*) AS n FROM ${QUEUE_TABLE} WHERE attempts < ?`,
    MAX_REPLAY_ATTEMPTS,
  );
  return row?.n ?? 0;
}

/** Test hook — drop the cached handle so a fresh DB can be opened. */
export function _resetDbHandleForTests(): void {
  _db = null;
}

export { QUEUE_TABLE, DB_NAME };
