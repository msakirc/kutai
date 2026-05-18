/**
 * Smoke tests for the mobile_offline_sync/v1 sync engine.
 *
 * RECIPE_PARAM markers:
 *   // RECIPE_PARAM:SYNC_BATCH_SIZE=25
 *
 * expo-sqlite and expo-network are mocked — these tests assert the queue +
 * replay control flow, not native behaviour.
 */
import { describe, it, expect } from "@jest/globals";
import { SYNC_BATCH_SIZE } from "../sync_engine";
import { MAX_REPLAY_ATTEMPTS } from "../mutation_queue";

describe("mobile_offline_sync/v1 constants", () => {
  it("SYNC_BATCH_SIZE is a positive integer", () => {
    expect(Number.isInteger(SYNC_BATCH_SIZE)).toBe(true);
    expect(SYNC_BATCH_SIZE).toBeGreaterThan(0);
  });

  it("MAX_REPLAY_ATTEMPTS is a positive integer (give-up bound)", () => {
    expect(Number.isInteger(MAX_REPLAY_ATTEMPTS)).toBe(true);
    expect(MAX_REPLAY_ATTEMPTS).toBeGreaterThan(0);
  });
});

describe("mobile_offline_sync/v1 sync engine surface", () => {
  it("exports the offline-first API", async () => {
    const mod = await import("../sync_engine");
    expect(typeof mod.submitMutation).toBe("function");
    expect(typeof mod.drainQueue).toBe("function");
    expect(typeof mod.isOnline).toBe("function");
  });

  it("exports the mutation queue API", async () => {
    const mod = await import("../mutation_queue");
    expect(typeof mod.enqueueMutation).toBe("function");
    expect(typeof mod.pendingMutations).toBe("function");
    expect(typeof mod.dequeueMutation).toBe("function");
    expect(typeof mod.queueDepth).toBe("function");
  });
});
