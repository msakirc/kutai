/**
 * @tanstack/react-query client wired to the offline sync engine.
 *
 * RECIPE_PARAM markers:
 *   // RECIPE_PARAM:STALE_TIME_MS=30000
 *
 * The template engine substitutes `KEY` tokens and inline literals
 * before this file is written. RECIPE_PARAM markers stay intact.
 *
 * The QueryClient is configured offline-first: queries keep serving the
 * cached value when the network is down, and a reconnect triggers both a
 * react-query refetch AND a queue drain via `useOfflineSync`.
 */
import { useEffect } from "react";
import { QueryClient } from "@tanstack/react-query";
import * as Network from "expo-network";
import { drainQueue } from "./sync_engine";

// Inline-substituted numeric literal — see _substitute_inline_params.
const STALE_TIME_MS = 30000; // RECIPE_PARAM:STALE_TIME_MS=30000

/**
 * Build the app's QueryClient. `networkMode:"offlineFirst"` makes queries
 * serve cached data when offline instead of sitting in a perpetual
 * loading state, and lets mutations fire so the sync engine can queue them.
 */
export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: STALE_TIME_MS,
        networkMode: "offlineFirst",
        retry: 2,
      },
      mutations: {
        networkMode: "offlineFirst",
      },
    },
  });
}

/**
 * Hook: subscribe to connectivity changes and drain the offline mutation
 * queue whenever the device comes back online. Mount once near the app
 * root, inside the QueryClientProvider.
 */
export function useOfflineSync(apiBase: string): void {
  useEffect(() => {
    let cancelled = false;

    // Drain once on mount in case mutations were queued in a prior session.
    void drainQueue(apiBase);

    const sub = Network.addNetworkStateListener((state) => {
      if (cancelled) return;
      if (state.isConnected && state.isInternetReachable) {
        void drainQueue(apiBase);
      }
    });

    return () => {
      cancelled = true;
      sub.remove();
    };
  }, [apiBase]);
}

export { STALE_TIME_MS };
