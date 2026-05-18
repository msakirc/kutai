/**
 * monitoring_kit_nextjs/v1 — error tracking + structured logging.
 *
 * `initObservability()` is a no-op when the Sentry DSN env var is unset, so
 * local/dev runs need no account. Wire `@sentry/nextjs` properly via its
 * `instrumentation.ts` hook for full browser+server coverage; this helper
 * is the minimal env-gated init + a JSON `log` helper that costs nothing.
 *
 * RECIPE_PARAM markers (leave intact):
 *   // RECIPE_PARAM:SENTRY_DSN_ENV=NEXT_PUBLIC_SENTRY_DSN
 *   // RECIPE_PARAM:ENVIRONMENT_ENV=NEXT_PUBLIC_APP_ENV
 */

const SENTRY_DSN_ENV = "NEXT_PUBLIC_SENTRY_DSN"; // RECIPE_PARAM:SENTRY_DSN_ENV=NEXT_PUBLIC_SENTRY_DSN
const ENVIRONMENT_ENV = "NEXT_PUBLIC_APP_ENV"; // RECIPE_PARAM:ENVIRONMENT_ENV=NEXT_PUBLIC_APP_ENV

type Level = "info" | "warn" | "error";

/** One JSON object per log line — parseable by any aggregator. */
export function log(level: Level, msg: string, extra?: Record<string, unknown>): void {
  const line = JSON.stringify({
    ts: new Date().toISOString(),
    level,
    msg,
    ...(extra ?? {}),
  });
  if (level === "error") console.error(line);
  else if (level === "warn") console.warn(line);
  else console.log(line);
}

/**
 * Initialise Sentry when a DSN is present. Returns true when wired.
 * Call once from `instrumentation.ts` (register hook) or app entry.
 */
export async function initObservability(): Promise<boolean> {
  const dsn = (process.env[SENTRY_DSN_ENV] ?? "").trim();
  if (!dsn) {
    log("info", "observability: Sentry DSN unset — error tracking off");
    return false;
  }
  try {
    const Sentry = await import("@sentry/nextjs");
    Sentry.init({
      dsn,
      environment: process.env[ENVIRONMENT_ENV] ?? "development",
      tracesSampleRate: Number(process.env.SENTRY_TRACES_SAMPLE_RATE ?? "0"),
      sendDefaultPii: false,
    });
    log("info", "observability: Sentry initialised");
    return true;
  } catch {
    log("warn", "observability: @sentry/nextjs not installed — error tracking off");
    return false;
  }
}
