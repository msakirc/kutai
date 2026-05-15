/**
 * analytics_instrumentation/v1 — posthog-js client shim (web).
 *
 * A thin `track_event(name, properties)` helper over posthog-js. Every event
 * auto-attaches mission/feature/cohort metadata so downstream digests
 * (Z9 weekly_analytics) can slice by mission, feature, variant and segment
 * without each call site repeating boilerplate.
 *
 * RECIPE_PARAM markers:
 *   // RECIPE_PARAM:POSTHOG_API_KEY_ENV=POSTHOG_API_KEY
 *   // RECIPE_PARAM:POSTHOG_HOST_ENV=POSTHOG_HOST
 *   // RECIPE_PARAM:POSTHOG_HOST_DEFAULT=https://us.i.posthog.com
 *
 * Env contract (read at init):
 *   NEXT_PUBLIC_POSTHOG_API_KEY  — PostHog project API key (public, write-only)
 *   NEXT_PUBLIC_POSTHOG_HOST     — PostHog ingestion host
 *
 * Usage:
 *   import { initAnalytics, track_event, setAnalyticsContext } from "./analytics";
 *   initAnalytics();
 *   setAnalyticsContext({ mission_id: "m-42", feature_id: "checkout", business_model: "b2c" });
 *   track_event("checkout_started", { plan: "pro", amount: 19, currency: "USD" });
 */
import posthog from "posthog-js";

/** Metadata auto-attached to every event. Set once via setAnalyticsContext. */
export interface AnalyticsContext {
  mission_id: string;
  feature_id: string;
  /** Present only when an A/B experiment is active (Z9 T5). */
  variant?: string;
  /** Present only when cohort targeting is active (Z9 T5). */
  segment?: string;
  /** "b2b" | "b2c" | "hybrid" — defaults to "b2c" when unset. */
  business_model?: string;
  /**
   * Z9 T5B — B2B account/org identifier. When business_model is "b2b",
   * track_event attaches BOTH account_id and user_id; B2C attaches
   * user_id only. Set the per-user distinct id via posthog.identify().
   */
  account_id?: string;
}

let _context: AnalyticsContext = {
  mission_id: "unknown",
  feature_id: "unknown",
};
let _initialized = false;

/**
 * Z9 T5A — optional cohort gate. When the analytics_instrumentation recipe
 * declares `segment_predicate`, the recipe applier substitutes it here. When
 * set, track_event tags every event with the active segment AND only emits
 * for users whose segment matches the predicate. Null → no gate (fire for
 * the whole user base).
 *
 *   // RECIPE_PARAM:SEGMENT_PREDICATE=null
 */
const SEGMENT_PREDICATE: string | null = null; // RECIPE_PARAM:SEGMENT_PREDICATE=null

/**
 * Returns true when the active segment satisfies SEGMENT_PREDICATE. A null
 * predicate always matches (no cohort gate). The predicate is matched
 * literally against `_context.segment`; recipe appliers may replace this
 * body with a richer expression for custom cohorts.
 */
function segmentMatches(): boolean {
  if (SEGMENT_PREDICATE === null) return true;
  return _context.segment === SEGMENT_PREDICATE;
}

/** Populate the metadata block attached to every subsequent track_event. */
export function setAnalyticsContext(ctx: Partial<AnalyticsContext>): void {
  _context = { ..._context, ...ctx };
}

/**
 * Initialize PostHog from env. Safe to call multiple times (no-ops after
 * first success). No-ops with a console warning when the API key is absent
 * so local/dev builds do not crash.
 */
export function initAnalytics(): void {
  if (_initialized) return;
  // RECIPE_PARAM:POSTHOG_API_KEY_ENV=POSTHOG_API_KEY
  const apiKey = process.env.NEXT_PUBLIC_POSTHOG_API_KEY;
  // RECIPE_PARAM:POSTHOG_HOST_ENV=POSTHOG_HOST
  const host =
    process.env.NEXT_PUBLIC_POSTHOG_HOST ||
    "https://us.i.posthog.com"; // RECIPE_PARAM:POSTHOG_HOST_DEFAULT=https://us.i.posthog.com
  if (!apiKey) {
    // eslint-disable-next-line no-console
    console.warn(
      "[analytics] NEXT_PUBLIC_POSTHOG_API_KEY not set — track_event is a no-op.",
    );
    return;
  }
  posthog.init(apiKey, {
    api_host: host,
    capture_pageview: false, // we emit landing_view explicitly
    persistence: "localStorage+cookie",
  });
  _initialized = true;
}

/**
 * Emit a standardized analytics event.
 *
 * @param name        One of the STANDARD_EVENTS taxonomy names (see events.ts).
 * @param properties  Event-specific properties (see EVENT_PROPERTY_HINTS).
 *
 * Auto-attaches: mission_id, feature_id, variant, segment, business_model.
 * For business_model "b2b" it additionally attaches account_id.
 *
 * Z9 T5A — when SEGMENT_PREDICATE is set, the event is dropped for users
 * whose segment does not match (cohort gate).
 */
export function track_event(
  name: string,
  properties: Record<string, unknown> = {},
): void {
  const businessModel = _context.business_model ?? "b2c";
  const enriched: Record<string, unknown> = {
    ...properties,
    mission_id: _context.mission_id,
    feature_id: _context.feature_id,
    business_model: businessModel,
  };
  if (_context.variant !== undefined) enriched.variant = _context.variant;
  if (_context.segment !== undefined) enriched.segment = _context.segment;
  // Z9 T5B — B2B events carry the account/org id alongside the user id.
  if (businessModel === "b2b" && _context.account_id !== undefined) {
    enriched.account_id = _context.account_id;
  }

  if (!_initialized) {
    // eslint-disable-next-line no-console
    console.warn(`[analytics] track_event("${name}") dropped — not initialized.`);
    return;
  }
  // Z9 T5A — cohort gate: skip emission for non-matching segments.
  if (!segmentMatches()) {
    return;
  }
  posthog.capture(name, enriched);
}
