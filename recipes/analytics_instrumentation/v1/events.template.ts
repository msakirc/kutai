/**
 * analytics_instrumentation/v1 — Standard AARRR event taxonomy.
 *
 * This file is the single source of truth for the standardized event names
 * the recipe ships. The Phase 13.4 `analytics_integration` agent maps every
 * `success_metrics.aarrr_metrics` entry onto these events and emits a
 * `track_event(name, properties)` call at the corresponding code site.
 *
 * RECIPE_PARAM markers:
 *   // RECIPE_PARAM:ACTIVATION_EVENT=first_value_event
 *
 * The `first_value_event` activation event is intentionally generic — rename
 * it per product to the moment a user first experiences core value
 * (e.g. `first_note_created`, `first_project_shared`). Substituted via
 * RECIPE_PARAM:ACTIVATION_EVENT.
 *
 * Every emitted event auto-attaches (handled by track_event in the shims):
 *   mission_id, feature_id, variant (if A/B active), segment (if cohort
 *   targeting active), business_model.
 */

/** AARRR funnel stage an event belongs to. */
export type AarrrStage =
  | "acquisition"
  | "activation"
  | "retention"
  | "revenue"
  | "referral";

/** The standardized event taxonomy. Keys are AARRR stages. */
export const STANDARD_EVENTS: Record<AarrrStage, readonly string[]> = {
  acquisition: ["landing_view", "signup_started", "signup_completed"],
  // Activation is product-specific; the recipe ships one slot, renamed via
  // RECIPE_PARAM:ACTIVATION_EVENT.
  activation: ["first_value_event"], // RECIPE_PARAM:ACTIVATION_EVENT=first_value_event
  retention: ["session_started"],
  revenue: [
    "checkout_started",
    "checkout_completed",
    "subscription_created",
    "subscription_cancelled",
  ],
  referral: ["share_initiated", "share_completed", "invite_redeemed"],
} as const;

/** Flat list of every standard event name. */
export const ALL_STANDARD_EVENTS: readonly string[] = Object.values(
  STANDARD_EVENTS,
).flat();

/**
 * Recommended properties per event. The Phase 13.4 agent should attach these
 * (in addition to the auto-attached metadata) when emitting track_event.
 */
export const EVENT_PROPERTY_HINTS: Record<string, readonly string[]> = {
  landing_view: ["referrer", "utm_source", "utm_campaign"],
  signup_started: ["method"],
  signup_completed: ["method", "time_to_signup_ms"],
  first_value_event: ["feature"],
  session_started: ["day_of_cohort"],
  checkout_started: ["plan", "amount", "currency"],
  checkout_completed: ["plan", "amount", "currency", "order_id"],
  subscription_created: ["plan", "amount", "currency", "interval"],
  subscription_cancelled: ["plan", "reason"],
  share_initiated: ["channel"],
  share_completed: ["channel"],
  invite_redeemed: ["inviter_id"],
};

/** Returns the AARRR stage a standard event belongs to, or null if unknown. */
export function stageForEvent(name: string): AarrrStage | null {
  for (const [stage, events] of Object.entries(STANDARD_EVENTS) as [
    AarrrStage,
    readonly string[],
  ][]) {
    if (events.includes(name)) return stage;
  }
  return null;
}
