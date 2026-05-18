/**
 * monitoring_kit_nextjs/v1 — health-check route handler.
 *
 * Place at `app/api/health/route.ts` (App Router). Exposes GET /api/health.
 * Returns 200 when the app is up and its dependencies answer, 503 otherwise.
 * Point a free external uptime monitor (UptimeRobot / cron-job.org) here.
 *
 * RECIPE_PARAM markers (leave intact — substituted at instantiation):
 *   // RECIPE_PARAM:HEALTH_ROUTE=app/api/health/route.ts
 */

// Always run this route dynamically — never cache a health check.
export const dynamic = "force-dynamic";

/**
 * Replace the body with the project's real dependency checks — e.g. a
 * `SELECT 1` against the database, or an upstream-API ping. Kept trivial
 * here so the template type-checks before the project wires its own client.
 */
async function checkDependencies(): Promise<Record<string, boolean>> {
  return { app: true };
}

export async function GET(): Promise<Response> {
  const checks = await checkDependencies();
  const ok = Object.values(checks).every(Boolean);
  return new Response(
    JSON.stringify({ status: ok ? "ok" : "degraded", checks }),
    {
      status: ok ? 200 : 503,
      headers: { "content-type": "application/json" },
    },
  );
}
