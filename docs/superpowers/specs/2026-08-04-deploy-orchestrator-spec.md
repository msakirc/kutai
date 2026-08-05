# Spec 2 — Mechanical deploy orchestrator (`deploy_staging`)

**Date:** 2026-08-04
**Status:** approved design → ready for implementation plan
**Depends on:** Spec 1 (`2026-08-04-deploy-adapters-spec.md`) — adapters + git-prereq chain
**Adversarial review:** GO-WITH-CHANGES — incorporates H1, H3, H4, H5, H7, H8, H9.

## Purpose
Turn Spec 1's atomic adapter actions into a **single deterministic mechanical executor** that
stands up a real $0 staging environment and writes m90 task 7.13's two artifacts. Per review
H1, this is **NOT** an LLM executor chaining `vendor_call` — 7.13's own instruction says run
"in mr_roboto," and the codebase precedent (`stripe_provision_products.py`) is a bespoke Python
executor. A stateless LLM re-deriving a poll-until-ready loop across ReAct iterations is exactly
the weak-model output-contract flakiness the project has engineered away from.

## Deliverable
`packages/mr_roboto/src/mr_roboto/executors/deploy_staging.py` with `async def run(task) -> dict`
returning `{ok: bool, ...}`, dispatched from `_run_dispatch` in
`packages/mr_roboto/src/mr_roboto/__init__.py`:

```python
if action == "deploy_staging":
    from mr_roboto.executors.deploy_staging import run as _deploy_run
    try:
        res = await _deploy_run(task)
        return Action(status="completed" if res.get("ok") else "failed",
                      error=None if res.get("ok") else res.get("reason"), result=res)
    except Exception as e:
        return Action(status="failed", error=str(e))
```

Add `VERB_REVERSIBILITY["deploy_staging"] = "irreversible"`.

## Rewire the 7.13-class step (review H1)
7.13 is currently `agent_type="executor"` (LLM) with an empty-action payload. Rewire the i2p
step to run mechanically:
- `agent: "mechanical"`
- `payload.action: "deploy_staging"` + params:
  `{repo, backend_arch:"nestjs_render", targets:{frontend:"vercel", backend:"render",
  db:"neon", cache:"upstash"}, workspace:"mission_{id}/", app:{backend_dir,frontend_dir}}`
- `real_tool_kind: "vercel|render"` (H9; drop railway/fly).
- For in-flight m90: `payload`/`agent_type` reach expanded rows via
  `task_refresh.refresh_workflow_step_payload` + `refresh_workflow_agent_type` (mechanical
  payload does NOT hot-reload through coulson — must use these dispatch-time refreshers).

## Ordering DAG (review H4 — the real integration)
Deterministic, abort-on-partial (mirror `stripe_provision_products.py`):

1. **preflight** — verify repo exists + reachable (Spec 1 git-prereq chain done). Abort if not.
2. **provision DB** — `neon.create_project` → capture `DATABASE_URL`. Idempotent: `list_projects`
   first, reuse if a project tagged for this mission exists.
3. **provision cache** — `upstash.create_redis` → capture `REDIS_URL`+`rest_token`. Idempotent
   via `list_redis`.
4. **deploy backend** — `render.create_service` (from repo, `plan:free`) with **env vars set
   BEFORE first boot** (`DATABASE_URL`, `REDIS_URL`) via `create_service` payload or
   `update_env_vars` then `trigger_deploy`. Note: Render env-var change triggers a redeploy —
   order matters.
5. **poll backend** — `poll_until(render.get_deploy, status=="live", timeout, backoff)`; on
   `build_failed` → abort with the build log ref.
6. **migrate** — run `prisma migrate deploy` against `DATABASE_URL` (out-of-band: Render release
   command, or a one-off shell step — Neon has no run-SQL REST, Spec 1 nuance). Verify exit 0.
7. **deploy frontend** — `vercel.deploy` with `NEXT_PUBLIC_API_URL` = backend public URL from
   step 5.
8. **poll frontend** — `poll_until(vercel.get_deployment, readyState=="READY", ...)`.
9. **health check** — HTTP GET the **public** backend `/health` (or root) + frontend URL.
10. **write artifacts** — `staging_environment{url, services:{frontend,backend,db,cache}}` +
    `staging_deployment_verified{deployed:true, health_check_passed:<bool>, checked_at, details}`.

## `poll_until` helper (review H1)
Bounded loop: `max_wait` (e.g. 10 min), exponential backoff (5→10→20→30s cap), a terminal-fail
predicate (`build_failed`/`ERROR`) that aborts early, and a hard timeout → `{ok:false,
reason:"deploy_timeout"}`. Lives in the executor module (or a small `mr_roboto/deploy_util.py`).
Do **not** encode polling in the declarative config — `HttpIntegration.execute` is one-shot.

## Anti-fake guard (review H3 — MANDATORY)
Mock mode is ON by default when `KUTAI_ENV != prod`. The mock envelope is tagged `mocked:true`.
The executor **must refuse to certify a live deploy from mocked responses**:
- If ANY provision/deploy/poll/health response carries `mocked:true` →
  `staging_deployment_verified.health_check_passed = false`,
  `reason = "mock_mode_active"`, and `ok=false` for a run that was supposed to be live.
- A **real** deploy run requires `KUTAI_VENDOR_LIVE=1` (explicit, audited) at the task boundary.
- This keeps 7.13's `must_be_true: health_check_passed` honest — no mock can satisfy it.
- Mock-mode runs are still valuable for **CI** (full-chain test proving the DAG + guard), they
  just cannot produce a `health_check_passed:true` artifact.

## Health check (review H7)
Use the **public** deployment URL from the poll result. `HttpIntegration._validate_url` blocks
private/internal hosts — if a provider hands back a `*.internal`/private preview URL, treat the
SSRF `ValueError` as a distinct **"url not yet public" retryable** state, not "deploy failed."
Health check via the raw `http_request` tool path or a direct httpx GET against the public URL.

## Idempotency + abort-on-partial
- List-before-create for every provisioned resource (Neon project, Upstash db, Render service);
  reuse a resource tagged for this mission (a `kutay_mission_{id}` name/label) instead of
  creating duplicates on retry.
- On any step failure, return `{ok:false, reason, partial:{provisioned:[...]}}` — do not leave
  the artifact half-written; downstream (7.14+) depends on a complete `staging_environment`.

## Confirmation gate decision (review H8)
7.13 is `reversibility:"irreversible"` but `cost_estimate=0` → the existing `z6_admission`
cost_ack gate is **skipped**. Decision for the plan: **gate an irreversible deploy on a founder
ack regardless of $0 cost** (either extend the admission check to fire on
`reversibility=="irreversible"` for deploy verbs, or set a nominal `cost_estimate_usd` to trip
`cost_ack`). Rationale: a live deploy publishes a public URL — an irreversible, outward-facing
action deserves a confirmation even at $0. Default: require ack; opt-out via env for autonomous
runs.

## backend_arch (review H5)
`backend_arch` is read from params but **hard-wired to `nestjs_render`** for now. Serverless
(`serverless_workers`) is a deferred separate track (different codegen + Prisma driver). The
executor asserts `backend_arch == "nestjs_render"` and returns a clear
`reason:"serverless_not_yet_supported"` otherwise — no silent wrong-path.

## Testing
- **Mock-mode full-chain e2e** (offline): drive `run(task)` end-to-end with all adapters mocked
  (`mocked:true`); assert the DAG executes in order, AND that the anti-fake guard forces
  `health_check_passed:false` + `reason:"mock_mode_active"`. This is the primary CI test.
- `poll_until` unit tests: ready-on-Nth-poll, terminal-fail early-abort, timeout.
- Ordering test: env vars set before backend deploy; frontend gets backend URL.
- Idempotency test: second run reuses provisioned resources, no duplicates.
- Dispatch test: `mr_roboto.run({payload:{action:"deploy_staging"}})` routes to the executor
  and returns an `Action`.
- **Live smoke** (founder-run, gated `KUTAI_VENDOR_LIVE=1` + stored creds): one real HabitHub
  staging deploy; verify a real `health_check_passed:true` artifact. NOT in CI.

## m90 7.13 completion path (real, honest)
1. Spec 1 lands (adapters + git-prereq). 2. Founder provisions: GitHub PAT, Render/Neon/Upstash
tokens via `/credential add`, `KUTAI_VENDOR_LIVE=1`. 3. This executor lands, 7.13 rewired to
mechanical. 4. `/restart` (`.py` change). 5. Re-pend 7.13 → executor runs the real DAG → writes
a genuine `health_check_passed:true`. No mock satisfies the gate.

## References
- Spec 1: `docs/superpowers/specs/2026-08-04-deploy-adapters-spec.md`
- Research: `docs/research/2026-08-04-free-deploy-stack-research.md`
- Precedent: `packages/mr_roboto/src/mr_roboto/executors/stripe_provision_products.py`,
  dispatch at `packages/mr_roboto/src/mr_roboto/__init__.py:2882`.
- Refreshers: `task_refresh.refresh_workflow_step_payload` / `refresh_workflow_agent_type`.
