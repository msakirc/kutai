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

Add `VERB_REVERSIBILITY["deploy_staging"] = "irreversible"`. **Required, not cosmetic:** without
it `get_reversibility` falls back to `DEFAULT_REVERSIBILITY="partial"` (under-gating), and
`test_reversibility_registry.py` enforces that every dispatched verb has an entry — so add the
verb entry + dispatch branch together or that test fails.

## Rewire the 7.13-class step (review H1)
7.13 is currently `agent_type="executor"` (LLM) with an empty-action payload. Rewire the i2p
step to run mechanically:
- `agent: "mechanical"`
- `payload.action: "deploy_staging"` + params:
  `{repo, backend_arch:"nestjs_render", targets:{frontend:"vercel", backend:"render",
  db:"neon", cache:"upstash"}, workspace:"mission_{id}/", app:{backend_dir,frontend_dir}}`
- `real_tool_kind: "vercel|render"` (H9; drop railway/fly).
- **Fresh missions:** the JSON edit is sufficient — the expander (`expander.py:480-495`) sets
  `context.executor="mechanical"`, `context.payload`, and `agent_type` from `agent:"mechanical"`
  at expansion time.
- **In-flight m90 7.13 (already-expanded row) — CORRECTED per review:** the named refreshers
  **cannot** do this. `refresh_workflow_agent_type` (`task_refresh.py:71`) explicitly EXCLUDES
  any transition into/out of `mechanical`, and `refresh_workflow_step_payload` only syncs
  `payload`, never `agent_type`/`executor`. So the frozen `agent_type="executor"` 7.13 row needs
  a **direct one-off DB reconcile**: set `agent_type` + `context.executor="mechanical"` +
  `context.payload` on that task row (or re-expand the step). Do NOT rely on the refreshers for
  the executor→mechanical flip.

## Ordering DAG (review H4 — the real integration)
Deterministic, abort-on-partial (mirror `stripe_provision_products.py`):

1. **preflight** — verify repo exists + reachable (Spec 1 git-prereq chain done). Abort if not.
2. **provision DB** — `neon.create_project` → capture `DATABASE_URL`. Idempotent: `list_projects`
   first, reuse if a project tagged for this mission exists.
3. **provision cache** — `upstash.create_redis` → capture `REDIS_URL`+`rest_token`. Idempotent
   via `list_redis`.
4. **deploy backend** — `render.create_service` from repo with **env vars set at create time**
   (`DATABASE_URL`, `REDIS_URL` in `serviceDetails.envVars`) so the first (auto-initiated) deploy
   boots with them. **CORRECTED per review:** create *auto-initiates* the first deploy (do NOT
   also `trigger_deploy` for first boot); and `update_env_vars` does **NOT** auto-deploy — if env
   changes AFTER boot, call `trigger_deploy` explicitly. Free tier = instance type in
   `serviceDetails`, not `plan:"free"`.
5. **poll backend** — `poll_until(render.get_deploy, status=="live", timeout, backoff)`; on
   `build_failed` → abort with the build log ref. **Size `timeout` to Render free-tier build
   reality (build+boot routinely several minutes)** — not the default 10 min if that's too tight.
6. **migrate** — run `prisma migrate deploy` against `DATABASE_URL` (out-of-band: Render release
   command, or a one-off shell step — Neon has no run-SQL REST, Spec 1 nuance). Verify exit 0.
7. **deploy frontend** — `vercel.deploy` with `NEXT_PUBLIC_API_URL` = backend public URL from
   step 5.
8. **poll frontend** — `poll_until(vercel.get_deployment, readyState=="READY", ...)`.
9. **health check** — HTTP GET the **public** backend `/health` (or root) + frontend URL.
10. **write artifacts** — `staging_environment{url, services:{frontend,backend,db,cache}}` +
    `staging_deployment_verified{deployed:true, health_check_passed:<bool>, checked_at, details}`.

## `poll_until` helper (review H1)
Bounded loop: **per-provider `max_wait`** — size to real build times (Render free build+boot can
run several minutes → e.g. 10–15 min, not a fixed 10), exponential backoff (5→10→20→30s cap), a
terminal-fail predicate (`build_failed`/`ERROR`) that aborts early, and a hard timeout →
`{ok:false, reason:"deploy_timeout"}`. Lives in the executor module (or a small
`mr_roboto/deploy_util.py`). Do **not** encode polling in the declarative config —
`HttpIntegration.execute` is one-shot.

## Anti-fake guard (review H3 — MANDATORY)
Mock mode is ON by default when `KUTAI_ENV != prod`. The registry tags the mock envelope
`mocked:true` (`registry.py:96`, `http_integration.py:271`).

> 🔴 **CORRECTED per Opus review — the tag does NOT survive the mr_roboto vendor_call path.**
> `mr_roboto/executors/vendor_call.py:272-278` returns `{ok, result:data, service, action,
> status_code}` and **drops `mocked`**. So if `deploy_staging` reaches vendors through that
> wrapper (as the stripe precedent does), the guard can NEVER fire — a full mock run could write
> `health_check_passed:true`. This is the exact failure the guard exists to prevent. **Two fixes
> (pick one, as an explicit task with a regression test):**
> - **(preferred) call `adapter.execute(action, params)` DIRECTLY** in `deploy_staging.py` and
>   inspect `result.get("mocked")` — bypass the vendor_call wrapper entirely for the deploy DAG.
> - **or** propagate the tag in the shared wrapper: `vendor_call.py:272` →
>   add `"mocked": result.get("mocked", False)` (fixes it for all vendor executors, but touches a
>   shared file → needs its own regression test).

The executor **must refuse to certify a live deploy from mocked responses**:
- If ANY provision/deploy/poll/health response carries `mocked:true` (after the fix above) →
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
**Also (review): Render free web services spin down on idle** — the first health request after
deploy can be a cold start (~1 min) or a transient 502-during-wake. Treat slow-first-response /
502-while-waking as **retryable**, distinct from a genuine health failure; give the health check
its own bounded retry with backoff.

## Idempotency + abort-on-partial
- List-before-create for every provisioned resource (Neon project, Upstash db, Render service);
  reuse a resource tagged for this mission (a `kutay_mission_{id}` name/label) instead of
  creating duplicates on retry.
- On any step failure, return `{ok:false, reason, partial:{provisioned:[...]}}` — do not leave
  the artifact half-written; downstream (7.14+) depends on a complete `staging_environment`.

## Confirmation gate decision (review H8)
7.13 is `reversibility:"irreversible"` but `cost_estimate=0` → the `z6_admission` cost_ack gate
is **skipped** (`z6_admission.py:313`, verified). **CORRECTED per review — prefer the already-wired
lever over faking cost:** `mr_roboto.run()` has a founder-confirmation gate that auto-arms on
`irreversible` **independent of cost** via `KUTAI_CONFIRM_POLICY=irreversible_only`
(`__init__.py:552-576`), routing through the clarification/`waiting_human` path. Set
`confirm_policy:"irreversible_only"` (task context or env) on the deploy step rather than
extending admission or setting a nominal `cost_estimate_usd`. Rationale: a live deploy publishes a
public URL — an irreversible outward-facing action deserves a confirmation even at $0.
**Interaction to handle in the plan:** this gate returns `needs_clarification` and parks the task,
then resumes on the founder's typed reply — the executor must tolerate a parked-then-resumed
dispatch (the `git_commit` two-pass mechanical precedent, `__init__.py:820-899`, shows it works).
Default: require ack; opt-out via env for fully-autonomous runs.

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
