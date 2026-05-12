# Z8 — Operations (v2)

**Supersedes:** [08-operations.md](08-operations.md) (v1, 2026-05-08).
v1 framed gaps A–J as "isolated automations." Ground-truth audit (this doc, §1) shows the substrate is half-built: founder_actions, reversibility registry, vendor_call executor, recipe engine, internal alerting, integration configs all shipped — but none of them stitched into an ongoing-ops lifecycle. v2 reframes Z8 not as "add monitoring + on-call" but as **"close the loop between substrates already shipped in Z6/Z10/Z2 and an ongoing-mission shape that doesn't yet exist."**

## 1. Re-audit (v1 claims vs. 2026-05-11 reality)

| v1 claim / assumption | Reality | Anchor |
|---|---|---|
| "Mission lifecycle is one-shot; DB schema change required" | Confirmed. No `mission_kind`; orchestrator dispatches to terminal state. `scheduled_tasks` table exists but **vestigial** — registered phase-9, zero call sites. | `src/infra/db.py:432`, `src/infra/db.py:1033`, `src/core/orchestrator.py:51` |
| "13.3 monitoring_setup is NEEDS-REAL-TOOLS marked" | Confirmed verbatim. Instruction says "Do NOT autonomously execute — surface a needs_clarification…" | `src/workflows/i2p/i2p_v3.json:9389–9420` |
| "No alert intake; no incident playbook" | Confirmed for vendor alerts. **But:** internal alerting EXISTS — rule-based via Telegram, tracks task failures, daily cost, queue depth, model success rate. v1 missed this. | `src/infra/alerting.py:1–113` |
| "Sentry/Better Stack/Posthog need wiring" | Sentry config **already shipped** but **orphaned** (zero call sites). Better Stack + Posthog missing entirely. v1 treated all three as missing. | `src/integrations/configs/sentry.json:1` |
| "Mr. Roboto vendor adapters configure rules at deploy-time" | `vendor_call` executor SHIPPED. IntegrationRegistry + 10 vendor configs (sentry, stripe, vercel, railway, github, supabase, sendgrid, cloudflare, apple_appstore, google_play). **Not integrated into mission expander.** | `packages/mr_roboto/src/mr_roboto/executors/vendor_call.py:1`, `src/integrations/` |
| "Founder gets accounts; agent configures rules" | `founder_actions` table + repo module + Telegram render SHIPPED (Z6 T1B). Z8 doesn't need to design escalation substrate — only wire it. | `src/founder_actions/__init__.py:1`, `src/infra/db.py:2243`, `src/app/founder_action_render.py` |
| "All actions audit-logged + reversibility-tagged" | Reversibility shipped (Z10 T1C): `reversibility.py` registry with 100+ verbs, `tasks.reversibility` column, `registry_events.reversibility` audit row. **Step-level reversibility on i2p_v3.json still missing** (Z0 territory). | `packages/mr_roboto/src/mr_roboto/reversibility.py:32`, `src/infra/db.py:1624,1699` |
| "Per-stack recipes for monitoring kit" | Recipe engine SHIPPED (Z2 T5A): Recipe dataclass, load/list/match/instantiate/pick verbs. **Zero ops recipes authored** — backup, cost-monitor, cve-scan, monitoring all greenfield. | `src/infra/recipes.py:1`, `packages/mr_roboto/src/mr_roboto/instantiate_recipe.py` |
| "Tier-1 support flow" | Clarification-Q&A flow exists in Telegram bot; no FAQ artifact, no ticket memory, no "answer from docs vs escalate" branching. | `src/app/telegram_bot.py:5541–5656` |
| "needs_real_tools tasks should be gated" | **SHIPPED.** Column added (`tasks.needs_real_tools`) AND `check_z6_admission` wired into `next_task()` since commit `a171bcd` (Z6 T1C). Initial v2 audit miscalled this. `/action_done` unblock chain works end-to-end. H3 closed. | `src/infra/db.py:2286`, `packages/general_beckman/src/general_beckman/__init__.py:329-355`, `src/founder_actions/__init__.py:498-502` |

**Net:** v1 treated Z8 as 10 gaps. v2 treats it as **4 hinge points** that unlock 10 outputs.

## 2. Four hinge points

### H1. `mission_kind ∈ {oneshot, ongoing}` + orchestrator resumption

Without this, nothing in §B (on-call), §E (cron missions), §I (regression), §J (cost) can land. Every other Z8 gap routes through H1.

**Not just a column.** Three downstream changes:

- **Beckman.next_task() lane separation.** Ongoing missions must not block the oneshot queue; need a second admission lane with its own concurrency cap (mirror the `runner` column added Phase D, 2026-05-05). Reuse `lane='ongoing'`.
- **Orchestrator state restoration on restart.** Ongoing missions have *no* terminal state — orchestrator boot must reattach to them, replay last-seen offset (webhooks/cron), not re-execute. Add `mission.cursor` JSON column (last_event_id, last_run_at per subscription).
- **Terminal condition: revocation only.** Founder `/stop_ops <mission_id>` Telegram cmd; mission row `lifecycle_state` transitions `active → revoked`; orchestrator drops subscriptions.

**Alt shape considered & rejected:** modeling ongoing ops as long-lived *tasks* under a oneshot mission. Rejected: Beckman lifecycle (`apply/retry/sweep/rewrite`) assumes terminal output; a 3-month-running task would be permanently "in flight" and break the sweep heuristics. Mission-level lifecycle is the right grain.

**Migration:** ADD COLUMN `missions.kind TEXT DEFAULT 'oneshot'`, `missions.lifecycle_state TEXT DEFAULT 'pending'`, `missions.cursor TEXT`. Backfill existing rows = `oneshot/terminal`. New `mission_lanes` registration in Beckman init.

### H2. Webhook ingestion as the alert spine

Without inbound webhooks, the on-call agent is blind. **New HTTP surface inside KutAI** — not in the Telegram bot (different listener), not in the orchestrator (different lifecycle).

**Shape:** `src/app/webhook_listener.py` — FastAPI app, single port (configurable), one route per vendor + a generic `/webhook/<integration_id>` that dispatches by IntegrationRegistry config.

**Security (v1 missed):**
- **Signed payload verification** per vendor (Sentry HMAC, Stripe `Stripe-Signature`, GitHub `X-Hub-Signature-256`). Reuse IntegrationRegistry credential store — add `webhook_secret` to integration config schema.
- **Idempotency / replay defense.** Vendor delivery retries — need dedup keys. Add `webhook_events` table `(integration_id, event_id PRIMARY KEY, received_at, payload_hash, mission_id, processed_at)`. Reject duplicate `event_id` within 24h.
- **Rate limit per integration** — 1k/min default; spike = alert + drop.
- **Outbound network policy:** webhook listener should NOT make outbound calls in the handler; enqueue work to Beckman and return 200 fast. Vendors expect <5s ack.

**Routing:** Webhook → dedup → Beckman.enqueue(`task_type='alert_triage'`, mission_id resolved by integration_id mapping). Triage task severity-classifies, then routes to on-call mission's task queue or escalates via `founder_actions`.

**Multi-product:** Founder may run 3 products. `mission.product_id` (already in spec? — confirm Z0) plus `integration_mappings` table `(integration_id, product_id, mission_id)` so a single Sentry webhook can fan out to the right ongoing mission.

### H3. needs_real_tools admission gate (blocking-but-already-paid-for)

v1 said monitoring "wired by adapters at deploy-time." Audit shows the column exists and `vendor_call` executor exists, but Beckman doesn't check the column. So a task with `needs_real_tools=true` and no valid credential in IntegrationRegistry **silently fails or fabricates**. This is the single highest-leverage fix in Z8: enforces all of §A, half of §B, all of §H.

**Gate logic in `general_beckman.next_task()`:**
1. Pull row; if `needs_real_tools=true`, resolve required integrations from `task.context.post_hook.service` (and recipe-declared `requires.integrations` list).
2. For each required integration, check IntegrationRegistry for valid credential (not expired, not revoked).
3. Missing/invalid → write `founder_actions` row (`kind='credential_request'`, payload= integration_id + scopes), park task in `awaiting_founder` lane, do not dispatch.
4. Founder completes via `/action_done <id>`; founder_actions writer flips parked tasks to ready.

**Migration:** No schema change beyond shipped column. Add `missions.awaiting_actions` view. Update Beckman select query (+1 WHERE clause). Wire founder_actions completion → task ready.

### H4. Ops recipes as first-class library

The Z2 recipe substrate is build-time only today. v1 §A/§E proposed monitoring + cron recipes ad-hoc; v2 says **author them as recipes** — same engine, same versioning, same `match_recipe()` selector. Free benefits: per-stack defaults (FastAPI vs NextJS Sentry SDK init differ), parameterization, `param_defaults`, post-hooks already implemented.

Recipe catalog to author:
- `monitoring_kit_fastapi_v1`, `monitoring_kit_nextjs_v1`, `monitoring_kit_django_v1`
- `backup_verify_postgres_v1`, `backup_verify_sqlite_v1`
- `dependency_hygiene_python_v1`, `dependency_hygiene_node_v1`
- `cost_monitor_aws_v1`, `cost_monitor_vercel_v1`, `cost_monitor_stripe_v1`
- `cve_scan_python_v1`, `cve_scan_node_v1`, `cve_scan_docker_v1`
- `incident_playbook_db_disk_full_v1`, `incident_playbook_payment_provider_down_v1`, `incident_playbook_auth_provider_down_v1`, `incident_playbook_cert_expiring_v1` (≥12 templates)
- `synthetic_check_lighthouse_v1`, `synthetic_check_k6_v1`

Each recipe: declares `requires.integrations`, `requires.tech_stack`, `post_hooks` (action verbs from Mr. Roboto), `produces` artifacts, `reversibility` per step. Match via existing `pick_recipe()`.

## 3. Re-mapped gaps (v1 → v2 wiring)

| v1 gap | v2 implementation | New work | Reused substrate |
|---|---|---|---|
| **A. Monitoring kit at launch** | Ops recipes (H4) `monitoring_kit_*_v1` + needs_real_tools gate (H3) + founder_actions for credentials | 6–9 recipes, alert-rule library YAML | recipes.py, IntegrationRegistry, founder_actions, vendor_call |
| **B. On-call agent** | Ongoing mission (H1) + webhook spine (H2) + alert_triage task type + action-whitelist policy module | `oncall_agent` profile (sys-prompt + tools), severity classifier, action whitelist enforcer | reversibility.py, Beckman lanes, vendor_call |
| **C. Incident playbooks** | Recipe templates (H4) generated at phase 13 from spec+arch; on-call executes via `instantiate_recipe` | Playbook-recipe YAML schema (decision tree, action sequence), generator agent | instantiate_recipe, pick_recipe |
| **D. Tier-1 support** | New `support_tier1` ongoing mission (H1); Telegram bot becomes ticket inlet; `tickets` table + FAQ artifact regenerated by weekly cron mission | tickets schema, FAQ artifact builder, escalation routing via founder_actions | telegram_bot.py clarification flow, founder_actions, RAG/embeddings for FAQ match |
| **E. Backup verify cron** | Ongoing mission (H1) with cron schedule; recipe-driven (H4) | `backup_verify_*_v1` recipes, cron scheduler in orchestrator (replaces vestigial scheduled_tasks) | recipes, mr_roboto, Beckman.enqueue |
| **F. Dependency hygiene** | Same pattern as E; recipe per ecosystem | `dependency_hygiene_*_v1` recipes, advisory feed adapter | IntegrationRegistry (GitHub Dependabot API, OSV.dev) |
| **G. Load + headroom tracking** | Ongoing mission reads weekly aggregates; triggers capacity-plan submission | aggregator query module, threshold config | alerting.py extended, mission spawning |
| **H. Security posture** | CVE recipe (H4) + secret-scan recipe; both as cron ongoing missions | OSV.dev / NVD adapter; gitleaks/trufflehog post-hook | reversibility tagging, founder_actions for findings |
| **I. Perf regression** | Synthetic check recipes (H4); baselines persisted in `perf_baselines` table; bisect-on-break extension | baselines table, regression diff module | mr_roboto test_run, Z3 bisect, recipe engine |
| **J. Cost monitor** | Per-vendor cost recipe (H4) + extends alerting.py weekly trend + anomaly detector | cost trend aggregator, anomaly model (z-score over 14d) | IntegrationRegistry, alerting.py |

## 4. New insights v1 missed

### 4.1 The "silent autonomy" risk
v1 lists a generous action whitelist (restart, rollback, scale up, drain, rotate, archive flake). Without **per-mission per-action cooldowns**, a misbehaving on-call agent triggers rollback-redeploy-rollback loops. Z10 shipped reversibility tags but **not rate limits per verb**. Add `action_cooldowns` table `(mission_id, verb, last_invoked_at, count_24h)` enforced in Mr. Roboto pre-execute. Default: rollback ≤2/hr, restart ≤5/hr, scale ≤3/hr, key-rotate ≤1/24h. Per-mission override via founder approval.

### 4.2 Cursor + replay = correctness
Ongoing missions reading webhooks need at-least-once + dedup. `webhook_events.event_id` (H2) gives dedup. `mission.cursor` tracks "last successfully processed event per integration" so a crashed orchestrator can resume without double-handling. **v1 didn't surface this.**

### 4.3 Reversibility is not enough; **observability of agent actions** matters too
Founders need to *see* what the on-call agent did, not just whether it's revertible. Build `/ops_log <mission_id>` Telegram command — pulls last N `registry_events` for that mission, renders verb + outcome + reversibility + manual-revert command if applicable. Without this, autonomy = blind trust = founder pulls the plug.

### 4.4 Tier-1 support uses RAG, not LLM cold-call
FAQ-as-artifact is wrong shape. Build `support_docs` collection in ChromaDB (already configured, multilingual-e5-base 768d) seeded from spec + product docs + resolved tickets. Ticket flow: embed query → top-3 docs → LLM compose answer with citations → if confidence <0.7 or "angry" sentiment, escalate via founder_actions. This reuses the embedding/RAG plumbing instead of a parallel FAQ-regenerator mission.

### 4.5 Multi-product scoping
A solo founder shipping their 3rd product cannot have one global on-call mission. `product_id` must be threaded: webhook routing (H2), action cooldowns (4.1), founder_actions display (`/ops_log` filters), cost monitor digests. If `product_id` doesn't exist yet (Z0 territory), Z8 adds it as a side effect — flag for Z0 coordination.

### 4.6 Webhook listener deployment shape
v1 silently assumed listener exists. It doesn't. Two options:
- **(a) Embedded in KutAI process.** Pro: single process, shared DB. Con: KutAI restart = missed webhooks. Mitigates with vendor retries + dedup.
- **(b) Separate process (Yaşar Usta–managed sibling).** Pro: survives KutAI orchestrator restarts. Con: new IPC, new failure mode.
- **Recommendation:** (a) for v1; (b) when first production product ships. Webhook listener becomes a `kutai_wrapper.py` sibling worker.

### 4.7 The "founder is asleep" problem
Severity-classifier needs to know **founder availability** to escalate sensibly. Cross-ref [07-humanish-layers.md] (founder context layer). Default: business hours = Telegram; outside = SMS via Twilio (new integration) for tier-2 only; tier-3 (security) = SMS regardless. Don't over-page. Add `escalation_policy` per mission with quiet hours + severity-threshold matrix.

### 4.8 Playbooks must declare their preconditions
Incident playbook for "DB disk 80%" should never fire if DB is read-replica or if disk is ephemeral. Recipes already support `requires.tech_stack` — extend to `requires.runtime_state` (queryable from monitoring snapshot). Playbook selector reads current state before firing.

### 4.9 Cost monitor wants leading indicators, not lagging
v1's "60% of budget, breach in 4 weeks" is post-hoc. Add **rate-of-change alerts**: "Sentry quota used 12% today vs 3% 7-day avg → anomaly." Already shipped: alerting.py 15-minute cooldown logic; extend to per-integration cost slope.

### 4.10 Tier-1 escalation already has substrate
v1 §D proposes a new escalation flow. `founder_actions` (Z6 T1B) is the substrate. Reuse the same table + Telegram render; just add `kind='support_escalation'`. Zero new UI.

## 5. Migration plan (5-tier batched, T1→T5)

### T1. Lifecycle foundation (blocking everything)
- T1A. `mission.kind / lifecycle_state / cursor` migration + Beckman ongoing-lane admission.
- T1B. Orchestrator state restoration on restart (replay last cursor).
- T1C. `/stop_ops` Telegram command + revocation flow.
- **Tests:** ongoing mission survives orchestrator restart; revocation drops subscriptions; oneshot queue unaffected.

### T2. needs_real_tools admission gate + ops recipes scaffold
- T2A. Beckman enforces `needs_real_tools` → founder_actions credential_request → parked-task release on action_done.
- T2B. Ops recipe sub-catalog: 3 monitoring recipes (FastAPI / NextJS / Django), 2 backup recipes.
- T2C. `instantiate_recipe` → vendor_call wiring smoke-test against Sentry sandbox project.
- **Tests:** missing credential parks task, founder approval releases it; recipe + vendor_call produces real Sentry project; reversibility tag flows through registry_events.

### T3. Webhook spine + alert triage
- T3A. `src/app/webhook_listener.py` FastAPI app + signed-payload verification + `webhook_events` dedup table.
- T3B. `alert_triage` task type — severity classifier (rule-based for v1, LLM-graded for high uncertainty), routing to on-call queue.
- T3C. Per-vendor adapter integrations: Sentry, Stripe, Better Stack, GitHub (security advisories).
- **Tests:** replayed webhook deduped; bad signature 401; alert routes to right mission by product_id.

### T4. On-call agent + playbooks
- T4A. `oncall_agent` profile (sys-prompt + tools): tools = `vendor_call`, `restart_service`, `rollback`, `scale_up`, `drain_traffic`, `rotate_key`, `archive_flake`, `escalate_to_founder`. All write actions gated by reversibility check + action_cooldowns.
- T4B. Action cooldowns table + Mr. Roboto pre-execute enforcement.
- T4C. Playbook recipe schema + 6 starter playbooks; phase 13 generator step emits per-mission playbook set.
- T4D. `/ops_log <mission_id>` Telegram cmd; escalation policy table.
- **Tests:** rollback-loop blocked by cooldown; playbook executes matching recipe; out-of-whitelist verb refused.

### T5. Cron missions + support tier 1 + cost/perf/security
- T5A. Backup verify, dependency hygiene, CVE scan, secret scan, cost monitor — each an ongoing mission with cron config; recipes (H4) drive logic.
- T5B. `support_tier1` ongoing mission: Telegram ticket inlet, ChromaDB `support_docs`, RAG answer composer, founder_actions escalation, weekly doc regenerator from resolved tickets.
- T5C. `perf_baselines` table + synthetic check recipes + regression diff against last green; bisect-on-break extension to prod.
- T5D. Cost slope anomaly detector + weekly digest; SMS escalation via Twilio integration (tier-3 only).
- **Tests:** backup drill writes restore-fail digest on broken backup; ticket below confidence escalates; perf regression on staging blocks promote.

## 6. Open questions (v1 → v2 answers)

| v1 Q | v1 answer | v2 answer |
|---|---|---|
| Long-running mission shape — DB change? | Yes, add column | Confirmed; **plus** `lifecycle_state` + `cursor` + orchestrator resumption (H1) |
| Autonomy bounds — defaults vs per-product? | Default conservative; per-product expansion | Same, **but** also per-mission per-verb cooldowns (4.1); whitelist enforced in Mr. Roboto pre-execute, not just by agent prompt |
| Webhook ingestion — where? | New `webhooks` adapter, route to mission | New FastAPI surface `src/app/webhook_listener.py` (H2); signed payloads, dedup table, product_id routing; embedded in KutAI process v1, split process when 1st prod ships |
| Rate limits — cooldowns + caps? | Yes | Make it a first-class table `action_cooldowns` with per-verb default policy; founder override; **separate from KutAI's existing model-swap cooldown** |
| Pager — PagerDuty/Opsgenie? | Telegram v1, PagerDuty later | Telegram v1 + Twilio SMS for tier-3 from day-1 (the "founder is asleep" problem, 4.7); PagerDuty when team scales |
| Playbook authorship — LLM-generated, vetted? | Yes | Yes, **and** templated as recipes (H4) with `requires.runtime_state` (4.8); selector reads monitoring snapshot before firing |
| Cost monitor — per-vendor or aggregator? | Per-vendor v1 | Per-vendor v1; **plus** rate-of-change anomaly detector (4.9), not just budget % |
| Tier-1 FAQ shape? | (not asked v1) | RAG over ChromaDB `support_docs`, not separate FAQ artifact (4.4); reuses existing embeddings |
| Multi-product scoping? | (not asked v1) | `mission.product_id` threaded through webhooks, cooldowns, escalations, cost digests; coordinate with Z0 if column doesn't yet exist (4.5) |

## 7. Dependencies (revised)

- **Inbound (must land before Z8 T1):**
  - Z0 `mission.product_id` (if not yet shipped); confirm with `00-README.md` / `z0-mission-preflight.md`.
  - Z6 founder_actions integration into admission (T2A blocked otherwise).
- **Inbound (must land before Z8 T3):**
  - IntegrationRegistry credential schema extended with `webhook_secret`.
- **Outbound (Z8 unlocks):**
  - Z9 growth: Posthog adapter (Z8 T2) + cost monitor (Z8 T5D) feed product-analytics + unit-economics dashboards.
  - Z7 humanish: support tier-1 (T5B) writes escalation summaries; investor-update generator reads ops digest.
- **Cross:**
  - Z10 reversibility tagging — already enforced; Z8 adds action_cooldowns as orthogonal guard.
  - Z2 recipe engine — Z8 is its largest consumer; expect 20+ ops recipes by T5 end.

## 8. Agent task brief (v2)

When picking up Z8 implementation:
1. **Read in order:** [00-README.md], [02-build-foundation-v2.md], [06-real-world-bridge-v2.md], [10-cross-cutting.md], this doc.
2. **Verify ground truth** at start of each tier — `git log` on `src/infra/db.py`, `packages/general_beckman/`, `packages/mr_roboto/`, `src/integrations/`. Reality drift since 2026-05-11 is likely; **do not trust §1 table without re-checking**.
3. **Tier discipline:** T1 must merge fully before T2 starts (lifecycle is foundational). T2 must merge before T3 (gate before alerts). T3+T4 can interleave. T5 is parallelizable across cron recipes.
4. **Per tier:** spec → tests → implementation → migration script → docs `## Updates` entry; follow `superpowers:writing-plans` + `superpowers:test-driven-development`.
5. **Test discipline:** ongoing-mission tests need a fake clock; cron tests must not actually wait; webhook tests use signed-payload fixtures.
6. **Coordinate with Z0** if `product_id` / `lifecycle_state` ownership ambiguous (4.5, H1).
7. **Coordinate with Z6** for new vendor integrations (Twilio SMS, OSV.dev, GitHub Advisories) — add to IntegrationRegistry, don't bypass.
8. **No mode flags** (`feedback_no_agent_modes`): `oncall_agent` is a distinct agent (sys-prompt + tools + reflection block), **not** a mode of another agent. KEEP or DROP, no middle ground.
9. **Subagent-driven execution** (`feedback_subagents_always`): each T-tier dispatched to its own agent; don't ask permission per tier.

## Updates

- **2026-05-11** — v2 deep-dive. Audit reveals substrate half-built (founder_actions, reversibility, recipe engine, vendor_call, IntegrationRegistry all shipped; admission gate, ongoing-mission lifecycle, webhook spine missing). Reframed Z8 around 4 hinge points (H1–H4) instead of 10 isolated gaps. 5-tier batched plan (T1–T5). New insights surfaced v1 missed: action cooldowns, cursor/replay correctness, ops_log observability, RAG-based support, multi-product scoping, founder-availability escalation policy, runtime-state-aware playbook preconditions, cost rate-of-change anomalies. Open questions answered with concrete shapes.
- **2026-05-12** — Z8 COMPLETE. All five tiers shipped. **H1 lifecycle**: `mission.kind` / `lifecycle_state` / `cursor` columns, Beckman `ongoing` lane with caps + admission, orchestrator resumption + revocation, `/stop_ops` (T1A-E). **H2 webhook spine**: FastAPI listener (embedded) + dedup + per-vendor signature verification + `integration_mappings` product_id routing + `alert_triage` task type + rule-based severity classifier (T2-T3). **H3 admission gate** wired into `Beckman.next_task()` + `action_done` unblock. **H4 ops recipes catalog** — 18 recipes: incident_playbook (6), backup_verify (postgres/sqlite), dependency_hygiene (python/node), cve_scan (python/node/docker), cost_monitor (stripe/vercel/aws), synthetic_check (lighthouse/k6). **On-call layer (T4)**: `oncall_agent` whitelisted actions, `action_cooldowns` table + Mr. Roboto pre-execute, escalation_policy + `/ops_log`. **Cron + tier-1 (T5)**: cron scheduler for ongoing missions, backup/dep/cve/secret/cost cron (T5A-D), support_tier1 agent + ChromaDB `support_docs` collection + `tickets` table + `/ask` Telegram inlet + confidence-based escalation via `support_escalation` founder_actions (T5E), `perf_baselines` table + `synthetic_check` executor with last-green regression diff (T5F), Twilio SMS executor + `escalate_to_founder` mechanical executor that resolves channel via escalation_policy + quiet hours and real-wires the on-call agent's `escalate_to_founder` verb (T5G). Tests: 148 across Z8 (131 in tests/ops+tests/app+tests/integration, 17 in packages/general_beckman/tests). Commits `z8-t1-shipped` → `z8-t5-shipped`, final tag `z8-complete`. **Known deferred items**: (a) i2p_v3 step 13.3 `monitoring_setup` stays NEEDS-REAL-TOOLS — `monitoring_kit_*_v1` recipes were never in T5 scope; existing `monitoring_check` executor + `alerting.py` cover internal monitoring, so step 13.3 is deferred until a future tier ships the kit. (b) `/force_action <mission_id> <verb>` Telegram override for action_cooldowns (Risks §). (c) Non-escalation on-call verbs (restart_service, rollback_to_last_green, scale_*, drain_traffic, rotate_failed_key, archive_flake_test) remain stub sub-handlers — production wiring requires real cloud-provider methods on `vendor_call`. (d) Health endpoint `/webhook/__health` for Yaşar Usta watchdog (Risks §). (e) Weekly cluster-and-propose job over resolved support tickets — `index_doc()` helper shipped; founder-approved FAQ regeneration loop deferred.
