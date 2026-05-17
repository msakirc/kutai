# Handoff — Wiring sweep of Z1 / Z6 / Z8 / Z9

**Date:** 2026-05-18
**Sibling of:** `docs/handoff/2026-05-18-wiring-sweep-z2-z3-z4-z5-z10.md` and
`docs/handoff/2026-05-17-wiring-audit-z0-handoff.md`. With this sweep the full
Z0–Z10 + yalayut wiring picture is closed. (Yalayut is being wired in a separate
parallel session — not covered here.)
**Method:** four parallel read-only agents, one per zone. Same method as the prior
sweeps: enumerate every shipped surface, trace for a real production trigger.

**Result:** uneven. **Z8 is the worst zone audited in any sweep** — two P0s, one of
which (recipes don't load) was predicted by the Z5 sweep and is now confirmed. **Z1 and
Z6 are the best-wired zones in the whole project** — Z1's 8 previously-dead post-hooks
are all genuinely fixed; Z6 has no P0/P1 orphans at all. Z9's earlier dead-pipeline fix
held; one real P1 remains (`metric_emit`).

---

## Priority table — all four zones

| Pri | Zone | Feature dead | Root cause | Fix |
|---|---|---|---|---|
| **P0** | Z8 | All 13 ops recipes (`backup_verify`, `cve_scan`, `cost_monitor`, `dependency_hygiene`, `synthetic_check`) undiscoverable | one-level dir layout (`backup_verify_sqlite_v1/recipe.yaml`); `list_recipes()` requires two-level `<name>/<version>/recipe.yaml` | restructure dirs |
| **P0** | Z8 | `alert_triage` classifies severity then stops — never reaches `oncall_agent` | `alert_triage.py:49` hardcodes `"oncall_routed": False`; no `enqueue()` of an oncall task | wire enqueue |
| **P1** | Z8 | `/ask` saves a ticket + acks the user but the answer never comes | `cmd_ask` (`telegram_bot.py:11158`) exits after reply; never enqueues a `support_tier1` task | add enqueue |
| **P1** | Z8 | `synthetic_check` executor unreachable | no `if action == "synthetic_check"` branch in `mr_roboto/__init__.py` dispatch | add branch |
| **P1** | Z8 | 5 ops crons never fire (backup_verify, dependency_hygiene, cve_scan, secret_scan, cost_pull) | none seeded in `cron_seed.INTERNAL_CADENCES` | add 5 seed rows |
| **P1** | Z9 | `metric_emit` growth_events never written → `/northstar` trend always empty, `roadmap_sync` always flags "untracked", `investor_bullets` starved | no production code writes `kind="metric_emit"`; PostHog events never sync back into `growth_events` | add metric sync |
| **P2** | Z8 | No ongoing on-call mission ever exists; `integration_mappings` always empty | nothing creates a `kind='ongoing'` on-call mission; webhooks route to `mission_id=None` | provision mechanism |
| **P2** | Z6 | `audit_completeness_check` post-hook path fails when dispatched | `audit_completeness_check` missing from the mr_roboto post-hook dispatch tuple (`__init__.py:3384`); cron path works, post-hook path doesn't | add to tuple |
| **P2** | Z6 | `vendor_call` LLM tool may be invisible to agents | tool is in `_optional_tools`, not main `TOOL_REGISTRY`; unverified whether agent tool-building scans `_optional_tools` | verify + fix |
| **P2** | Z1 | `derive_token_tag_signature` verb has no caller | no i2p step / cron / command dispatches it | add i2p step near 5.0 |
| **P2** | Z1 | `kill_preview_url` has no standalone caller | only invoked internally by `emit_preview_url` re-arm; no mission-end cleanup step | add cleanup step |
| **P3** | misc | Z1 `propose_spec_patch_from_html_diff` unwired; Z6 `expected_output_schema` never validated on resolve; Z9 non-i2p missions skip hypothesis/north-star; Z9 `score_backlog` skips silently on classifier failure; Z9 reinforce model-resolver fragile title-join | — | see below |

---

## Z8 — operations (worst zone; 2× P0)

**P0 — all 13 ops recipes are undiscoverable.** Confirmed the lead from the Z5 sweep.
`list_recipes()` (`src/infra/recipes.py:236`) iterates `root/<name>/<version>/recipe.yaml`
— a two-level layout. Every Z8 ops recipe is one level deep:
`recipes/backup_verify_sqlite_v1/recipe.yaml`, `cve_scan_*`, `cost_monitor_*`,
`dependency_hygiene_*`, `synthetic_check_*` (13 total). Runtime confirms `list_recipes()`
returns 14 recipes — all mobile/Z2/Z9 — and **zero Z8 ops recipes**. `match_recipe()` /
`pick_recipe()` never surface any of them. (The 6 incident playbooks are fine — they use
`playbook.yaml` + a two-level layout and a separate `list_playbooks()` loader.)
Fix: move each ops recipe into `<name>/v1/recipe.yaml`.

**P0 — `alert_triage` → `oncall_agent` routing gap.** Webhook → Beckman enqueues
`alert_triage` → `alert_triage.py` classifies severity → and stops. Line 49 hardcodes
`"oncall_routed": False` with a `# TODO T4` comment. Nothing ever enqueues an
`oncall_agent` task from the classified severity. The `oncall_agent` exists and is
registered but is never dispatched. The webhook→triage→oncall loop is open-circuit.

**P1 — `/ask` never answers.** `cmd_ask` (`telegram_bot.py:11077`) retrieves docs from
`support_docs`, saves a `tickets` row, acks the user "Full answer coming." — then exits.
No `general_beckman.enqueue()` of a `support_tier1` task. `support_tier1` is instantiated
but never dispatched. Tickets pile up `status='open'`, never answered.

**P1 — `synthetic_check` executor unreachable.** `executors/synthetic_check.py` (full
Lighthouse/k6 + `perf_baselines` regression diff) exists, but `mr_roboto/__init__.py`
has no `synthetic_check` dispatch branch — a triggered task would hit the unknown-action
error path. (Also doubly dead via the P0 recipe-layout bug.)

**P1 — 5 ops crons never fire.** `backup_verify`, `dependency_hygiene`/`dependency_scan`,
`cve_scan`, `secret_scan`, `cost_pull` all have mr_roboto dispatch branches but **no row
in `cron_seed.INTERNAL_CADENCES`**. 39 cadences are seeded (Z6/Z7/Z9/Z10) — zero Z8 ops
scans. The backup/CVE/cost/secret hygiene loop never runs.

**P2 — no ongoing on-call mission.** `missions.kind/lifecycle_state/cursor`, the
`ongoing` lane, `find_resumable()`, `/stop_ops`, `integration_mappings` all exist — but
no code ever *creates* a `kind='ongoing'` on-call mission or seeds `integration_mappings`.
`_route_to_mission()` returns `None` for every webhook → `alert_triage` runs with
`mission_id=None`. Needs a `/start_oncall <product>` command or an i2p step.

**P2 — `monitoring_kit_*_v1` recipes never authored** (known deferred; i2p step 13.3
correctly stays `[NEEDS-REAL-TOOLS]`).

Self-admitted deferred list re-verified: (d) `/webhook/__health` is actually SHIPPED
(`webhook_listener.py:45`) — the Updates log was wrong. (b) `/force_action` confirmed
not shipped. (c) on-call verb stubs confirmed fail-loud (by design).

Correctly wired: webhook listener boot (`run.py:437`, port 9882) + signature verify +
dedup; `alert_triage` mechanical dispatch; severity classifier; `action_cooldowns` +
enforcement; `oncall_action` dispatch + fail-loud stubs; `escalate_to_founder` +
`escalation_policy` + Twilio SMS; `/stop_ops`, `/ops_log`; lifecycle columns + `ongoing`
lane + resumption; `generate_playbooks` step + 6 incident playbooks.

---

## Z9 — growth (one real P1; earlier fix held)

**P1 — `metric_emit` growth_events are never written.** Confirmed (the Z9 Updates log
admits it). No production code writes `insert_growth_event(kind="metric_emit", ...)`. The
`analytics_instrumentation` recipe ships PostHog client/server shims, but nothing syncs
those PostHog events back into KutAI's `growth_events`. Downstream: `/northstar` trend
always shows "No measured values yet"; `analytics_digest` growth_events section is empty;
`roadmap_sync` always flags every north-star as "untracked"; `investor_bullets`
`_fetch_z6_metrics()` always returns `{}` (this is the same starvation flagged as Z7
item A9 in the 2026-05-17 handoff — common root cause). Fix: extend the already-weekly
`analytics_digest` executor to pull PostHog summaries and write them as `metric_emit`
rows.

**P2 — hypothesis + north-star dead for non-i2p missions.** `record_hypothesis` (i2p
step 7.0y) and `inject_north_star` (i2p step 8.0ns) are explicit i2p steps. Missions
created via `/task` or direct Beckman enqueue never hit Phase 7/8, so they record no
hypothesis and get no `north_star` context. Acceptable if all real missions go through
i2p; flag otherwise.

**P3 — `score_backlog` silently skips on classifier failure.** `classify_signals.py:254`
enqueues `score_backlog` only when `written > 0`. If the `signal_classifier` agent fails
or classifies nothing, the weekly backlog recompute is silently skipped — no retry, no
DLQ. Fix: enqueue `score_backlog` unconditionally, or add a fallback in the error path.

**P3 — reinforce model-resolver fragile join.** `record_verdict.py:194` joins
`tasks.title = model_pick_log.task_name` (free-form strings). On mismatch it falls back
to "most recent model overall" — may reinforce the wrong model. Fix: add `task_id` to
`model_pick_log` and join by id.

Note: the other 4 planned Z9 post-hook kinds (`record_hypothesis`, `verdict_check`,
`backlog_score_recompute`, `sunset_score_recompute`) were **legitimately re-implemented**
as explicit i2p steps + global crons — not gaps. Only `metric_emit` is a real hole.

Correctly wired (verified): signal-intake webhooks + `redact_user_pii`;
`classify_signals`→`score_backlog` chain (the earlier dead-pipeline fix held);
`signal_classify_sweep` / `verdict_window_sweep` / `dlq_signal_review` /
`sunset_score_recompute` / `roadmap_northstar_sync` crons; `record_verdict` full chain;
reinforce nudge write + `reinforce_bonus` read in selection; `analytics_digest` +
synthesis agent + `/digest`; `assign_variant` A/B; all Z9 Telegram commands (real, not
stubs).

---

## Z6 — real-world bridge (best-wired; no P0/P1)

**P2 — `audit_completeness_check` post-hook path is broken.** The handler module
(`posthook_handlers/audit_completeness_check.py`) is real, and the **cron path** works
(`mr_roboto/__init__.py:3342` handles `action == "audit_completeness_check"`). But the
**post-hook-via-mechanical path** routes through the `copy_compliance_review` /
`brand_voice_lint` / `briefing_compose` tuple at `__init__.py:3384` — and
`audit_completeness_check` is **not in that tuple**. A post-hook-triggered
`audit_completeness_check` fails to dispatch. Fix: add it to the tuple.

**P2 — `vendor_call` LLM tool may be invisible to agents.** The tool is registered in
`_optional_tools` (`src/tools/__init__.py:338`), not the main `TOOL_REGISTRY`. The
per-agent allowlist + audit context are correct, but if `build_tools_for_agent()` (or
equivalent) only scans `TOOL_REGISTRY`, agents never actually see `vendor_call`. Verify
and, if so, include `_optional_tools` in the agent tool set.

**P3 — `expected_output_schema_json` never enforced on founder-action resolution.** The
field is stored and rendered, but `/action_done` accepts any free-text payload without
validating it against the schema (credential-paste actions especially). Schema validation
exists at the credential-store level; wire it at `update_status()` for
`kind='credential_paste'`.

Correctly wired (verified): `founder_actions` table/repo/Telegram (`/actions`,
`/action_done`, inline buttons, notifier); `z6_admission` gate in `next_task()` + park +
unblock; Coulson detect-and-bail; all 10 vendor configs + 10 credential schemas; Apple
JWT + Google SA adapters; compliance step 12.1/12.1b + 8 templates + GDPR/CCPA; all 6
Stripe executors + their crons; `credential_rotation_reminder` + `compliance_template_
staleness` crons; `vendor_call` mechanical executor; `register_artifact`.

---

## Z1 — pre-code (cleanest zone; 8 prior dead hooks all fixed)

Verified: the 8 post-hooks that were dead at `z1-complete` tag time and fixed by
`171ffefd` (`compliance_template_present`, `compliance_blocker_check`,
`find_similar_missions`, `index_idea_fingerprint`, `surface_prior_mission_hints`,
`prior_art_min_coverage`, `verify_falsification_present`, `critic_gate`) are **all
genuinely wired now** — registry + apply.py dispatch + explicit i2p step each.

Three orphan verbs remain — registered dispatch branch, no production caller:

- **P2 — `derive_token_tag_signature`** (`mr_roboto/__init__.py:1140`) — no i2p step,
  cron, or command. Likely intended as a sibling of step 5.0 (`verify_taste_emphasis_
  shape`). Fix: add the i2p step.
- **P2 — `kill_preview_url`** (`__init__.py:1401`) — invoked only internally by
  `emit_preview_url`'s idempotent re-arm; no standalone mission-end cleanup step. Fix:
  add a cleanup step (e.g. `15.14z`) or accept implicit lifecycle.
- **P3 — `propose_spec_patch_from_html_diff`** (`__init__.py:1357`) — the C17/A20
  two-way HTML-edit-reflection verb; no caller. The `propagate:` Telegram button reaches
  `propagate_asset_change`, not this. Fix: wire a Telegram inline button on the
  `annotate_html_oids` / `regen_artifact` result.

Correctly wired: all phase 0–6 mechanical verify steps; `spec_consistency_check` on
wave-starts 7.0z–12.0z; `find_prior_art` tool (in `TOOL_REGISTRY` + researcher
`allowed_tools` + step 1.0 hint); `ingest_visual` (Telegram photo flow);
`propagate_asset_change` (`/propagate`); `reverse_pitch` + `generate_intake_todo` (i2p
steps with Telegram clarify gates); `sade_kalsin` bash audit (quarterly cron);
`c21_paraflow_diff` (`/paraflow_check` + weekly cron).

---

## Consolidated picture — full Z0–Z10 wiring sweep now complete

| Zone | Sweep handoff | Headline state |
|---|---|---|
| Z0 | 2026-05-17 | only ~35-40% built — finish, don't sweep |
| Z1 | this doc | clean; 3 minor orphan verbs |
| Z2 | 2026-05-18 (z2-z3-z4-z5-z10) | `inject_lessons` dead (P1) |
| Z3 | 2026-05-18 | multi-file expansion dead → cascade (P2); dials ignored |
| Z4 | 2026-05-18 | founder visual loop dead (P1, one line) |
| Z5 | 2026-05-18 | whole mobile submit chain dead (P1) |
| Z6 | this doc | best-wired; 2× P2, 1× P3 |
| Z7 | 2026-05-17 (z7-unwired) | 8 unwired features |
| Z8 | this doc | **worst; 2× P0, 3× P1** |
| Z9 | this doc | `metric_emit` dead (P1); rest holds |
| Z10 | 2026-05-18 | `/mission_cost` stub (P0); calibration loop dead (P1) |
| yalayut | 2026-05-16 + parallel session in progress | demand-signal subsystem |

## Suggested order (this batch)

1. **Z8 P0s** — recipe-dir restructure (mechanical move of 13 dirs) and the
   `alert_triage`→`oncall_agent` enqueue. Without these the entire ops zone is inert.
2. **Z8 P1s** — `/ask` enqueue, `synthetic_check` dispatch branch, 5 cron seed rows.
   All small, all "connect existing code."
3. **Z9 `metric_emit`** — fold a PostHog→`growth_events` sync into the existing weekly
   `analytics_digest` executor. Unblocks `/northstar`, `roadmap_sync`, and Z7's
   `investor_bullets` in one move.
4. **Z6 P2s** — one-line tuple add for `audit_completeness_check`; verify+fix
   `vendor_call` tool visibility.
5. **Z1 orphan verbs** — three small i2p-step / Telegram-button additions.

As before: every fix is "connect existing correct code", and each needs a host-path
test — the unit suites passed *with* all of these bugs.
