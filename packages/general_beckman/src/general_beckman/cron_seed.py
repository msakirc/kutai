"""Internal-cadence seeder for General Beckman.

Seeds a fixed set of internal scheduled tasks (kind='internal') into the
scheduled_tasks table on startup. Uses upsert-by-(title, kind) semantics so
repeated calls are idempotent. A module-level flag avoids redundant DB round-
trips when the process is long-running.
"""
from __future__ import annotations

import asyncio
import json
import os

from src.infra.logging_config import get_logger

logger = get_logger("beckman.cron_seed")

# Canonical internal cadences.  Nothing outside this module should hardcode
# these — downstream consumers query by kind='internal'.
INTERNAL_CADENCES: list[dict] = [
    {
        "title": "beckman_sweep",
        "description": "Periodic Beckman queue sweep",
        "interval_seconds": 300,
        "payload": {"_marker": "sweep"},
    },
    {
        "title": "hoca_benchmark_refresh",
        "description": "Refresh Fatih Hoca benchmark cache",
        "interval_seconds": 300,
        "payload": {"_marker": "benchmark_refresh"},
    },
    {
        "title": "todo_reminder",
        "description": "Remind user of pending todos",
        "interval_seconds": 7200,
        "payload": {"_executor": "todo_reminder"},
    },
    # daily_digest + api_discovery cadences intentionally omitted —
    # their mr_roboto handlers were dropped in the Phase 2b Beckman refactor
    # and never re-implemented. Re-add here once mr_roboto gains matching
    # action handlers; until then, seeding them just fills DLQ with
    # `unknown mechanical action` rows every day.
    {
        "title": "price_watch_check",
        "description": "Re-scrape watched products and notify on price drops",
        "interval_seconds": 86400,
        "payload": {"_executor": "price_watch_check"},
    },
    {
        "title": "nerd_herd_health_alert",
        "description": "Alert on Nerd Herd health anomalies",
        "interval_seconds": 600,
        "payload": {"_marker": "nerd_herd_health"},
    },
    {
        "title": "cloud_refresh",
        "description": "Re-run cloud provider /models discovery + bench refresh",
        "interval_seconds": 21600,  # 6h
        "payload": {"_executor": "cloud_refresh"},
    },
    # Z10 T1A — sweep orphan file_locks every 60s. Closes the
    # task-crashed-without-release reversibility hole.
    {
        "title": "file_locks_sweep",
        "description": "Release orphan file_locks (expired or owner-task-dead)",
        "interval_seconds": 60,
        "payload": {"_marker": "file_locks_sweep"},
    },
    {
        "title": "kdv_persist",
        "description": "Persist KDV rate-limit state (adapted limits, 429 history, daily counters) to kutai.db",
        "interval_seconds": 60,
        "payload": {"_executor": "kdv_persist"},
    },
    {
        "title": "btable_rollup",
        "description": "Aggregate model_call_tokens into step_token_stats percentiles (14-day window)",
        "interval_seconds": 3600,  # hourly
        "payload": {"_marker": "btable_rollup"},
    },
    # Z7 B9 — hourly external-comms audit completeness check. Finds
    # vendor_call / external-publish rows (reversibility != 'full') that
    # have no external_comms_log row within the audit window and escalates
    # one founder_action per gap. Pure read-only scan + alert; the
    # `audit_completeness_check` mr_roboto verb owns the logic.
    {
        "title": "audit_completeness_check",
        "description": "Hourly external-comms audit: flag external sends with no external_comms_log row",
        "interval_seconds": 3600,  # hourly
        "payload": {"_executor": "audit_completeness_check"},
    },
    {
        "title": "monitoring_check",
        "description": "URL uptime and GitHub repo poll; alerts via notify_user sub-tasks",
        "interval_seconds": int(os.getenv("MONITOR_INTERVAL", "300")),
        "payload": {"_executor": "monitoring_check"},
    },
    # Z7 A6 — daily cold-outreach deliverability sweep. The
    # outreach_deliverability_check verb scans bounce/complaint rates and
    # writes an outreach_pauses row when a domain degrades; outreach_send
    # Gate 2b then honors the pause. Its PostHookSpec has auto_wire_triggers=[]
    # and no workflow step declares it, so without this cadence the pause is
    # never written in production.
    {
        "title": "outreach_deliverability_check",
        "description": "Daily cold-outreach deliverability sweep: pause degraded sender domains",
        "interval_seconds": 86400,
        "payload": {"_executor": "outreach_deliverability_check"},
    },
    {
        "title": "vector_maint_wal",
        "description": "ChromaDB WAL checkpoint to release write-ahead log bloat",
        "interval_seconds": 1800,
        "payload": {"_executor": "vector_maint_wal"},
    },
    {
        "title": "vector_maint_snapshot",
        "description": "ChromaDB directory snapshot for crash recovery (daily)",
        "interval_seconds": 86400,
        "payload": {"_executor": "vector_maint_snapshot"},
    },
    # Z10 T2A — write mission_budget_alerts rows at 50/75/90% ceiling
    # breaches. T2B drains them onto the per-mission Telegram thread.
    {
        "title": "mission_budget_alerts",
        "description": "Check mission cost-ceiling breaches (50/75/90%) and queue alerts",
        "interval_seconds": 300,
        "payload": {"_marker": "mission_budget_alerts"},
    },
    # Z10 T3A — every 30 minutes: compute pacing for each running/pending
    # mission; if >75% burn + >25% scope, post a single [asking] tradeoff
    # event. Idempotent via UNIQUE(mission_id, DATE(posted_at)).
    {
        "title": "mission_pacing_check",
        "description": "Compute pacing per mission; post tradeoff [asking] at 75/25",
        "interval_seconds": 1800,
        "payload": {"_marker": "mission_pacing_check"},
    },
    # Z1 Tier 7A (B12) — quarterly bash-audit. Cron: first of Jan/Apr/Jul/Oct
    # at 09:00. cron_expression beats interval_seconds because quarterly
    # intervals don't fit 86400-second arithmetic cleanly across leap years
    # and DST boundaries.
    # Z10 T2B — drain pending T1C confirmations + T2A budget alerts every
    # 5s. Cheap mechanical executor; idempotent on every tick.
    {
        "title": "mission_event_drain",
        "description": "Drain pending action_confirmations + mission_budget_alerts to Telegram mission_events",
        "interval_seconds": 5,
        "payload": {"_executor": "mission_event_drain"},
    },
    # Z10 T4B — recompute confidence reliability scores every 6h. Cheap
    # aggregation over confidence_outcomes; pulled into prompt builder via
    # cache refresh at the end of the same job.
    {
        "title": "confidence_calibration_recompute",
        "description": "Roll up confidence_outcomes into reliability scores (T4B)",
        "interval_seconds": 21600,
        "payload": {"_marker": "confidence_calibration_recompute"},
    },
    {
        "title": "bash_audit",
        "description": "sade_kalsin scaffolding audit (quarterly): what does each layer do that bash + Claude can't?",
        "cron_expression": "0 9 1 jan,apr,jul,oct *",
        "payload": {"_executor": "run_bash_audit"},
    },
    # Z1 T7B — weekly paraflow-goldens regression sweep. Iterates every
    # mission_<id>/ with a `.paraflow_archetype` marker file and diffs
    # against the named golden bundle. Drift trend persisted to
    # paraflow_diff_log. Opt-in per mission (no marker = skip).
    {
        "title": "paraflow_audit_all",
        "description": "Weekly paraflow-goldens drift audit across all opt-in missions",
        "interval_seconds": 604800,
        "payload": {"_executor": "paraflow_audit_all"},
    },
    # Z6 T4D — weekly compliance-template staleness scan; emits
    # founder_action(kind='legal_counsel') for each template whose
    # .meta.json last_reviewed is >180 days old. Idempotent (skips dup titles).
    {
        "title": "compliance_template_staleness",
        "description": "Weekly scan of compliance template metadata; emit legal_counsel founder_action per stale template",
        "interval_seconds": 604800,
        "payload": {"_executor": "compliance_template_staleness"},
    },
    # Z6 T7A — weekly credential rotation reminder. Scans the credentials
    # table for rows that either (a) have expires_at within 14d or
    # (b) have rotated_at IS NULL AND created_at > 90d. Emits a
    # founder_action(kind='credential_paste') per service. Idempotent.
    {
        "title": "credential_rotation_reminder",
        "description": "Weekly scan of credentials for upcoming expiry / overdue rotation; emit credential_paste founder_action per service",
        "interval_seconds": 604800,
        "payload": {"_executor": "credential_rotation_reminder"},
    },
    # Z6 T5D — weekly Stripe dispute scan; emits legal_counsel
    # founder_action per new dispute. No-op when no Stripe integration.
    {
        "title": "stripe_dispute_check",
        "description": "Weekly Stripe dispute scan; legal_counsel founder_action per new dispute",
        "interval_seconds": 604800,
        "payload": {"_executor": "stripe_dispute_check"},
    },
    # Z6 T5D — weekly Stripe revenue digest (active subs + MRR + balance).
    # Writes digest_<YYYY-WW>.md and (best-effort) posts to mission thread.
    {
        "title": "stripe_revenue_digest",
        "description": "Weekly Stripe revenue digest (active subs + MRR + balance)",
        "interval_seconds": 604800,
        "payload": {"_executor": "stripe_revenue_digest"},
    },
    # Z6 T5E — monthly Stripe Tax CSV export ledger. Emits founder_action
    # prompting forward-to-accountant.
    {
        "title": "tax_export_ledger",
        "description": "Monthly Stripe Tax transactions → CSV ledger; founder_action to forward to accountant",
        "interval_seconds": 2592000,
        "payload": {"_executor": "tax_export_ledger"},
    },
    # Z2 T4B + Item-1 followup — daily DLQ→mission_lessons emitter.
    # Scans the last 30d of unresolved DLQ rows, groups by (stack,
    # error_category), and upserts lessons rows when occurrences ≥ 3.
    # Without this cron the lessons table sits empty until a human runs
    # `python -m src.infra.mission_lessons emit-dlq`.
    {
        "title": "mission_lessons_emit_dlq",
        "description": "Daily DLQ pattern detector — upserts mission_lessons rows for recurring failures",
        "interval_seconds": 86400,  # 24h
        "payload": {"_executor": "emit_dlq_lessons"},
    },
    # Z9 Growth T3D — weekly DLQ feedback hook. Mines recurring failure
    # patterns from the dead-letter queue and writes dlq_pattern
    # growth_events so they surface in the weekly analytics digest (T2).
    # Weekly cadence aligns with the digest cycle; idempotent per pattern_key.
    {
        "title": "dlq_signal_review",
        "description": "Weekly DLQ pattern-mining — emit dlq_pattern growth_events for recurring failures",
        "interval_seconds": 604800,  # 7d
        "payload": {"_executor": "mine_dlq_patterns"},
    },
    # Z9 Growth T3 — weekly signal→backlog refresh. Runs classify_signals,
    # which enqueues the signal_classifier agent; its completion continuation
    # persists classified_signal rows and chains into score_backlog, writing
    # backlog_candidate rows the founder reviews via /backlog. Without this
    # cron the whole T3 signal→backlog loop never runs — classify_signals and
    # score_backlog are otherwise orphaned executors. Weekly, digest-aligned.
    {
        "title": "signal_classify_sweep",
        "description": "Weekly signal→backlog refresh — classify raw_signal growth_events, then score into backlog_candidate rows",
        "interval_seconds": 604800,  # 7d
        "payload": {"_executor": "classify_signals"},
    },
    # Z9 Growth T4C — daily verdict window sweeper. Scans every pending
    # hypothesis and enqueues a record_verdict mechanical task for each one
    # whose measurement window has closed (created_at + window_seconds <=
    # now). Restart-safe: re-derives "due" from the DB each tick, so no
    # per-hypothesis scheduled rows are needed. Idempotent — a recorded
    # verdict flips verdict away from 'pending' so later sweeps skip it.
    {
        "title": "verdict_window_sweep",
        "description": "Daily scan of pending hypotheses — enqueue verdict checks for closed measurement windows",
        "interval_seconds": 86400,  # 24h
        "payload": {"_executor": "verdict_window_sweep"},
    },
    # Z9 Growth T5C — weekly feature sunset scorer. Computes per-feature
    # usage from the last 30d of growth_events (+ recipe_pin_log catalog)
    # and writes sunset_candidate growth_events for features whose distinct-
    # user reach is below the founder-set threshold yet still cost money to
    # maintain. Pure deterministic math — no LLM. The founder reviews
    # candidates via /sunset and approves a deprecation mission via
    # /approve_sunset; the cron never spawns a mission itself.
    {
        "title": "sunset_score_recompute",
        "description": "Weekly feature sunset scorer — write sunset_candidate growth_events for low-usage non-zero-cost features",
        "interval_seconds": 604800,  # 7d
        "payload": {"_executor": "score_sunset"},
    },
    # Z9 Growth T5C — weekly roadmap / north-star sync. Reads the
    # success_metrics artifact and checks the declared north-star metric
    # against recent reality; writes a northstar_review growth_events row
    # when the metric is undefined, untracked, or flat — prompting the
    # founder to refine it. Sibling to sunset_score_recompute (distinct
    # i2p loop: 15.13 technical_debt_tracking vs 15.14 roadmap_update).
    {
        "title": "roadmap_northstar_sync",
        "description": "Weekly north-star staleness check — write northstar_review when the declared metric is stale or flat",
        "interval_seconds": 604800,  # 7d
        "payload": {"_executor": "roadmap_sync"},
    },
    # Z7 A0 — daily founder briefing at 09:00. Aggregates in-flight missions,
    # pending founder_actions, cost burn. Idempotent (same-day row guard).
    # cron_expression beats interval_seconds because "09:00 local" requires
    # a wall-clock anchor rather than a rolling offset.
    {
        "title": "daily_briefing",
        "description": "Daily 09:00 founder briefing: in-flight missions, pending actions, cost burn",
        "cron_expression": "0 9 * * *",
        "payload": {"_executor": "daily_briefing"},
    },
    # Z7 T4 A10 — daily follow-up reminder at 09:00. Scans interactions WHERE
    # follow_up_at <= today+7 AND done=0; builds digest; notifies founder.
    # Shares the 09:00 anchor with daily_briefing so the founder sees both
    # together in the morning. Idempotent — digest is always rebuilt fresh.
    {
        "title": "follow_up_reminder",
        "description": "Daily 09:00 CRM follow-up reminder: pending follow-ups within 7 days",
        "cron_expression": "0 9 * * *",
        "payload": {"_executor": "follow_up_reminder"},
    },
    # Z7 T4 B4 — meeting brief dispatch (every 5 minutes).
    # Phase 1: picks meetings in [25, 35]min window with no brief yet;
    # enqueues MAIN_WORK beckman task for LLM brief generation.
    # Phase 2: picks meetings 20-60min past scheduled_for with no outcome logged;
    # fires outcome_prompt founder_action card (non-LLM).
    {
        "title": "meeting_brief_dispatch",
        "description": "Every-5min: generate meeting briefs 30min pre-meeting + prompt outcome 30min post",
        "interval_seconds": 300,
        "payload": {"_executor": "meeting_brief_dispatch"},
    },
    # Z7 T4 A8 — weekly FAQ regen (A8 + A8.r1 multilingual).
    # Pulls last 7 days of low-confidence + escalated tickets, groups by language,
    # LLM-clusters by topic (within-language only), drafts FAQ entries for
    # clusters > 3, and surfaces founder_action "approve FAQ entry?" per draft.
    # On approve: appends to faq_{lang}.md + re-indexes support_docs_{lang}.
    {
        "title": "faq_regen",
        "description": "Weekly FAQ regen: cluster low-confidence/escalated tickets by language, draft FAQ entries for approval",
        "interval_seconds": 604800,  # 7 days
        "payload": {"_executor": "faq_regen"},
    },
    # Z7 T4 A8 — monthly quote harvest.
    # Scans positive-resolution tickets; emits founder_action "request quote
    # consent?" for each candidate; on consent, inserts into press_kit_quotes.
    {
        "title": "quote_harvest",
        "description": "Monthly quote harvest: scan positive tickets, emit consent founder_action, insert into press_kit_quotes on approve",
        "interval_seconds": 2592000,  # 30 days
        "payload": {"_executor": "quote_harvest"},
    },
    # Z7 T5 B1 — lifecycle email send (every 5 minutes).
    # Picks email_sends rows where scheduled_for <= now AND sent_at IS NULL;
    # calls send_email per product config; marks sent_at on success.
    # Falls back to manual /lifecycle trigger when Z6 event stream is absent.
    {
        "title": "lifecycle_email_send",
        "description": "Every-5min: pick due email_sends + send via product ESP + mark sent_at",
        "interval_seconds": 300,
        "payload": {"_executor": "lifecycle_email_send"},
    },
    # Z7 T5 B8 — daily reviews harvest. Polls all configured platforms,
    # ingests new reviews (dedup via unique constraint), classifies unclassified
    # reviews (sentiment + theme). Surfaces 1-2-star founder_actions + queues
    # bug-tagged reviews for investigation.
    {
        "title": "reviews_poll_daily",
        "description": "Daily reviews harvest: poll configured platforms, ingest new reviews, classify sentiment + theme",
        "interval_seconds": 86400,  # 24h
        "payload": {"_executor": "reviews_poll_daily"},
    },
    # Z7 T5 A9 — monthly investor bullets (A9 + A9.r1 segmented).
    # At month-end: collects metrics from Z6/Z8/Z3/A8 (degrades when absent),
    # runs anomaly detection (±2σ vs trailing-3-month median), calls LLM for
    # hypotheses (OVERHEAD lane), renders structured Markdown bullets, emits
    # 3 segmented variants (pre_investor_pitch_bullets / current_investor_update /
    # advisor_check_in), surfaces founder_action for copy-to-clipboard review.
    # NOT auto-sent — founder writes the prose; A9 surfaces what they haven't noticed.
    {
        "title": "investor_bullets",
        "description": (
            "Monthly investor-bullet digest: metrics, ±2σ anomaly detection, "
            "LLM hypotheses, 3 segmented variants (investor/advisor). "
            "Surfaces founder_action for copy-to-clipboard review — no auto-send."
        ),
        "interval_seconds": 2592000,  # 30 days
        "payload": {"_executor": "investor_bullets"},
    },
    # Yalayut Phase 4 — daily catalog discovery (trusted cron-mode sources).
    # Backstop cadence: the orchestrator's in-memory _check_yalayut_discovery
    # gate resets on restart; this scheduled_tasks row survives restarts.
    {
        "title": "yalayut_discovery",
        "description": "Daily yalayut catalog discovery — pull trusted sources",
        "interval_seconds": 86400,  # 24h
        "payload": {"_executor": "yalayut_discovery", "mode": "daily"},
    },
    # Yalayut Phase 4 — source-scout candidate proposal (daily).
    {
        "title": "source_scout",
        "description": "Daily yalayut source-scout — propose candidate sources",
        "interval_seconds": 86400,  # 24h
        "payload": {"_executor": "source_scout"},
    },
]

# Fast-path: once seeded in this process, skip DB round-trips on subsequent calls.
_seeded: bool = False

# Serialises concurrent seed attempts so two near-simultaneous next_task()
# callers can't both pass the SELECT-before-INSERT check and create duplicate
# rows. Harmless under the serial pump today, but cheap insurance.
_seed_lock: asyncio.Lock = asyncio.Lock()


async def seed_internal_cadences() -> None:
    """Upsert all INTERNAL_CADENCES rows into scheduled_tasks.

    Safe to call multiple times — skips rows that already exist by
    (title, kind='internal').  Sets the module-level ``_seeded`` flag only
    after a successful pass so a crash mid-seed allows retry.

    Newly inserted rows have next_run set to now + interval_seconds so they
    don't fire immediately on first tick (avoids spurious task insertion in
    tests and on fresh deployments).
    """
    global _seeded
    if _seeded:
        return

    from datetime import timedelta
    from src.infra.db import get_db  # lazy to avoid circular import at module load
    from src.infra.times import utc_now, to_db

    async with _seed_lock:
        # Re-check inside the lock: another coroutine may have finished seeding
        # while we were waiting to acquire it.
        if _seeded:
            return

        db = await get_db()
        now = utc_now()
        for cadence in INTERNAL_CADENCES:
            cursor = await db.execute(
                "SELECT id FROM scheduled_tasks WHERE title = ? AND kind = 'internal'",
                (cadence["title"],),
            )
            existing = await cursor.fetchone()
            if existing:
                logger.debug("cron_seed: skipping existing row", title=cadence["title"])
                continue

            interval = cadence.get("interval_seconds")
            cron_expr = cadence.get("cron_expression")
            if interval:
                first_run = to_db(now + timedelta(seconds=interval))
            elif cron_expr:
                # croniter is optional; fall back to "fire in 1h" if absent
                # so cron's _advance_schedule can compute the next real slot
                # on the first tick.
                try:
                    from croniter import croniter
                    first_run = to_db(croniter(cron_expr, now).get_next(type(now)))
                except Exception:
                    first_run = to_db(now + timedelta(hours=1))
            else:
                first_run = to_db(now + timedelta(hours=1))
            await db.execute(
                """INSERT INTO scheduled_tasks
                   (title, description, interval_seconds, cron_expression, kind, context, enabled, next_run)
                   VALUES (?, ?, ?, ?, 'internal', ?, 1, ?)""",
                (
                    cadence["title"],
                    cadence["description"],
                    interval,
                    cron_expr,
                    json.dumps(cadence["payload"]),
                    first_run,
                ),
            )
            logger.info("cron_seed: inserted internal cadence", title=cadence["title"])

        await db.commit()
        _seeded = True
        logger.info("cron_seed: all internal cadences seeded", count=len(INTERNAL_CADENCES))
