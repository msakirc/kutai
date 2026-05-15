"""Verb reversibility taxonomy — Z10 Tier 1B.

Source of truth for the reversibility tag attached to every mr_roboto
action. T1C / Phase D wire per-action enforcement (confirmation flow,
audit log) on top of this registry.

Tags
----
- ``"full"``         — action is fully reversible (e.g. snapshot, pure
                       check, append-only log row, local file write that
                       can be `git checkout`-ed).
- ``"partial"``      — reversible but with side-effect cost (e.g.
                       `git push` to a remote we control, `run_cmd` whose
                       command may or may not be destructive).
- ``"irreversible"`` — visible side-effect that can't be undone
                       (Telegram message to founder, GitHub repo init in
                       public mode, anything the founder/user observes
                       and reacts to).

Default for unknown verbs is ``"partial"`` — conservative: gate logic
should treat unknown as "ask before proceeding" rather than "free pass".
"""
from __future__ import annotations

from typing import Literal

Reversibility = Literal["full", "partial", "irreversible"]

# Source of truth — every verb registered in mr_roboto.run() dispatcher.
# Cross-checked against actions.py `if action == "<verb>"` blocks; the
# registry test (test_reversibility_registry.py) enforces this.
VERB_REVERSIBILITY: dict[str, Reversibility] = {
    # ---- Snapshots & local writes (fully reversible) ---------------
    "workspace_snapshot": "full",  # read-only filesystem snapshot
    "git_commit": "full",  # local commit; reset HEAD~1 reverses it
    # ---- Pure verifiers / checks (no state change) -----------------
    "verify_artifacts": "full",
    "verify_schema_version": "full",
    "verify_charter_shape": "full",
    "verify_reverse_pitch_shape": "full",
    "verify_falsification_present": "full",
    "verify_non_goals_shape": "full",
    "check_against_non_goals": "full",
    "verify_adr_shape": "full",
    "verify_adr_register": "full",
    "verify_cost_curve_present": "full",
    "verify_competitive_positioning_shape": "full",
    "verify_interview_script_shape": "full",
    "verify_taste_emphasis_shape": "full",
    "verify_design_tokens_shape": "full",
    "verify_surfaces_shape": "full",
    "verify_user_flow_shape": "full",
    "verify_screen_inventory_shape": "full",
    "verify_shared_shell_shape": "full",
    "verify_screen_plan_shape": "full",
    "verify_html_prototype_shape": "full",
    "verify_screen_consistency": "full",
    "verify_premortem_shape": "full",
    "spec_consistency_check": "full",
    "prior_art_min_coverage": "full",
    "verify_against_paraflow_goldens": "full",  # comparison only
    "paraflow_audit_all": "full",  # iterates verify_against_paraflow_goldens
    "compliance_template_present": "full",
    "compliance_blocker_check": "full",
    "compliance_fingerprint_collection": "full",  # writes merged file, replayable
    "check_grounding": "full",  # log inspection
    "critic_gate": "full",  # advisory verdict, no side-effect
    # ---- Local file emitters (reversible via git) ------------------
    "regen_artifact": "full",  # versioned `.v{N}` siblings + log
    "regen_bundle": "full",  # delegates to regen_artifact
    "annotate_html_oids": "full",  # rewrites local HTML
    "propagate_asset_change": "full",  # writes a proposal report
    "propose_spec_patch_from_html_diff": "full",  # writes patch report
    "derive_token_tag_signature": "full",  # local hash file
    "generate_intake_todo": "full",  # local todo row + Telegram prompt; revoke via /cleartodos
    "ingest_visual": "full",  # writes visual_brief.md locally
    # ---- Read-only / network checks (no persistent state) ----------
    "http_check": "full",  # GET/HEAD/etc., we treat as side-effect-free
    "parse_og_tags": "full",  # one HTTP fetch, no write
    "monitoring_check": "full",
    "analytics_digest": "full",  # Z9 T2B — read-only pull + append-only growth_events row
    "arm_analytics_digest": "full",  # Z9 T2B — cron arm; disarm is a single cursor edit
    "social_preview_check": "full",
    "staging_smoke_check": "full",
    # ---- Datastore append/refresh (reversible by re-run) -----------
    "cloud_refresh": "full",  # idempotent provider-state pull
    "kdv_persist": "full",  # rate-table upsert; latest-wins
    "mission_event_drain": "full",  # idempotent post + UPDATE; no external write
    "find_similar_missions": "full",  # vector query only
    "index_idea_fingerprint": "full",  # vector upsert; idempotent
    "surface_prior_mission_hints": "full",  # query-only
    "index_mission_artifacts": "full",  # vector upsert; idempotent
    "vector_maint_wal": "full",
    "vector_maint_snapshot": "full",
    "attention_check": "full",  # read-only budget query
    "attention_debit": "full",  # append-only debit row
    "audit_completeness_check": "full",  # read-only gap scan + alert escalation
    "daily_briefing": "full",  # idempotent mission_briefings row write
    "z0_preflight_write": "full",  # idempotent JSON write + missions upsert
    "request_interview_data": "full",  # surfaces a Telegram prompt; cancellable
    "workflow_advance": "full",  # mission-state transition; backward by retry
    "run_pytest": "full",  # reads only, side-effect-free
    "run_semgrep": "full",  # read-only static analysis, no writes
    "apply_migration": "full",  # ephemeral DB only; no real-world write
    "run_bash_audit": "full",  # audit report writer
    "regen_and_diff": "full",  # generates to stdout only; never writes target
    # ---- Partial-reversible (caller may need to override) ----------
    "run_cmd": "partial",  # command may be destructive — caller knows
    "git_push": "partial",  # if remote is ours, force-push reverses; not perfect
    # ---- Irreversible (visible to founder / external) --------------
    "notify_user": "irreversible",  # Telegram message — user sees it
    "todo_reminder": "irreversible",  # Telegram nudge — user sees it
    "price_watch_check": "irreversible",  # may trigger price-drop alert
    "clarify": "irreversible",  # Telegram clarification prompt to user
    "init_mission_github_repo": "irreversible",  # public repo creation visible to world
    "emit_preview_url": "irreversible",  # tunneled URL discoverable by share
    "kill_preview_url": "partial",  # tears tunnel down; URL was already published
    "propose_spec_patch": "full",  # alias kept for compat with older payloads
    # ---- Z10 T3B sandbox-gate verbs (caller-opened confirmation rows) -----
    "sandbox_local_mode": "partial",  # mission requested host-mode shell
    "broader_egress": "partial",  # shell wants to reach a host outside whitelist
    # ---- Z10 T3C reset-to-green primitives ------------------------------
    "mark_green": "full",  # pure snapshot — git tag + DB+Chroma dump
    "rollback_mission": "irreversible",  # rolls workspace/DB/Chroma back; no roll-forward
    # ---- Z10 T4A end-of-mission demo deliverable ------------------------
    "record_demo": "full",  # additive artifact (data/missions/{id}/demo.mp4)
    "verify_demo_artifact": "full",  # pure check
    "mission_deliverable_bundle": "irreversible",  # posts to Telegram thread
    # ---- Z3 T5 integration replay + bisect ------------------------------
    "integration_replay": "full",  # checkout + pytest only; read-only against working tree
    "integration_bisect": "full",  # binary-search checkouts + pytest; restores HEAD
    # ---- Z2 T4C cross-mission lesson injector ---------------------------
    "inject_lessons": "full",  # read-only query + writes to mission context JSON only
    "emit_dlq_lessons": "full",  # DLQ scan + idempotent upsert into mission_lessons
    # ---- Z2 T5A recipe library substrate --------------------------------
    "pick_recipe": "full",  # read-only: scans recipes/ dir, no writes
    # ---- Z2 T5C recipe instantiation ------------------------------------
    "instantiate_recipe": "partial",  # writes files to target_dir; git-reversible
    "instantiate_picked_recipes": "partial",  # batch wrapper for instantiate_recipe
    # ---- Z6 T3A vendor-call mechanical ---------------------------------
    "vendor_call": "partial",  # real-world API call; per-call adapter knows
    # ---- Z6 T5 Stripe family --------------------------------------------
    "stripe_scaffold": "full",  # local file writes only; git-reversible
    "stripe_provision_products": "irreversible",  # creates real Stripe objects
    "stripe_payment_flow_test": "partial",  # test-mode only; remnant test data
    "stripe_dispute_check": "full",  # read-only API + local checkpoint file
    "stripe_revenue_digest": "irreversible",  # posts to mission Telegram thread
    "tax_export_ledger": "irreversible",  # surfaces founder_action visible to user
    # ---- Z8 T4B on-call action gateway ----------------------------------
    # `oncall_action` is the dispatcher wrapper; the verb-specific
    # reversibility lives below and is what the gate-logic should consult
    # once a verb is in flight.
    "oncall_action": "partial",  # gateway: real impact depends on the sub-verb
    "restart_service": "partial",  # transient downtime; idempotent restart
    "rollback_to_last_green": "irreversible",  # rolls live workspace back
    "scale_up": "partial",  # cost impact; manually reversible via scale_down
    "scale_down": "partial",  # may drop in-flight requests; scale_up restores
    "drain_traffic": "partial",  # traffic redirect; reversible by re-enabling
    "rotate_failed_key": "irreversible",  # old key invalidated; users see effect
    "archive_flake_test": "full",  # local pytest.ini / xfail marker only
    "escalate_to_founder": "irreversible",  # founder sees the alert
    # ---- Z8 T4C phase 13 playbook generator -----------------------------
    "generate_playbooks": "full",  # read-only mission inputs → artifact JSON
    # ---- Z8 T5A backup_verify cron --------------------------------------
    "backup_verify": "full",  # copy + read smoke SELECT, no external write
    "cron_backup_verify": "full",
    # ---- Z8 T5B dependency hygiene cron ---------------------------------
    "dependency_scan": "full",  # pip-audit / npm audit, read-only
    "cron_dep_hygiene": "full",
    # ---- Z8 T5C CVE + secret scan cron ----------------------------------
    "cve_scan": "full",  # OSV.dev HTTP query, no writes
    "cron_cve_scan": "full",
    "secret_scan": "full",  # gitleaks read-only scan
    "cron_secret_scan": "full",
    # ---- Z8 T5D cost monitor cron ---------------------------------------
    "cost_pull": "full",  # vendor read-only cost API
    "cron_cost_pull": "full",
    # ---- Z7 T3A launch playbook verbs (A2 + A2.r1) -------------------------
    # Draft verbs: fully reversible — produce local draft text only, no publish.
    "launch_drafts/hn": "full",
    "launch_drafts/ph": "full",
    "launch_drafts/twitter": "full",
    "launch_drafts/linkedin": "full",
    "launch_drafts/reddit": "full",
    # publish_synchronized: irreversible — channels see the post immediately.
    "publish_synchronized": "irreversible",
    # launch_response_monitor: full — enqueues a monitoring sub-mission; reversible.
    "launch_response_monitor": "full",
    # launch_lessons_writeback: full — append-only mission_lessons rows; idempotent.
    "launch_lessons_writeback": "full",
    # launch_readiness_gate: full — read-only checks + founder_action (advisory).
    "launch_readiness_gate": "full",
    # ---- Z7 T3B demo pipeline verbs (A3 + A3.r1) --------------------------
    "demo/storyboard": "full",        # writes storyboard.json to workspace; git-reversible
    "demo/record": "full",            # writes raw .webm per scene; git-reversible
    "demo/edit": "full",              # writes cuts/*.mp4; git-reversible
    "demo/caption": "full",           # writes demo.vtt; git-reversible
    "demo/accessibility_pass": "full", # writes accessibility_manifest.json; git-reversible
    # ---- Z9 growth verbs (registered T1C; implemented T2-T5) ------------
    "inject_north_star": "full",  # writes north-star row; deletable
    "emit_metric": "full",  # append-only metric row
    "record_hypothesis": "full",  # writes hypothesis row; deletable
    "record_verdict": "full",  # writes verdict row; re-runnable
    "suppress_hypothesis": "full",  # suppressed_until expires naturally
    "assign_variant": "partial",  # DB row reversible, but exposed users already saw the variant
    "retire_variant": "partial",  # stops the variant, but prior exposure cannot be unwound
    "score_backlog": "full",  # writes backlog scores; re-runnable
    "score_sunset": "full",  # writes sunset scores; re-runnable
    "classify_signals": "full",  # Z9 T3B — query + Beckman enqueue + append-only rows
    # ---- Z7 T3C press kit verbs (A4 + A4.r1) --------------------------------
    "press_kit/assemble": "full",   # writes zips + manifest to workspace; git-reversible
    "press_kit/publish": "partial", # copies zips to local store or S3; uploaded bytes persist
    # ---- Z7 T3D incident comms verbs (B3) -----------------------------------
    "incident/draft_update": "full",      # drafts text only; DB not yet written; git-reversible
    "incident/publish_status": "irreversible",  # inserts status_updates row; customers see it on /status
    "incident/draft_postmortem": "full",  # writes local .md artifact; git-reversible
    "incident_update_review": "irreversible",  # surfaces founder_action to Telegram; user sees it
    # ---- Z7 T3E crisis comms verbs (B6) -------------------------------------
    "crisis/freeze_marketing": "full",    # DB write; reversible via crisis/resume
    "crisis/draft_holding": "full",       # draft variants only; no publish; git-reversible
    "crisis/disclosure_timer": "irreversible",  # surfaces founder_action visible to user
}

DEFAULT_REVERSIBILITY: Reversibility = "partial"
"""Conservative default — unknown verbs are gated like ``run_cmd``."""


def get_reversibility(
    verb: str,
    override: Reversibility | None = None,
) -> Reversibility:
    """Return the reversibility tag for ``verb``.

    ``override`` (caller intent) wins when provided — used by ``run_cmd``
    callers who know whether their specific command is destructive.
    """
    if override is not None:
        return override
    return VERB_REVERSIBILITY.get(verb, DEFAULT_REVERSIBILITY)


__all__ = [
    "Reversibility",
    "VERB_REVERSIBILITY",
    "DEFAULT_REVERSIBILITY",
    "get_reversibility",
]
