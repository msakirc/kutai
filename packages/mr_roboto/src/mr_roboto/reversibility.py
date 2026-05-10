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
    "request_interview_data": "full",  # surfaces a Telegram prompt; cancellable
    "workflow_advance": "full",  # mission-state transition; backward by retry
    "run_pytest": "full",  # reads only, side-effect-free
    "run_bash_audit": "full",  # audit report writer
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
