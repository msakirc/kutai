"""Mr. Roboto — mechanical dispatcher: non-LLM task executors."""
from __future__ import annotations

from mr_roboto.actions import Action
from mr_roboto.workspace_snapshot import snapshot_workspace
from mr_roboto.git_commit import auto_commit
from mr_roboto.verify_artifacts import verify_artifacts
from mr_roboto.verify_schema_version import verify_schema_version
from mr_roboto.verify_charter_shape import verify_charter_shape
from mr_roboto.verify_reverse_pitch_shape import verify_reverse_pitch_shape
from mr_roboto.verify_falsification_present import verify_falsification_present
from mr_roboto.verify_non_goals_shape import verify_non_goals_shape
from mr_roboto.check_against_non_goals import check_against_non_goals
from mr_roboto.verify_screen_plan_shape import verify_screen_plan_shape
from mr_roboto.verify_html_prototype_shape import verify_html_prototype_shape
from mr_roboto.verify_screen_consistency import verify_screen_consistency
from mr_roboto.generate_intake_todo import generate_intake_todo
from mr_roboto.regen import (
    regen_artifact,
    regen_bundle,
    known_axes as known_regen_axes,
)
from mr_roboto.annotate_html_oids import annotate_html_oids
from mr_roboto.propagate_asset_change import propagate_asset_change
from mr_roboto.propose_spec_patch import propose_spec_patch_from_html_diff
from mr_roboto.run_cmd import run_cmd
from mr_roboto.run_pytest import run_pytest
from mr_roboto.parse_og_tags import parse_og_tags
from mr_roboto.http_check import http_check
from mr_roboto.emit_preview_url import emit_preview_url
from mr_roboto.kill_preview_url import kill_preview_url
from mr_roboto.compliance_fingerprint_collection import (
    compliance_fingerprint_collection,
)
from mr_roboto.compliance_template_present import compliance_template_present
from mr_roboto.compliance_blocker_check import compliance_blocker_check
from mr_roboto.attention_check import (
    attention_check,
    attention_debit,
    write_deferred_question,
)

__all__ = [
    "Action",
    "run",
    "snapshot_workspace",
    "auto_commit",
    "verify_artifacts",
    "verify_schema_version",
    "verify_charter_shape",
    "verify_reverse_pitch_shape",
    "verify_falsification_present",
    "verify_non_goals_shape",
    "check_against_non_goals",
    "verify_screen_plan_shape",
    "verify_html_prototype_shape",
    "verify_screen_consistency",
    "generate_intake_todo",
    "regen_artifact",
    "regen_bundle",
    "known_regen_axes",
    "annotate_html_oids",
    "propagate_asset_change",
    "propose_spec_patch_from_html_diff",
    "run_cmd",
    "run_pytest",
    "parse_og_tags",
    "http_check",
    "emit_preview_url",
    "kill_preview_url",
    "compliance_fingerprint_collection",
    "compliance_template_present",
    "compliance_blocker_check",
    "attention_check",
    "attention_debit",
    "write_deferred_question",
]


async def run(task: dict) -> Action:
    """Route a mechanical task to the appropriate executor.

    ``task["payload"]["action"]`` selects the executor:

    - ``"workspace_snapshot"`` → :func:`mr_roboto.snapshot_workspace`
    - ``"git_commit"``         → :func:`mr_roboto.auto_commit`

    Unknown actions return an ``Action(status="failed", error=...)``; the
    orchestrator is responsible for marking the task failed.
    """
    payload = task.get("payload") or {}
    action = payload.get("action")

    if action == "workspace_snapshot":
        snap = await snapshot_workspace(
            mission_id=task["mission_id"],
            task_id=task["id"],
            workspace_path=payload["workspace_path"],
            repo_path=payload.get("repo_path"),
        )
        if snap is None:
            return Action(status="failed", error="snapshot failed")
        return Action(status="completed", result=snap)

    if action == "git_commit":
        commit_info = await auto_commit(task, payload.get("result") or {})
        # Backwards-compatible default: empty diff is OK (no-op success).
        # Opt-in: when payload.require_diff is true, an empty diff is a
        # hard failure — surfaces the "step claimed file changes but
        # nothing actually changed" pattern observed in mission 57.
        if payload.get("require_diff") and (commit_info or {}).get("empty"):
            return Action(
                status="failed",
                error="empty diff: require_diff was set but nothing was committed",
                result=commit_info or {},
            )
        if (commit_info or {}).get("error"):
            # Best-effort path: keep prior behaviour (don't fail the task)
            # unless require_diff is set and we got nothing.
            return Action(status="completed", result=commit_info or {})
        return Action(status="completed", result=commit_info or {})

    if action == "check_grounding":
        # Layer 2 of G: declarative match between source task's tool_calls
        # audit log and the workflow step's declared `produces` paths.
        # Pass = every produces slot has a matching successful write call.
        # Fail = at least one produces slot was never written; source
        # retries with the missing paths in the feedback message.
        from mr_roboto.check_grounding import check_grounding as _ground
        try:
            res = _ground(
                tool_calls=payload.get("tool_calls") or [],
                produces=payload.get("produces") or [],
            )
            if not res.get("passed"):
                return Action(
                    status="failed",
                    error=(
                        f"check_grounding: missing={res.get('missing')} "
                        f"written={res.get('written')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_artifacts":
        from mr_roboto.verify_artifacts import verify_artifacts as _verify
        try:
            res = await _verify(
                mission_id=task.get("mission_id"),
                paths=payload.get("paths") or [],
                min_bytes=int(payload.get("min_bytes", 1)),
                compile_check=bool(payload.get("compile_check", False)),
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("all_ok"):
                return Action(
                    status="failed",
                    error=(
                        f"verify_artifacts: missing={res.get('missing')} "
                        f"failed={res.get('failed')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_schema_version":
        # P7 — assert each artifact carries `_schema_version` matching the
        # workflow step's declared expectation. Pure check; no I/O. Failure
        # surfaces the (artifact_name, found, expected) triple in error so
        # the source retries with the missing/mismatched version visible.
        from mr_roboto.verify_schema_version import (
            verify_schema_version as _verify_schema,
        )
        try:
            res = _verify_schema(
                artifacts=payload.get("artifacts") or {},
                expected_versions=payload.get("expected_versions") or {},
                legacy_pre_p7=bool(payload.get("legacy_pre_p7", False)),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"verify_schema_version: missing={res.get('missing')} "
                        f"mismatched={res.get('mismatched')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_charter_shape":
        # Z1 Tier 1 — paraflow-shape product_charter.md validator.
        # Auto-wired as post-hook on the charter-producing step (0.1).
        from mr_roboto.verify_charter_shape import (
            verify_charter_shape as _verify_charter,
        )
        try:
            res = _verify_charter(
                charter_text=payload.get("charter_text"),
                charter_paths=payload.get("charter_paths"),
                min_solutions=int(payload.get("min_solutions", 3)),
                max_solutions=int(payload.get("max_solutions", 7)),
                min_brand_keywords=int(payload.get("min_brand_keywords", 5)),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"verify_charter_shape: missing_sections={res.get('missing_sections')} "
                        f"solution_count={res.get('solution_count')} "
                        f"solution_problems={res.get('solution_problems')} "
                        f"placeholders={res.get('placeholders')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_reverse_pitch_shape":
        # Z1 Tier 1 — Amazon working-backwards reverse_pitch.md validator.
        from mr_roboto.verify_reverse_pitch_shape import (
            verify_reverse_pitch_shape as _verify_rp,
        )
        try:
            res = _verify_rp(
                pitch_text=payload.get("pitch_text"),
                pitch_paths=payload.get("pitch_paths"),
                ambition_tier=str(payload.get("ambition_tier", "private_beta")),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"verify_reverse_pitch_shape: missing={res.get('missing_sections')} "
                        f"placeholders={res.get('placeholders')} "
                        f"ack={res.get('acknowledged_no_users')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_falsification_present":
        # Z1 Tier 2 (P4) — assert every requirement-bundle item carries
        # the falsification triple (risk_if_wrong / validation_method /
        # falsification_signal). Auto-wired as sibling on phase-3 steps
        # that produce commitment-shaped artifacts.
        from mr_roboto.verify_falsification_present import (
            verify_falsification_present as _verify_fals,
        )
        try:
            res = _verify_fals(
                artifacts=payload.get("artifacts") or {},
                legacy_pre_falsification=bool(
                    payload.get("legacy_pre_falsification", False)
                ),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"verify_falsification_present: missing={res.get('missing')} "
                        f"critical_underspecified={res.get('critical_underspecified')} "
                        f"empty={res.get('empty')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_non_goals_shape":
        # Z1 Tier 2 (A2) — mission-wide non_goals.md shape validator.
        from mr_roboto.verify_non_goals_shape import (
            verify_non_goals_shape as _verify_ng,
        )
        try:
            res = _verify_ng(
                non_goals_text=payload.get("non_goals_text"),
                non_goals_paths=payload.get("non_goals_paths"),
                min_items=int(payload.get("min_items", 3)),
                max_items=int(payload.get("max_items", 7)),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"verify_non_goals_shape: problems={res.get('problems')} "
                        f"bullet_count={res.get('bullet_count')} "
                        f"placeholders={res.get('placeholders')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "check_against_non_goals":
        # Z1 Tier 2 (A2) — cheap heuristic pre-flag of non-goal overlap.
        # NOT a substitute for LLM reviewer at 3.11 / 4.16 / 5.10; just a
        # cheap signal the reviewer consumes as input.
        from mr_roboto.check_against_non_goals import (
            check_against_non_goals as _check_ng,
        )
        try:
            res = _check_ng(
                non_goals_text=payload.get("non_goals_text"),
                non_goals_paths=payload.get("non_goals_paths"),
                target_text=payload.get("target_text"),
                target_paths=payload.get("target_paths"),
                min_overlap_tokens=int(payload.get("min_overlap_tokens", 2)),
            )
            # This is a SOFT check — overlaps surface for the reviewer to
            # judge. Always returns completed; matches list is the signal.
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_adr_shape":
        # Z1 Tier 2 (P3) — universal Nygard-extended ADR validator.
        # Auto-wired as sibling step `<step_id>.verify` after every
        # phase-4 ADR-emitting step.
        from mr_roboto.verify_adr_shape import (
            verify_adr_shape as _verify_adr,
        )
        try:
            res = _verify_adr(
                adr_text=payload.get("adr_text"),
                adr_obj=payload.get("adr_obj"),
                adr_paths=payload.get("adr_paths"),
                expected_schema_version=str(
                    payload.get("expected_schema_version", "1")
                ),
                require_cost_curve=bool(payload.get("require_cost_curve", False)),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"verify_adr_shape: adr_id={res.get('adr_id')} "
                        f"missing={res.get('missing_fields')} "
                        f"options_problems={res.get('options_problems')} "
                        f"orphan_chosen={res.get('orphan_chosen_option_id')} "
                        f"falsification_missing={res.get('falsification_missing')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_adr_register":
        # Z1 Tier 2 (P3) — register.md vs on-disk ADR JSON consistency.
        from mr_roboto.verify_adr_register import (
            verify_adr_register as _verify_register,
        )
        try:
            res = _verify_register(
                register_text=payload.get("register_text"),
                register_path=payload.get("register_path"),
                adr_dir=payload.get("adr_dir"),
                allow_empty_register=bool(
                    payload.get("allow_empty_register", False)
                ),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"verify_adr_register: missing_files={res.get('missing_files')} "
                        f"orphan_files={res.get('orphan_files')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_cost_curve_present":
        # Z1 Tier 2 (A8) — stack-related ADR cost-curve presence guard.
        from mr_roboto.verify_cost_curve_present import (
            verify_cost_curve_present as _verify_curve,
        )
        try:
            res = _verify_curve(
                adr_text=payload.get("adr_text"),
                adr_obj=payload.get("adr_obj"),
                adr_paths=payload.get("adr_paths"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"verify_cost_curve_present: adr_id={res.get('adr_id')} "
                        f"options_missing_curve={res.get('options_missing_curve')} "
                        f"cost_at_target_missing={res.get('cost_at_target_missing')} "
                        f"cost_mitigation_field_missing={res.get('cost_mitigation_field_missing')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_competitive_positioning_shape":
        # Z1 Tier 2 (C2) — paraflow PRD §6 named-competitor positioning.
        from mr_roboto.verify_competitive_positioning_shape import (
            verify_competitive_positioning_shape as _verify_cp,
        )
        try:
            res = _verify_cp(
                positioning_text=payload.get("positioning_text"),
                positioning_paths=payload.get("positioning_paths"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"verify_competitive_positioning_shape: "
                        f"missing={res.get('missing_sections')} "
                        f"empty={res.get('empty_sections')} "
                        f"named_competitors={res.get('named_competitors')} "
                        f"placeholders={res.get('placeholders')} "
                        f"schema_version={res.get('schema_version')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_interview_script_shape":
        # Z1 Tier 2 (A4) — interview-script generator post-hook.
        from mr_roboto.verify_interview_script_shape import (
            verify_interview_script_shape as _verify_is,
        )
        try:
            res = _verify_is(
                script_text=payload.get("script_text"),
                script_paths=payload.get("script_paths"),
                min_questions=int(payload.get("min_questions", 5)),
                max_questions=int(payload.get("max_questions", 7)),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"verify_interview_script_shape: "
                        f"question_count={res.get('question_count')} "
                        f"problems={res.get('question_problems')} "
                        f"target_assumptions={res.get('target_assumptions')} "
                        f"placeholders={res.get('placeholders')} "
                        f"schema_version={res.get('schema_version')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "request_interview_data":
        # Z1 Tier 2 (A4) — surface interview-script path to founder via
        # Telegram and wait for DONE/SKIP. record_skip mode persists
        # missions.interview_skip_reason.
        from mr_roboto.request_interview_data import (
            request_interview_data as _req_iv,
        )
        try:
            res = await _req_iv(task)
            if res.get("status") == "failed":
                return Action(status="failed", error=str(res.get("error", "")))
            if (res.get("status") == "needs_clarification"
                    and res.get("keyboard_sent")):
                return Action(status="needs_clarification", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_taste_emphasis_shape":
        from mr_roboto.verify_taste_emphasis_shape import (
            verify_taste_emphasis_shape as _verify_taste,
        )
        try:
            res = await _verify_taste(
                mission_id=task.get("mission_id"),
                path=payload.get("path", ".style/taste_emphasis.json"),
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"verify_taste_emphasis_shape: {res.get('error')}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_design_tokens_shape":
        from mr_roboto.verify_design_tokens_shape import (
            verify_design_tokens_shape as _verify_tokens,
        )
        try:
            res = await _verify_tokens(
                mission_id=task.get("mission_id"),
                path=payload.get("path", ".style/design_tokens.json"),
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"verify_design_tokens_shape: {res.get('error')}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "derive_token_tag_signature":
        from mr_roboto.derive_token_tag_signature import (
            derive_token_tag_signature as _derive_tag,
        )
        try:
            res = await _derive_tag(
                mission_id=task.get("mission_id"),
                path=payload.get("path", ".style/taste_emphasis.json"),
                workspace_path=payload.get("workspace_path"),
                payload=payload.get("taste_payload"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"derive_token_tag_signature: {res.get('error')}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_surfaces_shape":
        from mr_roboto.verify_surfaces_shape import verify_surfaces_shape
        try:
            res = await verify_surfaces_shape(
                mission_id=task.get("mission_id"),
                path=payload.get("path"),
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"verify_surfaces_shape: {res.get('errors')}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_user_flow_shape":
        from mr_roboto.verify_user_flow_shape import verify_user_flow_shape
        try:
            res = await verify_user_flow_shape(
                mission_id=task.get("mission_id"),
                path=payload.get("path"),
                surfaces=payload.get("surfaces"),
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"verify_user_flow_shape: {res.get('errors')}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_screen_inventory_shape":
        from mr_roboto.verify_screen_inventory_shape import (
            verify_screen_inventory_shape,
        )
        try:
            res = await verify_screen_inventory_shape(
                mission_id=task.get("mission_id"),
                path=payload.get("path"),
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"verify_screen_inventory_shape: {res.get('errors')}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_shared_shell_shape":
        from mr_roboto.verify_shared_shell_shape import verify_shared_shell_shape
        try:
            res = await verify_shared_shell_shape(
                mission_id=task.get("mission_id"),
                path=payload.get("path"),
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"verify_shared_shell_shape: {res.get('errors')}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "regen_artifact":
        # Z1 Tier 4A (C11+A15) — single-artifact regen with versioned
        # `.v{N}` siblings + regen_log row. Founder change_description
        # required; emitter delegates to coulson when wired (mocked in
        # unit tests).
        from mr_roboto.regen import regen_artifact as _regen_one
        try:
            change = payload.get("change_description")
            if not change or not str(change).strip():
                return Action(
                    status="failed",
                    error="regen_artifact: change_description is required",
                )
            mid = task.get("mission_id") or payload.get("mission_id")
            res = await _regen_one(
                mission_id=int(mid) if mid is not None else 0,
                artifact_path=str(payload.get("artifact_path") or ""),
                change_description=str(change),
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=str(res.get("error") or "regen_artifact failed"),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "regen_bundle":
        # Z1 Tier 4A (C19) — directional bundle regen. Axis registry in
        # mr_roboto.regen_artifact._KNOWN_AXES; per-artifact emission goes
        # through regen_artifact (so each gets its own regen_log row).
        from mr_roboto.regen import regen_bundle as _regen_bundle
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            res = await _regen_bundle(
                mission_id=int(mid) if mid is not None else 0,
                axis=str(payload.get("axis") or ""),
                direction=str(payload.get("direction") or ""),
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=str(res.get("error") or "regen_bundle failed"),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "annotate_html_oids":
        # Z1 Tier 4 (T4B / C17+A20) — DOM post-processor that assigns
        # `data-oid` to every semantic block in a generated HTML
        # prototype. Runs AFTER verify_html_prototype_shape and BEFORE
        # verify_screen_consistency. Anchor for the spec-patch proposer.
        from mr_roboto.annotate_html_oids import (
            annotate_html_oids as _annotate,
        )
        try:
            res = _annotate(
                html_text=payload.get("html_text"),
                html_paths=payload.get("html_paths"),
                artifact_slug=payload.get("artifact_slug"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"annotate_html_oids: {res.get('error') or res.get('per_file')}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "propagate_asset_change":
        # Z1 Tier 4 (T4B / B2) — paraflow-style asset->spec propagation.
        # Walks produces/consumes graph for the asset_path and surfaces
        # dependents + suggested patches. Founder reviews; accepted
        # items dispatch to T4A's `regen_artifact`.
        from mr_roboto.propagate_asset_change import (
            propagate_asset_change as _propagate,
        )
        try:
            res = _propagate(
                asset_path=payload.get("asset_path") or "",
                change_description=payload.get("change_description") or "",
                steps=payload.get("steps"),
                workflow_path=payload.get("workflow_path"),
                mission_id=payload.get("mission_id") or task.get("mission_id") or "0",
                out_path=payload.get("out_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"propagate_asset_change: {res.get('error')}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "propose_spec_patch_from_html_diff":
        # Z1 Tier 4 (T4B / C17+A20) — DOM-aware HTML diff via data-oid.
        # Founder edits HTML offline; this proposer pairs nodes by oid
        # and surfaces upstream artifact patches for review.
        from mr_roboto.propose_spec_patch import (
            propose_spec_patch_from_html_diff as _propose,
        )
        try:
            res = _propose(
                html_path=payload.get("html_path") or "",
                edited_html_path=payload.get("edited_html_path") or "",
                out_path=payload.get("out_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"propose_spec_patch_from_html_diff: {res.get('error')}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "emit_preview_url":
        # Z1 Tier 4C (C10+A19) — emit a tunneled preview URL surface.
        # EMIT-ONLY: Z2 owns hosting. Fail-soft when env unset / binary missing.
        from mr_roboto.emit_preview_url import (
            emit_preview_url as _emit_preview,
        )
        try:
            res = await _emit_preview(
                mission_id=int(task.get("mission_id") or payload.get("mission_id")),
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"emit_preview_url: {res.get('error') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "kill_preview_url":
        # Z1 Tier 4C (C10) — terminate the per-mission preview tunnel.
        from mr_roboto.kill_preview_url import (
            kill_preview_url as _kill_preview,
        )
        try:
            res = await _kill_preview(
                mission_id=int(task.get("mission_id") or payload.get("mission_id")),
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"kill_preview_url: {res.get('error') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "generate_intake_todo":
        # Z1 Tier 1 (B1) — agent-generated todo as the only structured
        # intake gate. Returns needs_clarification with keyboard_sent so
        # general_beckman.result_router keeps the row waiting_human until
        # the founder confirms via Telegram.
        from mr_roboto.generate_intake_todo import (
            generate_intake_todo as _gen_intake,
        )
        try:
            res = await _gen_intake(task)
            if res.get("status") == "failed":
                return Action(status="failed", error=str(res.get("error", "")))
            if (res.get("status") == "needs_clarification"
                    and res.get("keyboard_sent")):
                return Action(status="needs_clarification", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "run_cmd":
        from mr_roboto.run_cmd import run_cmd as _run_cmd
        try:
            res = await _run_cmd(
                mission_id=task.get("mission_id"),
                cmd=payload.get("cmd") or [],
                cwd=payload.get("cwd"),
                timeout_s=float(payload.get("timeout_s", 60.0)),
                env=payload.get("env"),
                require_exit_zero=bool(payload.get("require_exit_zero", False)),
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"run_cmd: exit={res.get('exit')} "
                        f"timed_out={res.get('timed_out')} "
                        f"err={res.get('error') or ''}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "run_pytest":
        from mr_roboto.run_pytest import run_pytest as _run_pytest
        try:
            res = await _run_pytest(
                mission_id=task.get("mission_id"),
                target=payload.get("target"),
                cwd=payload.get("cwd"),
                timeout_s=float(payload.get("timeout_s", 600.0)),
                extra_args=payload.get("extra_args"),
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"run_pytest: passed={res.get('passed')} "
                        f"failed={res.get('failed')} errors={res.get('errors')} "
                        f"total={res.get('total')} exit={res.get('exit')} "
                        f"timed_out={res.get('timed_out')} "
                        f"err={res.get('error') or ''}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "parse_og_tags":
        from mr_roboto.parse_og_tags import parse_og_tags as _parse_og
        try:
            res = await _parse_og(
                url=payload.get("url") or "",
                timeout_s=float(payload.get("timeout_s", 15.0)),
                check_image=bool(payload.get("check_image", True)),
                required=payload.get("required"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"parse_og_tags: status={res.get('status')} "
                        f"missing={res.get('missing')} "
                        f"errors={res.get('errors')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "http_check":
        from mr_roboto.http_check import http_check as _http_check
        try:
            es = payload.get("expect_status") or (200, 299)
            if isinstance(es, list) and len(es) == 2 and all(
                isinstance(x, int) for x in es
            ) and payload.get("expect_status_as_range", True):
                # JSON can't carry tuples; default-interpret 2-int list as a range.
                es = (int(es[0]), int(es[1]))
            res = await _http_check(
                url=payload.get("url") or "",
                method=str(payload.get("method", "GET")),
                timeout_s=float(payload.get("timeout_s", 15.0)),
                max_attempts=int(payload.get("max_attempts", 5)),
                backoff_base_s=float(payload.get("backoff_base_s", 1.0)),
                backoff_cap_s=float(payload.get("backoff_cap_s", 8.0)),
                expect_status=es,
                expect_body_contains=payload.get("expect_body_contains"),
                headers=payload.get("headers"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"http_check: status={res.get('final_status')} "
                        f"attempts={res.get('attempts')} "
                        f"err={res.get('final_error') or ''}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "clarify":
        from mr_roboto.clarify import clarify
        try:
            res = await clarify(task)
            # variant_choice that successfully sent a keyboard is waiting
            # on a user tap — must return needs_clarification so beckman's
            # result router leaves the row as waiting_human instead of
            # flipping it back to completed (which caused the mission to
            # advance past the clarify step and produce "no results" for
            # every clarify-gated shopping mission). Plain question-clarify
            # stays completed as before.
            if (isinstance(res, dict)
                    and res.get("status") == "needs_clarification"
                    and res.get("keyboard_sent")):
                return Action(status="needs_clarification", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "notify_user":
        from mr_roboto.notify_user import notify_user
        try:
            res = await notify_user(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "todo_reminder":
        from mr_roboto.todo_reminder import run as todo_reminder_run
        try:
            res = await todo_reminder_run(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "price_watch_check":
        from mr_roboto.price_watch_check import run as price_watch_run
        try:
            res = await price_watch_run(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "workflow_advance":
        from mr_roboto.workflow_advance import run as workflow_advance_run
        try:
            res = await workflow_advance_run(task)
            if res.get("status") == "failed":
                return Action(status="failed", error=res.get("error", ""))
            return Action(status=res.get("status", "completed"), result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "cloud_refresh":
        from mr_roboto.cloud_refresh import run as cloud_refresh_run
        try:
            res = await cloud_refresh_run(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "kdv_persist":
        from mr_roboto.kdv_persist import run as kdv_persist_run
        try:
            res = await kdv_persist_run(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "monitoring_check":
        from mr_roboto.executors.monitoring_check import run as monitoring_check_run
        try:
            res = await monitoring_check_run(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "social_preview_check":
        from mr_roboto.executors.social_preview_check import run as social_preview_run
        try:
            res = await social_preview_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"social_preview_check: {res.get('error') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "staging_smoke_check":
        from mr_roboto.executors.staging_smoke_check import run as staging_smoke_run
        try:
            res = await staging_smoke_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"staging_smoke_check: {res.get('error') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "vector_maint_wal":
        from mr_roboto.executors.vector_maint import run_wal
        try:
            res = await run_wal(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "vector_maint_snapshot":
        from mr_roboto.executors.vector_maint import run_snapshot
        try:
            res = await run_snapshot(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "ingest_visual":
        # B7+C16: founder-uploaded sketch/photo → structured visual_brief.md
        from mr_roboto.ingest_visual import ingest_visual as _ingest_visual
        try:
            res = await _ingest_visual(
                mission_id=int(task.get("mission_id") or payload.get("mission_id")),
                file_paths=list(payload.get("file_paths") or []),
                purpose=str(payload.get("purpose") or ""),
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"ingest_visual: reason={res.get('reason')} "
                        f"detail={res.get('detail') or ''}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_screen_plan_shape":
        # Z1 Tier 3 (C3+A10+C14) — paraflow-shape per-screen plan validator.
        # Auto-wired as sibling on each chunked sub-step of `5.1
        # generate_per_screen_plans`.
        from mr_roboto.verify_screen_plan_shape import (
            verify_screen_plan_shape as _verify_sp,
        )
        try:
            res = _verify_sp(
                plan_text=payload.get("plan_text"),
                plan_paths=payload.get("plan_paths"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"verify_screen_plan_shape: problems={res.get('problems') or [pf.get('problems') for pf in res.get('per_file', [])]}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_html_prototype_shape":
        # Z1 Tier 3 (C9+A11) — paraflow-shape per-screen HTML validator.
        # Auto-wired as sibling on each chunked sub-step of `5.2
        # generate_html_prototypes`.
        from mr_roboto.verify_html_prototype_shape import (
            verify_html_prototype_shape as _verify_html,
        )
        try:
            res = _verify_html(
                html_text=payload.get("html_text"),
                html_paths=payload.get("html_paths"),
                design_tokens=payload.get("design_tokens"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"verify_html_prototype_shape: problems={res.get('problems') or [pf.get('problems') for pf in res.get('per_file', [])]}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_screen_consistency":
        # Z1 Tier 3 (C18+B8) — cross-screen inherits_shell consistency
        # check. Wired as sibling at the end of each 5.2 chunk-pair step.
        from mr_roboto.verify_screen_consistency import (
            verify_screen_consistency as _verify_consistency,
        )
        try:
            res = _verify_consistency(
                screen_plan_paths=payload.get("screen_plan_paths"),
                screen_plans=payload.get("screen_plans"),
                shared_shell_components=payload.get("shared_shell_components"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"verify_screen_consistency: mismatches={res.get('mismatches')} "
                        f"out_of_set={res.get('out_of_set')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "compliance_fingerprint_collection":
        # Z1 Tier 5A (P6) — read z0 preflight + emit merged fingerprint.
        from mr_roboto.compliance_fingerprint_collection import (
            compliance_fingerprint_collection as _collect,
        )
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            res = await _collect(
                mission_id=int(mid) if mid is not None else 0,
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"compliance_fingerprint_collection: {res.get('error')}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "compliance_template_present":
        # Z1 Tier 5A (P6) — post-hook on 1.11a; verify referenced templates
        # exist on disk under compliance_templates/.
        from mr_roboto.compliance_template_present import (
            compliance_template_present as _check_present,
        )
        try:
            res = _check_present(
                template_ids=payload.get("template_ids"),
                overlay_path=payload.get("overlay_path"),
                overlay_obj=payload.get("overlay_obj"),
                template_root=payload.get("template_root"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"compliance_template_present: missing={res.get('missing')} "
                        f"checked={res.get('checked')} root={res.get('root')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "compliance_blocker_check":
        # Z1 Tier 5A (P6) — phase-boundary blocker check post-hook on 6.6.
        from mr_roboto.compliance_blocker_check import (
            compliance_blocker_check as _blocker,
        )
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            res = _blocker(
                mission_id=int(mid) if mid is not None else 0,
                current_phase=int(payload.get("current_phase", 6)),
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"compliance_blocker_check: pending={res.get('pending')} "
                        f"phase={res.get('checked_phase')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "attention_check":
        # Z1 Tier 5A (A5) — founder attention budget pre-hook for clarify-shape steps.
        from mr_roboto.attention_check import attention_check as _att_check
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            res = await _att_check(
                mission_id=int(mid) if mid is not None else 0,
                reserve_minutes=int(payload.get("reserve_minutes", 5)),
            )
            # Never fail the dispatch — caller (orchestrator pre-hook) reads
            # `ok` and decides whether to skip or proceed. Return result.
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "attention_debit":
        # Z1 Tier 5A (A5) — record a debit row.
        from mr_roboto.attention_check import attention_debit as _att_debit
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            res = await _att_debit(
                mission_id=int(mid) if mid is not None else 0,
                step_id=str(payload.get("step_id") or ""),
                action=str(payload.get("debit_action") or "clarify"),
                minutes_debited=int(payload.get("minutes_debited", 5)),
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    return Action(status="failed", error=f"unknown mechanical action: {action!r}")
