"""Mr. Roboto — mechanical dispatcher: non-LLM task executors."""
from __future__ import annotations

from mr_roboto.actions import Action
from mr_roboto.reversibility import (
    Reversibility,
    VERB_REVERSIBILITY,
    DEFAULT_REVERSIBILITY,
    get_reversibility,
)
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
from mr_roboto.run_jest import run_jest
from mr_roboto.run_vitest import run_vitest
from mr_roboto.run_semgrep import run_semgrep
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
from mr_roboto.verify_premortem_shape import verify_premortem_shape
from mr_roboto.spec_consistency_check import spec_consistency_check
# NOTE: do NOT `from mr_roboto.init_mission_github_repo import init_mission_github_repo`
# here — that would shadow the submodule on the package namespace and break
# `monkeypatch.setattr("mr_roboto.init_mission_github_repo._persist_repo_url", ...)`
# style mocks. Import the submodule itself; consumers can do
# `from mr_roboto.init_mission_github_repo import init_mission_github_repo`
# at call time.
from mr_roboto import init_mission_github_repo as init_mission_github_repo_module  # noqa: F401
# Same submodule-import pattern for T6A (find_similar_missions + surface_prior_mission_hints).
from mr_roboto import find_similar_missions as find_similar_missions_module  # noqa: F401
from mr_roboto import surface_prior_mission_hints as surface_prior_mission_hints_module  # noqa: F401
from mr_roboto.prior_art_min_coverage import prior_art_min_coverage
from mr_roboto.pick_recipe import pick_recipe
# NOTE: do NOT `from mr_roboto.critic_gate import critic_gate` — that would
# shadow the submodule on the mr_roboto package namespace and break
# `patch("mr_roboto.critic_gate._persist")` style mocking. Import the
# submodule itself so `mr_roboto.critic_gate` keeps resolving to the module.
from mr_roboto import critic_gate as critic_gate_module  # noqa: F401
# Z10 T3C — reset-to-green primitives.
from mr_roboto import mark_green as mark_green_module  # noqa: F401
from mr_roboto import rollback_mission as rollback_mission_module  # noqa: F401
# Z10 T4A — end-of-mission demo deliverable.
# Submodule imports keep `mr_roboto.record_demo` resolving to the module
# (matches the mocking pattern used elsewhere, e.g. critic_gate / mark_green).
# `from mr_roboto import record_demo` returns the submodule; callers use
# `record_demo.run(...)`.
import mr_roboto.record_demo as record_demo  # noqa: F401
import mr_roboto.verify_demo_artifact as verify_demo_artifact  # noqa: F401
import mr_roboto.mission_deliverable_bundle as mission_deliverable_bundle  # noqa: F401
from mr_roboto.check_imports import check_imports
from mr_roboto.regen_and_diff import regen_and_diff
from mr_roboto.apply_migration import apply_migration
from mr_roboto.inject_lessons import inject_lessons
from mr_roboto.instantiate_recipe import instantiate_recipe_verb

__all__ = [
    "Action",
    "Reversibility",
    "VERB_REVERSIBILITY",
    "DEFAULT_REVERSIBILITY",
    "get_reversibility",
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
    "run_jest",
    "run_vitest",
    "run_semgrep",
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
    "verify_premortem_shape",
    "spec_consistency_check",
    "prior_art_min_coverage",
    "check_imports",
    "regen_and_diff",
    "apply_migration",
    "inject_lessons",
    "pick_recipe",
    "instantiate_recipe_verb",
]


async def run(task: dict) -> Action:
    """Route a mechanical task to the appropriate executor.

    ``task["payload"]["action"]`` selects the executor:

    - ``"workspace_snapshot"`` → :func:`mr_roboto.snapshot_workspace`
    - ``"git_commit"``         → :func:`mr_roboto.auto_commit`

    Unknown actions return an ``Action(status="failed", error=...)``; the
    orchestrator is responsible for marking the task failed.

    Z10-T1B: every returned Action carries a ``reversibility`` tag from
    :data:`VERB_REVERSIBILITY`, with per-invocation override via
    ``payload["reversibility_override"]``.

    Z10-T1C: every dispatch lands a row in ``registry_events``
    (scope='action') with verb + reversibility + mission/task ids. When
    ``payload['require_confirmation']`` is True AND the resolved
    reversibility is ``partial`` / ``irreversible``, the dispatcher opens
    a confirmation row and polls until the founder responds (or the 60s
    skeleton timeout fires — T2B will replace polling with the Telegram
    reactor).
    """
    payload = task.get("payload") or {}
    verb = payload.get("action") or ""
    override = payload.get("reversibility_override")
    if override not in ("full", "partial", "irreversible", None):
        override = None
    resolved_reversibility = get_reversibility(str(verb), override=override)

    # Skeleton confirmation gate. Default off; only the explicit caller
    # flag arms it for T1C. T2B will wire auto-arm + Telegram surface.
    require_confirmation = bool(payload.get("require_confirmation", False))
    if require_confirmation and resolved_reversibility in (
        "partial", "irreversible"
    ):
        gate_action = await _await_confirmation(
            task_id=task.get("id"),
            verb=str(verb),
            reversibility=resolved_reversibility,
            payload=payload,
        )
        if gate_action is not None:
            gate_action.reversibility = resolved_reversibility
            await _log_action_event(
                verb=str(verb),
                reversibility=resolved_reversibility,
                task=task,
                payload=payload,
                status=gate_action.status,
            )
            return gate_action

    action_obj = await _run_dispatch(task)
    action_obj.reversibility = resolved_reversibility

    await _log_action_event(
        verb=str(verb),
        reversibility=resolved_reversibility,
        task=task,
        payload=payload,
        status=action_obj.status,
    )
    return action_obj


async def _log_action_event(
    *,
    verb: str,
    reversibility: str,
    task: dict,
    payload: dict,
    status: str,
) -> None:
    """Best-effort audit-log writer. Never raises into the dispatch path."""
    try:
        from src.infra.db import record_action_event
        # Strip well-known noisy fields from the payload snapshot so the
        # audit row stays compact and JSON-serializable.
        snap = {
            k: v for k, v in (payload or {}).items()
            if k not in ("result",) and not callable(v)
        }
        await record_action_event(
            verb=verb,
            reversibility=reversibility,
            mission_id=task.get("mission_id"),
            task_id=task.get("id"),
            payload=snap,
            status=status,
        )
    except Exception:
        # Audit failures must never block the action itself.
        import logging
        logging.getLogger("mr_roboto.audit").debug(
            "record_action_event failed", exc_info=True
        )


async def _await_confirmation(
    *,
    task_id,
    verb: str,
    reversibility: str,
    payload: dict,
    max_wait_s: float = 60.0,
    poll_interval_s: float = 0.5,
) -> Action | None:
    """Open a confirmation request and poll for verdict.

    Returns:
        ``None`` when verdict='approved' — caller proceeds with dispatch.
        ``Action(status='rejected')`` when verdict='rejected'.
        ``Action(status='failed', error='confirmation_timeout')`` when the
        skeleton timeout fires (T2B replaces this with reactor delivery).
    """
    import asyncio
    try:
        from src.infra.db import (
            request_confirmation,
            check_confirmation,
        )
    except Exception as e:
        # If the DB isn't available we can't gate — fail closed.
        return Action(
            status="failed",
            error=f"confirmation_unavailable: {e}",
        )

    try:
        summary = payload.get("payload_summary") or payload.get("message") or ""
        summary = str(summary)[:500]
        confirmation_id = await request_confirmation(
            task_id=int(task_id) if task_id is not None else 0,
            verb=verb,
            reversibility=reversibility,
            payload_summary=summary,
        )
    except Exception as e:
        return Action(
            status="failed", error=f"confirmation_open_failed: {e}"
        )

    elapsed = 0.0
    while elapsed < max_wait_s:
        res = await check_confirmation(confirmation_id)
        verdict = res.get("verdict")
        if verdict == "approved":
            return None
        if verdict == "rejected":
            return Action(
                status="rejected",
                error="action rejected by confirmation gate",
                result={"confirmation_id": confirmation_id},
            )
        await asyncio.sleep(poll_interval_s)
        elapsed += poll_interval_s

    return Action(
        status="failed",
        error="confirmation_timeout",
        result={"confirmation_id": confirmation_id},
    )


async def _run_dispatch(task: dict) -> Action:
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
        # Z1 Tier 5C (B4) — Critic gate post-hook on `git_commit`.
        # Pattern: pre-stage, capture the planned commit message + staged
        # diff, gate-check; on veto unstage and return failed; on pass
        # let auto_commit perform the real commit.
        from mr_roboto.critic_gate import critic_gate as _critic_gate, _opt_out as _critic_opt_out
        from src.tools.workspace import get_mission_workspace_relative
        from src.tools.git_ops import _run_git, ensure_git_repo

        gate_enabled = (not _critic_opt_out()) and bool(payload.get("critic_gate", True))
        gate_result: dict | None = None
        if gate_enabled:
            try:
                mid = task.get("mission_id")
                repo_path = (
                    get_mission_workspace_relative(mid) if mid else ""
                )
                await ensure_git_repo(repo_path)
                # Stage everything so we can read the staged diff.
                from src.tools.git_ops import _resolve_repo
                target = _resolve_repo(repo_path) or ""
                if target:
                    await _run_git(["add", "-A"], cwd=target)
                    _, diff_out, _ = await _run_git(
                        ["diff", "--cached", "--stat"], cwd=target
                    )
                    _, diff_full, _ = await _run_git(
                        ["diff", "--cached"], cwd=target
                    )
                else:
                    diff_out = ""
                    diff_full = ""
                planned_msg = (
                    f"Task #{task.get('id')}: "
                    f"{(task.get('title') or 'untitled')[:60]}"
                )
                gate_result = await _critic_gate(
                    "git_commit",
                    {
                        "commit_message": planned_msg,
                        "diff_stat": (diff_out or "")[:2000],
                        "diff_excerpt": (diff_full or "")[:4000],
                    },
                    mission_id=task.get("mission_id"),
                )
                if gate_result.get("verdict") == "veto":
                    # Roll back the stage so the next attempt starts clean.
                    if target:
                        await _run_git(["reset"], cwd=target)
                    return Action(
                        status="failed",
                        error=(
                            "critic_gate vetoed git_commit: "
                            f"{gate_result.get('reasons')}"
                        ),
                        result={"critic": gate_result},
                    )
            except Exception as e:
                # Never block work on a broken gate — log and continue.
                from src.infra.logging_config import get_logger as _gl
                _gl("mr_roboto.critic_gate").warning(
                    f"git_commit critic gate failed open: {e}"
                )

        commit_info = await auto_commit(task, payload.get("result") or {})
        if gate_result is not None:
            (commit_info or {}).setdefault("critic", gate_result)

        # Z10 T1C: provenance — record each file changed by this commit.
        # Best-effort; failures must not break the commit action.
        try:
            if (commit_info or {}).get("committed") and (commit_info or {}).get("commit_sha"):
                from src.tools.git_ops import _run_git, _resolve_repo
                mid = task.get("mission_id")
                repo_path = (
                    get_mission_workspace_relative(mid) if mid else ""
                )
                target = _resolve_repo(repo_path) or ""
                if target:
                    _, names_out, _ = await _run_git(
                        [
                            "show",
                            "--no-renames",
                            "--pretty=",
                            "--name-only",
                            commit_info["commit_sha"],
                        ],
                        cwd=target,
                    )
                    changed = [
                        line.strip()
                        for line in (names_out or "").splitlines()
                        if line.strip()
                    ]
                    if changed:
                        from src.infra.db import record_artifact_write
                        retry_n = int(payload.get("retry_n") or 0)
                        model_id = (
                            payload.get("model_id")
                            or payload.get("model")
                            or None
                        )
                        step_id = (
                            payload.get("step_id")
                            or task.get("workflow_step_id")
                        )
                        for p in changed:
                            try:
                                await record_artifact_write(
                                    path=p,
                                    task_id=task.get("id"),
                                    step_id=step_id,
                                    model_id=model_id,
                                    retry_n=retry_n,
                                    reviewer_verdict_id=payload.get(
                                        "reviewer_verdict_id"
                                    ),
                                    mission_id=task.get("mission_id"),
                                )
                            except Exception:
                                pass
        except Exception:
            import logging
            logging.getLogger("mr_roboto.provenance").debug(
                "record_artifact_write failed", exc_info=True
            )

        # Z10 T3C — optional green-tag capture. When the step sets
        # `payload.mark_green=True`, snapshot workspace + DB + Chroma into
        # `mission_green_tags` so /rollback_mission can restore here.
        # Best-effort — failure does NOT fail the commit.
        if (
            payload.get("mark_green")
            and (commit_info or {}).get("committed")
            and task.get("mission_id") is not None
        ):
            try:
                from mr_roboto.mark_green import run as _mark_green
                _green = await _mark_green(
                    mission_id=int(task["mission_id"]),
                    task_id=int(task["id"]),
                    summary=(
                        f"{task.get('title', '')} | "
                        f"{(commit_info or {}).get('commit_sha', '')[:12]}"
                    ),
                    repo_path=payload.get("repo_path"),
                )
                (commit_info or {}).setdefault("green_tag", _green)
            except Exception as _green_exc:
                from src.infra.logging_config import get_logger as _gl
                _gl("mr_roboto.mark_green").warning(
                    f"mark_green from git_commit failed: {_green_exc}"
                )

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

    if action == "critic_gate":
        # Z1 Tier 5C (B4) — standalone critic-gate invocation. Useful
        # for non-mr_roboto actions (e.g. pre-deploy gates) and for
        # explicit workflow steps. Returns failed when verdict=veto.
        from mr_roboto.critic_gate import critic_gate as _standalone_critic
        try:
            res = await _standalone_critic(
                action_name=str(payload.get("action_name") or "unknown"),
                payload=payload.get("target_payload"),
                mission_id=task.get("mission_id") or payload.get("mission_id"),
            )
            if res.get("verdict") == "veto":
                return Action(
                    status="failed",
                    error=f"critic_gate veto: {res.get('reasons')}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

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
                reversibility_intent=payload.get("reversibility_override"),
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

    if action == "check_imports":
        # Z2 T2B — static import checker.
        from mr_roboto.check_imports import check_imports as _check_imports
        try:
            res = await _check_imports(
                mission_id=task.get("mission_id"),
                target_files=payload.get("target_files") or [],
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("ok"):
                missing = res.get("missing") or []
                summary = "; ".join(
                    f"{r['file']}:{r.get('line', '?')} {r['module']}"
                    for r in missing[:5]
                )
                if len(missing) > 5:
                    summary += f" … (+{len(missing) - 5} more)"
                return Action(
                    status="failed",
                    error=f"check_imports: missing={summary}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "run_semgrep":
        # Z2 T2C — pattern_lint post-hook. Soft-skips when semgrep absent.
        from mr_roboto.run_semgrep import run_semgrep as _run_semgrep
        try:
            res = await _run_semgrep(
                mission_id=task.get("mission_id"),
                target_files=payload.get("target_files"),
                rule_pack_path=payload.get("rule_pack_path"),
                workspace_path=payload.get("workspace_path"),
                timeout_s=float(payload.get("timeout_s", 120.0)),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"run_semgrep: exit={res.get('exit')} "
                        f"err={res.get('error') or ''}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "regen_and_diff":
        # Z2 T3B — openapi_sync / typescript_sync post-hook.
        from mr_roboto.regen_and_diff import regen_and_diff as _regen_and_diff
        try:
            res = await _regen_and_diff(
                mission_id=task.get("mission_id"),
                generator_cmd=list(payload.get("generator_cmd") or []),
                target_path=str(payload.get("target_path") or ""),
                workspace_path=str(payload.get("workspace_path") or ""),
                timeout_s=float(payload.get("timeout_s", 60.0)),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"regen_and_diff: exit={res.get('exit')} "
                        f"err={res.get('error') or ''}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "legal_document_render":
        # Z6 T4A — render legal docs from compliance_overlay.required_documents[]
        # at step 12.1. Mechanical predecessor to 12.1b's LLM placeholder-fill.
        from mr_roboto.executors.legal_document_render import (
            legal_document_render as _legal_render,
        )
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            res = await _legal_render(
                mission_id=int(mid) if mid is not None else 0,
                workspace_path=payload.get("workspace_path"),
                overlay_obj=payload.get("overlay_obj"),
                lang=str(payload.get("lang") or "en"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"legal_document_render: reason={res.get('reason')} "
                        f"errors={res.get('errors')} skipped={res.get('skipped')}"
                    )[:500],
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "apply_migration":
        # Z2 T3A — migration_apply post-hook.
        from mr_roboto.apply_migration import apply_migration as _apply_migration
        try:
            res = await _apply_migration(
                mission_id=task.get("mission_id"),
                target_files=list(payload.get("target_files") or []),
                workspace_path=str(payload.get("workspace_path") or ""),
                stack_hint=str(payload.get("stack_hint") or ""),
                timeout_s=float(payload.get("timeout_s", 120.0)),
                enable_testcontainers=bool(
                    payload.get("enable_testcontainers", False)
                ),
            )
            if res.get("skipped"):
                return Action(status="completed", result=res)
            if res.get("ok"):
                return Action(status="completed", result=res)
            return Action(
                status="failed",
                error=(
                    f"apply_migration: stack={res.get('stack_used')} "
                    f"err={res.get('error') or ''}"
                )[:500],
                result=res,
            )
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "run_tests":
        # test_run post-hook dispatch: picks the right runner from target_files.
        # Rules (applied in order):
        #   *.py            → run_pytest
        #   *.test.ts(x)    → run_jest if jest config detected, else run_vitest
        #   *.spec.ts       → same as above
        #   no match        → warning verdict (no runner detected)
        target_files: list[str] = list(payload.get("target_files") or [])
        stack_hint: str = str(payload.get("stack_hint") or "")
        timeout_s_val = float(payload.get("timeout_s", 600.0))

        # Classify target files by language
        py_targets = [f for f in target_files if f.endswith(".py")]
        ts_targets = [
            f for f in target_files
            if any(f.endswith(ext) for ext in (
                ".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx",
            ))
        ]

        if py_targets:
            from mr_roboto.run_pytest import run_pytest as _run_pytest
            try:
                res = await _run_pytest(
                    mission_id=task.get("mission_id"),
                    target=py_targets,
                    cwd=payload.get("cwd"),
                    timeout_s=timeout_s_val,
                    extra_args=payload.get("extra_args"),
                    workspace_path=payload.get("workspace_path"),
                )
            except Exception as e:
                return Action(status="failed", error=f"run_tests/pytest spawn: {e}")
        elif ts_targets:
            # Choose runner: jest if jest is detected, else vitest.
            # Detection: stack_hint contains "jest", or workspace package.json
            # has a jest key. Vitest is the default for TS (common in Vite stacks).
            use_jest = "jest" in stack_hint.lower()
            if not use_jest:
                # Best-effort workspace package.json probe
                import os, json as _json
                ws = payload.get("workspace_path") or ""
                pkg_path = os.path.join(ws, "package.json") if ws else "package.json"
                try:
                    with open(pkg_path, "r", encoding="utf-8") as _f:
                        pkg = _json.load(_f)
                    use_jest = "jest" in (pkg.get("dependencies") or {}) or \
                               "jest" in (pkg.get("devDependencies") or {}) or \
                               "jest" in pkg
                except (OSError, _json.JSONDecodeError):
                    pass

            if use_jest:
                from mr_roboto.run_jest import run_jest as _run_jest
                try:
                    res = await _run_jest(
                        mission_id=task.get("mission_id"),
                        target=ts_targets,
                        cwd=payload.get("cwd"),
                        timeout_s=timeout_s_val,
                        extra_args=payload.get("extra_args"),
                        workspace_path=payload.get("workspace_path"),
                    )
                except Exception as e:
                    return Action(status="failed", error=f"run_tests/jest spawn: {e}")
            else:
                from mr_roboto.run_vitest import run_vitest as _run_vitest
                try:
                    res = await _run_vitest(
                        mission_id=task.get("mission_id"),
                        target=ts_targets,
                        cwd=payload.get("cwd"),
                        timeout_s=timeout_s_val,
                        extra_args=payload.get("extra_args"),
                        workspace_path=payload.get("workspace_path"),
                    )
                except Exception as e:
                    return Action(status="failed", error=f"run_tests/vitest spawn: {e}")
        else:
            # No recognised test files — warn but don't block.
            return Action(
                status="completed",
                result={
                    "ok": True,
                    "warning": "no_runner_detected",
                    "target_files": target_files,
                    "message": (
                        "run_tests: no .py or .test.ts(x)/.spec.ts(x) targets found; "
                        "skipping test run."
                    ),
                },
            )

        # Slow-suite warning (>120s) but don't block; red test → blocker.
        duration = float(res.get("duration_s") or 0.0)
        warning = None
        if duration > 120.0 and res.get("ok"):
            warning = f"slow_suite: {duration:.1f}s > 120s threshold"

        if not res.get("ok"):
            return Action(
                status="failed",
                error=(
                    f"run_tests: passed={res.get('passed')} "
                    f"failed={res.get('failed')} errors={res.get('errors')} "
                    f"total={res.get('total')} exit={res.get('exit')} "
                    f"timed_out={res.get('timed_out')} "
                    f"err={res.get('error') or ''}"
                ),
                result=res,
            )
        result_out = dict(res)
        if warning:
            result_out["warning"] = warning
        return Action(status="completed", result=result_out)

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
        # Z1 Tier 5C (B4) — Critic gate pre-hook on `notify_user`.
        # Send is irreversible (user sees the message); gate the text
        # before dispatch. Veto = drop the message and return failed.
        from mr_roboto.notify_user import notify_user
        from mr_roboto.critic_gate import (
            critic_gate as _critic_gate,
            _opt_out as _critic_opt_out,
        )
        gate_enabled = (
            (not _critic_opt_out()) and bool(payload.get("critic_gate", True))
        )
        if gate_enabled:
            text = payload.get("message") or payload.get("text") or ""
            try:
                gate_result = await _critic_gate(
                    "notify_user",
                    {"message": text, "chat_id": payload.get("chat_id")},
                    mission_id=task.get("mission_id"),
                )
            except Exception as e:
                from src.infra.logging_config import get_logger as _gl
                _gl("mr_roboto.critic_gate").warning(
                    f"notify_user critic gate failed open: {e}"
                )
                gate_result = None
            if gate_result and gate_result.get("verdict") == "veto":
                return Action(
                    status="failed",
                    error=(
                        "critic_gate vetoed notify_user: "
                        f"{gate_result.get('reasons')}"
                    ),
                    result={"critic": gate_result, "sent": False},
                )
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

    if action == "mission_event_drain":
        # Z10 T2B: drain T1C confirmations + T2A budget alerts → mission_events.
        from mr_roboto.mission_event_drain import run as drain_run
        try:
            res = await drain_run(task)
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

    if action == "vendor_call":
        # Z6 T3A — drive a real-world vendor API via IntegrationRegistry.
        from mr_roboto.executors.vendor_call import run as _vendor_call_run
        try:
            res = await _vendor_call_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"vendor_call: reason={res.get('reason')} "
                        f"service={res.get('service')} "
                        f"action={res.get('action')} "
                        f"err={res.get('error') or res.get('detail') or ''}"
                    ),
                    result=res,
                )
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

    if action == "verify_premortem_shape":
        # Z1 Tier 5B (A6) — premortem envelope shape verifier. Auto-wired
        # as post-hook on step 6.5z failure_premortem.
        from mr_roboto.verify_premortem_shape import (
            verify_premortem_shape as _verify_premortem,
        )
        try:
            res = _verify_premortem(
                premortem=payload.get("premortem"),
                premortem_text=payload.get("premortem_text"),
                premortem_path=payload.get("premortem_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"verify_premortem_shape: problems={res.get('problems')} "
                        f"missing_kinds={res.get('missing_kinds')} "
                        f"per_item_problems={res.get('per_item_problems')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "spec_consistency_check":
        # Z1 Tier 5B (B5) — Augment Intent's "specs stay alive". Wave-start
        # mechanical step before every phase 7+ wave: re-reads phase-≤6
        # spec artifacts + current phase-N artifacts, surfaces drift.
        # On drift: returns needs_review with the report path so the
        # reviewer / source acknowledges before downstream phases proceed.
        from mr_roboto.spec_consistency_check import (
            spec_consistency_check as _spec_check,
        )
        try:
            mid = task.get("mission_id") or payload.get("mission_id") or "0"
            current_phase = (
                payload.get("current_phase")
                or task.get("phase")
                or "phase_unknown"
            )
            res = _spec_check(
                mission_id=mid,
                current_phase=str(current_phase),
                workspace_path=payload.get("workspace_path"),
                out_path=payload.get("out_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="needs_review",
                    error=(
                        f"spec_consistency_check: drift_count="
                        f"{len(res.get('drift_items') or [])} "
                        f"report={res.get('report_path')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "init_mission_github_repo":
        # Z1 Tier 6 (C18) — end-of-phase-6 GitHub repo init. Mechanical.
        from mr_roboto.init_mission_github_repo import (
            init_mission_github_repo as _init_gh,
        )
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            res = await _init_gh(
                mission_id=int(mid) if mid is not None else 0,
                repo_visibility=str(
                    payload.get("repo_visibility")
                    or payload.get("visibility")
                    or "public"
                ),
                workspace_path=payload.get("workspace_path"),
                skip=bool(payload.get("skip_github", False)),
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "find_similar_missions":
        # Z1 Tier 6A (A7) — idea fingerprint cross-mission dedup.
        from mr_roboto.find_similar_missions import (
            find_similar_missions as _find_similar,
        )
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            res = await _find_similar(
                mission_id=int(mid) if mid is not None else 0,
                idea_summary=payload.get("idea_summary"),
                workspace_path=payload.get("workspace_path"),
                top_k=int(payload.get("top_k", 3)),
                threshold=(
                    float(payload["threshold"])
                    if payload.get("threshold") is not None
                    else None
                ),
            )
            if not res.get("ok"):
                return Action(
                    status="needs_review",
                    error=(
                        f"find_similar_missions: matches="
                        f"{len(res.get('matches') or [])} "
                        f"report={res.get('report_path')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "index_idea_fingerprint":
        # Z1 Tier 6A (A7) — embed + store idea fingerprint after confirm.
        from mr_roboto.find_similar_missions import (
            index_idea_fingerprint as _idx_idea,
        )
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            res = await _idx_idea(
                mission_id=int(mid) if mid is not None else 0,
                idea_summary=payload.get("idea_summary"),
                workspace_path=payload.get("workspace_path"),
                title=str(payload.get("title") or ""),
                final_status_note=str(payload.get("final_status_note") or ""),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"index_idea_fingerprint: {res.get('reason')}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "surface_prior_mission_hints":
        # Z1 Tier 6A (P9) — cross-mission ADR + compliance inheritance.
        from mr_roboto.surface_prior_mission_hints import (
            surface_prior_mission_hints as _surface_hints,
        )
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            res = await _surface_hints(
                mission_id=int(mid) if mid is not None else 0,
                workspace_path=payload.get("workspace_path"),
                founder_id=str(payload.get("founder_id") or "default"),
                top_k=int(payload.get("top_k", 3)),
                jaccard_threshold=float(
                    payload.get("jaccard_threshold", 0.3)
                ),
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "index_mission_artifacts":
        # Z1 Tier 6A (P9) — phase-6-tail mechanical step.
        from mr_roboto.surface_prior_mission_hints import (
            index_mission_artifacts as _idx_artifacts,
        )
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            res = await _idx_artifacts(
                mission_id=int(mid) if mid is not None else 0,
                workspace_path=payload.get("workspace_path"),
                founder_id=str(payload.get("founder_id") or "default"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"index_mission_artifacts: {res.get('reason')}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "prior_art_min_coverage":
        # Z1 Tier 6B (P5) — post-hook on step 1.0 prior_art_search.
        from mr_roboto.prior_art_min_coverage import (
            prior_art_min_coverage as _pa_check,
        )
        try:
            res = _pa_check(
                report=payload.get("report"),
                report_path=payload.get("report_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"prior_art_min_coverage: problems={res.get('problems')} "
                        f"verdict={res.get('verdict')!r} "
                        f"attempted={res.get('attempted')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "run_bash_audit":
        # Z1 Tier 7A (B12) — quarterly sade_kalsin scaffolding audit.
        from mr_roboto.run_bash_audit import run as bash_audit_run
        try:
            res = await bash_audit_run(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_against_paraflow_goldens":
        # Z1 Tier 7B (C21) — bundle-quality regression vs Paraflow goldens.
        # NOT auto-wired into i2p; invoked manually via /paraflow_check
        # or by a future standing audit job.
        from mr_roboto.verify_against_paraflow_goldens import (
            verify_against_paraflow_goldens as _verify_paraflow,
        )
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            res = await _verify_paraflow(
                mission_id=int(mid) if mid is not None else 0,
                archetype=str(payload.get("archetype") or "truthrate"),
                workspace_path=payload.get("workspace_path"),
            )
            # Always completed: paraflow_gap is information, not error.
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "mark_green":
        # Z10 T3C — capture paired green checkpoint (git tag + DB snapshot +
        # Chroma snapshot + ledger row).
        from mr_roboto.mark_green import run as _mark_green
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            tid = task.get("id") or payload.get("task_id")
            if mid is None or tid is None:
                return Action(
                    status="failed",
                    error="mark_green requires mission_id and task_id",
                )
            res = await _mark_green(
                mission_id=int(mid),
                task_id=int(tid),
                summary=str(payload.get("summary") or task.get("title") or ""),
                repo_path=payload.get("repo_path"),
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "rollback_mission":
        # Z10 T3C — restore workspace/git + mission DB rows + Chroma to a
        # prior green-tag snapshot. Registered as `irreversible` in
        # VERB_REVERSIBILITY so T1C confirmation flow auto-gates when the
        # caller sets require_confirmation=True.
        from mr_roboto.rollback_mission import run as _rollback
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            if mid is None:
                return Action(
                    status="failed",
                    error="rollback_mission requires mission_id",
                )
            target = payload.get("target_task_id")
            res = await _rollback(
                mission_id=int(mid),
                target_task_id=int(target) if target is not None else None,
                repo_path=payload.get("repo_path"),
            )
            if not res.get("ok"):
                return Action(status="failed", error=res.get("error") or "rollback failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "record_demo":
        # Z10 T4A — Playwright e2e capture → ffmpeg-trimmed mp4 demo.
        from mr_roboto.record_demo import run as _record_demo
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            if mid is None:
                return Action(
                    status="failed",
                    error="record_demo requires mission_id (per-mission container)",
                )
            scenario = payload.get("scenario_path") or "tests/e2e/golden_path.spec.ts"
            max_s = int(payload.get("max_seconds") or 90)
            res = await _record_demo(
                mission_id=int(mid),
                scenario_path=str(scenario),
                max_seconds=max_s,
            )
            # Record provenance for demo.mp4 so it surfaces in artifact lineage.
            try:
                from src.infra.db import record_artifact_write
                await record_artifact_write(
                    path=res.get("video_path") or "",
                    task_id=task.get("id"),
                    step_id=payload.get("step_id") or task.get("workflow_step_id"),
                    model_id=None,
                    retry_n=int(payload.get("retry_n") or 0),
                    mission_id=int(mid),
                )
            except Exception:
                pass
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verify_demo_artifact":
        # Z10 T4A — sibling gate after record_demo.
        from mr_roboto.verify_demo_artifact import run as _verify_demo
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            if mid is None:
                return Action(
                    status="failed",
                    error="verify_demo_artifact requires mission_id",
                )
            res = _verify_demo(
                mission_id=int(mid),
                video_path=payload.get("video_path"),
                min_bytes=int(payload.get("min_bytes") or (1024 * 1024)),
                min_duration_s=float(payload.get("min_duration_s") or 5.0),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=res.get("reason") or "demo verify failed",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "mission_deliverable_bundle":
        # Z10 T4A — Telegram post of demo + commit + provenance + cost.
        from mr_roboto.mission_deliverable_bundle import run as _bundle
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            if mid is None:
                return Action(
                    status="failed",
                    error="mission_deliverable_bundle requires mission_id",
                )
            # Bot may be passed via payload (orchestrator) or via app singleton.
            bot = payload.get("bot")
            if bot is None:
                try:
                    from src.app.telegram_bot import get_bot
                    bot = await get_bot()
                except Exception:
                    bot = None
            if bot is None:
                return Action(
                    status="failed",
                    error="mission_deliverable_bundle: no telegram bot available",
                )
            res = await _bundle(
                bot=bot,
                mission_id=int(mid),
                video_path=payload.get("video_path"),
                repo_path=payload.get("repo_path"),
                chat_id=payload.get("chat_id"),
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "inject_lessons":
        # Z2 T4C — cross-mission lesson injector.
        from mr_roboto.inject_lessons import inject_lessons as _inject_lessons
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            if mid is None:
                return Action(
                    status="failed",
                    error="inject_lessons requires mission_id",
                )
            res = await _inject_lessons(
                mission_id=int(mid),
                stack=str(payload.get("stack") or ""),
                domain=payload.get("domain"),
                limit=int(payload.get("limit") or 5),
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "pick_recipe":
        # Z2 T5A — match recipe library against stack + feature_decl.
        from mr_roboto.pick_recipe import pick_recipe as _pick_recipe
        try:
            res = await _pick_recipe(
                mission_id=task.get("mission_id"),
                feature_decl=str(payload.get("feature_decl") or ""),
                stack=str(payload.get("stack") or ""),
                recipes_dir=str(payload.get("recipes_dir") or "recipes"),
                min_fit=float(payload.get("min_fit", 0.7)),
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "instantiate_recipe":
        # Z2 T5C — instantiate recipe templates into target directory.
        from mr_roboto.instantiate_recipe import instantiate_recipe_verb as _inst
        try:
            res = await _inst(
                recipe_name=str(payload.get("recipe_name") or ""),
                version=str(payload.get("version") or ""),
                target_dir=str(payload.get("target_dir") or ""),
                params=dict(payload.get("params") or {}),
                recipes_dir=str(payload.get("recipes_dir") or "recipes"),
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    return Action(status="failed", error=f"unknown mechanical action: {action!r}")
