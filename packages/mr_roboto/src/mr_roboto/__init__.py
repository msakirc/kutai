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
from mr_roboto.publish_preview_pages import publish_preview_pages
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
from mr_roboto.prior_art_fetch import prior_art_fetch
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
from mr_roboto.extract_signatures import extract_signatures
from mr_roboto.check_adr_drift import check_adr_drift  # Z3 T4B
from mr_roboto.integration_replay import integration_replay  # Z3 T5
from mr_roboto.integration_bisect import integration_bisect  # Z3 T5
# Z5 T4b — Maestro mobile-QA adapter (feeds the `mobile_smoke` post-hook)
from mr_roboto.maestro_run import maestro_run  # Z5 T4b
from mr_roboto.visual_review import visual_review  # Z4 T2B
from mr_roboto.capture_screenshots import capture_screenshots  # Z4 T1A
# Z7 T6 A7 — cold outreach + deliverability spine (A7 + A7.r1)
import mr_roboto.outreach_send as outreach_send_module  # noqa: F401
import mr_roboto.outreach_handle_reply as outreach_handle_reply_module  # noqa: F401
import mr_roboto.outreach_draft as outreach_draft_module  # noqa: F401
import mr_roboto.outreach_deliverability_check as outreach_deliverability_check_module  # noqa: F401
import mr_roboto.outreach_domain_verify as outreach_domain_verify_module  # noqa: F401
# Z7 T6 A12 — marketing copy generator (A12 / A1)
import mr_roboto.marketing_copy as marketing_copy_module  # noqa: F401
# Z7 T6D — demo distribution stage (A3 distribute)
import mr_roboto.demo_distribute as demo_distribute_module  # noqa: F401
# Z5 T3 — mobile build/distribution adapters
from mr_roboto.expo_cli import expo_cli  # Z5 T3
from mr_roboto.android_build import android_build  # Z5 T3
from mr_roboto.eas_build import eas_build  # Z5 T3
from mr_roboto.eas_submit import eas_submit  # Z5 T3
# Z5 T3b — free-first GitHub Actions CI + Fastlane mobile build path
from mr_roboto.gen_mobile_ci import gen_mobile_ci  # Z5 T3b
from mr_roboto.fastlane_run import fastlane  # Z5 T3b

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
    "publish_preview_pages",
    "compliance_fingerprint_collection",
    "compliance_template_present",
    "compliance_blocker_check",
    "attention_check",
    "attention_debit",
    "z0_preflight_write",
    "write_deferred_question",
    "verify_premortem_shape",
    "spec_consistency_check",
    "prior_art_min_coverage",
    "prior_art_fetch",
    "check_imports",
    "regen_and_diff",
    "apply_migration",
    "inject_lessons",
    "pick_recipe",
    "instantiate_recipe_verb",
    "extract_signatures",
    "maestro_run",
    "visual_review",
    "capture_screenshots",
    "expo_cli",
    "android_build",
    "eas_build",
    "eas_submit",
    "gen_mobile_ci",
    "fastlane",
]

# Actions that involve running arbitrary shell commands — these go through
# safety_guard.pre_action before the executor is invoked.
_SHELL_ACTIONS = {"run_cmd", "run_pytest"}


async def _safety_guard_check(task: dict) -> Action | None:
    """Return a blocked/waiting Action if safety_guard says no, else None (Allow)."""
    payload = task.get("payload") or {}
    action = payload.get("action")

    if action not in _SHELL_ACTIONS:
        return None  # non-shell actions skip the guard

    # Build a string command for the guard to inspect.
    cmd_raw = payload.get("cmd") or payload.get("command") or payload.get("shell") or ""
    cmd_is_argv = isinstance(cmd_raw, list)
    if cmd_is_argv:
        import shlex
        cmd_str = shlex.join(str(t) for t in cmd_raw)
    else:
        cmd_str = str(cmd_raw)

    if not cmd_str:
        return None  # nothing to inspect

    # Resolve workspace root.
    import os
    workspace_root = (
        payload.get("workspace_path")
        or os.environ.get("WORKSPACE_ROOT")
        or os.getcwd()
    )

    # For argv-style commands (list), `detect_shell_outside_workspace` is not
    # meaningful — the executable itself is a trusted binary, not a file target,
    # and run_cmd's own _resolve_cwd already prevents cwd escape.  Bypass that
    # specific check by widening workspace_root to the filesystem root.
    if cmd_is_argv:
        import pathlib
        workspace_root = str(pathlib.Path(workspace_root).anchor or workspace_root)

    # Best-effort current branch.
    current_branch = "unknown"
    try:
        import subprocess
        current_branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip() or "unknown"
    except Exception:
        pass

    # Load per-mission allowlist from missions.context.safety_allowlist.
    mission_allowlist: list[str] = []
    mission_id = task.get("mission_id")
    if mission_id is not None:
        try:
            from src.infra.db import get_db
            import json as _json
            db = await get_db()
            cur = await db.execute(
                "SELECT context FROM missions WHERE id = ?", (mission_id,)
            )
            row = await cur.fetchone()
            if row and row[0]:
                ctx = _json.loads(row[0])
                if isinstance(ctx, dict):
                    raw = ctx.get("safety_allowlist", [])
                    if isinstance(raw, list):
                        mission_allowlist = [str(p) for p in raw]
        except Exception:
            pass

    step = {
        "id": task.get("step_id") or task.get("title"),
        "reversibility": task.get("reversibility", "full"),
        "locked": task.get("locked", False),
    }
    sg_action = {"command": cmd_str}

    from safety_guard import pre_action, Allow, WaitForFounder, Block
    decision = pre_action(
        step,
        sg_action,
        workspace_root=workspace_root,
        current_branch=current_branch,
        founder_recently_active=True,  # TODO: wire real activity tracker
        mission_allowlist=mission_allowlist,
    )
    if isinstance(decision, Block):
        return Action(status="blocked", error=f"safety_guard blocked: {decision.reason}")
    if isinstance(decision, WaitForFounder):
        return Action(status="waiting_human", error=f"safety_guard waiting: {decision.reason}")
    return None  # Allow → proceed


def _resolve_path_list(paths):
    """Resolve verify_* `_paths` lists against WORKSPACE_DIR + flat fallback.

    Workflow JSON declares workspace-relative paths
    (`mission_<id>/.charter/foo.md`). Verifier helpers open with bare
    `open(p)` against CWD — those are project root, not the workspace —
    so the file is never found and the shape check reports "empty" /
    "missing sections" even when the artifact is on disk.

    Also handles the common drift where the agent wrote the file flat
    (`mission_<id>/<basename>`) despite the declared subdir
    (`mission_<id>/.charter/<basename>`). Cloud LLMs often skip the
    subdir; this helper picks up the flat fallback so the verifier
    doesn't false-fail when content exists.
    """
    if not paths:
        return paths
    from src.tools.workspace import WORKSPACE_DIR as _WSD
    import os.path as _osp
    out = []
    for p in paths:
        if not isinstance(p, str) or not p.strip():
            continue
        if _osp.isabs(p):
            out.append(p)
            continue
        candidate = _osp.join(_WSD, p)
        if _osp.isfile(candidate):
            out.append(candidate)
            continue
        # Flat fallback: agent wrote `mission_<id>/<basename>` instead of
        # `mission_<id>/<subdir>/<basename>`.
        norm = p.replace("\\", "/")
        parts = norm.split("/", 1)
        if (
            len(parts) == 2
            and parts[0].startswith("mission_")
            and "/" in parts[1]
        ):
            flat = _osp.join(_WSD, parts[0], _osp.basename(norm))
            if _osp.isfile(flat):
                out.append(flat)
                continue
        out.append(candidate)
    return out


async def _emit_bisect_lesson(
    *,
    mission_id,
    stack: str,
    bisect_result: dict,
    source_task_id=None,
) -> None:
    """Z3 R4 — write a rich mission_lessons row after integration_bisect.

    pattern  = "break at <sha8>: <failing_test>" (truncated)
    fix      = "suspect: <cluster_dir> (<count> files)\\n<diff_shortstat>"
    severity = "blocker"
    domain   = "integration_replay"
    source_ref records breaking_pair / cluster / shortstat / subject for
    later forensics.
    """
    from src.infra.mission_lessons import upsert_mission_lesson

    pair = bisect_result.get("breaking_pair") or []
    if len(pair) != 2:
        return
    commit_a, commit_b = pair[0], pair[1]

    failing_test = (bisect_result.get("failing_test") or "").strip()
    cluster = bisect_result.get("file_cluster") or []
    shortstat = (bisect_result.get("diff_shortstat") or "").strip()
    subject = (bisect_result.get("commit_subject") or "").strip()
    changed = bisect_result.get("changed_files") or []

    sha8 = (commit_b or "")[:8]
    if failing_test:
        # Strip the leading "FAILED " marker pytest emits.
        ft = failing_test.replace("FAILED", "").strip()[:80]
        pattern = f"break at {sha8}: {ft}"
    else:
        pattern = f"break at {sha8}: {subject[:80] or 'unknown'}"

    fix_parts: list[str] = []
    if cluster:
        top = cluster[0]
        fix_parts.append(
            f"suspect: {top.get('dir')} ({top.get('count')} file{'s' if top.get('count', 0) != 1 else ''})"
        )
    if shortstat:
        fix_parts.append(shortstat)
    fix = "\n".join(fix_parts)[:300]

    await upsert_mission_lesson(
        stack=stack or "unknown",
        domain="integration_replay",
        pattern=pattern[:120],
        fix=fix,
        severity="blocker",
        source_kind="bisect_break",
        source_ref={
            "mission_id": mission_id,
            "source_task_id": source_task_id,
            "breaking_pair": [commit_a, commit_b],
            "commit_subject": subject[:200],
            "file_cluster": cluster,
            "changed_files": changed[:20],
            "diff_shortstat": shortstat,
        },
    )


async def _surface_spec_patch_proposal(payload: dict, res: dict) -> None:
    """Z1 — surface a written spec-patch proposal back into Telegram.

    Side-effect of the ``propose_spec_patch_from_html_diff`` dispatch
    branch: enqueue a ``notify_user`` follow-up carrying the proposal body
    and Apply/Reject inline buttons so the founder can review→apply without
    leaving the chat.

    Callback tokens are kept short (``sp_apply:<mid>:<ts>`` /
    ``sp_rej:<mid>:<ts>``) — Telegram caps callback_data at 64 bytes and
    silently drops longer entries. The proposal path is deterministically
    reconstructable from mid+ts on the Apply side, so we never put a path
    in the token.

    Fail-soft: missing mid/ts → skip silently; any enqueue error is logged
    and swallowed so a notify failure never fails the proposer.
    """
    mid = payload.get("mission_id")
    ts = payload.get("ts")
    if mid is None or ts is None:
        return  # nothing to address the notify to — skip gracefully

    changes = res.get("changes") or []
    proposal_md = res.get("proposal_md") or ""
    # Telegram hard limit is 4096; leave headroom for the header + ellipsis.
    body = proposal_md[:3500]
    if len(proposal_md) > 3500:
        body += "\n…(truncated)"
    header = (
        f"📋 Spec patch proposal for mission #{mid} — "
        f"{len(changes)} change(s)\n\n"
    )
    message = header + body

    inline_buttons = []
    if changes:
        # Only offer Apply when there is something to apply; a lone Reject
        # button with nothing to reject is noise.
        inline_buttons = [
            {"label": "✅ Apply to spec",
             "callback_data": f"sp_apply:{mid}:{ts}"},
            {"label": "❌ Reject",
             "callback_data": f"sp_rej:{mid}:{ts}"},
        ]

    try:
        import general_beckman
        await general_beckman.enqueue({
            "title": f"notify_spec_patch:{mid}:{ts}",
            "description": f"Surface spec-patch proposal for mission #{mid}.",
            "agent_type": "mechanical",
            "kind": "main_work",
            "priority": 5,
            "mission_id": mid,
            "context": {
                "executor": "mechanical",
                "payload": {
                    "action": "notify_user",
                    "message": message,
                    "inline_buttons": inline_buttons,
                },
            },
        })
    except Exception as exc:  # pragma: no cover - logged, never fatal
        try:
            from src.infra.logging_config import get_logger as _gl
            _gl("mr_roboto.spec_patch").warning(
                "spec-patch notify enqueue failed", error=str(exc))
        except Exception:
            pass


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
    ``payload['require_confirmation']`` is True (or ``confirm_policy``
    auto-arms) AND the resolved reversibility is ``partial`` /
    ``irreversible``, the founder confirmation gate runs via
    :func:`_await_confirmation`. The gate parks the task through the
    existing clarification path (asks on Telegram, marks the task
    ``waiting_human``, returns ``needs_clarification``) and resumes when
    the founder's typed reply re-dispatches the task with
    ``context['user_clarification']`` set — no busy-poll, no worker-slot
    hold, no 60s hard-fail.
    """
    # Z0: safety guard pre-action check for shell-executing actions.
    guard_result = await _safety_guard_check(task)
    if guard_result is not None:
        return guard_result

    # payload can arrive as task["payload"] (orchestrator copy) or as
    # task["context"]["payload"] (raw expander shape before the copy step).
    payload = (task.get("payload")
               or (task.get("context") or {}).get("payload")
               or {})
    verb = payload.get("action") or ""
    override = payload.get("reversibility_override")
    # Z5 T3b — `fastlane` reversibility depends on the lane: build/match are
    # reversible, pilot/supply push a binary to a store track. The verb body
    # cannot influence the tag (it is resolved before dispatch), so derive
    # the override from the lane here and feed it through the standard
    # `reversibility_override` mechanism. An explicit caller override still
    # wins (it is read first, above).
    if verb == "fastlane" and override is None:
        from mr_roboto.fastlane_run import lane_reversibility

        override = lane_reversibility(payload.get("lane"))
    if override not in ("full", "partial", "irreversible", None):
        override = None
    resolved_reversibility = get_reversibility(str(verb), override=override)

    # Skeleton confirmation gate. Default off; only the explicit caller
    # flag arms it for T1C. T2B will wire auto-arm + Telegram surface.
    require_confirmation = bool(payload.get("require_confirmation", False))

    # Z10 P2 (2026-05-18 sweep) — auto-arm based on confirm_policy when the
    # caller didn't already set require_confirmation. Policy resolution
    # order: task context override → env var KUTAI_CONFIRM_POLICY → 'off'.
    #
    # Policy values:
    #   'off'                 — never auto-arm (current default; preserves
    #                           pre-fix behaviour for any caller that
    #                           hasn't opted in).
    #   'irreversible_only'   — auto-arm when reversibility == 'irreversible'.
    #   'partial_or_worse'    — auto-arm when reversibility in
    #                           ('partial', 'irreversible').
    if not require_confirmation:
        _ctx = task.get("context") or {}
        _ctx_policy = _ctx.get("confirm_policy") if isinstance(_ctx, dict) else None
        try:
            import os as _os
            _env_policy = (_os.environ.get("KUTAI_CONFIRM_POLICY") or "").strip().lower()
        except Exception:
            _env_policy = ""
        _policy = (_ctx_policy or _env_policy or "off").lower()
        if _policy == "irreversible_only" and resolved_reversibility == "irreversible":
            require_confirmation = True
        elif _policy == "partial_or_worse" and resolved_reversibility in (
            "partial", "irreversible",
        ):
            require_confirmation = True

    if require_confirmation and resolved_reversibility in (
        "partial", "irreversible"
    ):
        gate_action = await _await_confirmation(
            task=task,
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

    # Z7 B9 — external-comms audit trail. When an external-publish verb
    # completes successfully, land an immutable external_comms_log row
    # (content hash + gzip body + channel/recipient/mission). Best-effort:
    # log_publish_action never raises into the dispatch path.
    await _log_external_publish(str(verb), action_obj, task)

    return action_obj


async def _log_external_publish(verb: str, action_obj: Action, task: dict) -> None:
    """Best-effort B9 audit-log writer for external-publish verbs.

    Delegates to mr_roboto.audit_log.log_publish_action, which no-ops for
    non-publish verbs and for verbs that did not actually deliver content.
    Never raises into the dispatch path.
    """
    try:
        from mr_roboto.audit_log import log_publish_action
        await log_publish_action(verb, action_obj, task)
    except Exception:
        import logging
        logging.getLogger("mr_roboto.audit_log").debug(
            "log_publish_action failed", exc_info=True
        )


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


# Tokens the founder can type to approve / reject a gated action. Exact
# match against the lower-cased, stripped reply. Anything else is treated
# as ambiguous → fail-closed (an irreversible action must never proceed on
# an unclear answer).
_CONFIRM_APPROVE_TOKENS = frozenset({
    "yes", "y", "evet", "e", "onay", "onayla", "onaylıyorum",
    "approve", "approved", "ok", "tamam", "✅",
})
_CONFIRM_REJECT_TOKENS = frozenset({
    "no", "n", "hayır", "hayir", "h", "reddet", "iptal",
    "reject", "rejected", "❌",
})


async def _await_confirmation(
    *,
    task: dict,
    verb: str,
    reversibility: str,
    payload: dict,
) -> Action | None:
    """Founder confirmation gate via the clarification park/resume path.

    This is NOT a busy-poll. The gate reuses the existing clarification
    machinery — the same mechanism ``mr_roboto.clarify`` uses — so a slow
    founder never holds a worker slot or hard-fails on a timeout:

    * **First entry** (``context["user_clarification"]`` absent): send a
      Telegram question, mark the task ``waiting_human``, and return
      ``Action(status="needs_clarification")``. The orchestrator
      special-cases that status for mechanical tasks and skips
      ``on_task_finished`` (orchestrator.py:316-446), leaving the row
      parked. The founder's typed reply →
      ``telegram_bot._resume_with_clarification`` injects
      ``context["user_clarification"]`` and resets the task to ``pending``
      → Beckman re-dispatches → ``mr_roboto.run`` re-runs → this gate runs
      again, now on the resume path.
    * **Resume** (``user_clarification`` present): approve tokens → return
      ``None`` (caller proceeds with dispatch); reject tokens → return
      ``Action(status="rejected")``; anything else (ambiguous) →
      fail-closed ``Action(status="rejected")`` — an irreversible action
      must never proceed on an answer we cannot parse.

    A human gate that cannot reach the human is never silently skipped:
    if Telegram is unavailable (or the park fails), we return
    ``Action(status="failed")`` so the mission halts at the gate.

    NOTE: this gate no longer uses ``action_confirmations``
    (request_confirmation / check_confirmation / resolve_confirmation) —
    it parks via the clarify path instead. That table is NOT orphaned,
    though: ``src/infra/cost_wiring.py`` still opens rows for the
    cost-decision gate, ``mr_roboto.audit_log.pending_audit_gaps`` joins it
    for the Z7 B9 external-comms audit trail, and the seeded drain cron
    surfaces pending rows. Leave it intact.

    Returns:
        ``None`` — founder approved, caller proceeds with dispatch.
        ``Action(status="needs_clarification")`` — first entry, parked.
        ``Action(status="rejected")`` — founder rejected or answer ambiguous.
        ``Action(status="failed")`` — gate could not reach the founder.
    """
    context = task.get("context") or {}
    clar = context.get("user_clarification") if isinstance(context, dict) else None

    # ── Resume path: a founder reply was injected into the context. ──
    if clar:
        ans = str(clar).strip().lower()
        if ans in _CONFIRM_APPROVE_TOKENS:
            return None
        if ans in _CONFIRM_REJECT_TOKENS:
            return Action(
                status="rejected",
                error="action rejected by founder confirmation",
                result={"verb": verb},
            )
        # Ambiguous — fail closed. Irreversible/partial actions must never
        # proceed on an answer we cannot interpret.
        return Action(
            status="rejected",
            error=(
                f"confirmation answer not understood ({clar!r}); "
                f"fail-closed for {reversibility} action"
            ),
            result={"verb": verb},
        )

    # ── First entry: ask the founder, park the task, return parked. ──
    task_id = task.get("id")
    title = task.get("title", "")
    question = (
        f"⚠️ Onay gerekiyor — `{verb}` ({reversibility}). "
        f"Bu işlem geri alınamaz / kısmen geri alınabilir. "
        f"Çalıştırılsın mı? *Evet* / *Hayır* ile yanıtla.\n"
        f"(Confirmation required for `{verb}` — reply *Evet*/*Yes* to run, "
        f"*Hayır*/*No* to cancel.)"
    )

    try:
        from src.app.telegram_bot import get_telegram
        tg = get_telegram()
    except Exception as e:
        tg = None
        _tg_err = e
    else:
        _tg_err = None

    if tg is None:
        # A human gate that cannot reach the human must NOT be skippable.
        suffix = f" ({_tg_err})" if _tg_err else ""
        return Action(
            status="failed",
            error=(
                f"confirmation gate cannot reach founder (telegram "
                f"unavailable) for {verb}; human approval cannot be "
                f"skipped{suffix}"
            ),
            result={"verb": verb},
        )

    try:
        await tg.request_clarification(int(task_id), title, question)
        from src.infra.db import update_task
        await update_task(int(task_id), status="waiting_human")
    except Exception as e:
        # Park failed — fail closed rather than silently proceeding.
        return Action(
            status="failed",
            error=(
                f"confirmation gate failed to park task for {verb}: {e}"
            ),
            result={"verb": verb},
        )

    return Action(
        status="needs_clarification",
        result={
            "awaiting_confirmation": True,
            "verb": verb,
            "reversibility": reversibility,
            "question": question,
        },
    )



async def _run_dispatch(task: dict) -> Action:
    payload = (task.get("payload")
               or (task.get("context") or {}).get("payload")
               or {})
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

        gate_enabled = (
            (not _critic_opt_out())
            and bool(payload.get("critic_gate", True))
        )
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
                # Pass paths RAW — verify_artifacts resolves each under its
                # own `workspace_path` via `_resolve_under`, which *rejects*
                # absolute paths as a traversal guard. `_resolve_path_list`
                # always returns absolutes, so routing through it made every
                # verify_artifacts call fail "path rejected". It also can't
                # express the any_of / glob path entries this verb accepts.
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
                charter_paths=_resolve_path_list(payload.get("charter_paths")),
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
                pitch_paths=_resolve_path_list(payload.get("pitch_paths")),
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
                non_goals_paths=_resolve_path_list(payload.get("non_goals_paths")),
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
                non_goals_paths=_resolve_path_list(payload.get("non_goals_paths")),
                target_text=payload.get("target_text"),
                target_paths=_resolve_path_list(payload.get("target_paths")),
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
                adr_paths=_resolve_path_list(payload.get("adr_paths")),
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
                adr_paths=_resolve_path_list(payload.get("adr_paths")),
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
                positioning_paths=_resolve_path_list(payload.get("positioning_paths")),
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
                script_paths=_resolve_path_list(payload.get("script_paths")),
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
            # Z1 Tier 4B — fire artifact-emit notification with inline buttons
            # so the founder can iterate (re-regen with another change) or
            # propagate the change downstream. Best-effort; failure here must
            # not flip the regen action to failed.
            try:
                from mr_roboto._artifact_notify import enqueue_artifact_emit_notice
                await enqueue_artifact_emit_notice(
                    mission_id=int(mid) if mid is not None else 0,
                    artifact_path=str(payload.get("artifact_path") or ""),
                    change_description=str(change),
                    new_version=str(res.get("new_version") or ""),
                )
            except Exception as _notify_exc:
                logger = __import__(
                    "logging"
                ).getLogger("mr_roboto.regen_artifact")
                logger.warning("artifact-emit notice enqueue failed: %s", _notify_exc)
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
                html_paths=_resolve_path_list(payload.get("html_paths")),
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
            # Side-effect: surface the proposal to Telegram with Apply/Reject
            # buttons. Never let a notify failure fail the proposer.
            await _surface_spec_patch_proposal(payload, res)
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

    if action == "publish_preview_pages":
        # Unit B — push preview root to GitHub Pages (gh-pages branch force-push).
        # Fail-soft: any failure → pending, never DLQ.
        from mr_roboto.publish_preview_pages import publish_preview_pages as _pub
        try:
            res = await _pub(
                mission_id=task.get("mission_id"),
                workspace_path=payload.get("workspace_path"),
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

    if action == "credential_rotation_reminder":
        # Z6 T7A — weekly scan of the credentials table; emit one
        # founder_action(kind='credential_paste') per service due for rotation.
        from mr_roboto.executors.credential_rotation_reminder import (
            credential_rotation_reminder as _rot_reminder,
        )
        try:
            res = await _rot_reminder()
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"credential_rotation_reminder: "
                        f"{res.get('error', 'failed')}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "compliance_template_staleness":
        # Z6 T4D — weekly scan of compliance template .meta.json files; emit
        # founder_action(kind='legal_counsel') per stale entry. Idempotent.
        from mr_roboto.executors.compliance_template_staleness import (
            compliance_template_staleness as _staleness,
        )
        try:
            res = await _staleness(
                template_root=payload.get("template_root"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"compliance_template_staleness: {res.get('error', 'failed')}"
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
            # A clarify gate that returned `failed` (e.g. could not reach
            # the founder) must surface as failed — completing it would
            # silently skip a human approval gate.
            if isinstance(res, dict) and res.get("status") == "failed":
                return Action(
                    status="failed",
                    error=str(res.get("error") or "clarify failed"),
                    result=res,
                )
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
            (not _critic_opt_out())
            and bool(payload.get("critic_gate", True))
        )
        if gate_enabled:
            text = payload.get("message") or payload.get("text") or ""
            try:
                # Gate the MESSAGE CONTENT only. The critic judges (a) spec
                # break, (b) founder fury, (c) secret/PII leak — all properties
                # of the text. chat_id (the recipient) is irrelevant to those,
                # and a null chat_id is the NORMAL case (notify_user defaults it
                # to the admin chat). Passing it in only baited the critic into
                # spurious "chat_id is null → will fail" validity-vetoes that
                # DLQ'd valid notifications (task #261969, 2026-06-02).
                gate_result = await _critic_gate(
                    "notify_user",
                    {"message": text},
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

    if action == "yalayut_recipe":
        # Phase 3 — preempt lane: run a yalayut shell_recipe mechanically.
        from mr_roboto.executors.yalayut_recipe import run as _yalayut_recipe
        try:
            res = await _yalayut_recipe(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"yalayut_recipe: {res.get('reason') or 'recipe failed'}",
                    result=res,
                )
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

    if action == "emit_dlq_lessons":
        # Z2 T4B + Item-1 followup — daily DLQ→mission_lessons emitter.
        from src.infra.mission_lessons import emit_lessons_from_dlq_patterns
        try:
            count = await emit_lessons_from_dlq_patterns()
            return Action(
                status="completed",
                result={"ok": True, "lessons_count": int(count)},
            )
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "mine_dlq_patterns":
        # Z9 Growth T3D — weekly DLQ feedback hook. Mines recurring failure
        # patterns from the dead-letter queue and writes dlq_pattern
        # growth_events that surface in the analytics digest. Idempotent.
        from src.infra.dlq_feedback import mine_dlq_patterns
        try:
            count = await mine_dlq_patterns()
            return Action(
                status="completed",
                result={"ok": True, "patterns_count": int(count)},
            )
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

    if action == "classify_signals":
        # Z9 T3B — fetch unclassified raw_signal growth_events + enqueue the
        # signal_classifier agent via Beckman. Mechanical orchestration only;
        # NO direct LLM call.
        from mr_roboto.executors.classify_signals import run as _classify_run
        try:
            res = await _classify_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"classify_signals: {res.get('error') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "score_backlog":
        # Z9 T3C — deterministic scorer over classified_signal rows; writes
        # top-N backlog_candidate rows. Pure math, no LLM.
        from mr_roboto.executors.score_backlog import run as _score_run
        try:
            res = await _score_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"score_backlog: {res.get('error') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "score_sunset":
        # Z9 T5C — deterministic feature-lifecycle scorer. Computes per-feature
        # usage from growth_events + recipe_pin_log; writes sunset_candidate
        # rows for low-usage non-zero-cost features. Pure math, no LLM.
        from mr_roboto.executors.score_sunset import run as _sunset_run
        try:
            res = await _sunset_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"score_sunset: {res.get('error') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "roadmap_sync":
        # Z9 T5C — north-star sync. Reads the success_metrics artifact and
        # checks the declared north-star against recent reality; writes a
        # northstar_review row when it is undefined / untracked / flat.
        from mr_roboto.executors.roadmap_sync import run as _roadmap_run
        try:
            res = await _roadmap_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"roadmap_sync: {res.get('error') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "record_hypothesis":
        # Z9 T4A — capture the mission's predicted metric impact as a
        # pending hypotheses row at Phase 7 spec finalization. Pure
        # deterministic spec parsing, NO LLM.
        from mr_roboto.executors.record_hypothesis import run as _rec_hyp
        try:
            res = await _rec_hyp(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"record_hypothesis: {res.get('error') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "inject_north_star":
        # Z9 T4B — read the success_metrics artifact (i2p step 2.9) and
        # merge north_star_metric + aarrr_metrics into mission.context so
        # Phase 8+ feature-scoring steps see the north-star. Mechanical.
        from mr_roboto.executors.inject_north_star import run as _inj_ns
        try:
            res = await _inj_ns(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"inject_north_star: {res.get('error') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "validate_target_segment":
        # Z9 T5A — cohort-awareness nag: WARN (never block) when a Phase 8+
        # mission reaches the implementation backlog with no explicit
        # mission.context['target_segment']. Back-fills the 'any' default.
        from mr_roboto.executors.validate_target_segment import (
            run as _val_seg,
        )
        try:
            res = await _val_seg(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        f"validate_target_segment: "
                        f"{res.get('error') or 'failed'}"
                    ),
                    result=res,
                )
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

    if action == "stripe_scaffold":
        # Z6 T5A — emit Stripe checkout/webhook scaffolds under mission_<id>/api/.
        from mr_roboto.executors.stripe_scaffold import run as _stripe_scaffold_run
        try:
            res = await _stripe_scaffold_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"stripe_scaffold: {res.get('reason') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "stripe_provision_products":
        # Z6 T5B — provision Stripe products+prices from monetization_strategy.
        from mr_roboto.executors.stripe_provision_products import (
            run as _stripe_provision_run,
        )
        try:
            res = await _stripe_provision_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"stripe_provision_products: {res.get('reason') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "stripe_payment_flow_test":
        # Z6 T5C — exercise Stripe sandbox via vendor_call.
        from mr_roboto.executors.stripe_payment_flow_test import (
            run as _stripe_payment_run,
        )
        try:
            res = await _stripe_payment_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"stripe_payment_flow_test: {res.get('reason') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "stripe_dispute_check":
        # Z6 T5D — weekly Stripe dispute scan.
        from mr_roboto.executors.stripe_dispute_check import (
            run as _stripe_dispute_run,
        )
        try:
            res = await _stripe_dispute_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"stripe_dispute_check: {res.get('reason') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "stripe_revenue_digest":
        # Z6 T5D — weekly Stripe revenue digest.
        from mr_roboto.executors.stripe_revenue_digest import (
            run as _stripe_revenue_run,
        )
        try:
            res = await _stripe_revenue_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"stripe_revenue_digest: {res.get('reason') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "tax_export_ledger":
        # Z6 T5E — monthly Stripe Tax CSV export.
        from mr_roboto.executors.tax_export_ledger import (
            run as _tax_export_run,
        )
        try:
            res = await _tax_export_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"tax_export_ledger: {res.get('reason') or 'failed'}",
                    result=res,
                )
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
                plan_paths=_resolve_path_list(payload.get("plan_paths")),
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
                html_paths=_resolve_path_list(payload.get("html_paths")),
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
                screen_plan_paths=_resolve_path_list(payload.get("screen_plan_paths")),
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
                _detail = res.get("error")
                return Action(
                    status="failed",
                    error=(
                        f"compliance_template_present: "
                        + (f"{_detail} " if _detail else "")
                        + f"missing={res.get('missing')} "
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

    if action == "z0_preflight_write":
        # Z0 minimal slice — write mission preflight JSON + set
        # ambition_tier / cost_ceiling_usd / attention budget on
        # missions row. Drives downstream Z1 gates.
        from mr_roboto.z0_preflight import z0_preflight_write as _z0_write
        try:
            mid = task.get("mission_id") or payload.get("mission_id")
            res = await _z0_write(
                mission_id=int(mid) if mid is not None else 0,
                ambition_tier=payload.get("ambition_tier"),
                cost_ceiling_usd=payload.get("cost_ceiling_usd"),
                attention_budget_minutes=payload.get(
                    "attention_budget_minutes"
                ),
                jurisdictions=payload.get("jurisdictions"),
                user_classes=payload.get("user_classes"),
                workspace_path=payload.get("workspace_path"),
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=str(res.get("error") or "z0_preflight_write failed"),
                    result=res,
                )
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
                # Z1 T6A — surface inline Continue / Branch / Abort buttons
                # in Telegram. Best-effort: notification failure must not
                # mask the needs_review verdict.
                try:
                    from mr_roboto._similar_review_notify import (
                        enqueue_similar_review_notice,
                    )
                    await enqueue_similar_review_notice(
                        mission_id=int(mid) if mid is not None else 0,
                        matches=res.get("matches") or [],
                        report_path=res.get("report_path"),
                    )
                except Exception as _notice_exc:
                    import logging as _logging
                    _logging.getLogger("mr_roboto.find_similar").warning(
                        "similar-review notify failed: %s", _notice_exc,
                    )
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
                candidates_path=payload.get("candidates_path"),
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

    if action == "prior_art_fetch":
        from mr_roboto.prior_art_fetch import prior_art_fetch as _paf
        try:
            res = await _paf(
                queries_path=payload.get("queries_path"),
                candidates_path=payload.get("candidates_path"),
            )
            if not res.get("ok"):
                return Action(status="failed", error=res.get("error"), result=res)
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

    if action == "paraflow_audit_all":
        # Z1 Tier 7B (C21) — workspace-wide weekly audit. Iterates
        # missions with a `.paraflow_archetype` marker; per-mission
        # verify call persists to paraflow_diff_log via the standard
        # path. Returns summary counts for cron telemetry.
        from mr_roboto.verify_against_paraflow_goldens import (
            paraflow_audit_all as _audit_all,
        )
        try:
            res = await _audit_all(
                workspace_root=payload.get("workspace_root"),
            )
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
            # Z10 wire-fix F3: do NOT hardcode tests/e2e/golden_path.spec.ts.
            # When payload omits scenario_path, record_demo resolves via
            # missions.demo_scenario_path → newest tests/e2e/*.spec.[tj]s →
            # no_e2e_specs skip path.
            scenario = payload.get("scenario_path")
            max_s = int(payload.get("max_seconds") or 90)
            res = await _record_demo(
                mission_id=int(mid),
                scenario_path=(str(scenario) if scenario else None),
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

    if action == "instantiate_picked_recipes":
        # Z2 Item-2 followup — consume recipe_picks.json, instantiate
        # each non-null pick, emit recipe_instantiations.json manifest.
        from mr_roboto.instantiate_picked_recipes import (
            instantiate_picked_recipes as _inst_picks,
        )
        try:
            res = await _inst_picks(
                mission_id=task.get("mission_id"),
                recipe_picks_path=str(payload.get("recipe_picks_path") or "mission/recipe_picks.json"),
                manifest_path=str(payload.get("manifest_path") or "mission/recipe_instantiations.json"),
                recipes_dir=str(payload.get("recipes_dir") or "recipes"),
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "extract_signatures":
        # Z3 T2C — AST-based signature extraction + cross-file mismatch detection.
        # Used as a mechanical pre-check inside the integration_review post-hook.
        from mr_roboto.extract_signatures import extract_signatures as _extract_sigs
        try:
            res = await _extract_sigs(
                target_files=list(payload.get("target_files") or []),
                workspace_path=payload.get("workspace_path") or None,
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "run_bandit":
        from mr_roboto.run_bandit import run_bandit as _run_bandit
        try:
            res = await _run_bandit(
                target_files=list(payload.get("target_files") or []),
                workspace_path=payload.get("workspace_path") or None,
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "run_npm_audit":
        from mr_roboto.run_npm_audit import run_npm_audit as _run_npm_audit
        try:
            res = await _run_npm_audit(workspace_path=payload.get("workspace_path") or "")
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "security_review":
        # Z3 T3A — composite: semgrep + bandit + npm audit.
        from mr_roboto.security_review import security_review as _sec_review
        try:
            res = await _sec_review(
                target_files=list(payload.get("target_files") or []),
                workspace_path=payload.get("workspace_path") or None,
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "run_axe":
        # Z3 T3B — accessibility scan via @axe-core/cli against preview URL.
        from mr_roboto.run_axe import run_axe as _run_axe
        try:
            res = await _run_axe(
                preview_url=str(payload.get("preview_url") or ""),
                target_paths=list(payload.get("target_paths") or []),
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "run_schemathesis":
        # Z3 T3C — contract fuzz via schemathesis.
        from mr_roboto.run_schemathesis import run_schemathesis as _run_st
        try:
            res = await _run_st(
                spec_path=str(payload.get("spec_path") or ""),
                base_url=str(payload.get("base_url") or ""),
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "performance_review":
        # Z3 T3C — composite: lighthouse (mode=web) or k6 (mode=api).
        from mr_roboto.performance_review import performance_review as _perf
        try:
            res = await _perf(**payload)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "check_adr_drift":
        # Z3 T4B — mechanical ADR drift gate.
        from mr_roboto.check_adr_drift import check_adr_drift as _check_drift
        try:
            res = await _check_drift(
                adr_register_path=str(payload.get("adr_register_path") or ""),
                produced_files=list(payload.get("produced_files") or []),
                workspace_path=payload.get("workspace_path") or None,
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "integration_replay":
        # Z3 T5 — re-run test suite against commits in shuffle; soft-skip on no git/tests.
        from mr_roboto.integration_replay import integration_replay as _replay
        try:
            res = await _replay(
                commits=list(payload.get("commits") or []),
                suite_glob=str(payload.get("suite_glob") or "tests/integration/**"),
                shuffle_seed=int(payload.get("shuffle_seed") or 0),
                mode=str(payload.get("mode") or "standard"),
                workspace_path=payload.get("workspace_path") or None,
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "integration_bisect":
        # Z3 T5 — binary-search breaking commit pair after integration_replay fail.
        from mr_roboto.integration_bisect import integration_bisect as _bisect
        try:
            res = await _bisect(
                commits=list(payload.get("commits") or []),
                suite_glob=str(payload.get("suite_glob") or "tests/integration/**"),
                workspace_path=str(payload.get("workspace_path") or ""),
            )
            # Z3 R4 — emit mission_lessons row when a breaking_pair is found.
            # Best-effort: never let lesson-emit failure cascade.
            try:
                if res.get("breaking_pair") and payload.get("mission_id"):
                    await _emit_bisect_lesson(
                        mission_id=payload.get("mission_id"),
                        stack=str(payload.get("stack") or "unknown"),
                        bisect_result=res,
                        source_task_id=payload.get("source_task_id"),
                    )
            except Exception as _lesson_exc:
                from src.infra.logging_config import get_logger as _gl
                _gl("mr_roboto.integration_bisect").debug(
                    "lesson emit skipped: %s", _lesson_exc
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "run_semgrep_layer_filtered":
        # Z3 T4C — semgrep with layer-filter pre-pass (e.g. forbidden_in_domain).
        from mr_roboto.run_semgrep_layer_filtered import run_semgrep_layer_filtered as _sgflt
        try:
            res = await _sgflt(
                mission_id=payload.get("mission_id"),
                target_files=list(payload.get("target_files") or []),
                rule_pack_path=str(payload.get("rule_pack_path") or ""),
                required_layer=str(payload.get("required_layer") or "domain"),
                workspace_path=payload.get("workspace_path") or None,
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "alert_triage":
        # Z8 T3D — webhook → severity classification (rule-based + LLM stub).
        from mr_roboto.executors.alert_triage import run as _alert_triage_run
        try:
            res = await _alert_triage_run(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "synthetic_check":
        # Z8 T5F (2026-05-18 sweep) — Lighthouse / k6 synthetic check with
        # perf_baselines regression diff. The executor was complete but
        # had no dispatch branch — every triggered task hit the
        # unknown-action path. Doubly dead pre-fix because the Z8 ops
        # recipes (which schedule this verb) were also undiscoverable; the
        # recipe-layout restructure in this same merge re-enables them.
        from mr_roboto.executors.synthetic_check import run as _synthetic_run
        try:
            res = await _synthetic_run(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "oncall_action":
        # Z8 T4B — on-call verb gateway: whitelist + cooldown check, then
        # delegate to the verb sub-handler. See executors/oncall_action.py.
        from mr_roboto.executors.oncall_action import run as _oncall_run
        try:
            res = await _oncall_run(task)
            # Cooldown blocks and whitelist refusals are completed actions
            # from the dispatcher's view — they carry their own status field
            # for downstream interpretation (the agent must re-evaluate).
            # not_implemented verbs are propagated as failed so the workflow
            # engine and on-call agent see a genuine failure and escalate.
            if res.get("status") == "not_implemented":
                return Action(status="failed", error=res.get("error", "not_implemented"), result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "generate_playbooks":
        # Z8 T4C — phase 13 playbook generator: matches incident_playbook
        # recipes against the mission's tech_stack and emits the
        # incident_playbooks artifact.
        from mr_roboto.executors.generate_playbooks import run as _gen_pb_run
        try:
            res = await _gen_pb_run(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action in ("backup_verify", "cron_backup_verify"):
        # Z8 T5A — sqlite copy + smoke SELECT, or postgres pg_restore.
        from mr_roboto.executors.backup_verify import run as _bv_run
        try:
            res = await _bv_run(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action in ("dependency_scan", "cron_dep_hygiene"):
        # Z8 T5B — pip-audit / npm audit subprocess wrapper.
        from mr_roboto.executors.dependency_scan import run as _dep_run
        try:
            res = await _dep_run(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action in ("cve_scan", "cron_cve_scan"):
        # Z8 T5C — OSV.dev API CVE lookup.
        from mr_roboto.executors.cve_scan import run as _cve_run
        try:
            res = await _cve_run(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action in ("secret_scan", "cron_secret_scan"):
        # Z8 T5C — gitleaks subprocess wrapper.
        from mr_roboto.executors.secret_scan import run as _sec_run
        try:
            res = await _sec_run(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action in ("cost_pull", "cron_cost_pull"):
        # Z8 T5D — vendor cost pull via vendor_call + aggregation.
        from mr_roboto.executors.cost_pull import run as _cost_run
        try:
            res = await _cost_run(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "analytics_digest":
        # Z9 T2B — weekly analytics data-pull. Mechanical (no LLM): pulls
        # PostHog analytics + DB aggregates, then enqueues the
        # growth_digest_synthesizer agent via Beckman for LLM synthesis.
        from mr_roboto.executors.analytics_digest import run as _analytics_run
        try:
            res = await _analytics_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"analytics_digest: {res.get('reason') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "arm_analytics_digest":
        # Z9 T2B — Phase 14 launch step: arm the weekly analytics_digest
        # cron for the mission (idempotent — durable cursor entry + live arm).
        from mr_roboto.executors.arm_analytics_digest import (
            run as _arm_digest_run,
        )
        try:
            res = await _arm_digest_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"arm_analytics_digest: {res.get('reason') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "verdict_window_sweep":
        # Z9 T4C — daily sweep of pending hypotheses; enqueues a
        # record_verdict mechanical task for each closed measurement
        # window. Mechanical orchestration only — NO direct LLM call.
        from mr_roboto.executors.verdict_window_sweep import (
            run as _verdict_sweep_run,
        )
        try:
            res = await _verdict_sweep_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"verdict_window_sweep: {res.get('reason') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "record_verdict":
        # Z9 T4D — pull the actual metric, compute a Bayesian verdict,
        # persist it, mirror refuted/inconclusive to mission_lessons, and
        # fire the reinforce loop on confirmed. Deterministic — no LLM.
        # Z9 T5D also runs A/B result evaluation here when the mission had
        # experiment_variants rows.
        from mr_roboto.executors.record_verdict import run as _record_verdict_run
        try:
            res = await _record_verdict_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"record_verdict: {res.get('reason') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "assign_variant":
        # Z9 T5D/T5E — split a Phase-8+ feature mission into control +
        # treatment variants, wire a PostHog feature-flag, honour the
        # insufficient-N guard (<100 DAU → 100% rollout). Deterministic —
        # no LLM. A/B winner math reuses src/growth/verdict_stats.py.
        from mr_roboto.executors.assign_variant import run as _assign_variant_run
        try:
            res = await _assign_variant_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"assign_variant: {res.get('reason') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "retire_variant":
        # Z9 T5D/T5E — mark winner/loser variants and flip the PostHog
        # flag (winner → 100%, loser → 0%). Founder-gated: invoked only by
        # /experiment_ship and /experiment_rollback, never auto-fired.
        from mr_roboto.executors.retire_variant import run as _retire_variant_run
        try:
            res = await _retire_variant_run(task)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"retire_variant: {res.get('reason') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "audit_completeness_check":
        # Z7 B9 — hourly completeness check (invoked as a standalone
        # mechanical action by the audit cron, not as a per-step posthook).
        # Finds vendor_call rows with reversibility != 'full' that have no
        # external_comms_log row within the window; raises an ops alert per gap.
        #
        # Z6 P2 (2026-05-18 sweep) — when this verb arrives via the
        # POST-HOOK path (the apply layer always sets source_task_id),
        # delegate to general_beckman.posthook_handlers.audit_completeness_check
        # instead of the cron scanner. The handler scopes its check to the
        # source task's emitted vendor_call rows and returns the
        # standard {status: ok|fail, ...} verdict shape that the apply
        # layer's posthook_verdict path expects. Without this branch the
        # post-hook always ran the cron scanner, which never persisted a
        # verdict for the source task.
        if payload.get("source_task_id"):
            try:
                import importlib
                _mod = importlib.import_module(
                    "general_beckman.posthook_handlers.audit_completeness_check"
                )
                from src.infra.db import get_task as _get_task
                _src_id = int(payload.get("source_task_id"))
                _src = await _get_task(_src_id)
                _src_task = dict(_src) if _src else {}
                _res = await _mod.handle(_src_task, {})
                _status = (_res or {}).get("status", "ok")
                return Action(
                    status=("completed" if _status == "ok" else "failed"),
                    result=_res or {},
                )
            except Exception as e:
                return Action(status="failed", error=str(e))
        from mr_roboto.audit_log import pending_audit_gaps
        try:
            window_minutes = int(payload.get("window_minutes", 5))
            gaps = await pending_audit_gaps(window_minutes=window_minutes)
            if not gaps:
                return Action(status="completed", result={"gaps_found": 0, "ok": True})
            from mr_roboto.executors.escalate_to_founder import run as _ef_run
            alerts_sent = 0
            for gap in gaps:
                try:
                    await _ef_run({
                        "mission_id": gap.get("mission_id"),
                        "payload": {
                            "severity": "high",
                            "title": "Audit gap: external send not logged",
                            "summary": (
                                f"vendor_call id={gap.get('vendor_call_id')} "
                                f"verb={gap.get('verb')!r} at {gap.get('created_at')} "
                                f"has no external_comms_log row after {window_minutes}min. "
                                "Manual audit required."
                            ),
                        },
                    })
                    alerts_sent += 1
                except Exception as _ae:
                    from src.infra.logging_config import get_logger as _gl
                    _gl("mr_roboto.audit_completeness_check").error(
                        "audit_completeness_check: alert failed for gap %s: %s",
                        gap.get("vendor_call_id"), _ae,
                    )
            return Action(
                status="completed",
                result={"gaps_found": len(gaps), "alerts_sent": alerts_sent, "ok": True},
            )
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action in (
        "copy_compliance_review",
        "brand_voice_lint",
        "briefing_compose",
    ):
        # Z7 T1.0 — humanish-layers posthook handlers.
        # Each handler lives in general_beckman.posthook_handlers.<action>.
        # The handler receives the full task dict and an empty result dict,
        # and returns {status: ok|fail, ...}.
        try:
            import importlib
            _mod = importlib.import_module(
                f"general_beckman.posthook_handlers.{action}"
            )
            source_task_id = payload.get("source_task_id")
            source_task: dict = {}
            if source_task_id:
                try:
                    from src.infra.db import get_task as _get_task
                    _src = await _get_task(int(source_task_id))
                    if _src:
                        source_task = dict(_src)
                except Exception:
                    pass
            # Merge payload fields into source task context so handler can
            # read jurisdiction/channel/workspace_path/etc.
            import json as _json
            src_ctx: dict = {}
            raw_ctx = source_task.get("context") or {}
            if isinstance(raw_ctx, str):
                try:
                    src_ctx = _json.loads(raw_ctx)
                except Exception:
                    src_ctx = {}
            elif isinstance(raw_ctx, dict):
                src_ctx = dict(raw_ctx)
            for _k in (
                "workspace_path", "jurisdiction", "channel",
                "artifact_metadata", "copy_path", "privacy_policy_path", "produces",
            ):
                if payload.get(_k):
                    src_ctx.setdefault(_k, payload[_k])
            source_task["context"] = src_ctx

            res = await _mod.handle(source_task, {})
            if res.get("status") == "fail":
                return Action(status="failed", error=str(res), result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "daily_briefing":
        # Z7 A0 — daily founder briefing: aggregates in-flight missions,
        # pending founder_actions, cost burn; writes mission_briefings kind=daily.
        # Idempotent: no-ops if today's row already exists.
        from src.app.jobs.daily_briefing import run_daily_briefing
        try:
            res = await run_daily_briefing()
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"daily_briefing: {res.get('reason') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "follow_up_reminder":
        # Z7 T4 A10 — daily CRM follow-up reminder: scans interactions where
        # follow_up_at <= today+7 AND done=0; builds digest; notifies founder.
        from src.app.jobs.follow_up_reminder import run_follow_up_reminder
        try:
            res = await run_follow_up_reminder()
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"follow_up_reminder: {res.get('reason') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 T4 A8: FAQ flywheel + quote harvest executors ────────────────────────

    if action == "faq_regen":
        # Z7 T4 A8 — weekly FAQ regen (A8 + A8.r1 multilingual).
        # Pulls last 7d of low-confidence/escalated tickets, groups by language,
        # LLM-clusters, drafts FAQ entries for clusters > 3, emits founder_actions.
        from src.app.jobs.faq_regen import run_faq_regen
        try:
            res = await run_faq_regen()
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"faq_regen: {res.get('reason') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "quote_harvest":
        # Z7 T4 A8 — monthly quote harvest.
        # Scans positive tickets; emits consent founder_action per candidate.
        from src.app.jobs.quote_harvest import run_quote_harvest
        try:
            res = await run_quote_harvest()
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"quote_harvest: {res.get('reason') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "investor_bullets":
        # Z7 T5 A9 — monthly investor bullets (A9 + A9.r1 segmented templates).
        # Collects metrics from Z6/Z8/Z3/A8 (degrades gracefully when absent),
        # runs anomaly detection (±2σ vs 3-month median), generates LLM hypotheses
        # (OVERHEAD lane), emits 3 segmented variants, surfaces founder_action.
        from src.app.jobs.investor_bullets import run_investor_bullets
        try:
            product_id = payload.get("product_id")
            if product_id and product_id != "default":
                res = await run_investor_bullets(product_id=product_id)
                if not res.get("ok"):
                    return Action(
                        status="failed",
                        error=f"investor_bullets: {res.get('reason') or 'failed'}",
                        result=res,
                    )
                return Action(status="completed", result=res)
            # Cron tick — no specific product. Z7 #7: product_id now defaults
            # to mission_id, so sweep every distinct product. Without this the
            # cron ran for the literal "default" which matches no mission.
            from src.infra.db import get_db
            _db = await get_db()
            _cur = await _db.execute(
                "SELECT DISTINCT product_id FROM missions "
                "WHERE product_id IS NOT NULL"
            )
            _products = [r[0] for r in await _cur.fetchall()]
            await _cur.close()
            ran = 0
            for _pid in _products:
                try:
                    _r = await run_investor_bullets(product_id=str(_pid))
                    if _r.get("ok"):
                        ran += 1
                except Exception as _e:  # noqa: BLE001 — one product must not abort
                    logger.warning("investor_bullets failed for %s: %s", _pid, _e)
            return Action(status="completed",
                          result={"ok": True, "products_swept": len(_products),
                                  "products_ran": ran})
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "documentation_gap_detect":
        # Z7 T4 A8 — documentation_gap_detect posthook handler.
        # Semantic-search question against per-language support_docs collection;
        # writes docs_gap_log row when no match found.
        try:
            import importlib
            _mod = importlib.import_module(
                "general_beckman.posthook_handlers.documentation_gap_detect"
            )
            source_task_id = payload.get("source_task_id")
            source_task: dict = {}
            if source_task_id:
                try:
                    from src.infra.db import get_task as _get_task
                    source_task = await _get_task(source_task_id) or {}
                except Exception:
                    pass
            res = await _mod.handle(source_task, {})
            if res.get("status") == "fail":
                return Action(status="failed", error=str(res), result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 T4 A10: CRM data-layer mechanical verbs ───────────────────────────

    if action == "crm/add_contact":
        # Add or update a CRM contact. Upserts by (product_id, handle).
        from src.app.crm import add_contact
        try:
            contact_id = await add_contact(
                product_id=payload.get("product_id", "default"),
                handle=payload["handle"],
                display_name=payload.get("display_name", payload["handle"]),
                category=payload.get("category", "other"),
                email=payload.get("email"),
                notes_md=payload.get("notes_md"),
            )
            return Action(status="completed", result={"contact_id": contact_id}, reversibility="full")
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "crm/log_interaction":
        # Log a CRM interaction with optional relative follow-up window.
        from src.app.crm import log_interaction
        try:
            iid = await log_interaction(
                product_id=payload.get("product_id", "default"),
                contact_id=int(payload["contact_id"]),
                kind=payload.get("kind", "other"),
                summary=payload.get("summary", ""),
                follow_up=payload.get("follow_up"),
                next_action=payload.get("next_action"),
                mission_id=payload.get("mission_id"),
            )
            return Action(status="completed", result={"interaction_id": iid}, reversibility="full")
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "crm/grant_consent":
        # Grant consent for a purpose.
        from src.app.crm import grant_consent
        try:
            cid = await grant_consent(
                product_id=payload.get("product_id", "default"),
                contact_id=int(payload["contact_id"]),
                purpose=payload["purpose"],
                source_evidence_url=payload.get("source_evidence_url", ""),
                expires_at=payload.get("expires_at"),
            )
            return Action(status="completed", result={"consent_id": cid}, reversibility="full")
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "crm/revoke_consent":
        # Revoke a previously granted consent.
        from src.app.crm import revoke_consent
        try:
            await revoke_consent(
                product_id=payload.get("product_id", "default"),
                contact_id=int(payload["contact_id"]),
                purpose=payload["purpose"],
            )
            return Action(status="completed", result={}, reversibility="full")
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 T4 B4: meeting brief auto-generation verbs ────────────────────────

    if action == "meeting/brief":
        # Z7 B4 — LLM-bound brief generation via CPS (SP2 Task 4).
        # The mechanical step builds the brief context, then enqueues the
        # LLM brief draft via Beckman with on_complete/on_error
        # continuations. The actual brief_md write + Telegram notify happen
        # in `meetings._brief_persist_resume` / `_brief_persist_err`.
        from src.app.meetings import (
            build_brief_context,
            enqueue_meeting_brief,
        )
        try:
            meeting_id = int(payload.get("meeting_id") or 0)
            product_id = str(payload.get("product_id") or "default")

            ctx = await build_brief_context(
                meeting_id=meeting_id, product_id=product_id,
            )
            child_id = await enqueue_meeting_brief(
                ctx, meeting_id=meeting_id, product_id=product_id,
            )
            return Action(
                status="completed",
                result={
                    "meeting_id": meeting_id,
                    "queued": True,
                    "child_task_id": child_id,
                },
            )
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "meeting/outcome_prompt":
        # Z7 B4 — surface a founder_action card asking for meeting outcome.
        # Non-LLM mechanical: creates the card and returns.
        from src.app.meetings import emit_outcome_prompt
        try:
            meeting_id = int(payload.get("meeting_id") or 0)
            product_id = str(payload.get("product_id") or "default")
            res = await emit_outcome_prompt(meeting_id=meeting_id, product_id=product_id)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"meeting/outcome_prompt: {res.get('reason') or 'failed'}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "meeting_brief_dispatch":
        # Z7 B4 — 5-min cron: brief generation window + outcome prompt phase.
        from src.app.jobs.meeting_brief_dispatch import run_meeting_brief_dispatch
        try:
            res = await run_meeting_brief_dispatch()
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"meeting_brief_dispatch: failed",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 T5 B1 — lifecycle email engine verbs ─────────────────────────────

    if action == "lifecycle_email_send":
        # B1 — 5-min cron: pick due email_sends + send + mark sent_at.
        from src.app.jobs.lifecycle_email_send import run_lifecycle_email_send
        try:
            res = await run_lifecycle_email_send()
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error="lifecycle_email_send: failed",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "email/send_via_provider":
        # B1 — one-shot send via the product's configured ESP.
        # payload: {product_id, to, subject, body_md, headers?, idempotency_key?}
        from src.integrations.email.service import send_email
        try:
            product_id = str(payload.get("product_id") or "")
            to = str(payload.get("to") or "")
            subject = str(payload.get("subject") or "")
            body_md = str(payload.get("body_md") or "")
            headers = payload.get("headers") or None
            idempotency_key = payload.get("idempotency_key") or None
            res = await send_email(
                product_id=product_id,
                to=to,
                subject=subject,
                body_md=body_md,
                headers=headers,
                idempotency_key=idempotency_key,
            )
            status = res.get("status", "error")
            if status not in ("sent", "suppressed", "quota_blocked"):
                return Action(
                    status="failed",
                    error=f"email/send_via_provider: {res.get('error') or status}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 T4 B7 — customer interview pipeline verbs ─────────────────────────

    if action == "interview/transcribe":
        # B7 step 1: Whisper-CPU transcription (pluggable).
        # Reads audio_path from interview_notes, writes transcript_md back.
        from src.app.interview import transcribe_interview
        try:
            note_id = int(payload.get("note_id") or 0)
            product_id = str(payload.get("product_id") or "default")
            model_size = payload.get("model_size") or None
            res = await transcribe_interview(
                note_id=note_id,
                product_id=product_id,
                model_size=model_size,
            )
            if not res.get("ok"):
                return Action(status="failed", error=res.get("error", "transcribe failed"), result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "interview/summarize":
        # B7 step 2: LLM-bound summarization (OVERHEAD lane via beckman.enqueue).
        # Reads transcript_md, writes summary_md / quotes_json / insights_md /
        # action_items_json to interview_notes.
        from src.app.interview import summarize_interview
        try:
            note_id = int(payload.get("note_id") or 0)
            product_id = str(payload.get("product_id") or "default")
            res = await summarize_interview(note_id=note_id, product_id=product_id)
            if not res.get("ok"):
                return Action(status="failed", error=res.get("error", "summarize failed"), result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "interview/cross_link":
        # B7 step 3: non-LLM cross-linker.
        # a. Writes interactions row (kind='interview') via crm.log_interaction.
        # b. Enqueues action items as candidate backlog tasks.
        # c. Pushes quotes to press_kit_quotes gated on quote_use consent;
        #    emits founder_action requesting consent when absent.
        from src.app.interview import cross_link_interview
        try:
            note_id = int(payload.get("note_id") or 0)
            product_id = str(payload.get("product_id") or "default")
            contact_id = int(payload.get("contact_id") or 0)
            mission_id = payload.get("mission_id")
            if mission_id is not None:
                mission_id = int(mission_id)
            res = await cross_link_interview(
                note_id=note_id,
                product_id=product_id,
                contact_id=contact_id,
                mission_id=mission_id,
            )
            if not res.get("ok"):
                return Action(status="failed", error=res.get("error", "cross_link failed"), result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 T3B — demo pipeline verbs (A3 + A3.r1) ────────────────────────────

    if action == "demo/storyboard":
        # Generate a demo storyboard (ordered scenes) from product spec via LLM.
        from mr_roboto.demo_storyboard import run as _demo_storyboard
        try:
            res = await _demo_storyboard(
                mission_id=payload.get("mission_id"),
                spec_text=payload.get("spec_text") or "",
                workspace_path=payload.get("workspace_path") or "",
                parent_task_id=task.get("id"),
            )
            if not res.get("ok"):
                return Action(status="failed", error=res.get("error") or "demo/storyboard failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "demo/record":
        # Record each storyboard scene with Playwright --video on.
        from mr_roboto.demo_record import run as _demo_record
        try:
            res = await _demo_record(
                mission_id=payload.get("mission_id"),
                workspace_path=payload.get("workspace_path") or "",
                storyboard_path=payload.get("storyboard_path") or "",
                base_url=payload.get("base_url") or "",
                playwright_timeout=float(payload.get("playwright_timeout") or 300.0),
            )
            if not res.get("ok"):
                return Action(status="failed", error=res.get("error") or "demo/record failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "demo/edit":
        # Concat + trim scene recordings into three cut lengths via ffmpeg.
        from mr_roboto.demo_edit import run as _demo_edit
        try:
            res = await _demo_edit(
                mission_id=payload.get("mission_id"),
                workspace_path=payload.get("workspace_path") or "",
                storyboard_path=payload.get("storyboard_path") or "",
                scene_recordings=payload.get("scene_recordings") or [],
            )
            if not res.get("ok"):
                return Action(status="failed", error=res.get("error") or "demo/edit failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "demo/caption":
        # Generate script-driven WebVTT captions from storyboard narrator_text.
        from mr_roboto.demo_caption import run as _demo_caption
        try:
            res = await _demo_caption(
                mission_id=payload.get("mission_id"),
                workspace_path=payload.get("workspace_path") or "",
                storyboard_path=payload.get("storyboard_path") or "",
            )
            if not res.get("ok"):
                return Action(status="failed", error=res.get("error") or "demo/caption failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "demo/accessibility_pass":
        # A3.r1 — generate accessibility manifest (alt text, audio-desc, keyboard-nav).
        from mr_roboto.demo_accessibility_pass import run as _demo_a11y
        try:
            res = await _demo_a11y(
                mission_id=payload.get("mission_id"),
                workspace_path=payload.get("workspace_path") or "",
                storyboard_path=payload.get("storyboard_path") or "",
            )
            if not res.get("ok"):
                return Action(status="failed", error=res.get("error") or "demo/accessibility_pass failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action in ("demo_artifact_check", "demo_accessibility_check"):
        # Z7 T3B posthook handlers — route through general_beckman.posthook_handlers.
        try:
            import importlib
            _mod = importlib.import_module(
                f"general_beckman.posthook_handlers.{action}"
            )
            source_task_id = payload.get("source_task_id")
            source_task: dict = {}
            if source_task_id:
                try:
                    from src.infra.db import get_task as _get_task
                    _src = await _get_task(int(source_task_id))
                    if _src:
                        source_task = dict(_src)
                except Exception:
                    pass
            import json as _json
            src_ctx: dict = {}
            raw_ctx = source_task.get("context") or {}
            if isinstance(raw_ctx, str):
                try:
                    src_ctx = _json.loads(raw_ctx)
                except Exception:
                    src_ctx = {}
            elif isinstance(raw_ctx, dict):
                src_ctx = dict(raw_ctx)
            for _k in (
                "workspace_path", "demo_cuts", "demo_vtt_path", "demo_cut_targets",
                "demo_accessibility_manifest_path",
            ):
                if payload.get(_k):
                    src_ctx.setdefault(_k, payload[_k])
            source_task["context"] = src_ctx

            res = await _mod.handle(source_task, {})
            if res.get("status") == "failed":
                return Action(status="failed", error=str(res.get("error") or res), result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 T3C — press kit verbs (A4 + A4.r1) ────────────────────────────────

    if action == "press_kit/assemble":
        # Assemble a versioned press kit with 4 audience variants.
        from mr_roboto.press_kit_assemble import run as _pk_assemble
        try:
            res = await _pk_assemble(
                mission_id=payload.get("mission_id") or task.get("mission_id") or 0,
                product_id=payload.get("product_id") or "",
                spec_text=payload.get("spec_text") or "",
                workspace_path=payload.get("workspace_path") or "",
                logo_path=payload.get("logo_path") or "",
                screenshot_paths=payload.get("screenshot_paths") or [],
                founder_bio=payload.get("founder_bio") or "",
                fact_sheet_md=payload.get("fact_sheet_md") or "",
                quotes=payload.get("quotes") or [],
                past_mentions=payload.get("past_mentions") or [],
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=res.get("error") or "press_kit/assemble failed",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "press_kit/publish":
        # Upload assembled kit to S3/R2 or local fallback; return permanent URLs.
        from mr_roboto.press_kit_publish import run as _pk_publish
        try:
            manifest = payload.get("manifest") or {}
            if isinstance(manifest, str):
                import json as _json
                try:
                    manifest = _json.loads(manifest)
                except Exception:
                    manifest = {}
            res = await _pk_publish(
                mission_id=payload.get("mission_id") or task.get("mission_id") or 0,
                product_id=payload.get("product_id") or manifest.get("product_id") or "",
                manifest=manifest,
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=res.get("error") or "press_kit/publish failed",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "press_kit_freshness":
        # Z7 T3C posthook handler — route through general_beckman.posthook_handlers.
        try:
            import importlib
            _mod = importlib.import_module(
                "general_beckman.posthook_handlers.press_kit_freshness"
            )
            source_task_id = payload.get("source_task_id")
            source_task: dict = {}
            if source_task_id:
                try:
                    from src.infra.db import get_task as _get_task
                    _src = await _get_task(int(source_task_id))
                    if _src:
                        source_task = dict(_src)
                except Exception:
                    pass
            import json as _json
            src_ctx: dict = {}
            raw_ctx = source_task.get("context") or {}
            if isinstance(raw_ctx, str):
                try:
                    src_ctx = _json.loads(raw_ctx)
                except Exception:
                    src_ctx = {}
            elif isinstance(raw_ctx, dict):
                src_ctx = dict(raw_ctx)
            for _k in ("product_id",):
                if payload.get(_k):
                    src_ctx.setdefault(_k, payload[_k])
            source_task["context"] = src_ctx

            res = await _mod.handle(source_task, {})
            if res.get("status") == "failed":
                return Action(
                    status="failed",
                    error=str(res.get("error") or res),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 T3D — B3: incident comms verbs + posthook handler ─────────────────

    if action == "incident_update_review":
        # Z7 T3D B3 posthook handler — founder-review gate for draft status updates.
        # Route to general_beckman.posthook_handlers.incident_update_review.
        try:
            import importlib
            _mod = importlib.import_module(
                "general_beckman.posthook_handlers.incident_update_review"
            )
            source_task_id = payload.get("source_task_id")
            source_task: dict = {}
            if source_task_id:
                try:
                    from src.infra.db import get_task as _get_task
                    _src = await _get_task(int(source_task_id))
                    if _src:
                        source_task = dict(_src)
                except Exception:
                    pass
            import json as _json
            src_ctx: dict = {}
            raw_ctx = source_task.get("context") or {}
            if isinstance(raw_ctx, str):
                try:
                    src_ctx = _json.loads(raw_ctx)
                except Exception:
                    src_ctx = {}
            elif isinstance(raw_ctx, dict):
                src_ctx = dict(raw_ctx)
            # Merge payload fields so handler can read draft/incident_id/product_id.
            for _k in ("incident_id", "product_id", "draft", "status_kind"):
                if payload.get(_k):
                    src_ctx.setdefault(_k, payload[_k])
            source_task["context"] = src_ctx
            res = await _mod.handle(source_task, src_ctx)
            if res.get("status") == "failed":
                return Action(
                    status="failed",
                    error=str(res.get("error") or res),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "incident/draft_update":
        # Draft a customer-friendly status update (LLM-bound via beckman.enqueue).
        # CRITICAL: redacts internal hostnames / stack traces / customer PII.
        try:
            from mr_roboto.incident_draft_update import run as _draft_update
            res = await _draft_update(payload)
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or "draft_update failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "incident/publish_status":
        # Publish a founder-reviewed status update to the status_updates table.
        # Invalidates the /status page cache so the next request re-renders.
        try:
            from mr_roboto.incident_publish_status import run as _publish_status
            res = await _publish_status(payload)
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or "publish_status failed", result=res)
            # Bust cache so /status page shows the new update immediately.
            try:
                from src.app.status_page import invalidate_cache as _bust_cache
                _bust_cache()
            except Exception:
                pass
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "incident/draft_postmortem":
        # Auto-draft postmortem template at incident resolve.
        # Emits a founder_action (7-day review SLA).
        try:
            from mr_roboto.incident_draft_postmortem import run as _draft_postmortem
            res = await _draft_postmortem(payload)
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or "draft_postmortem failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 T3A — A2: Launch playbook verbs ─────────────────────────────────

    if action in (
        "launch_drafts/hn", "launch_drafts/ph", "launch_drafts/twitter",
        "launch_drafts/linkedin", "launch_drafts/reddit",
    ):
        # LLM-bound per-channel draft verb.
        # Enqueues a Beckman task (OVERHEAD lane) that generates the channel draft.
        channel = action.split("/", 1)[1]
        try:
            from mr_roboto.launch_drafts import run as _launch_drafts_run
            res = await _launch_drafts_run(channel, payload)
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or f"launch_drafts/{channel} failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "publish_synchronized":
        # Publish all approved channel drafts at T-0.
        # Checks is_marketing_frozen(product_id) first.
        try:
            from mr_roboto.launch_publish_synchronized import run as _pub_sync
            res = await _pub_sync(payload)
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or "publish_synchronized failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "launch_response_monitor":
        # Enqueue a sub-mission that polls channels for engagement + sentiment.
        try:
            from mr_roboto.launch_response_monitor import run as _launch_monitor
            res = await _launch_monitor(payload)
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or "launch_response_monitor failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "launch_lessons_writeback":
        # T+7d: emit 3-5 mission_lessons rows from engagement data.
        try:
            from mr_roboto.launch_lessons_writeback import run as _launch_lessons
            res = await _launch_lessons(payload)
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or "launch_lessons_writeback failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "launch_readiness_gate":
        # Pre-T-0 readiness gate: 7 hard checks.
        # Routes to posthook handler via the run() function.
        try:
            from mr_roboto.launch_readiness_gate import run as _launch_gate
            res = await _launch_gate(payload)
            if res.get("status") == "blocked":
                return Action(status="failed", error="launch_readiness_gate: blocked", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 T3E — B6: crisis comms verbs ─────────────────────────────────────

    if action == "crisis/freeze_marketing":
        # Pause in-flight A2/B1/A7 for the affected product.
        # Writes a per-product freeze flag into marketing_freeze table.
        # A2/B1/A7 check is_marketing_frozen(product_id) before proceeding.
        # Reversible via /crisis resume (sets resumed_at).
        try:
            from mr_roboto.crisis_freeze_marketing import run as _freeze
            res = await _freeze(payload)
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or "freeze_marketing failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "crisis/draft_holding":
        # LLM-bound: reads tier playbook + event context, outputs holding-statement variants.
        # Returns variants for founder selection — never publishes automatically.
        try:
            from mr_roboto.crisis_draft_holding import run as _draft_holding
            res = await _draft_holding(payload)
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or "draft_holding failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "crisis/disclosure_timer":
        # Tier 3 only: jurisdiction-aware 72h disclosure timer.
        # Runs every 6h; emits escalating founder_action reminder.
        try:
            from mr_roboto.crisis_disclosure_timer import run as _disclosure_timer
            res = await _disclosure_timer(payload)
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or "disclosure_timer failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 T5 B2 — changelog verbs ───────────────────────────────────────────

    if action == "changelog/draft":
        # Draft a changelog entry from recent git commits (KAC format).
        # Runs A5 brand_voice_lint + A6 copy_compliance (degrade gracefully).
        # Writes a draft row (published=0) and surfaces a founder_action.
        try:
            from mr_roboto.changelog_draft import run as _changelog_draft
            res = await _changelog_draft(payload)
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or "changelog/draft failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "changelog/publish":
        # Mark the entry published; invalidate cache; queue B1 announcement email.
        # Degrades gracefully when no announcement email sequence exists.
        try:
            from mr_roboto.changelog_publish import run as _changelog_publish
            res = await _changelog_publish(payload)
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or "changelog/publish failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "changelog_freshness":
        # Monthly check: goal:public_release missions without changelog entries.
        # Routes to general_beckman.posthook_handlers.changelog_freshness.
        try:
            import importlib
            _mod = importlib.import_module(
                "general_beckman.posthook_handlers.changelog_freshness"
            )
            source_task_id = payload.get("source_task_id")
            source_task: dict = {}
            if source_task_id:
                try:
                    from src.infra.db import get_task as _get_task
                    _src = await _get_task(int(source_task_id))
                    if _src:
                        source_task = dict(_src)
                except Exception:
                    pass
            import json as _json
            src_ctx: dict = {}
            raw_ctx = source_task.get("context") or {}
            if isinstance(raw_ctx, str):
                try:
                    src_ctx = _json.loads(raw_ctx)
                except Exception:
                    src_ctx = {}
            elif isinstance(raw_ctx, dict):
                src_ctx = dict(raw_ctx)
            for _k in ("product_id",):
                if payload.get(_k):
                    src_ctx.setdefault(_k, payload[_k])
            source_task["context"] = src_ctx
            res = await _mod.handle(source_task, {})
            if res.get("status") == "failed":
                return Action(status="failed", error=str(res.get("error") or res), result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 T5 B8 — reviews harvest verbs ────────────────────────────────────

    if action and action.startswith("reviews/poll/"):
        # Per-platform poll: reviews/poll/<platform>
        # Ingests new reviews from the platform; dedup via UNIQUE constraint.
        platform = action.split("/", 2)[2] if action.count("/") >= 2 else ""
        try:
            from mr_roboto.reviews_poll import run as _reviews_poll_run
            poll_payload = dict(payload)
            poll_payload["platform"] = platform
            res = await _reviews_poll_run(poll_payload)
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or f"reviews/poll/{platform} failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "reviews/classify":
        # LLM-bound: classify sentiment + theme_tag; side-effects for 1-2-star + bug.
        try:
            from mr_roboto.reviews_classify import run as _reviews_classify_run
            res = await _reviews_classify_run(payload)
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or "reviews/classify failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "reviews/draft_reply":
        # LLM-bound: draft a reply per brand voice + platform conventions.
        # NEVER auto-posts — founder approves before any reply is sent.
        try:
            from mr_roboto.reviews_draft_reply import run as _reviews_draft_reply_run
            res = await _reviews_draft_reply_run(payload)
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or "reviews/draft_reply failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "reviews_poll_daily":
        # Daily cron: poll all configured platforms + classify unclassified reviews.
        try:
            from src.app.jobs.reviews_poll_daily import run_reviews_poll_daily
            config = payload.get("config") or None
            res = await run_reviews_poll_daily(config=config)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error="reviews_poll_daily failed",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "capture_screenshots":
        # Z4 T1A — capture preview screenshots across breakpoints + color modes.
        from mr_roboto.capture_screenshots import (
            capture_screenshots as _capture_screenshots,
        )
        try:
            res = await _capture_screenshots(
                mission_id=int(task.get("mission_id") or payload.get("mission_id") or 0),
                step_id=str(step_id if (step_id := payload.get("step_id")) else task.get("id") or ""),
                routes=list(payload.get("routes")) if payload.get("routes") else None,
                components=list(payload.get("components")) if payload.get("components") else None,
                capture_mode=str(payload.get("capture_mode") or "viewport"),
                produces=list(payload.get("produces")) if payload.get("produces") else None,
                workspace_path=payload.get("workspace_path") or None,
            )
            if res.get("ok") is False:
                return Action(status="failed", error=res.get("error") or "capture_screenshots failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "visual_review":
        # Z4 T2B / T3D — diff captured screenshots against baselines or audit
        # against design tokens using a vision-capable model.
        # T3C: when captured_paths is empty/absent, the verb self-captures via
        # capture_screenshots (routes + produces forwarded for route inference).
        from mr_roboto.visual_review import visual_review as _visual_review
        try:
            res = await _visual_review(
                mission_id=int(task.get("mission_id") or payload.get("mission_id") or 0),
                step_id=str(step_id if (step_id := payload.get("step_id")) else task.get("id") or ""),
                captured_paths=list(payload.get("captured_paths") or []) or None,
                baseline_dir=payload.get("baseline_dir") or None,
                workspace_path=payload.get("workspace_path") or None,
                routes=list(payload.get("routes")) if payload.get("routes") else None,
                produces=list(payload.get("produces")) if payload.get("produces") else None,
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 T6 A7 — cold outreach verbs ──────────────────────────────────────
    if action == "outreach/send":
        try:
            from mr_roboto.outreach_send import run_outreach_send
            res = await run_outreach_send(
                product_id=payload.get("product_id") or "",
                list_id=payload.get("list_id") or "",
                target_email=payload.get("target_email") or "",
                template_id=payload.get("template_id") or "",
                subject=payload.get("subject") or "",
                body_md=payload.get("body_md") or "",
                postal_address=payload.get("postal_address") or "",
                unsubscribe_base_url=payload.get("unsubscribe_base_url") or "",
                jurisdiction=payload.get("jurisdiction"),
                has_explicit_opt_in=bool(payload.get("has_explicit_opt_in", False)),
            )
            if res.get("status") in ("error",):
                return Action(status="failed", error=res.get("error") or "outreach/send failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "outreach/handle_reply":
        try:
            from mr_roboto.outreach_handle_reply import run_outreach_handle_reply
            res = await run_outreach_handle_reply(
                product_id=payload.get("product_id") or "",
                send_id=int(payload.get("send_id") or 0),
                reply_body=payload.get("reply_body") or "",
                reply_from=payload.get("reply_from") or "",
                mission_id=task.get("mission_id"),
            )
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or "outreach/handle_reply failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "outreach/draft":
        try:
            from mr_roboto.outreach_draft import run_outreach_draft
            res = await run_outreach_draft(
                product_id=payload.get("product_id") or "",
                mission_id=int(task.get("mission_id") or payload.get("mission_id") or 0),
                prospect_data=payload.get("prospect_data") or {},
                template_id=payload.get("template_id") or "",
                list_id=payload.get("list_id") or "",
            )
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or "outreach/draft failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "outreach_deliverability_check":
        try:
            from mr_roboto.outreach_deliverability_check import handle as _dc_handle
            res = await _dc_handle(task, payload)
            # warning posthook: never fails the source
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="completed", result={"status": "skip", "reason": str(e)})

    if action == "outreach/domain_verify":
        try:
            from mr_roboto.outreach_domain_verify import run_domain_verify
            res = await run_domain_verify(
                product_id=payload.get("product_id") or "",
                mission_id=int(task.get("mission_id") or payload.get("mission_id") or 0),
                domain=payload.get("domain") or "",
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 T6 A11: mention_polls/<source> — mention monitor poll verbs ───────
    if action and action.startswith("mention_polls/"):
        source = action[len("mention_polls/"):]
        try:
            from mr_roboto.mention_polls import run as _mp_run
            res = await _mp_run({**payload, "source": source})
            if res.get("status") == "failed":
                return Action(status="failed", error=res.get("error") or "mention_polls failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 A11: mention_monitor_sweep — hourly cron over registered products ─
    if action == "mention_monitor_sweep":
        try:
            from mr_roboto.mention_monitor_sweep import run as _mms_run
            res = await _mms_run(task)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 T6 A11.r2: internal_signal_poll — proxy until Z6 event stream ────
    if action == "internal_signal_poll":
        try:
            from mr_roboto.internal_signal_poll import run as _isp_run
            res = await _isp_run(payload)
            if res.get("status") == "failed":
                return Action(status="failed", error=res.get("error") or "internal_signal_poll failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 T6 A12: marketing_copy — marketing copy generator ─────────────────
    if action == "marketing_copy":
        try:
            from mr_roboto.marketing_copy import run_marketing_copy
            _ctx = task.get("context") or {}
            res = await run_marketing_copy(
                product_id=payload.get("product_id") or str(task.get("mission_id") or ""),
                mission_id=int(task.get("mission_id") or payload.get("mission_id") or 0),
                product_spec=payload.get("product_spec") or {},
                brand_voice_audience=payload.get("brand_voice_audience"),
                faq_artifact_path=payload.get("faq_artifact_path"),
                task_id=task.get("id"),
                workspace_path=_ctx.get("workspace_path") or payload.get("workspace_path"),
            )
            if res.get("status") == "error":
                return Action(status="failed", error=res.get("error") or "marketing_copy failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z7 T6D — demo/distribute (A3 distribution stage) ─────────────────────
    if action == "demo/distribute":
        # Upload demo cuts to YouTube (unlisted), extract thumbnails, og:video snippet.
        from mr_roboto.demo_distribute import run as _demo_distribute
        try:
            cuts = payload.get("cuts") or {}
            if isinstance(cuts, str):
                import json as _json
                try:
                    cuts = _json.loads(cuts)
                except Exception:
                    cuts = {}
            # Workspace-convention fallback: when an i2p step does not pass an
            # explicit `cuts` map, resolve the three cuts demo/edit produces
            # under workspace/demo/cuts/. Keeps the i2p step payload minimal.
            if not cuts:
                import os as _os
                _ws = payload.get("workspace_path") or ""
                _cuts_dir = _os.path.join(_ws, "demo", "cuts")
                for _lbl in ("30s", "60s", "3min"):
                    _p = _os.path.join(_cuts_dir, f"{_lbl}.mp4")
                    if _ws and _os.path.exists(_p):
                        cuts[_lbl] = _p
            res = await _demo_distribute(
                mission_id=int(payload.get("mission_id") or task.get("mission_id") or 0),
                workspace_path=payload.get("workspace_path") or "",
                cuts=cuts,
                product_name=payload.get("product_name") or "",
                description=payload.get("description") or "",
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=res.get("error") or "demo/distribute failed",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    # ── Z5 T3 — mobile build/distribution adapters ───────────────────────────
    if action == "expo_cli":
        # Wrap the Expo CLI (prebuild / export / doctor). Soft-skips when
        # the Expo CLI / npx is not installed.
        from mr_roboto.expo_cli import expo_cli as _expo_cli
        try:
            res = await _expo_cli(
                mission_id=task.get("mission_id"),
                subcommand=str(payload.get("subcommand") or ""),
                workspace_path=payload.get("workspace_path") or None,
                extra_args=list(payload.get("extra_args") or []) or None,
                timeout_s=float(payload.get("timeout_s", 600.0)),
            )
            if res.get("skipped"):
                return Action(status="completed", result=res)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=res.get("error") or "expo_cli failed",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "android_build":
        # Wrap local gradle / adb. Runs natively on Windows. Soft-skips
        # when the gradle wrapper or adb is absent.
        from mr_roboto.android_build import android_build as _android_build
        try:
            # NOTE: payload["action"] is the verb selector ("android_build")
            # — the gradle/adb sub-action comes from a dedicated key.
            res = await _android_build(
                mission_id=task.get("mission_id"),
                action=str(
                    payload.get("android_action")
                    or payload.get("subcommand")
                    or ""
                ),
                workspace_path=payload.get("workspace_path") or None,
                variant=str(payload.get("variant") or "release"),
                extra_args=list(payload.get("extra_args") or []) or None,
                timeout_s=float(payload.get("timeout_s", 600.0)),
            )
            if res.get("skipped"):
                return Action(status="completed", result=res)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=res.get("error") or "android_build failed",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "eas_build":
        # Wrap `eas build` — cloud builds (iOS + Android). Soft-skips when
        # eas-cli / npx is not installed.
        from mr_roboto.eas_build import eas_build as _eas_build
        try:
            res = await _eas_build(
                mission_id=task.get("mission_id"),
                platform=str(payload.get("platform") or "all"),
                profile=str(payload.get("profile") or "production"),
                workspace_path=payload.get("workspace_path") or None,
                non_interactive=bool(payload.get("non_interactive", True)),
                extra_args=list(payload.get("extra_args") or []) or None,
                timeout_s=float(payload.get("timeout_s", 1800.0)),
            )
            if res.get("skipped"):
                return Action(status="completed", result=res)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=res.get("error") or "eas_build failed",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "eas_submit":
        # Wrap `eas submit` — uploads a build to TestFlight / Play
        # internal. Irreversible (a binary lands on a store track).
        # Soft-skips when eas-cli / npx is not installed.
        from mr_roboto.eas_submit import eas_submit as _eas_submit
        try:
            res = await _eas_submit(
                mission_id=task.get("mission_id"),
                platform=str(payload.get("platform") or ""),
                workspace_path=payload.get("workspace_path") or None,
                build_id=payload.get("build_id") or None,
                latest=bool(payload.get("latest", False)),
                profile=str(payload.get("profile") or "production"),
                non_interactive=bool(payload.get("non_interactive", True)),
                extra_args=list(payload.get("extra_args") or []) or None,
                timeout_s=float(payload.get("timeout_s", 900.0)),
            )
            if res.get("skipped"):
                return Action(status="completed", result=res)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=res.get("error") or "eas_submit failed",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "gen_mobile_ci":
        # Z5 T3b — generate .github/workflows/mobile.yml (free-first GH
        # Actions mobile CI: macOS-runner iOS job + Linux Android job).
        # Reversibility `full` — writes one local file.
        from mr_roboto.gen_mobile_ci import gen_mobile_ci as _gen_mobile_ci
        try:
            res = await _gen_mobile_ci(
                mission_id=task.get("mission_id"),
                workspace_path=payload.get("workspace_path") or None,
                platforms=list(payload.get("platforms"))
                if payload.get("platforms") is not None
                else None,
                bundle_id=str(payload.get("bundle_id") or "com.example.app"),
                scheme=payload.get("scheme") or None,
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=res.get("error") or "gen_mobile_ci failed",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "fastlane":
        # Z5 T3b — run a Fastlane lane (build / match / pilot / supply).
        # Per-lane reversibility: build/match → full, pilot/supply →
        # irreversible (resolved in run() via reversibility_override).
        # Soft-skips when the fastlane CLI is not installed.
        from mr_roboto.fastlane_run import fastlane as _fastlane
        try:
            res = await _fastlane(
                mission_id=task.get("mission_id"),
                lane=str(payload.get("lane") or ""),
                workspace_path=payload.get("workspace_path") or None,
                extra_args=list(payload.get("extra_args") or []) or None,
                timeout_s=float(payload.get("timeout_s", 1200.0)),
            )
            if res.get("skipped"):
                return Action(status="completed", result=res)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=res.get("error") or "fastlane failed",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "maestro":
        # Z5 T4b — run one or more Maestro mobile-QA flow YAMLs. Drives the
        # `mobile_smoke` post-hook (sign in → onboard → core action → sign
        # out). Read-only test run — reversibility `full`. Soft-skips when
        # the Maestro CLI is not installed (treated as a soft pass).
        from mr_roboto.maestro_run import maestro_run as _maestro_run
        try:
            res = await _maestro_run(
                mission_id=task.get("mission_id"),
                flow_paths=list(payload.get("flow_paths") or []),
                workspace_path=payload.get("workspace_path") or None,
                extra_args=list(payload.get("extra_args") or []) or None,
                timeout_s=float(payload.get("timeout_s", 600.0)),
            )
            if res.get("skipped"):
                return Action(status="completed", result=res)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=res.get("error") or "maestro flows failed",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))

    if action == "yalayut_discovery":
        from mr_roboto.executors.yalayut_discovery import run as _yal_disc_run
        res = await _yal_disc_run(task)
        return Action(status="completed", result=res)

    if action == "source_scout":
        from mr_roboto.executors.source_scout import run as _scout_run
        res = await _scout_run(task)
        return Action(status="completed", result=res)

    if action == "capture_hint":
        from mr_roboto.executors.capture_hint import run as _capture_run
        res = await _capture_run(task)
        return Action(status="completed", result=res)

    if action == "yalayut_demand":
        from mr_roboto.executors.yalayut_demand import run as _yal_demand_run
        res = await _yal_demand_run(task)
        return Action(status="completed", result=res)

    return Action(status="failed", error=f"unknown mechanical action: {action!r}")
