"""Mr. Roboto — mechanical dispatcher: non-LLM task executors."""
from __future__ import annotations

from mr_roboto.actions import Action
from mr_roboto.workspace_snapshot import snapshot_workspace
from mr_roboto.git_commit import auto_commit
from mr_roboto.verify_artifacts import verify_artifacts
from mr_roboto.verify_schema_version import verify_schema_version
from mr_roboto.verify_charter_shape import verify_charter_shape
from mr_roboto.verify_reverse_pitch_shape import verify_reverse_pitch_shape
from mr_roboto.generate_intake_todo import generate_intake_todo
from mr_roboto.run_cmd import run_cmd
from mr_roboto.run_pytest import run_pytest
from mr_roboto.parse_og_tags import parse_og_tags
from mr_roboto.http_check import http_check

__all__ = [
    "Action",
    "run",
    "snapshot_workspace",
    "auto_commit",
    "verify_artifacts",
    "verify_schema_version",
    "verify_charter_shape",
    "verify_reverse_pitch_shape",
    "generate_intake_todo",
    "run_cmd",
    "run_pytest",
    "parse_og_tags",
    "http_check",
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

    return Action(status="failed", error=f"unknown mechanical action: {action!r}")
