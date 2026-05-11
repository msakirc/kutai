"""Z2 T3B — openapi_sync + typescript_sync post-hook kind tests.

Covers:
1. Registry has both kinds with correct fields + shared verb.
2. Auto-wire: step produces route file → openapi_sync wired; produces
   openapi.json → both openapi_sync AND typescript_sync wired; idempotent.
3. regen_and_diff verb: no-drift case, drift case, soft-skip (generator absent).
4. _posthook_agent_and_payload maps both kinds to regen_and_diff payload.
5. Verdict: drift → blocker retry; ok/no-drift → advance; skipped → advance.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_step(produces: list[str], post_hooks: list[str] | None = None) -> dict:
    step: dict = {
        "id": "1.1",
        "phase": "phase_1",
        "name": "write_routes",
        "agent": "coder",
        "instruction": "Write routes.",
        "depends_on": [],
        "input_artifacts": [],
        "output_artifacts": [],
        "produces": produces,
    }
    if post_hooks is not None:
        step["post_hooks"] = post_hooks
    return step


def _make_task(action: str, **payload_fields) -> dict:
    return {
        "id": 1,
        "mission_id": None,
        "payload": {"action": action, **payload_fields},
    }


# ---------------------------------------------------------------------------
# 1.  Registry
# ---------------------------------------------------------------------------

def test_registry_has_openapi_sync():
    from general_beckman.posthooks import POST_HOOK_REGISTRY, PostHookSpec
    assert "openapi_sync" in POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["openapi_sync"]
    assert isinstance(spec, PostHookSpec)
    assert spec.kind == "openapi_sync"
    assert spec.verb == "regen_and_diff"
    assert spec.default_severity == "blocker"


def test_registry_has_typescript_sync():
    from general_beckman.posthooks import POST_HOOK_REGISTRY, PostHookSpec
    assert "typescript_sync" in POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["typescript_sync"]
    assert isinstance(spec, PostHookSpec)
    assert spec.kind == "typescript_sync"
    assert spec.verb == "regen_and_diff"
    assert spec.default_severity == "blocker"


def test_both_kinds_share_verb():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    assert (
        POST_HOOK_REGISTRY["openapi_sync"].verb
        == POST_HOOK_REGISTRY["typescript_sync"].verb
        == "regen_and_diff"
    )


def test_openapi_sync_triggers():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    triggers = POST_HOOK_REGISTRY["openapi_sync"].auto_wire_triggers
    assert "openapi.json" in triggers
    assert "openapi.yaml" in triggers
    # Route file globs
    assert any("routes" in t for t in triggers)


def test_typescript_sync_triggers():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    triggers = POST_HOOK_REGISTRY["typescript_sync"].auto_wire_triggers
    assert "openapi.json" in triggers
    assert "types/api.ts" in triggers


def test_both_in_post_hook_kinds():
    from general_beckman.posthooks import POST_HOOK_KINDS
    assert "openapi_sync" in POST_HOOK_KINDS
    assert "typescript_sync" in POST_HOOK_KINDS


# ---------------------------------------------------------------------------
# 2.  Auto-wire via expander
# ---------------------------------------------------------------------------

def test_autowire_route_file_gets_openapi_sync():
    """Step produces a route file → openapi_sync auto-wired (in post_hooks)."""
    from src.workflows.engine.expander import _auto_wire_posthooks

    ctx = {"produces": ["app/routes/users.py"]}
    _auto_wire_posthooks(ctx)
    post_hooks = ctx.get("post_hooks") or []
    assert "openapi_sync" in post_hooks
    # typescript_sync does not trigger on py route files
    assert "typescript_sync" not in post_hooks


def test_autowire_openapi_json_gets_both_kinds():
    """Step produces openapi.json → both openapi_sync AND typescript_sync wired."""
    from src.workflows.engine.expander import _auto_wire_posthooks

    ctx = {"produces": ["openapi.json"]}
    _auto_wire_posthooks(ctx)
    post_hooks = ctx.get("post_hooks") or []
    assert "openapi_sync" in post_hooks
    assert "typescript_sync" in post_hooks


def test_autowire_idempotent():
    """Running _auto_wire_posthooks twice doesn't duplicate entries."""
    from src.workflows.engine.expander import _auto_wire_posthooks

    ctx = {"produces": ["openapi.json"]}
    _auto_wire_posthooks(ctx)
    count_before = len(ctx.get("post_hooks") or [])
    _auto_wire_posthooks(ctx)
    assert len(ctx.get("post_hooks") or []) == count_before


# ---------------------------------------------------------------------------
# 3.  regen_and_diff verb behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_regen_no_drift(tmp_path):
    """Generator output matches committed file → diff_present=False."""
    from mr_roboto.regen_and_diff import regen_and_diff

    target = tmp_path / "openapi.json"
    target.write_text("hello", encoding="utf-8")

    res = await regen_and_diff(
        mission_id=None,
        generator_cmd=[sys.executable, "-c", "print('hello', end='')"],
        target_path="openapi.json",
        workspace_path=str(tmp_path),
        timeout_s=10.0,
    )
    assert res["ok"] is True
    assert res["diff_present"] is False
    assert res["diff_excerpt"] == ""
    assert res["skipped"] is False


@pytest.mark.asyncio
async def test_regen_drift(tmp_path):
    """Generator output differs from committed file → diff_present=True, diff populated."""
    from mr_roboto.regen_and_diff import regen_and_diff

    target = tmp_path / "openapi.json"
    target.write_text("different content", encoding="utf-8")

    res = await regen_and_diff(
        mission_id=None,
        generator_cmd=[sys.executable, "-c", "print('generated content', end='')"],
        target_path="openapi.json",
        workspace_path=str(tmp_path),
        timeout_s=10.0,
    )
    assert res["ok"] is True
    assert res["diff_present"] is True
    assert res["diff_excerpt"] != ""
    assert "committed" in res["diff_excerpt"] or "regenerated" in res["diff_excerpt"]


@pytest.mark.asyncio
async def test_regen_file_missing(tmp_path):
    """Target file does not exist → diff_present=True with 'file missing' message."""
    from mr_roboto.regen_and_diff import regen_and_diff

    res = await regen_and_diff(
        mission_id=None,
        generator_cmd=[sys.executable, "-c", "print('content', end='')"],
        target_path="openapi.json",
        workspace_path=str(tmp_path),
        timeout_s=10.0,
    )
    assert res["ok"] is True
    assert res["diff_present"] is True
    assert "file missing" in res["diff_excerpt"]


@pytest.mark.asyncio
async def test_regen_soft_skip_nonexistent_cmd(tmp_path):
    """Generator command not found → skipped=True, ok=True (v1 ramp)."""
    from mr_roboto.regen_and_diff import regen_and_diff

    target = tmp_path / "openapi.json"
    target.write_text("x", encoding="utf-8")

    res = await regen_and_diff(
        mission_id=None,
        generator_cmd=["nonexistent_cmd_xyz_abc"],
        target_path="openapi.json",
        workspace_path=str(tmp_path),
        timeout_s=5.0,
    )
    assert res["ok"] is True
    assert res["skipped"] is True
    assert res["reason"] == "generator not installed"
    assert res["diff_present"] is False


# ---------------------------------------------------------------------------
# 4.  _posthook_agent_and_payload mapping
# ---------------------------------------------------------------------------

def test_openapi_sync_payload():
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    source_ctx = {"workspace_path": "/ws"}
    a = RequestPostHook(source_task_id=99, kind="openapi_sync", source_ctx=source_ctx)
    agent_type, ctx = _posthook_agent_and_payload(a, {}, source_ctx)
    assert agent_type == "mechanical"
    assert ctx["posthook_kind"] == "openapi_sync"
    payload = ctx["payload"]
    assert payload["action"] == "regen_and_diff"
    assert isinstance(payload["generator_cmd"], list)
    assert payload["target_path"] == "openapi.json"
    assert payload["workspace_path"] == "/ws"


def test_typescript_sync_payload():
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    source_ctx = {"workspace_path": "/ws"}
    a = RequestPostHook(source_task_id=99, kind="typescript_sync", source_ctx=source_ctx)
    agent_type, ctx = _posthook_agent_and_payload(a, {}, source_ctx)
    assert agent_type == "mechanical"
    assert ctx["posthook_kind"] == "typescript_sync"
    payload = ctx["payload"]
    assert payload["action"] == "regen_and_diff"
    assert isinstance(payload["generator_cmd"], list)
    assert payload["target_path"] == "types/api.ts"
    assert payload["workspace_path"] == "/ws"


def test_openapi_sync_context_override():
    """regen_cmd and openapi_target_path overrides are respected."""
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    source_ctx = {
        "workspace_path": "/ws",
        "regen_cmd": ["make", "openapi"],
        "openapi_target_path": "docs/openapi.json",
    }
    a = RequestPostHook(source_task_id=10, kind="openapi_sync", source_ctx=source_ctx)
    _, ctx = _posthook_agent_and_payload(a, {}, source_ctx)
    payload = ctx["payload"]
    assert payload["generator_cmd"] == ["make", "openapi"]
    assert payload["target_path"] == "docs/openapi.json"


def test_typescript_sync_context_override():
    """regen_cmd and types_target_path overrides are respected."""
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    source_ctx = {
        "workspace_path": "/ws",
        "regen_cmd": ["npx", "openapi-typescript", "docs/openapi.json"],
        "types_target_path": "src/types/api.ts",
    }
    a = RequestPostHook(source_task_id=10, kind="typescript_sync", source_ctx=source_ctx)
    _, ctx = _posthook_agent_and_payload(a, {}, source_ctx)
    payload = ctx["payload"]
    assert payload["generator_cmd"] == ["npx", "openapi-typescript", "docs/openapi.json"]
    assert payload["target_path"] == "src/types/api.ts"


# ---------------------------------------------------------------------------
# 5.  Verdict: drift → blocker retry; no-drift → advance; skipped → advance
# ---------------------------------------------------------------------------

def _make_source_task(worker_attempts: int = 0, status: str = "ungraded") -> dict:
    return {
        "id": 42,
        "mission_id": None,
        "status": status,
        "worker_attempts": worker_attempts,
        "max_worker_attempts": 15,
        "agent_type": "coder",
        "title": "Write routes",
        "result": None,
        "context": json.dumps({"_pending_posthooks": ["openapi_sync"]}),
        "task_state": None,
    }


@pytest.mark.asyncio
async def test_verdict_no_drift_advances():
    """No drift → source marked completed (no other pending hooks)."""
    import src.infra.db as _db_mod
    from general_beckman.apply import _apply_type_sync_verdict
    from general_beckman.result_router import PostHookVerdict

    updated: dict = {}

    async def fake_update_task(task_id, **kwargs):
        updated.update(kwargs)

    original = getattr(_db_mod, "update_task", None)
    _db_mod.update_task = fake_update_task
    try:
        with patch("general_beckman.apply._spawn_workflow_advance_if_mission", new=AsyncMock()):
            source = _make_source_task()
            ctx = {"_pending_posthooks": ["openapi_sync"]}
            verdict = PostHookVerdict(
                source_task_id=42,
                kind="openapi_sync",
                passed=True,
                raw={"ok": True, "diff_present": False, "diff_excerpt": ""},
            )
            await _apply_type_sync_verdict(
                kind="openapi_sync",
                source=source,
                ctx=ctx,
                pending=["openapi_sync"],
                verdict=verdict,
            )
    finally:
        if original is not None:
            _db_mod.update_task = original

    assert updated.get("status") == "completed"


@pytest.mark.asyncio
async def test_verdict_drift_retries():
    """Drift detected → source set to pending with quality error + schema feedback."""
    import src.infra.db as _db_mod
    from general_beckman.apply import _apply_type_sync_verdict
    from general_beckman.result_router import PostHookVerdict

    updated: dict = {}

    async def fake_update_task(task_id, **kwargs):
        updated.update(kwargs)

    original = getattr(_db_mod, "update_task", None)
    _db_mod.update_task = fake_update_task
    try:
        source = _make_source_task()
        ctx = {"_pending_posthooks": ["openapi_sync"]}
        verdict = PostHookVerdict(
            source_task_id=42,
            kind="openapi_sync",
            passed=False,
            raw={
                "ok": True,
                "diff_present": True,
                "diff_excerpt": "@@ -1 +1 @@\n-old\n+new",
                "target_path": "openapi.json",
                "generator_cmd": ["python", "-c", "..."],
            },
        )
        await _apply_type_sync_verdict(
            kind="openapi_sync",
            source=source,
            ctx=ctx,
            pending=["openapi_sync"],
            verdict=verdict,
        )
    finally:
        if original is not None:
            _db_mod.update_task = original

    assert updated.get("status") == "pending"
    assert updated.get("error_category") == "quality"
    # Schema feedback injected into ctx
    assert "openapi_sync" in (updated.get("context") or "")


@pytest.mark.asyncio
async def test_verdict_skipped_advances():
    """Generator not installed (skipped) → source advances without retry."""
    import src.infra.db as _db_mod
    from general_beckman.apply import _apply_type_sync_verdict
    from general_beckman.result_router import PostHookVerdict

    updated: dict = {}

    async def fake_update_task(task_id, **kwargs):
        updated.update(kwargs)

    original = getattr(_db_mod, "update_task", None)
    _db_mod.update_task = fake_update_task
    try:
        with patch("general_beckman.apply._spawn_workflow_advance_if_mission", new=AsyncMock()):
            source = _make_source_task()
            ctx = {"_pending_posthooks": ["openapi_sync"]}
            verdict = PostHookVerdict(
                source_task_id=42,
                kind="openapi_sync",
                passed=False,
                raw={
                    "ok": True,
                    "diff_present": False,
                    "skipped": True,
                    "reason": "generator not installed",
                },
            )
            await _apply_type_sync_verdict(
                kind="openapi_sync",
                source=source,
                ctx=ctx,
                pending=["openapi_sync"],
                verdict=verdict,
            )
    finally:
        if original is not None:
            _db_mod.update_task = original

    assert updated.get("status") == "completed"


# ---------------------------------------------------------------------------
# 6.  _run_dispatch routes regen_and_diff action
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_dispatch_regen_and_diff_no_drift(tmp_path):
    """_run_dispatch returns completed Action for no-drift regen_and_diff."""
    import mr_roboto

    target = tmp_path / "openapi.json"
    target.write_text("hello", encoding="utf-8")

    task = {
        "id": 1,
        "mission_id": None,
        "payload": {
            "action": "regen_and_diff",
            "generator_cmd": [sys.executable, "-c", "print('hello', end='')"],
            "target_path": "openapi.json",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result is not None
    assert action.result.get("ok") is True
    assert action.result.get("diff_present") is False


@pytest.mark.asyncio
async def test_run_dispatch_regen_and_diff_drift(tmp_path):
    """_run_dispatch returns completed Action even on drift (verdict fn decides)."""
    import mr_roboto

    target = tmp_path / "openapi.json"
    target.write_text("different", encoding="utf-8")

    task = {
        "id": 1,
        "mission_id": None,
        "payload": {
            "action": "regen_and_diff",
            "generator_cmd": [sys.executable, "-c", "print('generated', end='')"],
            "target_path": "openapi.json",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    # Drift is not an internal failure — completed so verdict fn can react.
    assert action.status == "completed"
    assert action.result.get("diff_present") is True
