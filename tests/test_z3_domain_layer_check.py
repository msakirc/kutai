"""Z3 — domain_layer_check post-hook wiring tests.

Covers:
(a) domain_layer_check is in POST_HOOK_REGISTRY with correct verb/severity/triggers.
(b) _auto_wire_posthooks prepends domain_layer_check for a step that produces
    src/domain/user.py (constructed inline — no i2p step dependency).
(c) Payload builder produces action="run_semgrep_layer_filtered",
    rule_pack_path="forbidden_in_domain.yml", required_layer="domain".
(d) Verdict path: real findings cascade the source (blocker);
    semgrep-crash result soft-drops without cascading.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ────────────────────────────────────────────────────────────────────────────
# (a) Registry
# ────────────────────────────────────────────────────────────────────────────

def test_domain_layer_check_in_registry():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    assert "domain_layer_check" in POST_HOOK_REGISTRY, (
        "domain_layer_check must be registered in POST_HOOK_REGISTRY"
    )


def test_domain_layer_check_verb():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["domain_layer_check"]
    assert spec.verb == "run_semgrep_layer_filtered", (
        f"expected verb='run_semgrep_layer_filtered', got {spec.verb!r}"
    )


def test_domain_layer_check_severity_blocker():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["domain_layer_check"]
    assert spec.default_severity == "blocker", (
        f"expected default_severity='blocker', got {spec.default_severity!r}"
    )


def test_domain_layer_check_cost_band_cheap():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["domain_layer_check"]
    assert spec.cost_band == "cheap", (
        f"expected cost_band='cheap', got {spec.cost_band!r}"
    )


def test_domain_layer_check_triggers_present():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["domain_layer_check"]
    triggers = spec.resolve_triggers()
    assert triggers, "auto_wire_triggers must be non-empty"
    # Must cover the two declared patterns
    assert "**/domain/*.py" in triggers or any("domain" in t for t in triggers), (
        f"expected a domain glob in triggers, got {triggers}"
    )


def test_domain_layer_check_trigger_matches_domain_file():
    """**/domain/*.py or src/domain/**/*.py must match src/domain/user.py."""
    import fnmatch
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["domain_layer_check"]
    triggers = spec.resolve_triggers()
    assert any(
        fnmatch.fnmatchcase("src/domain/user.py", t) for t in triggers
    ), f"No trigger matched src/domain/user.py in {triggers}"


def test_domain_layer_check_trigger_matches_nested_domain_file():
    """src/domain/**/*.py must match src/domain/models/order.py."""
    import fnmatch
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["domain_layer_check"]
    triggers = spec.resolve_triggers()
    assert any(
        fnmatch.fnmatchcase("src/domain/models/order.py", t) for t in triggers
    ), f"No trigger matched src/domain/models/order.py in {triggers}"


def test_domain_layer_check_trigger_does_not_match_infra():
    """Triggers must NOT match src/infra/db.py."""
    import fnmatch
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["domain_layer_check"]
    triggers = spec.resolve_triggers()
    assert not any(
        fnmatch.fnmatchcase("src/infra/db.py", t) for t in triggers
    ), f"A trigger incorrectly matched src/infra/db.py: {triggers}"


# ────────────────────────────────────────────────────────────────────────────
# (b) Auto-wire via expander
# ────────────────────────────────────────────────────────────────────────────

def test_auto_wire_prepends_domain_layer_check_for_domain_produces():
    """_auto_wire_posthooks prepends domain_layer_check when produces=src/domain/user.py."""
    from src.workflows.engine.expander import _auto_wire_posthooks

    context: dict[str, Any] = {
        "produces": ["src/domain/user.py"],
        "post_hooks": [],
    }
    _auto_wire_posthooks(context)
    assert "domain_layer_check" in context["post_hooks"], (
        f"domain_layer_check not in post_hooks after auto-wire: {context['post_hooks']}"
    )


def test_auto_wire_does_not_wire_domain_layer_check_for_infra_produces():
    """_auto_wire_posthooks must NOT add domain_layer_check for infra produces."""
    from src.workflows.engine.expander import _auto_wire_posthooks

    context: dict[str, Any] = {
        "produces": ["src/infra/db.py"],
        "post_hooks": [],
    }
    _auto_wire_posthooks(context)
    assert "domain_layer_check" not in context["post_hooks"], (
        f"domain_layer_check incorrectly wired for infra produces: {context['post_hooks']}"
    )


def test_auto_wire_idempotent_domain_layer_check():
    """Calling _auto_wire_posthooks twice must not duplicate domain_layer_check."""
    from src.workflows.engine.expander import _auto_wire_posthooks

    context: dict[str, Any] = {
        "produces": ["src/domain/user.py"],
        "post_hooks": [],
    }
    _auto_wire_posthooks(context)
    _auto_wire_posthooks(context)
    count = context["post_hooks"].count("domain_layer_check")
    assert count == 1, (
        f"domain_layer_check appears {count} times (expected 1): {context['post_hooks']}"
    )


# ────────────────────────────────────────────────────────────────────────────
# (c) Payload builder
# ────────────────────────────────────────────────────────────────────────────

def test_payload_builder_action():
    """Payload builder must set action='run_semgrep_layer_filtered'."""
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    source = {"id": 42, "mission_id": 7, "title": "build domain"}
    source_ctx = {
        "produces": ["src/domain/user.py"],
        "workspace_path": "/tmp/mission_7",
    }
    a = RequestPostHook(source_task_id=42, kind="domain_layer_check", source_ctx=source_ctx)
    agent_type, payload = _posthook_agent_and_payload(a, source, source_ctx)

    assert agent_type == "mechanical"
    assert payload["payload"]["action"] == "run_semgrep_layer_filtered"


def test_payload_builder_rule_pack_path():
    """Payload must set rule_pack_path='forbidden_in_domain.yml'."""
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    source = {"id": 42, "mission_id": 7}
    source_ctx = {
        "produces": ["src/domain/user.py"],
        "workspace_path": "/tmp/mission_7",
    }
    a = RequestPostHook(source_task_id=42, kind="domain_layer_check", source_ctx=source_ctx)
    _, payload = _posthook_agent_and_payload(a, source, source_ctx)

    assert payload["payload"]["rule_pack_path"] == "forbidden_in_domain.yml", (
        f"expected 'forbidden_in_domain.yml', got {payload['payload']['rule_pack_path']!r}"
    )


def test_payload_builder_required_layer():
    """Payload must set required_layer='domain'."""
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    source = {"id": 42, "mission_id": 7}
    source_ctx = {
        "produces": ["src/domain/user.py"],
        "workspace_path": "/tmp/mission_7",
    }
    a = RequestPostHook(source_task_id=42, kind="domain_layer_check", source_ctx=source_ctx)
    _, payload = _posthook_agent_and_payload(a, source, source_ctx)

    assert payload["payload"]["required_layer"] == "domain", (
        f"expected 'domain', got {payload['payload']['required_layer']!r}"
    )


def test_payload_builder_posthook_kind():
    """Payload must carry posthook_kind='domain_layer_check'."""
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    source = {"id": 42, "mission_id": 7}
    source_ctx = {
        "produces": ["src/domain/user.py"],
        "workspace_path": "/tmp/mission_7",
    }
    a = RequestPostHook(source_task_id=42, kind="domain_layer_check", source_ctx=source_ctx)
    _, payload = _posthook_agent_and_payload(a, source, source_ctx)

    assert payload["posthook_kind"] == "domain_layer_check"
    assert payload["executor"] == "mechanical"


def test_payload_builder_target_files_from_produces():
    """target_files must be the source step's produces list."""
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    source = {"id": 42, "mission_id": 7}
    produces = ["src/domain/user.py", "src/domain/order.py"]
    source_ctx = {
        "produces": produces,
        "workspace_path": "/tmp/mission_7",
    }
    a = RequestPostHook(source_task_id=42, kind="domain_layer_check", source_ctx=source_ctx)
    _, payload = _posthook_agent_and_payload(a, source, source_ctx)

    assert payload["payload"]["target_files"] == produces


def test_payload_builder_workspace_path():
    """workspace_path must be forwarded from source_ctx."""
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    source = {"id": 42, "mission_id": 7}
    source_ctx = {
        "produces": ["src/domain/user.py"],
        "workspace_path": "/mission/workspace",
    }
    a = RequestPostHook(source_task_id=42, kind="domain_layer_check", source_ctx=source_ctx)
    _, payload = _posthook_agent_and_payload(a, source, source_ctx)

    assert payload["payload"]["workspace_path"] == "/mission/workspace"


def test_payload_builder_mission_id():
    """mission_id must be forwarded from source."""
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    source = {"id": 42, "mission_id": 99}
    source_ctx = {
        "produces": ["src/domain/user.py"],
        "workspace_path": "/tmp/m",
    }
    a = RequestPostHook(source_task_id=42, kind="domain_layer_check", source_ctx=source_ctx)
    _, payload = _posthook_agent_and_payload(a, source, source_ctx)

    assert payload["payload"]["mission_id"] == 99


# ────────────────────────────────────────────────────────────────────────────
# (d) Verdict round-trip
# ────────────────────────────────────────────────────────────────────────────

def _make_source_and_ctx(pending=None, attempts=1):
    """Build minimal source task + context dicts for verdict tests."""
    if pending is None:
        pending = ["domain_layer_check"]
    source = {
        "id": 1,
        "mission_id": 5,
        "title": "build domain",
        "worker_attempts": attempts,
        "max_worker_attempts": 15,
        "result": "",
    }
    ctx = {
        "_pending_posthooks": list(pending),
        "produces": ["src/domain/user.py"],
    }
    return source, ctx


@pytest.mark.asyncio
async def test_verdict_semgrep_crash_soft_drops():
    """When domain_layer_check's mechanical task DLQs with a crash error,
    the source is NOT cascaded — it soft-drops and advances."""
    import json as _json
    from general_beckman.apply import _posthook_dlq_cascade

    source, ctx = _make_source_and_ctx()
    # _posthook_dlq_cascade reads source from get_task and source_ctx from
    # _parse_ctx(source). We need source.context to carry the ctx with
    # _pending_posthooks so the soft-drop can strip domain_layer_check.
    source["context"] = _json.dumps(ctx)
    # source must be 'ungraded' for the DLQ handler to proceed.
    source["status"] = "ungraded"

    # The posthook task carries posthook_kind and source_task_id in context.
    posthook_task = {
        "id": 2,
        "agent_type": "mechanical",
        "context": _json.dumps({
            "posthook_kind": "domain_layer_check",
            "source_task_id": 1,
        }),
    }

    updated_tasks: list[dict] = []

    async def fake_update_task(task_id, **kwargs):
        updated_tasks.append({"id": task_id, **kwargs})

    async def fake_get_task(task_id):
        return source

    # Patch src.infra.db lazily-imported symbols + spawn advance
    with (
        patch("src.infra.db.update_task", new_callable=AsyncMock, side_effect=fake_update_task),
        patch("src.infra.db.get_task", new_callable=AsyncMock, side_effect=fake_get_task),
        patch("general_beckman.apply._spawn_workflow_advance_if_mission",
              new_callable=AsyncMock),
    ):
        await _posthook_dlq_cascade(
            task=posthook_task,
            error="semgrep: command not found",
        )

    # Source must NOT be marked failed (soft-drop)
    failed_calls = [u for u in updated_tasks if u.get("status") == "failed"]
    assert not failed_calls, (
        f"domain_layer_check semgrep crash must soft-drop, not cascade source to failed. "
        f"Got: {failed_calls}"
    )
    # Some update must have been called (completed or context-only update)
    assert updated_tasks, "Expected at least one update_task call after DLQ"


@pytest.mark.asyncio
async def test_verdict_real_findings_cascade_source():
    """When domain_layer_check returns real blocker findings, source retries."""
    from general_beckman.apply import _apply_domain_layer_check_verdict
    from general_beckman.result_router import PostHookVerdict

    source, ctx = _make_source_and_ctx()
    pending = list(ctx["_pending_posthooks"])

    findings = [
        {
            "path": "src/domain/user.py",
            "line": 5,
            "rule_id": "requests-import-in-domain",
            "message": "Domain layer must not import requests",
            "severity": "ERROR",
        }
    ]

    verdict = PostHookVerdict(
        source_task_id=1,
        kind="domain_layer_check",
        passed=False,
        raw={
            "ok": True,
            "skipped": False,
            "findings": findings,
            "blocker_count": 1,
            "warning_count": 0,
            "exit": 1,
        },
    )

    with (
        patch("src.infra.db.update_task", new_callable=AsyncMock) as mock_update,
        patch("general_beckman.apply._spawn_workflow_advance_if_mission",
              new_callable=AsyncMock),
        patch("general_beckman.apply._stamp_retry_feedback"),
    ):
        await _apply_domain_layer_check_verdict(
            source=source, ctx=ctx, pending=pending, verdict=verdict,
        )

    mock_update.assert_called_once()
    call_kwargs = mock_update.call_args[1]
    # ERROR severity >= WARNING threshold → retry (pending) or DLQ (failed)
    assert call_kwargs.get("status") in ("pending", "failed"), (
        f"Findings must trigger retry (pending) or DLQ (failed), got: {call_kwargs}"
    )
    # Must NOT be completed when findings are present
    assert call_kwargs.get("status") != "completed", (
        f"Source must not complete with blocker findings"
    )
    # Findings must be stored in context
    assert "_domain_layer_check_findings" in ctx, (
        "Findings must be stored under _domain_layer_check_findings in ctx"
    )


@pytest.mark.asyncio
async def test_verdict_no_findings_completes_source():
    """When domain_layer_check returns no findings, source advances to completed."""
    from general_beckman.apply import _apply_domain_layer_check_verdict
    from general_beckman.result_router import PostHookVerdict

    source, ctx = _make_source_and_ctx()
    pending = list(ctx["_pending_posthooks"])

    verdict = PostHookVerdict(
        source_task_id=1,
        kind="domain_layer_check",
        passed=True,
        raw={
            "ok": True,
            "skipped": False,
            "findings": [],
            "blocker_count": 0,
            "warning_count": 0,
            "exit": 0,
        },
    )

    with (
        patch("src.infra.db.update_task", new_callable=AsyncMock) as mock_update,
        patch("general_beckman.apply._spawn_workflow_advance_if_mission",
              new_callable=AsyncMock),
    ):
        await _apply_domain_layer_check_verdict(
            source=source, ctx=ctx, pending=pending, verdict=verdict,
        )

    mock_update.assert_called_once()
    call_kwargs = mock_update.call_args[1]
    assert call_kwargs.get("status") == "completed", (
        f"No findings → source should complete. Got: {call_kwargs}"
    )
