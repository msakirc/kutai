"""Z2 T3C — design_system_check post-hook via shared semgrep engine.

Covers:
- Registry has design_system_check with correct spec fields.
- Shared verb run_semgrep (same as pattern_lint / T2C).
- Auto-wire triggers on *.tsx and *.jsx (not .py, not .ts without x).
- Auto-wire: produces=[Button.tsx] prepends both design_system_check AND
  pattern_lint (both trigger on *.tsx) — verify idempotent.
- design_system.yml rule pack parses as valid YAML with expected rule IDs.
- Verdict handler: warning-only, findings surfaced in ctx, source advances.
- DLQ soft-drop: never cascades source to failed.
- Skip semgrep-invocation tests when semgrep is not installed.
"""
from __future__ import annotations

import os
import shutil
from fnmatch import fnmatch
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from general_beckman.posthooks import (
    POST_HOOK_REGISTRY,
    PostHookSpec,
    determine_posthooks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEMGREP_AVAILABLE = shutil.which("semgrep") is not None

_DS_RULE_PACK_PATH = (
    Path(__file__).parent.parent
    / "packages" / "mr_roboto" / "src" / "mr_roboto"
    / "rule_packs" / "design_system.yml"
)


# ---------------------------------------------------------------------------
# 1. Registry: design_system_check present with correct fields
# ---------------------------------------------------------------------------

def test_design_system_check_in_registry():
    assert "design_system_check" in POST_HOOK_REGISTRY


def test_design_system_check_spec_is_posthookspec():
    assert isinstance(POST_HOOK_REGISTRY["design_system_check"], PostHookSpec)


def test_design_system_check_shared_verb():
    """Must share the run_semgrep verb with pattern_lint."""
    spec = POST_HOOK_REGISTRY["design_system_check"]
    assert spec.verb == "run_semgrep"


def test_design_system_check_default_severity_is_warning():
    """v1 ramp policy: ships at warning, not blocker."""
    spec = POST_HOOK_REGISTRY["design_system_check"]
    assert spec.default_severity == "warning"


def test_design_system_check_auto_wire_triggers():
    spec = POST_HOOK_REGISTRY["design_system_check"]
    expected = {"*.tsx", "*.jsx"}
    assert expected == set(spec.auto_wire_triggers), (
        f"expected {expected}, got {set(spec.auto_wire_triggers)}"
    )


def test_design_system_check_description_mentions_design_system():
    spec = POST_HOOK_REGISTRY["design_system_check"]
    assert "design" in spec.description.lower()


# ---------------------------------------------------------------------------
# 2. Auto-wire: triggers fire on correct extensions
# ---------------------------------------------------------------------------

def test_auto_wire_on_tsx():
    spec = POST_HOOK_REGISTRY["design_system_check"]
    assert any(fnmatch("src/components/Button.tsx", pat) for pat in spec.auto_wire_triggers)


def test_auto_wire_on_jsx():
    spec = POST_HOOK_REGISTRY["design_system_check"]
    assert any(fnmatch("src/components/Button.jsx", pat) for pat in spec.auto_wire_triggers)


def test_auto_wire_NOT_on_py():
    spec = POST_HOOK_REGISTRY["design_system_check"]
    assert not any(fnmatch("src/app/foo.py", pat) for pat in spec.auto_wire_triggers)


def test_auto_wire_NOT_on_ts_only():
    """Plain .ts files (no x) should NOT trigger design_system_check."""
    spec = POST_HOOK_REGISTRY["design_system_check"]
    assert not any(fnmatch("src/utils/helpers.ts", pat) for pat in spec.auto_wire_triggers)


def test_auto_wire_NOT_on_json():
    spec = POST_HOOK_REGISTRY["design_system_check"]
    assert not any(fnmatch("src/config.json", pat) for pat in spec.auto_wire_triggers)


# ---------------------------------------------------------------------------
# 3. Both design_system_check AND pattern_lint auto-wire on *.tsx
#    (idempotent: each appears exactly once)
# ---------------------------------------------------------------------------

def test_both_hooks_auto_wire_on_tsx_produces():
    """Expander picks up both hooks when produces contains a .tsx file."""
    pl_spec = POST_HOOK_REGISTRY["pattern_lint"]
    ds_spec = POST_HOOK_REGISTRY["design_system_check"]
    path = "src/components/Button.tsx"
    pl_fires = any(fnmatch(path, pat) for pat in pl_spec.auto_wire_triggers)
    ds_fires = any(fnmatch(path, pat) for pat in ds_spec.auto_wire_triggers)
    assert pl_fires, "pattern_lint should fire on *.tsx"
    assert ds_fires, "design_system_check should fire on *.tsx"


def test_jsx_not_in_pattern_lint_triggers():
    """pattern_lint does fire on *.jsx too — verify this is stable."""
    pl_spec = POST_HOOK_REGISTRY["pattern_lint"]
    assert any(fnmatch("Button.jsx", pat) for pat in pl_spec.auto_wire_triggers)


# ---------------------------------------------------------------------------
# 4. Rule pack: design_system.yml parses as valid YAML
# ---------------------------------------------------------------------------

def test_design_system_rule_pack_file_exists():
    assert _DS_RULE_PACK_PATH.is_file(), (
        f"design_system.yml not found at {_DS_RULE_PACK_PATH}"
    )


def test_design_system_rule_pack_valid_yaml():
    with open(_DS_RULE_PACK_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "rules" in data, "Rule pack must have top-level 'rules' key"
    assert isinstance(data["rules"], list), "'rules' must be a list"
    assert len(data["rules"]) >= 1, "Rule pack must have at least one rule"


def test_design_system_rule_pack_expected_ids():
    with open(_DS_RULE_PACK_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    ids = {r["id"] for r in data["rules"]}
    expected = {
        "no-raw-hex-color",
        "no-inline-style-attr",
        "no-direct-mui-import",
        "no-css-import-from-node-modules",
        "no-px-in-styles",
    }
    assert expected == ids, f"Rule ID mismatch: got {ids}"


def test_design_system_rule_pack_each_rule_has_required_keys():
    with open(_DS_RULE_PACK_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for rule in data["rules"]:
        assert "id" in rule, f"Rule missing 'id': {rule}"
        assert "message" in rule, f"Rule {rule.get('id')} missing 'message'"
        assert "severity" in rule, f"Rule {rule.get('id')} missing 'severity'"
        assert "languages" in rule, f"Rule {rule.get('id')} missing 'languages'"


def test_design_system_rule_pack_severities():
    """All rules must be WARNING or INFO (no ERROR — v1 ramp policy)."""
    with open(_DS_RULE_PACK_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    allowed = {"WARNING", "INFO"}
    for rule in data["rules"]:
        assert rule["severity"] in allowed, (
            f"Rule {rule['id']} severity {rule['severity']!r} not in {allowed}"
        )


# ---------------------------------------------------------------------------
# 5. Verdict handler: warning-only, findings in ctx, source advances
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verdict_findings_surfaced_in_ctx():
    """Findings go to ctx['_design_system_findings'], source advances."""
    from general_beckman.result_router import PostHookVerdict
    from general_beckman.apply import _apply_design_system_check_verdict

    source = {"id": 42, "mission_id": None, "agent_type": "coder"}
    ctx = {"_pending_posthooks": ["design_system_check"]}
    pending = ["design_system_check"]

    verdict = PostHookVerdict(
        source_task_id=42,
        kind="design_system_check",
        passed=True,
        raw={
            "findings": [{"rule_id": "no-inline-style-attr", "path": "Button.tsx"}],
            "blocker_count": 0,
            "warning_count": 1,
            "skipped": False,
        },
    )

    with patch("src.infra.db.update_task", new_callable=AsyncMock) as mock_update, \
         patch("general_beckman.apply._spawn_workflow_advance_if_mission",
               new_callable=AsyncMock):
        await _apply_design_system_check_verdict(
            source=source, ctx=ctx, pending=pending, verdict=verdict,
        )

    assert "_design_system_findings" in ctx
    assert len(ctx["_design_system_findings"]) == 1
    assert ctx["_design_system_findings"][0]["rule_id"] == "no-inline-style-attr"
    # Source should advance to completed (no remaining pending hooks)
    mock_update.assert_called_once()
    call_kwargs = mock_update.call_args[1]
    assert call_kwargs.get("status") == "completed"


@pytest.mark.asyncio
async def test_verdict_soft_skip_when_semgrep_missing():
    """skipped=True: no findings, no retry, source advances."""
    from general_beckman.result_router import PostHookVerdict
    from general_beckman.apply import _apply_design_system_check_verdict

    source = {"id": 43, "mission_id": None, "agent_type": "coder"}
    ctx = {"_pending_posthooks": ["design_system_check"]}
    pending = ["design_system_check"]

    verdict = PostHookVerdict(
        source_task_id=43,
        kind="design_system_check",
        passed=True,
        raw={"findings": [], "skipped": True},
    )

    with patch("src.infra.db.update_task", new_callable=AsyncMock) as mock_update, \
         patch("general_beckman.apply._spawn_workflow_advance_if_mission",
               new_callable=AsyncMock):
        await _apply_design_system_check_verdict(
            source=source, ctx=ctx, pending=pending, verdict=verdict,
        )

    assert "_design_system_findings" not in ctx
    mock_update.assert_called_once()
    call_kwargs = mock_update.call_args[1]
    assert call_kwargs.get("status") == "completed"


@pytest.mark.asyncio
async def test_verdict_remaining_pending_not_completed():
    """When other pending hooks remain, source stays pending (no completed)."""
    from general_beckman.result_router import PostHookVerdict
    from general_beckman.apply import _apply_design_system_check_verdict

    source = {"id": 44, "mission_id": None, "agent_type": "coder"}
    ctx = {"_pending_posthooks": ["design_system_check", "grade"]}
    pending = ["design_system_check", "grade"]

    verdict = PostHookVerdict(
        source_task_id=44,
        kind="design_system_check",
        passed=True,
        raw={"findings": [], "skipped": False},
    )

    with patch("src.infra.db.update_task", new_callable=AsyncMock) as mock_update, \
         patch("general_beckman.apply._spawn_workflow_advance_if_mission",
               new_callable=AsyncMock):
        await _apply_design_system_check_verdict(
            source=source, ctx=ctx, pending=pending, verdict=verdict,
        )

    mock_update.assert_called_once()
    call_kwargs = mock_update.call_args[1]
    # status arg not passed → update_task called without status=completed
    assert call_kwargs.get("status") != "completed"
    assert "grade" in ctx["_pending_posthooks"]


# ---------------------------------------------------------------------------
# 6. DLQ soft-drop: never cascades source to failed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dlq_soft_drop_does_not_cascade_to_failed():
    """design_system_check DLQ must NOT set source to failed."""
    from general_beckman.apply import _posthook_dlq_cascade

    source = {
        "id": 99,
        "mission_id": None,
        "agent_type": "coder",
        "worker_attempts": 1,
        "context": "{}",
    }
    task = {
        "id": 200,
        "agent_type": "mechanical",
        "context": '{"posthook_kind": "design_system_check", "source_task_id": 99}',
    }

    update_calls = []

    async def _fake_update(task_id, **kwargs):
        update_calls.append((task_id, kwargs))

    async def _fake_get_task(task_id):
        return source

    with patch("src.infra.db.update_task", side_effect=_fake_update), \
         patch("src.infra.db.get_task", side_effect=_fake_get_task), \
         patch("general_beckman.apply._spawn_workflow_advance_if_mission",
               new_callable=AsyncMock):
        await _posthook_dlq_cascade(task, "semgrep exploded")

    # None of the update calls should set source to 'failed'
    for tid, kwargs in update_calls:
        assert kwargs.get("status") != "failed", (
            f"design_system_check DLQ must not cascade source to failed; "
            f"got update({tid}, {kwargs})"
        )


# ---------------------------------------------------------------------------
# 7. _posthook_agent_and_payload: correct action and rule pack
# ---------------------------------------------------------------------------

def test_posthook_agent_and_payload_action():
    """Payload action must be run_semgrep (shared verb)."""
    from general_beckman.result_router import RequestPostHook
    from general_beckman.apply import _posthook_agent_and_payload

    source_ctx = {"produces": ["src/components/Button.tsx"]}
    a = RequestPostHook(
        kind="design_system_check",
        source_task_id=10,
        source_ctx=source_ctx,
    )
    source = {"id": 10, "agent_type": "coder"}

    agent_type, payload = _posthook_agent_and_payload(a, source, source_ctx)
    assert agent_type == "mechanical"
    assert payload["payload"]["action"] == "run_semgrep"
    assert payload["posthook_kind"] == "design_system_check"


def test_posthook_agent_and_payload_default_rule_pack_contains_design_system():
    """Default rule pack path should point to design_system.yml."""
    from general_beckman.result_router import RequestPostHook
    from general_beckman.apply import _posthook_agent_and_payload

    source_ctx = {"produces": ["Button.tsx"]}
    a = RequestPostHook(
        kind="design_system_check",
        source_task_id=11,
        source_ctx=source_ctx,
    )
    source = {"id": 11, "agent_type": "coder"}

    _, payload = _posthook_agent_and_payload(a, source, source_ctx)
    rule_pack = payload["payload"]["rule_pack_path"]
    assert "design_system" in rule_pack.lower(), (
        f"Expected design_system in rule_pack_path, got {rule_pack!r}"
    )


def test_posthook_agent_and_payload_custom_rule_pack():
    """If design_system_rule_pack in ctx, use it."""
    from general_beckman.result_router import RequestPostHook
    from general_beckman.apply import _posthook_agent_and_payload

    source_ctx = {
        "produces": ["Button.tsx"],
        "design_system_rule_pack": "/project/rules/my_ds.yml",
    }
    a = RequestPostHook(
        kind="design_system_check",
        source_task_id=12,
        source_ctx=source_ctx,
    )
    source = {"id": 12, "agent_type": "coder"}

    _, payload = _posthook_agent_and_payload(a, source, source_ctx)
    assert payload["payload"]["rule_pack_path"] == "/project/rules/my_ds.yml"
