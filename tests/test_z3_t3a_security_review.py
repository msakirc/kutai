"""Z3 T3A — security_review composite post-hook.

Covers:
- security_review kind registered with correct cost_band=moderate, severity=blocker
- Callable auto_wire_triggers: returns [] when qa_dial=off
- Callable auto_wire_triggers: returns expected globs when qa_dial=standard
- resolve_triggers() on PostHookSpec works for both list and callable forms
- run_bandit parses bandit JSON correctly (mock subprocess)
- run_npm_audit parses npm audit JSON v2 correctly (mock subprocess)
- run_npm_audit parses npm audit JSON v1 (advisories) format
- run_npm_audit soft-skips when no package.json
- security_review composite aggregates findings from all three tools (mock)
- security_review returns verdict=fail when any blocker found
- security_review returns verdict=pass when no blockers
- Soft-skip behaviour when tools not installed (mock FileNotFoundError)
- security.yml rule pack valid YAML with expected rule IDs
- Apply verdict: pass → source advances, findings in ctx
- Apply verdict: fail → source retried with feedback
- DLQ: blocker kind cascades source to failed
"""
from __future__ import annotations

import json
from fnmatch import fnmatch
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import importlib
import sys
import pytest

# 3 tests below assume the agent's apply.py shape (its own _apply_security_review_verdict
# path + dict-context for dials).  Canonical T1A puts dials in MissionDialContext and the
# generic _apply_simple_blocker_verdict helper handles all 4 T3 kinds.  These tests are
# kept for the agent's intent but skip-marked — coverage is provided by the generic
# helper's tests in tests/test_z3_t3c_*.
_SKIP_NON_CANONICAL = pytest.mark.skip(
    reason="Z3 T3A: test asserts agent-specific apply path; canonical uses _apply_simple_blocker_verdict via T1A MissionDialContext"
)

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers to get the actual submodules (not the re-exported functions
# that shadow them on the mr_roboto package namespace after __init__ imports).
# ---------------------------------------------------------------------------

def _get_submodule(dotted_name: str):
    """Return the actual module object, bypassing __init__ re-exports."""
    importlib.import_module(dotted_name)
    return sys.modules[dotted_name]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SECURITY_RULE_PACK_PATH = (
    Path(__file__).parent.parent
    / "packages" / "mr_roboto" / "src" / "mr_roboto"
    / "rule_packs" / "security.yml"
)

# ---------------------------------------------------------------------------
# 1. Registry: security_review registered with correct fields
# ---------------------------------------------------------------------------

def test_security_review_in_registry():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    assert "security_review" in POST_HOOK_REGISTRY


def test_security_review_cost_band_moderate():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["security_review"]
    assert spec.cost_band == "moderate"


def test_security_review_default_severity_blocker():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["security_review"]
    assert spec.default_severity == "blocker"


def test_security_review_verb():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["security_review"]
    assert spec.verb == "security_review"


# ---------------------------------------------------------------------------
# 2. Callable auto_wire_triggers: qa_dial gate
# ---------------------------------------------------------------------------

def test_callable_auto_wire_returns_empty_when_qa_dial_off():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["security_review"]
    assert callable(spec.auto_wire_triggers)
    result = spec.resolve_triggers({"qa_dial": "off"})
    assert result == []


def test_callable_auto_wire_returns_globs_when_qa_dial_standard():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["security_review"]
    result = spec.resolve_triggers({"qa_dial": "standard"})
    assert isinstance(result, list)
    assert len(result) > 0
    # Should include common source-code extensions
    assert "*.py" in result
    assert "*.ts" in result
    assert "package.json" in result


def test_callable_auto_wire_returns_globs_when_no_qa_dial():
    """Missing qa_dial (default): should fire."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["security_review"]
    result = spec.resolve_triggers({})
    assert len(result) > 0


def test_callable_auto_wire_fires_on_py_file():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["security_review"]
    triggers = spec.resolve_triggers({})
    assert any(fnmatch("src/app/foo.py", pat) for pat in triggers)


def test_callable_auto_wire_fires_on_package_json():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["security_review"]
    triggers = spec.resolve_triggers({})
    assert any(fnmatch("package.json", pat) for pat in triggers)


# ---------------------------------------------------------------------------
# 3. PostHookSpec.resolve_triggers for plain list (back-compat)
# ---------------------------------------------------------------------------

def test_resolve_triggers_plain_list():
    from general_beckman.posthooks import PostHookSpec
    spec = PostHookSpec(kind="test_kind", verb="test_verb", auto_wire_triggers=["*.py", "*.ts"])
    assert spec.resolve_triggers({}) == ["*.py", "*.ts"]


def test_resolve_triggers_callable():
    from general_beckman.posthooks import PostHookSpec
    spec = PostHookSpec(
        kind="test_kind",
        verb="test_verb",
        auto_wire_triggers=lambda ctx: [] if ctx.get("off") else ["*.py"],
    )
    assert spec.resolve_triggers({"off": True}) == []
    assert spec.resolve_triggers({}) == ["*.py"]


# ---------------------------------------------------------------------------
# 4. security.yml rule pack
# ---------------------------------------------------------------------------

def test_security_rule_pack_exists():
    assert _SECURITY_RULE_PACK_PATH.is_file(), (
        f"security.yml not found at {_SECURITY_RULE_PACK_PATH}"
    )


def test_security_rule_pack_valid_yaml():
    with open(_SECURITY_RULE_PACK_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "rules" in data
    assert isinstance(data["rules"], list)
    assert len(data["rules"]) >= 10


def test_security_rule_pack_has_expected_ids():
    with open(_SECURITY_RULE_PACK_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    ids = {r["id"] for r in data["rules"]}
    expected_subset = {
        "sql-injection-format-string",
        "command-injection",
        "weak-crypto-md5",
        "hardcoded-aws-access-key",
        "eval-misuse",
    }
    missing = expected_subset - ids
    assert not missing, f"Missing rule IDs: {missing}"


def test_security_rule_pack_each_rule_has_required_keys():
    with open(_SECURITY_RULE_PACK_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for rule in data["rules"]:
        assert "id" in rule
        assert "message" in rule
        assert "severity" in rule
        assert "languages" in rule


def test_security_rule_pack_severities_are_valid():
    with open(_SECURITY_RULE_PACK_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    allowed = {"ERROR", "WARNING", "INFO"}
    for rule in data["rules"]:
        assert rule["severity"] in allowed, (
            f"Rule {rule['id']} has invalid severity {rule['severity']!r}"
        )


def test_security_rule_pack_has_error_rules():
    """Must have at least some ERROR-severity rules for CVSS>=7 patterns."""
    with open(_SECURITY_RULE_PACK_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    error_rules = [r for r in data["rules"] if r["severity"] == "ERROR"]
    assert len(error_rules) >= 5, (
        f"Expected >= 5 ERROR rules, got {len(error_rules)}"
    )


# ---------------------------------------------------------------------------
# 5. run_bandit: JSON parsing
# ---------------------------------------------------------------------------

_BANDIT_JSON = json.dumps({
    "results": [
        {
            "filename": "src/app/shell.py",
            "line_number": 42,
            "issue_text": "Use of subprocess with shell=True",
            "issue_severity": "HIGH",
            "test_id": "B602",
        },
        {
            "filename": "src/app/auth.py",
            "line_number": 10,
            "issue_text": "Use of MD5",
            "issue_severity": "MEDIUM",
            "test_id": "B303",
        },
    ],
    "metrics": {},
})


@pytest.mark.asyncio
async def test_run_bandit_parses_json():
    """run_bandit correctly maps HIGH→blocker, MEDIUM→warning."""
    _bandit_mod = _get_submodule("mr_roboto.run_bandit")
    from mr_roboto.run_bandit import run_bandit

    mock_run_cmd_result = {
        "ok": True,
        "exit": 1,
        "timed_out": False,
        "error": None,
        "stdout_tail": _BANDIT_JSON,
        "stderr_tail": "",
        "duration_s": 1.2,
    }

    with patch.object(_bandit_mod, "_locate_bandit", return_value="/usr/bin/bandit"), \
         patch.object(_bandit_mod, "run_cmd", new_callable=AsyncMock, return_value=mock_run_cmd_result):
        result = await run_bandit(target_files=["src/app/shell.py", "src/app/auth.py"])

    assert result["ok"] is True
    assert result["skipped"] is False
    findings = result["findings"]
    assert len(findings) == 2

    high_finding = next(f for f in findings if f["rule_id"] == "B602")
    assert high_finding["severity"] == "blocker"
    assert high_finding["source"] == "bandit"
    assert high_finding["file"] == "src/app/shell.py"
    assert high_finding["line"] == 42

    med_finding = next(f for f in findings if f["rule_id"] == "B303")
    assert med_finding["severity"] == "warning"

    assert result["blocker_count"] == 1
    assert result["warning_count"] == 1


@pytest.mark.asyncio
async def test_run_bandit_soft_skip_when_not_installed():
    """FileNotFoundError from _locate_bandit → skipped=True."""
    _bandit_mod = _get_submodule("mr_roboto.run_bandit")
    from mr_roboto.run_bandit import run_bandit

    with patch.object(_bandit_mod, "_locate_bandit", side_effect=FileNotFoundError("bandit not found")):
        result = await run_bandit()

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["findings"] == []


@pytest.mark.asyncio
async def test_run_bandit_no_python_files_skips():
    """Non-.py files → no scan needed → soft-skip."""
    _bandit_mod = _get_submodule("mr_roboto.run_bandit")
    from mr_roboto.run_bandit import run_bandit

    with patch.object(_bandit_mod, "_locate_bandit", return_value="/usr/bin/bandit"):
        result = await run_bandit(target_files=["src/components/Button.tsx", "package.json"])

    assert result["skipped"] is True


# ---------------------------------------------------------------------------
# 6. run_npm_audit: JSON parsing
# ---------------------------------------------------------------------------

_NPM_AUDIT_JSON_V2 = json.dumps({
    "vulnerabilities": {
        "lodash": {
            "severity": "critical",
            "via": [{"title": "Prototype Pollution", "name": "lodash"}],
            "effects": [],
        },
        "axios": {
            "severity": "moderate",
            "via": [{"title": "SSRF via redirect", "name": "axios"}],
            "effects": [],
        },
    },
    "metadata": {"vulnerabilities": {"total": 2}},
})

_NPM_AUDIT_JSON_V1 = json.dumps({
    "advisories": {
        "1234": {
            "module_name": "lodash",
            "severity": "high",
            "title": "Prototype Pollution in lodash",
        },
    },
    "metadata": {},
})


@pytest.mark.asyncio
async def test_run_npm_audit_parses_v2_json():
    """npm audit JSON v2: critical→blocker, moderate→warning."""
    _npm_mod = _get_submodule("mr_roboto.run_npm_audit")
    from mr_roboto.run_npm_audit import run_npm_audit

    mock_run_cmd_result = {
        "ok": True,
        "exit": 1,
        "timed_out": False,
        "error": None,
        "stdout_tail": _NPM_AUDIT_JSON_V2,
        "stderr_tail": "",
        "duration_s": 2.0,
    }

    with patch.object(_npm_mod, "_locate_npm", return_value="/usr/bin/npm"), \
         patch.object(_npm_mod, "_has_package_json", return_value=True), \
         patch.object(_npm_mod, "run_cmd", new_callable=AsyncMock, return_value=mock_run_cmd_result):
        result = await run_npm_audit(workspace_path="/workspace")

    assert result["ok"] is True
    assert result["skipped"] is False
    findings = result["findings"]
    assert len(findings) == 2

    lodash = next(f for f in findings if "lodash" in f["rule_id"])
    assert lodash["severity"] == "blocker"
    assert lodash["source"] == "npm_audit"

    axios = next(f for f in findings if "axios" in f["rule_id"])
    assert axios["severity"] == "warning"

    assert result["blocker_count"] == 1
    assert result["warning_count"] == 1


@pytest.mark.asyncio
async def test_run_npm_audit_parses_v1_json():
    """npm audit JSON v1 (advisories): high→blocker."""
    _npm_mod = _get_submodule("mr_roboto.run_npm_audit")
    from mr_roboto.run_npm_audit import run_npm_audit

    mock_run_cmd_result = {
        "ok": True,
        "exit": 1,
        "timed_out": False,
        "error": None,
        "stdout_tail": _NPM_AUDIT_JSON_V1,
        "stderr_tail": "",
        "duration_s": 1.5,
    }

    with patch.object(_npm_mod, "_locate_npm", return_value="/usr/bin/npm"), \
         patch.object(_npm_mod, "_has_package_json", return_value=True), \
         patch.object(_npm_mod, "run_cmd", new_callable=AsyncMock, return_value=mock_run_cmd_result):
        result = await run_npm_audit(workspace_path="/workspace")

    findings = result["findings"]
    # v1 may produce findings from advisories
    lodash_findings = [f for f in findings if "lodash" in f["rule_id"]]
    assert len(lodash_findings) >= 1
    assert lodash_findings[0]["severity"] == "blocker"


@pytest.mark.asyncio
async def test_run_npm_audit_soft_skip_no_package_json():
    """No package.json → skipped=True, no subprocess."""
    _npm_mod = _get_submodule("mr_roboto.run_npm_audit")
    from mr_roboto.run_npm_audit import run_npm_audit

    with patch.object(_npm_mod, "_has_package_json", return_value=False):
        result = await run_npm_audit(workspace_path="/empty-workspace")

    assert result["skipped"] is True
    assert result["findings"] == []


@pytest.mark.asyncio
async def test_run_npm_audit_soft_skip_npm_not_installed():
    """npm not on PATH → skipped=True."""
    _npm_mod = _get_submodule("mr_roboto.run_npm_audit")
    from mr_roboto.run_npm_audit import run_npm_audit

    with patch.object(_npm_mod, "_has_package_json", return_value=True), \
         patch.object(_npm_mod, "_locate_npm", side_effect=FileNotFoundError("npm not found")):
        result = await run_npm_audit()

    assert result["skipped"] is True
    assert result["findings"] == []


# ---------------------------------------------------------------------------
# 7. security_review composite: aggregation and verdict
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_security_review_aggregates_findings_from_all_tools():
    """Composite aggregates semgrep + bandit + npm_audit findings."""
    _sg_mod = _get_submodule("mr_roboto.run_semgrep")
    _bd_mod = _get_submodule("mr_roboto.run_bandit")
    _npm_mod = _get_submodule("mr_roboto.run_npm_audit")
    from mr_roboto.security_review import security_review

    semgrep_result = {
        "ok": True, "skipped": False,
        "findings": [
            {"severity": "blocker", "path": "app.py", "line": 5,
             "message": "eval misuse", "rule_id": "eval-misuse",
             "semgrep_severity": "ERROR"},
        ],
    }
    bandit_result = {
        "ok": True, "skipped": False,
        "findings": [
            {"severity": "warning", "file": "auth.py", "line": 10,
             "why": "weak hash", "source": "bandit", "rule_id": "B303"},
        ],
        "blocker_count": 0, "warning_count": 1,
    }
    npm_result = {
        "ok": True, "skipped": False,
        "findings": [
            {"severity": "blocker", "file": "package.json", "line": 0,
             "why": "lodash: Prototype Pollution", "source": "npm_audit",
             "rule_id": "npm/lodash"},
        ],
        "blocker_count": 1, "warning_count": 0,
    }

    # Patch the underlying module-level functions that security_review imports lazily.
    with patch.object(_sg_mod, "run_semgrep", new_callable=AsyncMock, return_value=semgrep_result), \
         patch.object(_bd_mod, "run_bandit", new_callable=AsyncMock, return_value=bandit_result), \
         patch.object(_npm_mod, "run_npm_audit", new_callable=AsyncMock, return_value=npm_result):
        result = await security_review(
            target_files=["app.py", "auth.py", "package.json"],
            workspace_path="/workspace",
        )

    assert result["verdict"] == "fail"
    assert result["blocker_count"] >= 1


@pytest.mark.asyncio
async def test_security_review_verdict_fail_when_any_blocker():
    """Any blocker finding → verdict=fail."""
    _sg_mod = _get_submodule("mr_roboto.run_semgrep")
    _bd_mod = _get_submodule("mr_roboto.run_bandit")
    _npm_mod = _get_submodule("mr_roboto.run_npm_audit")
    from mr_roboto.security_review import security_review

    semgrep_result = {
        "ok": True, "skipped": False,
        "findings": [
            {"severity": "blocker", "path": "app.py", "line": 1,
             "message": "sql injection", "rule_id": "sql-injection-format-string",
             "semgrep_severity": "ERROR"},
        ],
    }
    bandit_result = {"ok": True, "skipped": True, "findings": [], "blocker_count": 0, "warning_count": 0}
    npm_result = {"ok": True, "skipped": True, "findings": [], "blocker_count": 0, "warning_count": 0}

    with patch.object(_sg_mod, "run_semgrep", new_callable=AsyncMock, return_value=semgrep_result), \
         patch.object(_bd_mod, "run_bandit", new_callable=AsyncMock, return_value=bandit_result), \
         patch.object(_npm_mod, "run_npm_audit", new_callable=AsyncMock, return_value=npm_result):
        result = await security_review()

    assert result["verdict"] == "fail"
    assert result["blocker_count"] >= 1


@pytest.mark.asyncio
async def test_security_review_verdict_pass_when_no_blockers():
    """No blockers → verdict=pass even with warnings."""
    _sg_mod = _get_submodule("mr_roboto.run_semgrep")
    _bd_mod = _get_submodule("mr_roboto.run_bandit")
    _npm_mod = _get_submodule("mr_roboto.run_npm_audit")
    from mr_roboto.security_review import security_review

    semgrep_result = {
        "ok": True, "skipped": False,
        "findings": [
            {"severity": "warning", "path": "app.py", "line": 1,
             "message": "sha1 usage", "rule_id": "weak-crypto-sha1",
             "semgrep_severity": "WARNING"},
        ],
    }
    bandit_result = {"ok": True, "skipped": True, "findings": [], "blocker_count": 0, "warning_count": 0}
    npm_result = {"ok": True, "skipped": True, "findings": [], "blocker_count": 0, "warning_count": 0}

    with patch.object(_sg_mod, "run_semgrep", new_callable=AsyncMock, return_value=semgrep_result), \
         patch.object(_bd_mod, "run_bandit", new_callable=AsyncMock, return_value=bandit_result), \
         patch.object(_npm_mod, "run_npm_audit", new_callable=AsyncMock, return_value=npm_result):
        result = await security_review()

    assert result["verdict"] == "pass"
    assert result["blocker_count"] == 0
    assert result["warning_count"] >= 1


@pytest.mark.asyncio
async def test_security_review_all_tools_skipped_is_pass():
    """All tools skipped (not installed) → verdict=pass (no findings)."""
    _sg_mod = _get_submodule("mr_roboto.run_semgrep")
    _bd_mod = _get_submodule("mr_roboto.run_bandit")
    _npm_mod = _get_submodule("mr_roboto.run_npm_audit")
    from mr_roboto.security_review import security_review

    skipped = {"ok": True, "skipped": True, "findings": [], "blocker_count": 0, "warning_count": 0}

    with patch.object(_sg_mod, "run_semgrep", new_callable=AsyncMock, return_value=skipped), \
         patch.object(_bd_mod, "run_bandit", new_callable=AsyncMock, return_value=skipped), \
         patch.object(_npm_mod, "run_npm_audit", new_callable=AsyncMock, return_value=skipped):
        result = await security_review()

    assert result["verdict"] == "pass"
    assert result["findings"] == []
    assert result["tools_used"] == []


# ---------------------------------------------------------------------------
# 8. Apply verdict: pass → source advances
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_apply_security_review_verdict_pass_source_advances():
    from general_beckman.result_router import PostHookVerdict
    from general_beckman.apply import _apply_security_review_verdict

    source = {"id": 50, "mission_id": None, "agent_type": "coder"}
    ctx = {"_pending_posthooks": ["security_review"]}
    pending = ["security_review"]

    verdict = PostHookVerdict(
        source_task_id=50,
        kind="security_review",
        passed=True,
        raw={
            "verdict": "pass",
            "findings": [],
            "tools_used": ["semgrep"],
            "blocker_count": 0,
            "warning_count": 0,
        },
    )

    with patch("src.infra.db.update_task", new_callable=AsyncMock) as mock_update, \
         patch("general_beckman.apply._spawn_workflow_advance_if_mission",
               new_callable=AsyncMock):
        await _apply_security_review_verdict(
            source=source, ctx=ctx, pending=pending, verdict=verdict,
        )

    mock_update.assert_called_once()
    call_kwargs = mock_update.call_args[1]
    assert call_kwargs.get("status") == "completed"


@pytest.mark.asyncio
@_SKIP_NON_CANONICAL
async def test_apply_security_review_verdict_pass_stores_findings():
    """Non-empty findings stored in ctx even on pass (warnings only)."""
    from general_beckman.result_router import PostHookVerdict
    from general_beckman.apply import _apply_security_review_verdict

    source = {"id": 51, "mission_id": None, "agent_type": "coder"}
    ctx = {"_pending_posthooks": ["security_review"]}
    pending = ["security_review"]

    verdict = PostHookVerdict(
        source_task_id=51,
        kind="security_review",
        passed=True,
        raw={
            "verdict": "pass",
            "findings": [{"severity": "warning", "file": "app.py", "line": 1, "why": "sha1", "source": "semgrep", "rule_id": "weak-crypto-sha1"}],
            "tools_used": ["semgrep"],
            "blocker_count": 0,
            "warning_count": 1,
        },
    )

    with patch("src.infra.db.update_task", new_callable=AsyncMock), \
         patch("general_beckman.apply._spawn_workflow_advance_if_mission",
               new_callable=AsyncMock):
        await _apply_security_review_verdict(
            source=source, ctx=ctx, pending=pending, verdict=verdict,
        )

    assert "_security_review_findings" in ctx
    assert len(ctx["_security_review_findings"]) == 1


# ---------------------------------------------------------------------------
# 9. Apply verdict: fail → source retried with feedback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_apply_security_review_verdict_fail_retries_source():
    from general_beckman.result_router import PostHookVerdict
    from general_beckman.apply import _apply_security_review_verdict

    source = {
        "id": 52, "mission_id": None, "agent_type": "coder",
        "worker_attempts": 1, "max_worker_attempts": 15,
        "result": "",
    }
    ctx = {"_pending_posthooks": ["security_review"]}
    pending = ["security_review"]

    verdict = PostHookVerdict(
        source_task_id=52,
        kind="security_review",
        passed=False,
        raw={
            "verdict": "fail",
            "findings": [
                {"severity": "blocker", "file": "app.py", "line": 5,
                 "why": "sql injection", "source": "semgrep", "rule_id": "sql-injection-format-string"},
            ],
            "tools_used": ["semgrep"],
            "blocker_count": 1,
            "warning_count": 0,
        },
    )

    with patch("src.infra.db.update_task", new_callable=AsyncMock) as mock_update, \
         patch("general_beckman.apply._spawn_workflow_advance_if_mission",
               new_callable=AsyncMock):
        await _apply_security_review_verdict(
            source=source, ctx=ctx, pending=pending, verdict=verdict,
        )

    mock_update.assert_called_once()
    call_kwargs = mock_update.call_args[1]
    assert call_kwargs.get("status") == "pending"
    assert "security_review" in (call_kwargs.get("error") or "")
    assert call_kwargs.get("error_category") == "quality"


# ---------------------------------------------------------------------------
# 10. DLQ: blocker kind cascades source to failed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@_SKIP_NON_CANONICAL
async def test_dlq_security_review_cascades_source_to_failed():
    from general_beckman.apply import _posthook_dlq_cascade

    source = {
        "id": 99,
        "mission_id": None,
        "agent_type": "coder",
        "worker_attempts": 1,
        "status": "ungraded",
        "context": "{}",
    }
    task = {
        "id": 201,
        "agent_type": "mechanical",
        "context": '{"posthook_kind": "security_review", "source_task_id": 99}',
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
        await _posthook_dlq_cascade(task, "security_review exploded")

    # Source must be cascaded to failed for blocker kind
    source_updates = [(tid, kw) for tid, kw in update_calls if tid == 99]
    assert any(kw.get("status") == "failed" for _, kw in source_updates), (
        f"security_review DLQ must cascade source to failed; got {source_updates}"
    )


# ---------------------------------------------------------------------------
# 11. _posthook_agent_and_payload for security_review
# ---------------------------------------------------------------------------

def test_posthook_agent_and_payload_security_review():
    from general_beckman.result_router import RequestPostHook
    from general_beckman.apply import _posthook_agent_and_payload

    source_ctx = {
        "produces": ["src/app/main.py", "package.json"],
        "workspace_path": "/workspace",
    }
    a = RequestPostHook(
        kind="security_review",
        source_task_id=20,
        source_ctx=source_ctx,
    )
    source = {"id": 20, "agent_type": "coder"}

    agent_type, payload = _posthook_agent_and_payload(a, source, source_ctx)
    assert agent_type == "mechanical"
    assert payload["payload"]["action"] == "security_review"
    assert payload["posthook_kind"] == "security_review"
    assert "src/app/main.py" in payload["payload"]["target_files"]


# ---------------------------------------------------------------------------
# 12. Expander: callable triggers fire on .py file when qa_dial not off
# ---------------------------------------------------------------------------

def test_expander_security_review_auto_wires_on_py_produces():
    """Auto-wire: produces=[main.py] should prepend security_review."""
    from src.workflows.engine.expander import _auto_wire_posthooks

    context = {
        "produces": ["src/app/main.py"],
        "post_hooks": [],
    }
    _auto_wire_posthooks(context)
    assert "security_review" in context.get("post_hooks", [])


@_SKIP_NON_CANONICAL
def test_expander_security_review_does_not_wire_when_qa_dial_off():
    """qa_dial=off: security_review should NOT be auto-wired."""
    from src.workflows.engine.expander import _auto_wire_posthooks

    context = {
        "produces": ["src/app/main.py"],
        "post_hooks": [],
        "qa_dial": "off",
    }
    _auto_wire_posthooks(context)
    assert "security_review" not in context.get("post_hooks", [])
