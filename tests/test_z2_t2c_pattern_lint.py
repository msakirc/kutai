"""Z2 T2C — pattern_lint post-hook via shared semgrep engine.

Covers:
- Registry has pattern_lint with correct spec fields.
- Auto-wire triggers on py/ts/tsx/js/jsx produces paths.
- run_semgrep verb: soft-skip when semgrep not installed.
- run_semgrep verb: finding emitted on time.sleep in test path (semgrep required).
- Rule pack forbidden.yml parses as valid YAML.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from general_beckman.posthooks import (
    POST_HOOK_REGISTRY,
    PostHookSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEMGREP_AVAILABLE = shutil.which("semgrep") is not None


# ---------------------------------------------------------------------------
# 1. Registry: pattern_lint present with correct fields
# ---------------------------------------------------------------------------

def test_pattern_lint_in_registry():
    assert "pattern_lint" in POST_HOOK_REGISTRY


def test_pattern_lint_spec_is_posthookspec():
    assert isinstance(POST_HOOK_REGISTRY["pattern_lint"], PostHookSpec)


def test_pattern_lint_verb():
    assert POST_HOOK_REGISTRY["pattern_lint"].verb == "run_semgrep"


def test_pattern_lint_default_severity_is_blocker():
    """Z2 T6: promoted to blocker (2026-05-12). Soft-skip preserved when semgrep absent."""
    assert POST_HOOK_REGISTRY["pattern_lint"].default_severity == "blocker"


def test_pattern_lint_auto_wire_triggers():
    spec = POST_HOOK_REGISTRY["pattern_lint"]
    expected = {"*.py", "*.ts", "*.tsx", "*.js", "*.jsx"}
    assert expected == set(spec.auto_wire_triggers), (
        f"expected {expected}, got {set(spec.auto_wire_triggers)}"
    )


# ---------------------------------------------------------------------------
# 2. Auto-wire: pattern_lint fires on matching produces
# ---------------------------------------------------------------------------

def test_auto_wire_on_python_produces():
    """Expander should auto-wire pattern_lint when produces contains a .py file."""
    from fnmatch import fnmatch
    spec = POST_HOOK_REGISTRY["pattern_lint"]
    assert any(
        fnmatch("src/app/foo.py", pat) for pat in spec.auto_wire_triggers
    )


def test_auto_wire_on_ts_produces():
    from fnmatch import fnmatch
    spec = POST_HOOK_REGISTRY["pattern_lint"]
    assert any(
        fnmatch("src/app/bar.ts", pat) for pat in spec.auto_wire_triggers
    )


def test_auto_wire_not_on_json():
    """JSON files should NOT trigger pattern_lint auto-wire."""
    from fnmatch import fnmatch
    spec = POST_HOOK_REGISTRY["pattern_lint"]
    assert not any(
        fnmatch("src/config.json", pat) for pat in spec.auto_wire_triggers
    )


# ---------------------------------------------------------------------------
# 3. Rule pack: parses as valid YAML
# ---------------------------------------------------------------------------

def test_rule_pack_is_valid_yaml():
    from mr_roboto.run_semgrep import DEFAULT_RULE_PACK
    assert os.path.isfile(DEFAULT_RULE_PACK), (
        f"Rule pack not found: {DEFAULT_RULE_PACK}"
    )
    with open(DEFAULT_RULE_PACK, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "rules" in data, "Rule pack must have top-level 'rules' key"
    assert isinstance(data["rules"], list), "'rules' must be a list"
    assert len(data["rules"]) >= 1, "Rule pack must have at least one rule"


def test_rule_pack_rule_ids():
    from mr_roboto.run_semgrep import DEFAULT_RULE_PACK
    with open(DEFAULT_RULE_PACK, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    ids = {r["id"] for r in data["rules"]}
    expected = {
        "no-console-log",
        "no-time-sleep-in-tests",
        "no-assert-true",
        "no-bare-exec",
        "no-bare-eval",
    }
    assert expected == ids, f"Rule IDs mismatch: got {ids}"


def test_rule_pack_each_rule_has_required_keys():
    from mr_roboto.run_semgrep import DEFAULT_RULE_PACK
    with open(DEFAULT_RULE_PACK, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for rule in data["rules"]:
        assert "id" in rule, f"Rule missing 'id': {rule}"
        assert "message" in rule, f"Rule {rule.get('id')} missing 'message'"
        assert "severity" in rule, f"Rule {rule.get('id')} missing 'severity'"
        assert "languages" in rule, f"Rule {rule.get('id')} missing 'languages'"


# ---------------------------------------------------------------------------
# 4. run_semgrep verb: soft-skip when semgrep missing
# ---------------------------------------------------------------------------

def _get_semgrep_module():
    """Return the mr_roboto.run_semgrep submodule (not the function of the same name)."""
    import sys
    import importlib
    importlib.import_module("mr_roboto.run_semgrep")
    return sys.modules["mr_roboto.run_semgrep"]


@pytest.mark.asyncio
async def test_run_semgrep_soft_skip_when_missing():
    """When semgrep is not on PATH, verb returns ok=True, skipped=True."""
    _mod = _get_semgrep_module()
    from mr_roboto.run_semgrep import run_semgrep

    orig = _mod._locate_semgrep

    def _raise():
        raise FileNotFoundError("semgrep not found")

    _mod._locate_semgrep = _raise
    try:
        result = await run_semgrep(
            mission_id=None,
            target_files=["some_file.py"],
            workspace_path="/tmp",
        )
    finally:
        _mod._locate_semgrep = orig

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["findings"] == []
    assert result["blocker_count"] == 0
    assert result.get("error") is None


@pytest.mark.asyncio
async def test_run_semgrep_exit_127_soft_skip():
    """Exit 127 (command not found via shell) also yields soft-skip."""
    _mod = _get_semgrep_module()
    from mr_roboto.run_semgrep import run_semgrep

    mock_raw = {
        "ok": False,
        "exit": 127,
        "stdout_tail": "",
        "stderr_tail": "semgrep: command not found",
        "duration_s": 0.0,
        "timed_out": False,
        "error": None,
    }

    orig_locate = _mod._locate_semgrep
    orig_run_cmd = _mod.run_cmd
    _mod._locate_semgrep = lambda: "/usr/bin/semgrep"

    async def _fake_run_cmd(**kw):
        return mock_raw

    _mod.run_cmd = _fake_run_cmd
    try:
        result = await run_semgrep(
            mission_id=None,
            target_files=["some_file.py"],
            workspace_path="/tmp",
        )
    finally:
        _mod._locate_semgrep = orig_locate
        _mod.run_cmd = orig_run_cmd

    assert result["ok"] is True
    assert result["skipped"] is True


# ---------------------------------------------------------------------------
# 5. run_semgrep verb: finding emitted on actual code (semgrep required)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _SEMGREP_AVAILABLE, reason="semgrep not installed")
@pytest.mark.asyncio
async def test_run_semgrep_finds_time_sleep_in_test():
    """no-time-sleep-in-tests rule fires on a test file with time.sleep()."""
    from mr_roboto.run_semgrep import run_semgrep, DEFAULT_RULE_PACK

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test_bad.py")
        with open(test_file, "w") as f:
            f.write("import time\n\ndef test_foo():\n    time.sleep(1)\n    assert True\n")

        result = await run_semgrep(
            mission_id=None,
            target_files=[test_file],
            rule_pack_path=DEFAULT_RULE_PACK,
            workspace_path=tmpdir,
        )

    assert result["ok"] is True
    assert result["skipped"] is False
    findings = result["findings"]
    assert len(findings) >= 1, f"Expected at least 1 finding, got: {findings}"
    rule_ids = {f["rule_id"].split(".")[-1] if "." in f["rule_id"] else f["rule_id"]
                for f in findings}
    # Accept either "no-time-sleep-in-tests" or a dotted variant
    matched = any("no-time-sleep-in-tests" in rid for rid in rule_ids)
    assert matched, f"Expected no-time-sleep-in-tests finding, got rule_ids={rule_ids}"


@pytest.mark.skipif(not _SEMGREP_AVAILABLE, reason="semgrep not installed")
@pytest.mark.asyncio
async def test_run_semgrep_clean_file_no_findings():
    """A file with no forbidden patterns yields empty findings."""
    from mr_roboto.run_semgrep import run_semgrep, DEFAULT_RULE_PACK

    with tempfile.TemporaryDirectory() as tmpdir:
        clean_file = os.path.join(tmpdir, "clean.py")
        with open(clean_file, "w") as f:
            f.write("def add(a, b):\n    return a + b\n")

        result = await run_semgrep(
            mission_id=None,
            target_files=[clean_file],
            rule_pack_path=DEFAULT_RULE_PACK,
            workspace_path=tmpdir,
        )

    assert result["ok"] is True
    assert result["findings"] == []


# ---------------------------------------------------------------------------
# 6. run_semgrep JSON-parse fail is non-fatal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_semgrep_json_parse_fail_returns_ok():
    """Malformed JSON stdout yields ok=True with empty findings (not a crash)."""
    _mod = _get_semgrep_module()
    from mr_roboto.run_semgrep import run_semgrep

    mock_raw = {
        "ok": True,
        "exit": 0,
        "stdout_tail": "not-json-at-all",
        "stderr_tail": "",
        "duration_s": 0.1,
        "timed_out": False,
        "error": None,
    }

    orig_locate = _mod._locate_semgrep
    orig_run_cmd = _mod.run_cmd
    _mod._locate_semgrep = lambda: "/usr/bin/semgrep"

    async def _fake_run_cmd(**kw):
        return mock_raw

    _mod.run_cmd = _fake_run_cmd
    try:
        result = await run_semgrep(
            mission_id=None,
            target_files=["clean.py"],
            workspace_path="/tmp",
        )
    finally:
        _mod._locate_semgrep = orig_locate
        _mod.run_cmd = orig_run_cmd

    assert result["ok"] is True
    assert result["findings"] == []
    assert result["blocker_count"] == 0
