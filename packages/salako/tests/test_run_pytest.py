"""Tests for salako.run_pytest — the pytest runner verb.

These exercise the parse logic in process (parsing both the JSON report and
the stdout summary fallback). The end-to-end integration test runs an actual
pytest subprocess against a tmpdir test file to confirm the pipe works on
the real platform.
"""
from __future__ import annotations

import json
import sys
import textwrap

import pytest

import salako
from salako.run_pytest import (
    _parse_json_report,
    _parse_stdout_summary,
    run_pytest,
)


def test_parse_json_report_happy(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({
        "summary": {"passed": 3, "failed": 1, "errors": 0, "skipped": 1, "total": 5}
    }))
    out = _parse_json_report(str(p))
    assert out == {"passed": 3, "failed": 1, "errors": 0, "skipped": 1, "total": 5}


def test_parse_json_report_missing_returns_none(tmp_path):
    assert _parse_json_report(str(tmp_path / "missing.json")) is None


def test_parse_json_report_corrupt_returns_none(tmp_path):
    p = tmp_path / "r.json"
    p.write_text("not json")
    assert _parse_json_report(str(p)) is None


def test_parse_stdout_summary_passed_only():
    out = _parse_stdout_summary("=== 5 passed in 0.42s ===")
    assert out["passed"] == 5
    assert out["failed"] == 0
    assert out["total"] == 5


def test_parse_stdout_summary_mixed():
    out = _parse_stdout_summary("=== 2 failed, 3 passed, 1 skipped in 1.0s ===")
    assert out["passed"] == 3
    assert out["failed"] == 2
    assert out["skipped"] == 1
    assert out["total"] == 6


def test_parse_stdout_summary_no_summary():
    out = _parse_stdout_summary("nothing here")
    assert out["total"] == 0
    assert out["passed"] == 0


@pytest.mark.asyncio
async def test_run_pytest_passes_on_real_test(tmp_path):
    """End-to-end: write a passing pytest file, run it, expect ok=True."""
    test_file = tmp_path / "test_demo.py"
    test_file.write_text(textwrap.dedent("""
        def test_truth():
            assert 1 + 1 == 2
    """).strip())

    res = await run_pytest(
        mission_id=None,
        target=["test_demo.py"],
        workspace_path=str(tmp_path),
        timeout_s=60.0,
    )
    assert res["exit"] == 0, res
    assert res["ok"] is True, res
    assert res["passed"] == 1
    assert res["failed"] == 0
    assert res["total"] == 1


@pytest.mark.asyncio
async def test_run_pytest_fails_on_failing_test(tmp_path):
    test_file = tmp_path / "test_bad.py"
    test_file.write_text(textwrap.dedent("""
        def test_lie():
            assert 1 == 2
    """).strip())

    res = await run_pytest(
        mission_id=None,
        target=["test_bad.py"],
        workspace_path=str(tmp_path),
        timeout_s=60.0,
    )
    assert res["ok"] is False
    assert res["failed"] >= 1


@pytest.mark.asyncio
async def test_run_pytest_zero_collected_is_failure(tmp_path):
    """A green exit on an empty test dir must NOT count as ok — silent skip
    is the regression class this verb exists to catch."""
    res = await run_pytest(
        mission_id=None,
        target=["."],
        workspace_path=str(tmp_path),
        timeout_s=60.0,
    )
    # pytest exits non-zero on no-collected by default (exit 5), but even if
    # the platform somehow returns 0, total==0 must keep ok==False.
    assert res["ok"] is False
    assert res["total"] == 0


@pytest.mark.asyncio
async def test_run_pytest_via_dispatcher(tmp_path):
    """Confirm salako.run() routes the run_pytest action correctly."""
    test_file = tmp_path / "test_ok.py"
    test_file.write_text("def test_ok():\n    assert True\n")

    action = await salako.run({
        "mission_id": None,
        "payload": {
            "action": "run_pytest",
            "target": ["test_ok.py"],
            "workspace_path": str(tmp_path),
            "timeout_s": 60.0,
        },
    })
    assert action.status == "completed", action
    assert action.result["ok"] is True
