"""Tests for run_semgrep Windows platform-skip behaviour.

Covers the key invariant from the Z2/Z3 bug fix:
  - When semgrep is absent on Windows, skipped_platform=True (not silent green)
  - When semgrep is absent on non-Windows, skipped_platform=False (soft-miss)
  - When Docker fallback succeeds, skipped=False and results are parsed normally
  - skipped_platform=True triggers WARNING log + _<kind>_platform_skip ctx flag
    in _apply_semgrep_blocker_verdict

Patching note
-------------
mr_roboto/__init__.py re-exports ``run_semgrep`` as a name, so
``patch("mr_roboto.run_semgrep._is_windows")`` fails — mock resolves
``mr_roboto.run_semgrep`` to the re-exported *function*, not the submodule.
We import the submodule explicitly first and use patch.object on the module
object retrieved from sys.modules.
"""
from __future__ import annotations

import json
import logging
import sys
import pytest
from unittest.mock import AsyncMock, patch


def _rs():
    """Return the run_semgrep *module* (not the re-exported function)."""
    import mr_roboto.run_semgrep  # noqa: F401 — ensure in sys.modules
    return sys.modules["mr_roboto.run_semgrep"]


# ---------------------------------------------------------------------------
# run_semgrep unit tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_windows_no_semgrep_no_docker_returns_platform_skip(tmp_path):
    """On Windows without semgrep or Docker, skipped_platform must be True."""
    rs_mod = _rs()

    with (
        patch.object(rs_mod, "_is_windows", return_value=True),
        patch.object(rs_mod, "_locate_semgrep", side_effect=FileNotFoundError),
        patch.object(rs_mod, "_run_semgrep_via_docker", new=AsyncMock(return_value=None)),
    ):
        result = await rs_mod.run_semgrep(mission_id=None, workspace_path=str(tmp_path))

    assert result["skipped"] is True, "must be marked skipped"
    assert result["skipped_platform"] is True, (
        "skipped_platform must be True on Windows without Docker — gate did not run"
    )
    assert result["ok"] is True, "ok stays True so action handler doesn't fail the task"
    assert result["findings"] == []
    assert result["blocker_count"] == 0


@pytest.mark.asyncio
async def test_non_windows_no_semgrep_returns_platform_skip_false(tmp_path):
    """On non-Windows without semgrep, skipped_platform must be False (soft-miss)."""
    rs_mod = _rs()

    with (
        patch.object(rs_mod, "_is_windows", return_value=False),
        patch.object(rs_mod, "_locate_semgrep", side_effect=FileNotFoundError),
        patch.object(rs_mod, "_run_semgrep_via_docker", new=AsyncMock(return_value=None)),
    ):
        result = await rs_mod.run_semgrep(mission_id=None, workspace_path=str(tmp_path))

    assert result["skipped"] is True
    assert result["skipped_platform"] is False, (
        "skipped_platform must be False on Linux — semgrep just not installed"
    )
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_windows_docker_fallback_success_returns_findings(tmp_path):
    """When Docker succeeds on Windows, results are parsed and skipped=False."""
    rs_mod = _rs()

    findings_json = json.dumps({
        "results": [
            {
                "check_id": "forbidden.eval",
                "path": "src/foo.py",
                "start": {"line": 10},
                "extra": {
                    "message": "eval is forbidden",
                    "severity": "ERROR",
                },
            }
        ]
    })
    docker_raw = {
        "exit": 1,
        "ok": True,
        "timed_out": False,
        "error": None,
        "stdout_tail": findings_json,
        "stderr_tail": "",
        "duration_s": 2.5,
    }

    with (
        patch.object(rs_mod, "_is_windows", return_value=True),
        patch.object(rs_mod, "_locate_semgrep", side_effect=FileNotFoundError),
        patch.object(rs_mod, "_run_semgrep_via_docker", new=AsyncMock(return_value=docker_raw)),
    ):
        result = await rs_mod.run_semgrep(mission_id=None, workspace_path=str(tmp_path))

    assert result["skipped"] is False
    assert result["skipped_platform"] is False
    assert result["ok"] is True
    assert len(result["findings"]) == 1
    assert result["findings"][0]["rule_id"] == "forbidden.eval"
    assert result["findings"][0]["severity"] == "blocker"  # ERROR → blocker
    assert result["blocker_count"] == 1


@pytest.mark.asyncio
async def test_windows_docker_timeout_returns_ok_false(tmp_path):
    """Docker timeout produces ok=False, not a silent skip."""
    rs_mod = _rs()

    docker_raw = {
        "exit": -1,
        "ok": False,
        "timed_out": True,
        "error": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "duration_s": 180.0,
    }

    with (
        patch.object(rs_mod, "_is_windows", return_value=True),
        patch.object(rs_mod, "_locate_semgrep", side_effect=FileNotFoundError),
        patch.object(rs_mod, "_run_semgrep_via_docker", new=AsyncMock(return_value=docker_raw)),
    ):
        result = await rs_mod.run_semgrep(mission_id=None, workspace_path=str(tmp_path))

    assert result["ok"] is False
    assert result["skipped"] is False
    assert "timed out" in result["error"]


@pytest.mark.asyncio
async def test_native_semgrep_run_includes_skipped_platform_false(tmp_path):
    """A successful native semgrep run always sets skipped_platform=False."""
    rs_mod = _rs()

    clean_json = json.dumps({"results": []})
    run_cmd_result = {
        "exit": 0,
        "ok": True,
        "timed_out": False,
        "error": None,
        "stdout_tail": clean_json,
        "stderr_tail": "",
        "duration_s": 0.5,
    }

    with (
        patch.object(rs_mod, "_is_windows", return_value=False),
        patch.object(rs_mod, "_locate_semgrep", return_value="/usr/bin/semgrep"),
        patch.object(rs_mod, "run_cmd", new=AsyncMock(return_value=run_cmd_result)),
    ):
        result = await rs_mod.run_semgrep(mission_id=None, workspace_path=str(tmp_path))

    assert result["ok"] is True
    assert result["skipped"] is False
    assert result["skipped_platform"] is False
    assert result["findings"] == []


@pytest.mark.asyncio
async def test_windows_platform_skip_emits_warning_log(tmp_path, caplog):
    """Platform skip on Windows must emit a WARNING — not debug/info."""
    rs_mod = _rs()

    with (
        patch.object(rs_mod, "_is_windows", return_value=True),
        patch.object(rs_mod, "_locate_semgrep", side_effect=FileNotFoundError),
        patch.object(rs_mod, "_run_semgrep_via_docker", new=AsyncMock(return_value=None)),
        caplog.at_level(logging.WARNING, logger="mr_roboto.run_semgrep"),
    ):
        await rs_mod.run_semgrep(mission_id=None, workspace_path=str(tmp_path))

    platform_warnings = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING
        and (
            "DID NOT RUN" in r.message
            or "platform" in r.message.lower()
            or "gate" in r.message.lower()
        )
    ]
    assert platform_warnings, (
        f"Expected WARNING about platform skip, got: {[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# apply.py verdict handler tests
# ---------------------------------------------------------------------------

def _make_verdict(skipped: bool, skipped_platform: bool, findings: list | None = None):
    """Build a PostHookVerdict for testing verdict handlers.

    PostHookVerdict(source_task_id, kind, passed, raw) — matches the
    actual dataclass/namedtuple signature in apply.py.
    """
    from general_beckman.apply import PostHookVerdict
    raw = {
        "ok": True,
        "skipped": skipped,
        "skipped_platform": skipped_platform,
        "findings": findings or [],
        "blocker_count": sum(1 for f in (findings or []) if f.get("severity") == "blocker"),
        "warning_count": 0,
        "exit": -1,
        "stdout_tail": "",
        "stderr_tail": "",
        "duration_s": 0.0,
        "error": None,
    }
    return PostHookVerdict(
        source_task_id=42,
        kind="pattern_lint",
        passed=True,
        raw=raw,
    )


@pytest.mark.asyncio
async def test_blocker_verdict_platform_skip_stamps_ctx(monkeypatch):
    """_apply_semgrep_blocker_verdict must stamp _<kind>_platform_skip=True in ctx."""
    import general_beckman.apply as apply_mod

    saved_calls: list = []

    async def fake_update_task(task_id, **kwargs):
        saved_calls.append(("update_task", task_id, kwargs))

    async def fake_spawn(*a, **kw):
        pass

    async def fake_get_task(tid):
        return None

    monkeypatch.setattr(apply_mod, "_spawn_workflow_advance_if_mission", fake_spawn)
    monkeypatch.setattr("src.infra.db.update_task", fake_update_task, raising=False)
    monkeypatch.setattr("src.infra.db.get_task", fake_get_task, raising=False)

    ctx: dict = {"_pending_posthooks": ["pattern_lint"]}
    verdict = _make_verdict(skipped=True, skipped_platform=True)
    await apply_mod._apply_semgrep_blocker_verdict(
        kind="pattern_lint",
        findings_ctx_key="_pattern_lint_findings",
        dlq_reason_ctx_key="_pattern_lint_dlq_reason",
        blocker_threshold="blocker",
        source={"id": 1, "mission_id": None},
        ctx=ctx,
        pending=["pattern_lint"],
        verdict=verdict,
    )

    assert ctx.get("_pattern_lint_platform_skip") is True, (
        "ctx must record _pattern_lint_platform_skip=True for platform skips"
    )
    assert len(saved_calls) > 0, "update_task should have been called"


@pytest.mark.asyncio
async def test_blocker_verdict_non_platform_skip_does_not_stamp_ctx(monkeypatch):
    """Non-platform skip (Linux, semgrep not installed) must NOT stamp platform_skip."""
    import general_beckman.apply as apply_mod

    async def fake_update_task(task_id, **kwargs):
        pass

    async def fake_spawn(*a, **kw):
        pass

    async def fake_get_task(tid):
        return None

    monkeypatch.setattr(apply_mod, "_spawn_workflow_advance_if_mission", fake_spawn)
    monkeypatch.setattr("src.infra.db.update_task", fake_update_task, raising=False)
    monkeypatch.setattr("src.infra.db.get_task", fake_get_task, raising=False)

    ctx: dict = {"_pending_posthooks": ["pattern_lint"]}
    verdict = _make_verdict(skipped=True, skipped_platform=False)
    await apply_mod._apply_semgrep_blocker_verdict(
        kind="pattern_lint",
        findings_ctx_key="_pattern_lint_findings",
        dlq_reason_ctx_key="_pattern_lint_dlq_reason",
        blocker_threshold="blocker",
        source={"id": 1, "mission_id": None},
        ctx=ctx,
        pending=["pattern_lint"],
        verdict=verdict,
    )

    assert "_pattern_lint_platform_skip" not in ctx, (
        "Non-platform skip must NOT stamp platform_skip flag in ctx"
    )


@pytest.mark.asyncio
async def test_blocker_verdict_platform_skip_warning_log(monkeypatch, caplog):
    """Platform skip must log at WARNING level in _apply_semgrep_blocker_verdict."""
    import general_beckman.apply as apply_mod

    async def fake_update_task(task_id, **kwargs):
        pass

    async def fake_spawn(*a, **kw):
        pass

    async def fake_get_task(tid):
        return None

    monkeypatch.setattr(apply_mod, "_spawn_workflow_advance_if_mission", fake_spawn)
    monkeypatch.setattr("src.infra.db.update_task", fake_update_task, raising=False)
    monkeypatch.setattr("src.infra.db.get_task", fake_get_task, raising=False)

    with caplog.at_level(logging.WARNING):
        ctx: dict = {"_pending_posthooks": ["pattern_lint"]}
        verdict = _make_verdict(skipped=True, skipped_platform=True)
        await apply_mod._apply_semgrep_blocker_verdict(
            kind="pattern_lint",
            findings_ctx_key="_pattern_lint_findings",
            dlq_reason_ctx_key="_pattern_lint_dlq_reason",
            blocker_threshold="blocker",
            source={"id": 1, "mission_id": None},
            ctx=ctx,
            pending=["pattern_lint"],
            verdict=verdict,
        )

    warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "DID NOT RUN" in m or "platform skip" in m.lower() or "NOT enforced" in m
        for m in warning_msgs
    ), (
        f"Expected WARNING containing 'DID NOT RUN'/'platform skip'/'NOT enforced', "
        f"got: {warning_msgs}"
    )
