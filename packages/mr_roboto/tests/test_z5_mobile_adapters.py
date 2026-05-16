"""Z5 T3 — mr_roboto mobile build/distribution adapter tests.

Covers the four mobile verbs: ``expo_cli``, ``android_build``,
``eas_build``, ``eas_submit``.

For each verb:
  (a) a fake ``run_cmd`` returning success → asserts the structured
      result shape;
  (b) a fake ``run_cmd`` returning ``{ok:False, error:"executable not
      found"}`` → asserts a soft-skip (status ``completed`` / ``skipped``,
      never ``failed``);
plus ``eas_submit`` reversibility resolves to ``irreversible``.

``run_cmd`` is monkeypatched at the *module* level of each adapter (each
adapter does ``from mr_roboto.run_cmd import run_cmd`` at import time, so
the bound name lives on the adapter module).
"""
from __future__ import annotations

import pytest

import importlib

import mr_roboto
from mr_roboto.reversibility import get_reversibility

# NOTE: mr_roboto/__init__.py does `from mr_roboto.expo_cli import expo_cli`,
# which rebinds the `mr_roboto.expo_cli` *attribute* to the function. To
# monkeypatch the module-level `run_cmd` we need the actual submodule
# object — fetch it via importlib (returns the cached module from
# sys.modules, not the shadowing attribute).
expo_cli_mod = importlib.import_module("mr_roboto.expo_cli")
android_build_mod = importlib.import_module("mr_roboto.android_build")
eas_build_mod = importlib.import_module("mr_roboto.eas_build")
eas_submit_mod = importlib.import_module("mr_roboto.eas_submit")


# --------------------------------------------------------------------------
# Fake run_cmd factories
# --------------------------------------------------------------------------

def _ok_run_cmd(stdout: str = "", stderr: str = "", exit_code: int = 0):
    """Return an async fake run_cmd that simulates a successful subprocess."""
    async def _fake(*args, **kwargs):
        return {
            "exit": exit_code,
            "stdout_tail": stdout,
            "stderr_tail": stderr,
            "duration_s": 1.5,
            "timed_out": False,
            "ok": exit_code == 0,
        }
    return _fake


def _missing_exe_run_cmd():
    """Return an async fake run_cmd that simulates a missing executable."""
    async def _fake(*args, **kwargs):
        return {
            "exit": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 0.0,
            "timed_out": False,
            "ok": False,
            "error": "executable not found: [Errno 2] No such file or directory: 'npx'",
        }
    return _fake


def _timeout_run_cmd():
    async def _fake(*args, **kwargs):
        return {
            "exit": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 99.0,
            "timed_out": True,
            "ok": False,
        }
    return _fake


# ==========================================================================
# expo_cli
# ==========================================================================

@pytest.mark.asyncio
async def test_expo_cli_success_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(
        expo_cli_mod, "run_cmd", _ok_run_cmd(stdout="Web Bundling complete"),
    )
    res = await expo_cli_mod.expo_cli(
        mission_id=None,
        subcommand="export",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is True
    assert res["skipped"] is False
    assert res["subcommand"] == "export"
    assert res["exit"] == 0
    assert "Web Bundling complete" in res["stdout_tail"]
    assert "duration_s" in res
    assert res["error"] is None


@pytest.mark.asyncio
async def test_expo_cli_missing_cli_soft_skips(monkeypatch, tmp_path):
    monkeypatch.setattr(expo_cli_mod, "run_cmd", _missing_exe_run_cmd())
    res = await expo_cli_mod.expo_cli(
        mission_id=None,
        subcommand="doctor",
        workspace_path=str(tmp_path),
    )
    # Missing CLI is a soft-skip — never a hard failure.
    assert res["skipped"] is True
    assert res["ok"] is True
    assert res["error"] is None


@pytest.mark.asyncio
async def test_expo_cli_rejects_unknown_subcommand(tmp_path):
    res = await expo_cli_mod.expo_cli(
        mission_id=None,
        subcommand="run-the-thing",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert res["skipped"] is False
    assert "unsupported" in (res["error"] or "")


@pytest.mark.asyncio
async def test_expo_cli_nonzero_exit_is_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        expo_cli_mod, "run_cmd", _ok_run_cmd(stderr="boom", exit_code=1),
    )
    res = await expo_cli_mod.expo_cli(
        mission_id=None,
        subcommand="prebuild",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert res["skipped"] is False
    assert res["exit"] == 1


@pytest.mark.asyncio
async def test_expo_cli_dispatch_via_run(monkeypatch, tmp_path):
    monkeypatch.setattr(expo_cli_mod, "run_cmd", _ok_run_cmd(stdout="ok"))
    task = {
        "id": 1,
        "mission_id": 42,
        "payload": {
            "action": "expo_cli",
            "subcommand": "doctor",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["subcommand"] == "doctor"
    assert action.reversibility == "full"


@pytest.mark.asyncio
async def test_expo_cli_dispatch_missing_cli_not_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(expo_cli_mod, "run_cmd", _missing_exe_run_cmd())
    task = {
        "id": 1,
        "mission_id": 42,
        "payload": {
            "action": "expo_cli",
            "subcommand": "export",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["skipped"] is True


# ==========================================================================
# android_build
# ==========================================================================

@pytest.mark.asyncio
async def test_android_build_success_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(
        android_build_mod, "run_cmd", _ok_run_cmd(stdout="BUILD SUCCESSFUL"),
    )
    res = await android_build_mod.android_build(
        mission_id=None,
        action="assembleRelease",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is True
    assert res["skipped"] is False
    assert res["action"] == "assembleRelease"
    assert res["tool"] == "gradle"
    assert res["exit"] == 0
    assert "BUILD SUCCESSFUL" in res["stdout_tail"]


@pytest.mark.asyncio
async def test_android_build_adb_devices_success(monkeypatch, tmp_path):
    monkeypatch.setattr(
        android_build_mod, "run_cmd",
        _ok_run_cmd(stdout="List of devices attached\nemulator-5554\tdevice"),
    )
    res = await android_build_mod.android_build(
        mission_id=None,
        action="devices",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is True
    assert res["tool"] == "adb"
    assert "emulator-5554" in res["stdout_tail"]


@pytest.mark.asyncio
async def test_android_build_missing_cli_soft_skips(monkeypatch, tmp_path):
    monkeypatch.setattr(android_build_mod, "run_cmd", _missing_exe_run_cmd())
    res = await android_build_mod.android_build(
        mission_id=None,
        action="assembleDebug",
        workspace_path=str(tmp_path),
    )
    assert res["skipped"] is True
    assert res["ok"] is True
    assert res["error"] is None


@pytest.mark.asyncio
async def test_android_build_rejects_unknown_action(tmp_path):
    res = await android_build_mod.android_build(
        mission_id=None,
        action="nukeEverything",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert "unsupported" in (res["error"] or "")


@pytest.mark.asyncio
async def test_android_build_dispatch_via_run(monkeypatch, tmp_path):
    monkeypatch.setattr(android_build_mod, "run_cmd", _ok_run_cmd(stdout="ok"))
    task = {
        "id": 2,
        "mission_id": 42,
        "payload": {
            "action": "android_build",
            "android_action": "assembleRelease",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["tool"] == "gradle"
    assert action.reversibility == "full"


@pytest.mark.asyncio
async def test_android_build_dispatch_missing_cli_not_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(android_build_mod, "run_cmd", _missing_exe_run_cmd())
    task = {
        "id": 2,
        "mission_id": 42,
        "payload": {
            "action": "android_build",
            "android_action": "devices",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["skipped"] is True


# ==========================================================================
# eas_build
# ==========================================================================

_EAS_BUILD_STDOUT = (
    "Build details: https://expo.dev/accounts/acme/projects/app/builds/"
    "abcd1234-5678-90ab-cdef-1234567890ab\n"
    "Build finished."
)


@pytest.mark.asyncio
async def test_eas_build_success_shape_and_id_parse(monkeypatch, tmp_path):
    monkeypatch.setattr(
        eas_build_mod, "run_cmd", _ok_run_cmd(stdout=_EAS_BUILD_STDOUT),
    )
    res = await eas_build_mod.eas_build(
        mission_id=None,
        platform="all",
        profile="production",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is True
    assert res["skipped"] is False
    assert res["platform"] == "all"
    assert res["profile"] == "production"
    assert res["build_id"] == "abcd1234-5678-90ab-cdef-1234567890ab"
    assert res["build_url"] and "builds/" in res["build_url"]
    assert res["exit"] == 0


@pytest.mark.asyncio
async def test_eas_build_missing_cli_soft_skips(monkeypatch, tmp_path):
    monkeypatch.setattr(eas_build_mod, "run_cmd", _missing_exe_run_cmd())
    res = await eas_build_mod.eas_build(
        mission_id=None,
        platform="ios",
        workspace_path=str(tmp_path),
    )
    assert res["skipped"] is True
    assert res["ok"] is True
    assert res["error"] is None
    assert res["build_id"] is None


@pytest.mark.asyncio
async def test_eas_build_rejects_unknown_platform(tmp_path):
    res = await eas_build_mod.eas_build(
        mission_id=None,
        platform="windows-phone",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert "unsupported" in (res["error"] or "")


@pytest.mark.asyncio
async def test_eas_build_timeout_is_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(eas_build_mod, "run_cmd", _timeout_run_cmd())
    res = await eas_build_mod.eas_build(
        mission_id=None,
        platform="android",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert res["skipped"] is False
    assert "timed out" in (res["error"] or "")


@pytest.mark.asyncio
async def test_eas_build_dispatch_via_run(monkeypatch, tmp_path):
    monkeypatch.setattr(
        eas_build_mod, "run_cmd", _ok_run_cmd(stdout=_EAS_BUILD_STDOUT),
    )
    task = {
        "id": 3,
        "mission_id": 42,
        "payload": {
            "action": "eas_build",
            "platform": "all",
            "profile": "production",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["build_id"] == "abcd1234-5678-90ab-cdef-1234567890ab"
    assert action.reversibility == "full"


# ==========================================================================
# eas_submit
# ==========================================================================

_EAS_SUBMIT_STDOUT = (
    "Submission details: https://expo.dev/accounts/acme/projects/app/"
    "submissions/11112222-3333-4444-5555-666677778888\n"
    "Submitted to App Store Connect."
)


@pytest.mark.asyncio
async def test_eas_submit_success_shape_and_id_parse(monkeypatch, tmp_path):
    monkeypatch.setattr(
        eas_submit_mod, "run_cmd", _ok_run_cmd(stdout=_EAS_SUBMIT_STDOUT),
    )
    res = await eas_submit_mod.eas_submit(
        mission_id=None,
        platform="ios",
        latest=True,
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is True
    assert res["skipped"] is False
    assert res["platform"] == "ios"
    assert res["submission_id"] == "11112222-3333-4444-5555-666677778888"
    assert res["submission_url"] and "submissions/" in res["submission_url"]
    assert res["exit"] == 0


@pytest.mark.asyncio
async def test_eas_submit_missing_cli_soft_skips(monkeypatch, tmp_path):
    monkeypatch.setattr(eas_submit_mod, "run_cmd", _missing_exe_run_cmd())
    res = await eas_submit_mod.eas_submit(
        mission_id=None,
        platform="android",
        build_id="abcd1234-5678-90ab-cdef-1234567890ab",
        workspace_path=str(tmp_path),
    )
    assert res["skipped"] is True
    assert res["ok"] is True
    assert res["error"] is None


@pytest.mark.asyncio
async def test_eas_submit_rejects_all_platform(tmp_path):
    # `all` is valid for build but not for submit (one store at a time).
    res = await eas_submit_mod.eas_submit(
        mission_id=None,
        platform="all",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert "unsupported" in (res["error"] or "")


@pytest.mark.asyncio
async def test_eas_submit_dispatch_via_run(monkeypatch, tmp_path):
    monkeypatch.setattr(
        eas_submit_mod, "run_cmd", _ok_run_cmd(stdout=_EAS_SUBMIT_STDOUT),
    )
    task = {
        "id": 4,
        "mission_id": 42,
        "payload": {
            "action": "eas_submit",
            "platform": "ios",
            "latest": True,
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["submission_id"] == "11112222-3333-4444-5555-666677778888"


@pytest.mark.asyncio
async def test_eas_submit_dispatch_missing_cli_not_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(eas_submit_mod, "run_cmd", _missing_exe_run_cmd())
    task = {
        "id": 4,
        "mission_id": 42,
        "payload": {
            "action": "eas_submit",
            "platform": "android",
            "latest": True,
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["skipped"] is True


def test_eas_submit_reversibility_is_irreversible():
    """eas_submit pushes a binary to a store track — irreversible."""
    assert get_reversibility("eas_submit") == "irreversible"


def test_mobile_build_verbs_reversibility_is_full():
    """Build/CLI verbs produce only disposable/local artifacts — full."""
    assert get_reversibility("expo_cli") == "full"
    assert get_reversibility("android_build") == "full"
    assert get_reversibility("eas_build") == "full"
