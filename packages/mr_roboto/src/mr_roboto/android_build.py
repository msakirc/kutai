"""Wrap local Android tooling — Z5 T3 mobile adapter.

Mechanical executor. No LLM. Shells out (via :func:`run_cmd`) to the
project-local Gradle wrapper and to ``adb``. Unlike iOS, the Android
toolchain runs natively on Windows, so this adapter drives a *local*
build/install path — no cloud required.

Actions
-------
``assembleRelease`` / ``assembleDebug``
    Run the Gradle wrapper (``gradlew``) to build an APK. The
    ``variant`` payload field is the short name (``release`` / ``debug``)
    and the Gradle task is derived as ``assemble<Variant>``.
``install``
    ``adb install -r`` an APK onto a connected device/emulator.
``devices``
    ``adb devices`` — list attached devices/emulators.

Invocation
----------
The Gradle wrapper is workspace-local: on Windows it is ``gradlew.bat``,
elsewhere ``./gradlew``. ``run_cmd`` resolves it relative to the workspace
cwd. ``adb`` is resolved off PATH. If either binary is absent, ``run_cmd``
returns ``{ok:False, error:"executable not found: ..."}`` and this verb
soft-skips (``skipped:True``) rather than hard-failing.

Reversibility: ``full`` — a Gradle build writes only to ``build/``
(git-ignored, disposable); ``adb install -r`` replaces an app on a dev
device and ``adb devices`` is read-only. No durable real-world side effect.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from src.infra.logging_config import get_logger
from mr_roboto.run_cmd import run_cmd

logger = get_logger("mr_roboto.android_build")

# Build actions → Gradle task name. ``install`` / ``devices`` are adb.
_GRADLE_ACTIONS = {
    "assembleRelease": "assembleRelease",
    "assembleDebug": "assembleDebug",
}
_ADB_ACTIONS = frozenset({"install", "devices"})
_ALLOWED_ACTIONS = frozenset(_GRADLE_ACTIONS) | _ADB_ACTIONS

_TAIL_CHARS = 4000
DEFAULT_TIMEOUT_S = 600.0


def _tail(text: str | None) -> str:
    text = text or ""
    return text[-_TAIL_CHARS:] if len(text) > _TAIL_CHARS else text


def _is_missing_exe(raw: dict[str, Any]) -> bool:
    err = (raw.get("error") or "").lower()
    return "executable not found" in err or "not found" in err


def _gradle_wrapper_cmd() -> list[str]:
    """Argv prefix for the workspace-local Gradle wrapper.

    On Windows the wrapper is ``gradlew.bat``; ``run_cmd`` resolves it
    relative to the workspace cwd. Elsewhere it is ``./gradlew``.
    """
    if sys.platform.startswith("win"):
        return [os.path.join(".", "gradlew.bat")]
    return [os.path.join(".", "gradlew")]


async def android_build(
    mission_id: int | None,
    action: str,
    workspace_path: str | None = None,
    variant: str = "release",
    extra_args: list[str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run a local Android build/install/list action.

    Parameters
    ----------
    mission_id:
        Forwarded to ``run_cmd`` for workspace resolution.
    action:
        One of ``assembleRelease`` / ``assembleDebug`` / ``install`` /
        ``devices``.
    workspace_path:
        Explicit workspace root. The Gradle wrapper is expected at its
        root; ``adb`` need only be on PATH.
    variant:
        Short build variant name. Only consulted when ``action`` is a bare
        ``assemble`` request without an explicit task — the explicit
        ``assembleRelease`` / ``assembleDebug`` actions take precedence.
    extra_args:
        Extra argv appended after the gradle task / adb subcommand
        (e.g. an APK path for ``install``).
    timeout_s:
        Hard cap. Default 600 s.

    Returns
    -------
    dict with keys ``ok, skipped, action, tool, exit, stdout_tail,
    stderr_tail, duration_s, error``.
    """
    act = (action or "").strip()
    if act not in _ALLOWED_ACTIONS:
        return {
            "ok": False,
            "skipped": False,
            "action": act,
            "tool": "",
            "exit": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 0.0,
            "error": (
                f"unsupported android action {act!r}; "
                f"allowed: {sorted(_ALLOWED_ACTIONS)}"
            ),
        }

    extra = list(extra_args or [])
    if act in _GRADLE_ACTIONS:
        tool = "gradle"
        gradle_task = _GRADLE_ACTIONS[act]
        cmd = [*_gradle_wrapper_cmd(), gradle_task, *extra]
    elif act == "install":
        tool = "adb"
        cmd = ["adb", "install", "-r", *extra]
    else:  # devices
        tool = "adb"
        cmd = ["adb", "devices", *extra]

    raw = await run_cmd(
        mission_id=mission_id,
        cmd=cmd,
        cwd=None,
        timeout_s=timeout_s,
        require_exit_zero=False,
        workspace_path=workspace_path,
        reversibility_intent="full",
    )

    if _is_missing_exe(raw):
        logger.warning(
            "android tooling not installed — android_build skipped",
            action=act, tool=tool,
        )
        return {
            "ok": True,
            "skipped": True,
            "action": act,
            "tool": tool,
            "exit": int(raw.get("exit", -1)),
            "stdout_tail": _tail(raw.get("stdout_tail")),
            "stderr_tail": _tail(raw.get("stderr_tail")),
            "duration_s": float(raw.get("duration_s", 0.0)),
            "error": None,
        }

    exit_code = int(raw.get("exit", -1))
    timed_out = bool(raw.get("timed_out"))

    if timed_out:
        return {
            "ok": False,
            "skipped": False,
            "action": act,
            "tool": tool,
            "exit": exit_code,
            "stdout_tail": _tail(raw.get("stdout_tail")),
            "stderr_tail": _tail(raw.get("stderr_tail")),
            "duration_s": float(raw.get("duration_s", 0.0)),
            "error": f"{tool} {act} timed out after {timeout_s}s",
        }

    ok = exit_code == 0
    return {
        "ok": ok,
        "skipped": False,
        "action": act,
        "tool": tool,
        "exit": exit_code,
        "stdout_tail": _tail(raw.get("stdout_tail")),
        "stderr_tail": _tail(raw.get("stderr_tail")),
        "duration_s": float(raw.get("duration_s", 0.0)),
        "error": None if ok else f"{tool} {act} exited {exit_code}",
    }
