"""Wrap the Expo CLI — Z5 T3 mobile adapter.

Mechanical executor. No LLM. Shells out (via :func:`run_cmd`) to the Expo
CLI to run ``prebuild`` / ``export`` / ``doctor`` against a mission
workspace.

Invocation
----------
Expo ships as a project-local dev dependency, so the canonical invocation
is ``npx expo <subcommand>``. ``run_cmd`` resolves ``npx`` off PATH; if
``npx`` (Node) is not installed at all the run soft-skips for free —
``run_cmd`` returns ``{ok:False, error:"executable not found: ..."}`` which
this verb maps to ``skipped`` rather than ``failed``.

Host-OS note
------------
Every Expo subcommand wrapped here runs fine on Windows. ``prebuild``
generates the native ``android/`` + ``ios/`` projects; ``export`` produces
a static web/JS bundle; ``doctor`` validates the project. None of these
require macOS-local iOS tooling — actual iOS *builds* go through EAS (see
``eas_build``), never through this verb.

Reversibility: ``full`` — all three subcommands write only local files
inside the workspace (or, for ``doctor``, nothing), all git-reversible.
"""

from __future__ import annotations

from typing import Any

from yazbunu import get_logger
from mr_roboto.run_cmd import run_cmd

logger = get_logger("mr_roboto.expo_cli")

# Subcommands this adapter is allowed to drive. Anything else is rejected
# with a structured error rather than passed blindly to the CLI.
_ALLOWED_SUBCOMMANDS = frozenset({"prebuild", "export", "doctor"})

_TAIL_CHARS = 4000  # keep last ~4 KB of each stream in the structured result

DEFAULT_TIMEOUT_S = 600.0


def _tail(text: str | None) -> str:
    text = text or ""
    return text[-_TAIL_CHARS:] if len(text) > _TAIL_CHARS else text


def _is_missing_exe(raw: dict[str, Any]) -> bool:
    """True when run_cmd could not find the executable to spawn."""
    err = (raw.get("error") or "").lower()
    return "executable not found" in err or "not found" in err


async def expo_cli(
    mission_id: int | None,
    subcommand: str,
    workspace_path: str | None = None,
    extra_args: list[str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run an Expo CLI subcommand and return a structured result.

    Parameters
    ----------
    mission_id:
        Forwarded to ``run_cmd`` for workspace resolution. May be None when
        ``workspace_path`` is supplied (tests).
    subcommand:
        One of ``prebuild`` / ``export`` / ``doctor``.
    workspace_path:
        Explicit workspace root. Optional when ``mission_id`` is set.
    extra_args:
        Additional argv tokens appended after the subcommand
        (e.g. ``["--platform", "android"]``).
    timeout_s:
        Hard cap passed to ``run_cmd``. Default 600 s.

    Returns
    -------
    dict with keys:

    ``ok``
        True when the CLI ran and exited 0. False on a non-zero exit or an
        internal error.
    ``skipped``
        True when the Expo CLI / ``npx`` is not installed. Callers MUST
        treat this as a soft pass, never a blocker.
    ``subcommand``
        Echo of the requested subcommand.
    ``exit``
        Raw CLI exit code. -1 on spawn failure / timeout.
    ``stdout_tail`` / ``stderr_tail``
        Tail of each stream.
    ``duration_s``
        Wall time of the subprocess.
    ``error``
        Human-readable error string when ``ok=False`` and not skipped.
    """
    sub = (subcommand or "").strip()
    if sub not in _ALLOWED_SUBCOMMANDS:
        return {
            "ok": False,
            "skipped": False,
            "subcommand": sub,
            "exit": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 0.0,
            "error": (
                f"unsupported expo subcommand {sub!r}; "
                f"allowed: {sorted(_ALLOWED_SUBCOMMANDS)}"
            ),
        }

    # npx expo <sub> [extra...]. --yes keeps npx non-interactive (no
    # "install package?" prompt when the binary is fetched on demand).
    cmd = ["npx", "--yes", "expo", sub, *(list(extra_args or []))]

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
        logger.warning("expo CLI / npx not installed — expo_cli skipped", subcommand=sub)
        return {
            "ok": True,
            "skipped": True,
            "subcommand": sub,
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
            "subcommand": sub,
            "exit": exit_code,
            "stdout_tail": _tail(raw.get("stdout_tail")),
            "stderr_tail": _tail(raw.get("stderr_tail")),
            "duration_s": float(raw.get("duration_s", 0.0)),
            "error": f"expo {sub} timed out after {timeout_s}s",
        }

    ok = exit_code == 0
    return {
        "ok": ok,
        "skipped": False,
        "subcommand": sub,
        "exit": exit_code,
        "stdout_tail": _tail(raw.get("stdout_tail")),
        "stderr_tail": _tail(raw.get("stderr_tail")),
        "duration_s": float(raw.get("duration_s", 0.0)),
        "error": None if ok else f"expo {sub} exited {exit_code}",
    }
