"""Wrap ``fastlane <lane>`` — Z5 T3b mobile adapter.

Mechanical executor. No LLM. Shells out (via :func:`run_cmd`) to
`Fastlane <https://fastlane.tools>`_, the open-source mobile build /
release automation tool. Fastlane is the **free-first** signing + upload
path the founder picked over EAS Submit (see
``docs/i2p-evolution/05-build-mobile-track-v2.md``).

Lanes
-----
A "lane" is a named Fastlane workflow. This adapter recognises four:

- ``build``  — compile the app (``gym`` / ``gradle``). Produces a
  disposable local artifact → **reversible** (``full``).
- ``match``  — sync code-signing certificates + provisioning profiles
  from the encrypted ``match`` git repo and install them into the CI
  keychain. Local keychain mutation, re-runnable → **reversible**
  (``full``).
- ``pilot``  — upload a build to **TestFlight**. Real testers receive it;
  Apple ingests the binary. Cannot be cleanly un-done → **irreversible**.
- ``supply`` — upload to the Google **Play internal** track. Same story
  → **irreversible**.

Per-lane reversibility
----------------------
``run()`` resolves the reversibility tag *before* dispatch, so this
module exposes :func:`lane_reversibility` and the dispatcher injects the
lane-derived value into ``payload["reversibility_override"]`` before the
tag is computed. The base ``VERB_REVERSIBILITY["fastlane"]`` entry is the
conservative default (``irreversible``) for the case where no lane is
resolvable.

Invocation
----------
``fastlane <lane> [extra_args...]``. Always run inside the workspace.
If ``fastlane`` is not installed the run soft-skips for free via
``run_cmd``'s missing-exe handling.
"""

from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger
from mr_roboto.run_cmd import run_cmd

logger = get_logger("mr_roboto.fastlane_run")

# Recognised lanes → reversibility tag.
#   build / match  → local, disposable, re-runnable        → full
#   pilot / supply → binary lands on a real store track    → irreversible
_LANE_REVERSIBILITY: dict[str, str] = {
    "build": "full",
    "match": "full",
    "pilot": "irreversible",
    "supply": "irreversible",
}

ALLOWED_LANES = frozenset(_LANE_REVERSIBILITY)

_TAIL_CHARS = 6000
DEFAULT_TIMEOUT_S = 1200.0  # builds + uploads; slower than a check


def _tail(text: str | None) -> str:
    text = text or ""
    return text[-_TAIL_CHARS:] if len(text) > _TAIL_CHARS else text


def _is_missing_exe(raw: dict[str, Any]) -> bool:
    err = (raw.get("error") or "").lower()
    return "executable not found" in err or "not found" in err


def lane_reversibility(lane: str | None) -> str:
    """Return the reversibility tag for a Fastlane lane.

    ``build`` / ``match`` → ``full``; ``pilot`` / ``supply`` →
    ``irreversible``. Unknown / missing lane → ``irreversible`` (the
    conservative default — an unrecognised lane might publish).
    """
    return _LANE_REVERSIBILITY.get((lane or "").strip().lower(), "irreversible")


async def fastlane(
    mission_id: int | None,
    lane: str,
    workspace_path: str | None = None,
    extra_args: list[str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run a Fastlane lane and return a structured result.

    Parameters
    ----------
    mission_id:
        Forwarded to ``run_cmd`` for workspace resolution.
    lane:
        One of ``build``, ``match``, ``pilot``, ``supply``.
    workspace_path:
        Explicit workspace root (the project containing ``fastlane/``).
    extra_args:
        Extra argv appended after the lane name (e.g. ``["--verbose"]``).
    timeout_s:
        Hard cap. Default 1200 s.

    Returns
    -------
    dict with keys ``ok, skipped, lane, reversibility, exit, stdout_tail,
    stderr_tail, duration_s, error``.
    """
    ln = (lane or "").strip().lower()
    if ln not in ALLOWED_LANES:
        return {
            "ok": False,
            "skipped": False,
            "lane": ln,
            "reversibility": "irreversible",
            "exit": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 0.0,
            "error": (
                f"unsupported fastlane lane {ln!r}; "
                f"allowed: {sorted(ALLOWED_LANES)}"
            ),
        }

    reversibility = lane_reversibility(ln)

    cmd = ["fastlane", ln]
    cmd.extend(list(extra_args or []))

    raw = await run_cmd(
        mission_id=mission_id,
        cmd=cmd,
        cwd=None,
        timeout_s=timeout_s,
        require_exit_zero=False,
        workspace_path=workspace_path,
        reversibility_intent=reversibility,
    )

    if _is_missing_exe(raw):
        logger.warning("fastlane CLI not installed — fastlane skipped", lane=ln)
        return {
            "ok": True,
            "skipped": True,
            "lane": ln,
            "reversibility": reversibility,
            "exit": int(raw.get("exit", -1)),
            "stdout_tail": _tail(raw.get("stdout_tail")),
            "stderr_tail": _tail(raw.get("stderr_tail")),
            "duration_s": float(raw.get("duration_s", 0.0)),
            "error": None,
        }

    exit_code = int(raw.get("exit", -1))
    timed_out = bool(raw.get("timed_out"))
    stdout = raw.get("stdout_tail") or ""
    stderr = raw.get("stderr_tail") or ""

    if timed_out:
        return {
            "ok": False,
            "skipped": False,
            "lane": ln,
            "reversibility": reversibility,
            "exit": exit_code,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "duration_s": float(raw.get("duration_s", 0.0)),
            "error": f"fastlane {ln} timed out after {timeout_s}s",
        }

    ok = exit_code == 0
    return {
        "ok": ok,
        "skipped": False,
        "lane": ln,
        "reversibility": reversibility,
        "exit": exit_code,
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
        "duration_s": float(raw.get("duration_s", 0.0)),
        "error": None if ok else f"fastlane {ln} exited {exit_code}",
    }
