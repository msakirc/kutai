"""Wrap ``fastlane <lane>`` — Z5 T3b mobile CI adapter.

Mechanical executor. No LLM. Shells out (via :func:`run_cmd`) to Fastlane,
the open-source mobile build/release automation tool. This is the
**free-first** distribution path the founder picked over EAS Submit: a
Fastlane ``Fastfile`` runs on a GitHub Actions runner (see
:mod:`mr_roboto.gen_mobile_ci`), so there is no per-build cloud charge.

Lanes
-----
Only four lanes are recognised -- they map onto the standard Fastlane
release pipeline:

- ``build``  -- compile the app (``gym`` / ``gradle``). Local, re-runnable.
- ``match``  -- sync code-signing certs/profiles from the encrypted git
               repo. Local; idempotent (re-running re-fetches the same
               material).
- ``pilot``  -- upload an iOS build to TestFlight. Testers receive it.
- ``supply`` -- upload an Android build to the Play internal track.

Reversibility
-------------
Reversibility is **per-lane**, not per-verb:

- ``build`` / ``match`` -> ``full`` -- they touch only local state
  (a build artifact, the keychain) and re-running yields an equivalent
  result.
- ``pilot`` / ``supply`` -> ``irreversible`` -- a binary lands on a real
  store track. Testers/Google/Apple ingest it; the upload cannot be
  cleanly undone (a build is at best expired or withdrawn, never erased).

The dispatcher resolves the tag *before* the verb runs, so the verb body
cannot influence it. :func:`lane_reversibility` derives the tag from the
lane; :func:`mr_roboto.run` feeds it through the standard
``reversibility_override`` payload mechanism. The base
``VERB_REVERSIBILITY["fastlane"]`` entry is the conservative default
(``irreversible``) -- used only when the lane cannot be resolved.

Invocation
----------
``fastlane <lane> [extra_args...]``. If the ``fastlane`` CLI is absent the
run soft-skips for free via :func:`run_cmd`'s missing-exe handling.
"""

from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger
from mr_roboto.run_cmd import run_cmd

logger = get_logger("mr_roboto.fastlane")

# Recognised Fastlane lanes. `build`/`match` are local + re-runnable;
# `pilot`/`supply` push a binary to a real store track.
_ALLOWED_LANES = frozenset({"build", "match", "pilot", "supply"})

# Lanes that push a binary onto a store track -- irreversible.
_IRREVERSIBLE_LANES = frozenset({"pilot", "supply"})

_TAIL_CHARS = 6000
DEFAULT_TIMEOUT_S = 1200.0  # Fastlane builds/uploads are slow


def _tail(text: str | None) -> str:
    text = text or ""
    return text[-_TAIL_CHARS:] if len(text) > _TAIL_CHARS else text


def _is_missing_exe(raw: dict[str, Any]) -> bool:
    err = (raw.get("error") or "").lower()
    return "executable not found" in err or "not found" in err


def lane_reversibility(lane: str | None) -> str:
    """Return the reversibility tag for a Fastlane lane.

    ``build`` / ``match`` -> ``"full"`` (local, re-runnable).
    ``pilot`` / ``supply`` -> ``"irreversible"`` (binary lands on a store
    track).

    An unknown / missing lane returns ``"irreversible"`` -- the
    conservative default, so the dispatcher gates it.
    """
    norm = (lane or "").strip().lower()
    if norm in _IRREVERSIBLE_LANES:
        return "irreversible"
    if norm in _ALLOWED_LANES:
        return "full"
    return "irreversible"


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
        One of ``build`` / ``match`` / ``pilot`` / ``supply``.
    workspace_path:
        Explicit workspace root (the project containing ``fastlane/Fastfile``).
    extra_args:
        Extra argv appended after the lane name.
    timeout_s:
        Hard cap. Default 1200 s.

    Returns
    -------
    dict with keys ``ok, skipped, lane, reversibility, exit, stdout_tail,
    stderr_tail, duration_s, error``.
    """
    norm = (lane or "").strip().lower()
    if norm not in _ALLOWED_LANES:
        return {
            "ok": False,
            "skipped": False,
            "lane": norm,
            "reversibility": "irreversible",
            "exit": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 0.0,
            "error": (
                f"unsupported fastlane lane {norm!r}; "
                f"allowed: {sorted(_ALLOWED_LANES)}"
            ),
        }

    reversibility = lane_reversibility(norm)

    cmd = ["fastlane", norm]
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
        logger.warning("fastlane CLI not installed -- fastlane skipped", lane=norm)
        return {
            "ok": True,
            "skipped": True,
            "lane": norm,
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
            "lane": norm,
            "reversibility": reversibility,
            "exit": exit_code,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "duration_s": float(raw.get("duration_s", 0.0)),
            "error": f"fastlane {norm} timed out after {timeout_s}s",
        }

    ok = exit_code == 0
    return {
        "ok": ok,
        "skipped": False,
        "lane": norm,
        "reversibility": reversibility,
        "exit": exit_code,
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
        "duration_s": float(raw.get("duration_s", 0.0)),
        "error": None if ok else f"fastlane {norm} exited {exit_code}",
    }
